from __future__ import annotations

import math
import time
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import torch
import wandb
from epochraft import CheckpointableDataset, CheckpointableIterator, Sample
from epochraft_hf_fsdp import fsdp, lr_schedulers
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
from torch.distributed.fsdp import ShardingStrategy  # type: ignore
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


LogValue = Union[float, torch.Tensor, None]
LogDict = Dict[str, LogValue]

logger = getLogger(__name__)


def get_num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass
class TrainerConfig:
    load_dir: Optional[Path]  # Path to a checkpoint directory
    save_dir: Path

    model: str
    transformer_blocks_path: str
    fsdp_sharding_strategy: ShardingStrategy
    fsdp_cpu_offload: bool

    seq_len: int
    global_batch_size: int
    micro_batch_size: int
    steps: int

    zerolr_warmup_steps: int
    linear_warmup_steps: int
    cooldown_steps: int
    max_lr: float
    min_lr: float
    weight_decay: float
    grad_clip: float
    beta1: float
    beta2: float
    eps: float

    val_steps: int

    ckpt_steps: int


# This is the state that is saved to the checkpoint in addition to the model and optimizer states.
@dataclass
class TrainerState:
    step: int
    next_batch: Optional[Sample]


class Trainer:
    def __init__(
        self,
        config: TrainerConfig,
        model: FSDP,
        num_params: int,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: lr_schedulers.CosineScheduler,
        tokenizer: AutoTokenizer,
        train_dataset: CheckpointableDataset,
        val_datasets: Sequence[tuple[str, CheckpointableDataset]],
        state: TrainerState,
        train_iter_state_dict: Optional[torch.Tensor] = None,
    ) -> None:
        self.config = config
        self.model = model
        self.num_params = num_params
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_datasets = val_datasets
        self.state = state
        self.train_iter_state_dict = train_iter_state_dict

        world_size = fsdp.get_world_size()
        assert config.global_batch_size % (config.micro_batch_size * world_size) == 0
        self.grad_acc_steps = config.global_batch_size // config.micro_batch_size // world_size

        self.last_iter_completion_time: Optional[float] = None  # To measure iteration times

    @classmethod
    def from_config(
        cls,
        config: TrainerConfig,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: CheckpointableDataset,
        val_datasets: Sequence[tuple[str, CheckpointableDataset]],
    ) -> Trainer:
        rank = fsdp.get_rank()

        # Model
        model = AutoModelForCausalLM.from_pretrained(
            config.model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_flash_attention_2=True,
            use_cache=False,
        )
        num_params = get_num_params(model)  # Need to do this before FSDP
        layer_cls = fsdp.get_transformer_block_class(model, config.transformer_blocks_path)
        logger.info(f"Model class: {type(model)}, block class: {layer_cls}, params: {num_params}")

        if config.load_dir:
            ckpt_path = config.load_dir / "model.pth"
            logger.info(f"Loading model state_dict from: {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state_dict)
            del state_dict

        model = fsdp.setup_fsdp(
            model, config.fsdp_sharding_strategy, layer_cls, cpu_offload=config.fsdp_cpu_offload
        )
        fsdp.apply_fsdp_checkpointing(model, layer_cls)

        # Optimization
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.0,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
        lr_scheduler = lr_schedulers.CosineScheduler(
            config.max_lr,
            config.min_lr,
            config.steps,
            config.zerolr_warmup_steps,
            config.linear_warmup_steps,
            config.cooldown_steps,
        )

        if config.load_dir:
            ckpt_path = config.load_dir / "optimizer.pth"
            logger.info(f"Loading optimizer state_dict from: {ckpt_path}")
            fsdp.load_optimizer_state_dict(model, optimizer, ckpt_path)

        # Datasets
        if config.load_dir:
            rank = fsdp.get_rank()
            ckpt_path = config.load_dir / f"train_iter_rank{rank:04d}.pth"
            logger.info("Loading train_iter state_dict from: {ckpt_path}")
            train_iter_state_dict = torch.load(ckpt_path)
        else:
            train_iter_state_dict = None

        # State
        if config.load_dir:
            ckpt_path = config.load_dir / "trainer.pth"
            logger.info(f"Loading trainer state from: {ckpt_path}")
            state_dict = torch.load(ckpt_path)
            state: TrainerState = state_dict
        else:
            state = TrainerState(step=0, next_batch=None)

        return cls(
            config=config,
            model=model,
            num_params=num_params,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            val_datasets=val_datasets,
            state=state,
            train_iter_state_dict=train_iter_state_dict,
        )

    def save_checkpoint(self, train_iter: CheckpointableIterator) -> None:
        rank = fsdp.get_rank()
        path = self.config.save_dir / f"ckpt/step_{self.state.step:07d}/"
        path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving checkpoint to: {path}")
        torch.distributed.barrier()

        fsdp.save_model_state_dict(self.model, path / "model.pth")
        fsdp.save_optimizer_state_dict(self.model, self.optimizer, path / "optimizer.pth")
        torch.save(self.state, path / "trainer.pth")
        torch.save(train_iter.state_dict(), path / f"train_iter_rank{rank:04d}.pth")

        torch.distributed.barrier()
        logger.info(f"Saved checkpoint to: {path}")

    def train(self) -> None:
        rank = fsdp.get_rank()

        with self.train_dataset.iter(self.train_iter_state_dict) as train_iter:
            if self.state.next_batch is None:
                logger.info("Preparing the first batch")
                self.state.next_batch = next(train_iter)

            logger.info("Waiting for other ranks")
            torch.distributed.barrier()
            logger.info("Training start")
            self.last_iter_completion_time = None

            tbar = tqdm(
                total=self.config.steps,
                disable=rank != 0,
                position=0,
                leave=True,
            )
            tbar.update(self.state.step)
            train_loss = math.nan

            while self.state.step <= self.config.steps:
                # To be logged to wandb
                trained_tokens = (
                    self.state.step * self.config.global_batch_size * self.config.seq_len
                )
                log_dict: LogDict = {
                    "step": self.state.step,
                    "train/tokens": trained_tokens,
                }

                # Checkpointing
                if self.state.step % self.config.ckpt_steps == 0 and self.state.step > 0:
                    self.save_checkpoint(train_iter)
                    self.last_iter_completion_time = None  # Checkpointing breaks iteration times

                # Validation
                if self.state.step % self.config.val_steps == 0:
                    scores = self.validate()
                    log_dict.update({f"val/{key}": value for key, value in scores.items()})
                    self.last_iter_completion_time = None  # Validation breaks iteration times

                # Training
                if self.state.step < self.config.steps:
                    log_dict.update(self.train_step(train_iter))
                    train_loss = float(log_dict["train/loss"] or 0.0)

                # Logging
                if rank == 0:
                    flops_per_sec = log_dict.get("train/flops_per_sec", None) or 0.0
                    tflops_per_sec = flops_per_sec / 1e12
                    tbar.set_description(
                        f"[ tokens={trained_tokens:.3e} | loss={train_loss:.3f} | "
                        f"tflops={tflops_per_sec:.1f} ]"
                    )
                    tbar.update()
                    wandb.log(log_dict, step=self.state.step)

                self.state.step += 1

            self.train_iter_state_dict = train_iter.state_dict()

        self.save_hf()

    def train_step(self, train_iter: CheckpointableIterator) -> LogDict:
        lr = self.lr_scheduler(self.state.step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        self.model.train()
        loss_sum = torch.zeros((), dtype=torch.float32, device=self.model.device)
        for _ in range(self.grad_acc_steps):
            batch = self.state.next_batch
            assert batch
            input_ids = batch["input_ids"].to(self.model.device)

            out = self.model(input_ids=input_ids, labels=input_ids, return_dict=False)
            loss = out[0]
            loss_sum += loss.detach()
            loss.backward()

            self.state.next_batch = next(train_iter)
            del loss

        self.model.clip_grad_norm_(self.config.grad_clip)
        self.optimizer.step()
        self.optimizer.zero_grad()

        torch.distributed.all_reduce(loss_sum, torch.distributed.ReduceOp.SUM)
        loss_avg = loss_sum / self.config.global_batch_size

        # FLOPs
        time_now = time.time()
        if self.last_iter_completion_time is not None:
            sec_per_iter = time_now - self.last_iter_completion_time
            flops_per_iter = (
                6
                * self.config.micro_batch_size
                * self.grad_acc_steps
                * self.config.seq_len
                * self.num_params
            )
            flops_per_sec = flops_per_iter / sec_per_iter
        else:
            sec_per_iter = None
            flops_per_sec = None
        self.last_iter_completion_time = time_now

        return {
            "train/lr": lr,
            "train/loss": loss_avg,
            "train/sec_per_iter": sec_per_iter,
            "train/flops_per_sec": flops_per_sec,
        }

    @torch.no_grad()
    def validate(self) -> LogDict:
        rank = fsdp.get_rank()
        logger.info("Validation starting")
        self.model.eval()
        scores = {}
        for dataset_name, dataset in tqdm(self.val_datasets, disable=rank != 0):
            # (sum, cnt)
            loss_agg = torch.zeros(2, dtype=torch.float32, device=self.model.device)

            with iter(dataset) as it:
                for batch in it:
                    input_ids = batch["input_ids"].to(self.model.device)
                    out = self.model(input_ids=input_ids, labels=input_ids, return_dict=False)
                    loss = out[0]
                    loss_agg[0] += loss
                    loss_agg[1] += 1.0

            torch.distributed.all_reduce(loss_agg, torch.distributed.ReduceOp.SUM)
            loss = loss_agg[0] / loss_agg[1]
            scores[dataset_name] = loss.item()
        logger.info(f"Validation done: {scores}")

        return scores

    def save_hf(self) -> None:
        logger.info("Saving HF model")
        state_dict = fsdp.get_model_state_dict(self.model)

        if fsdp.get_rank() == 0:
            out_path = self.config.save_dir / "hf"
            out_path.mkdir(parents=True, exist_ok=True)

            hf_model = AutoModelForCausalLM.from_pretrained(
                self.config.model, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            hf_model.load_state_dict(state_dict)
            hf_model.save_pretrained(out_path, safe_serialization=True)

            self.tokenizer.save_pretrained(out_path)

        torch.distributed.barrier()
        logger.info("Saved HF model")
