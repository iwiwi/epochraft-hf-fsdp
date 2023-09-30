from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Dict, Optional, Union

import torch
import wandb
from epochraft import CheckpointableIterator, Sample
from epochraft_hf_fsdp import data, fsdp, utils
from omegaconf import OmegaConf
from torch.distributed.fsdp import ShardingStrategy  # type: ignore
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


LogValue = Union[float, torch.Tensor]
LogDict = Dict[str, LogValue]

logger = getLogger(__name__)


@dataclass
class Config:
    out_dir: Path
    wandb_url: Optional[str]
    wandb_entity: Optional[str]
    wandb_project: str
    wandb_name: str

    model: str
    tokenizer: str
    transformer_blocks_path: str
    fsdp_sharding_strategy: ShardingStrategy

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

    train_dataset: list[data.DataSource]
    val_dataset: list[data.DataSource]

    val_samples: int
    val_steps: int

    ckpt_steps: int

    @staticmethod
    def from_cli() -> Config:
        parser = argparse.ArgumentParser()
        parser.add_argument("config_path", type=str, nargs="+")
        parser.add_argument("-m", "--modify", type=str, nargs="*")
        args = parser.parse_args()

        config = OmegaConf.merge(
            OmegaConf.structured(Config),
            *[OmegaConf.load(path) for path in args.config_path],
            OmegaConf.from_cli(args.modify or []),
        )
        OmegaConf.set_readonly(config, True)

        return config  # type: ignore


# This is the state that is saved to the checkpoint in addition to the model and optimizer states.
@dataclass
class TrainerState:
    step: int
    next_batch: Optional[Sample]


class Trainer:
    def __init__(self, config: Config) -> None:
        world_size = fsdp.get_world_size()
        rank = fsdp.get_rank()

        self.config = config

        # Model
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
        model = AutoModelForCausalLM.from_pretrained(
            config.model, torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        layer_cls = fsdp.get_transformer_block_class(model, config.transformer_blocks_path)
        logger.info(f"Model class: {type(model)}, block class: {layer_cls}")
        model = fsdp.setup_fsdp(model, config.fsdp_sharding_strategy, layer_cls)
        fsdp.apply_fsdp_checkpointing(model, layer_cls)
        self.model = model

        # Optimization
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.0,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
        self.lr_scheduler = utils.CosineScheduler(
            config.max_lr,
            config.min_lr,
            config.steps,
            config.zerolr_warmup_steps,
            config.linear_warmup_steps,
            config.cooldown_steps,
        )
        assert config.global_batch_size % (config.micro_batch_size * world_size) == 0
        self.grad_acc_steps = config.global_batch_size // config.micro_batch_size // world_size

        # Datasets
        self.train_dataset = data.construct_training_dataset(
            config.train_dataset,
            self.tokenizer,
            config.seq_len,
            rank,
            world_size,
            config.micro_batch_size,
        )
        self.val_datasets = data.construct_val_dataset(
            config.val_dataset,
            self.tokenizer,
            config.seq_len,
            rank,
            world_size,
            config.micro_batch_size,
            config.val_samples,
        )

        self.state = TrainerState(step=0, next_batch=None)

    def load_checkpoint(self, path: Path) -> None:
        # TODO: implement this when necessary
        pass

    def save_checkpoint(self, train_iter: CheckpointableIterator) -> None:
        rank = fsdp.get_rank()
        path = self.config.out_dir / f"ckpt/step_{self.state.step:07d}/"
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
        tbar = tqdm(
            total=self.config.steps,
            disable=rank != 0,
            position=0,
            leave=True,
        )
        tbar.update(self.state.step)
        train_loss = math.nan

        with iter(self.train_dataset) as train_iter:
            if self.state.next_batch is None:
                logger.info("Preparing the first batch")
                self.state.next_batch = next(train_iter)

            logger.info("Waiting for other ranks")
            torch.distributed.barrier()
            logger.info("Training start")

            while self.state.step <= self.config.steps:
                # To be logged to wandb
                trained_tokens = (
                    self.state.step * self.config.global_batch_size * self.config.seq_len
                )
                log_dict: LogDict = {
                    "step": self.state.step,
                    "train/tokens": trained_tokens,
                }

                # Validation
                if self.state.step % self.config.val_steps == 0:
                    scores = self.validate()
                    log_dict.update({f"val/{key}": value for key, value in scores.items()})

                # Checkpointing
                if self.state.step % self.config.ckpt_steps == 0 and self.state.step > 0:
                    self.save_checkpoint(train_iter)

                # Training
                if self.state.step < self.config.steps:
                    log_dict.update(self.train_step(train_iter))
                    train_loss = float(log_dict["train/loss"])

                # Logging
                if rank == 0:
                    tbar.set_description(
                        f"[ tokens {trained_tokens:.3e} | loss {train_loss:.3f} ]"
                    )
                    tbar.update()
                    wandb.log(log_dict)

                self.state.step += 1

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

        return {
            "train/lr": lr,
            "train/loss": loss_avg,
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
            out_path = self.config.out_dir / "hf"
            out_path.mkdir(parents=True, exist_ok=True)

            hf_model = AutoModelForCausalLM.from_pretrained(
                self.config.model, torch_dtype=torch.bfloat16, trust_remote_code=True
            )
            hf_model.load_state_dict(state_dict)
            hf_model.save_pretrained(out_path, safe_serialization=True)

            self.tokenizer.save_pretrained(out_path)

        torch.distributed.barrier()
        logger.info("Saved HF model")


def main() -> None:
    world_size = fsdp.get_world_size()
    rank = fsdp.get_rank()
    local_rank = fsdp.get_local_rank()
    print(f"World={world_size} Rank={rank}, Local rank={local_rank}", flush=True)

    config = Config.from_cli()
    fsdp.init_process_group()
    utils.setup_logger(config.out_dir)
    logger.info(f"World={world_size}, Rank={rank}, Local rank={local_rank}")
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

    # Wandb
    if rank == 0:
        wandb_config_dump = {
            "world_size": world_size,
            **OmegaConf.to_container(config, resolve=True),  # type: ignore
        }

        wandb.login(host=config.wandb_url)
        wandb.init(
            entity=config.wandb_entity,
            project=config.wandb_project,
            name=config.wandb_name,
            dir=config.out_dir,
            config=wandb_config_dump,
        )

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
