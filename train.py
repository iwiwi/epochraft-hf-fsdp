from __future__ import annotations

import dataclasses
from typing import Optional

from epochraft_hf_fsdp import config as config_module
from epochraft_hf_fsdp import fsdp, logging, pretraining_data
from epochraft_hf_fsdp.trainer import Trainer, TrainerConfig
from transformers import AutoTokenizer


@dataclasses.dataclass
class Config:
    wandb: logging.WandbConfig
    trainer: TrainerConfig

    tokenizer: Optional[str]
    train_dataset: list[pretraining_data.DataSource]
    val_dataset: list[pretraining_data.DataSource]
    val_samples: int


def main() -> None:
    config = config_module.load_config_from_cli(Config)
    fsdp.init_process_group()
    logging.setup_logger(config.trainer.save_dir)
    logging.setup_wandb(config.wandb, config.trainer.save_dir, dataclasses.asdict(config))

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer or config.trainer.model)
    train_dataset = pretraining_data.construct_training_dataset(
        sources=config.train_dataset,
        tokenizer=tokenizer,
        seq_len=config.trainer.seq_len,
        dp_rank=fsdp.get_rank(),
        dp_world=fsdp.get_world_size(),
        micro_batch_size=config.trainer.micro_batch_size,
    )
    val_datasets = pretraining_data.construct_val_dataset(
        config.val_dataset,
        tokenizer,
        config.trainer.seq_len,
        dp_rank=fsdp.get_rank(),
        dp_world=fsdp.get_world_size(),
        micro_batch_size=config.trainer.micro_batch_size,
        global_val_samples=config.val_samples,
    )

    trainer = Trainer.from_config(config.trainer, tokenizer, train_dataset, val_datasets)
    trainer.train()


if __name__ == "__main__":
    main()
