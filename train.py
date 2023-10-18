from __future__ import annotations

import dataclasses

from epochraft_hf_fsdp import fsdp, logging
from epochraft_hf_fsdp.trainer import Trainer, TrainerConfig


def main() -> None:
    config = TrainerConfig.from_cli()

    fsdp.init_process_group()
    logging.setup_logger(config.save_dir)
    logging.setup_wandb(
        config.wandb_url,
        config.wandb_entity,
        config.wandb_project,
        config.wandb_name,
        config.save_dir,
        dataclasses.asdict(config),
    )

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
