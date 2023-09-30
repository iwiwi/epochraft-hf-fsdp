from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Optional

import colorlog
import torch.distributed as dist
import wandb


logger = logging.getLogger(__name__)


#
# Logger configuration
#


def setup_file_logger(out_dir: Path) -> None:
    comm_rank = dist.get_rank()
    root_logger = logging.getLogger()

    (out_dir / "log").mkdir(exist_ok=True, parents=True)

    for level, name in [
        (logging.INFO, "info"),
        (logging.DEBUG, "debug"),
        (logging.WARNING, "warning"),
        (logging.ERROR, "error"),
    ]:
        log_path = out_dir / f"log/{name}_rank{comm_rank}.txt"
        file_handler = logging.FileHandler(filename=log_path)
        file_handler.setFormatter(
            logging.Formatter("[%(levelname)1.1s:%(asctime)s:%(pathname)s:%(lineno)d] %(message)s")
        )
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)


def setup_logger(out_dir: Optional[Path]) -> None:
    comm_rank = dist.get_rank()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(
        colorlog.ColoredFormatter(
            f"%(log_color)s[%(levelname).1s|%(asctime)s|{comm_rank}|%(name)s]%(reset)s %(message)s"
        )
    )
    stderr_handler.setLevel(logging.INFO if comm_rank == 0 else logging.WARNING)
    root_logger.addHandler(stderr_handler)

    if out_dir:
        setup_file_logger(out_dir)

    logger.info("Out: {}".format(out_dir))


def setup_wandb(wandb_init_kwargs: dict[str, Any]) -> bool:
    rank = dist.get_rank()
    if rank == 0:
        wandb.login(host="https://stability.wandb.io")
        wandb.init(**wandb_init_kwargs)
        wandb.define_metric("num_tokens")
        wandb.define_metric("loss", step_metric="num_tokens")
        return True
    else:
        return False


#
# Learning rate scheduler
#


class CosineScheduler:
    def __init__(
        self,
        max_lr: float,
        min_lr: float,
        steps: int,
        zerolr_warmup_steps: int,
        linear_warmup_steps: int,
        cooldown_steps: int,
    ) -> None:
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.steps = steps
        self.zerolr_warmup_steps = zerolr_warmup_steps
        self.linear_warmup_steps = linear_warmup_steps
        self.cooldown_steps = cooldown_steps

    def __call__(self, step: int) -> float:
        if step < self.zerolr_warmup_steps:
            return 0.0
        step -= self.zerolr_warmup_steps

        if step < self.linear_warmup_steps:
            return self.max_lr * step / self.linear_warmup_steps
        step -= self.linear_warmup_steps

        cosine_steps = (
            self.steps - self.zerolr_warmup_steps - self.linear_warmup_steps - self.cooldown_steps
        )
        if step < cosine_steps:
            return (
                self.min_lr
                + (self.max_lr - self.min_lr) * (1 + math.cos(step / cosine_steps * math.pi)) / 2
            )
        step -= cosine_steps

        if step < self.cooldown_steps:
            return self.min_lr * (self.cooldown_steps - step) / self.cooldown_steps
        else:
            logger.warning(f"Step is out of range: {step}")
            return 0.0
