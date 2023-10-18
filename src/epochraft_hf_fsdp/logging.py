from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import colorlog
import torch.distributed as dist
import wandb

from .fsdp import get_local_rank, get_rank, get_world_size


logger = logging.getLogger(__name__)


def setup_file_logger(out_dir: Path) -> None:
    comm_rank = get_rank()
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
    rank = get_rank()

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    stderr_handler = logging.StreamHandler()
    stderr_handler.setFormatter(
        colorlog.ColoredFormatter(
            f"%(log_color)s[%(levelname).1s|%(asctime)s|{rank}|%(name)s]%(reset)s %(message)s"
        )
    )
    stderr_handler.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    root_logger.addHandler(stderr_handler)

    if out_dir:
        setup_file_logger(out_dir)

    logger.info("Save dir: {}".format(out_dir))
    logger.info(f"World={get_world_size()}, Rank={rank}, Local rank={get_local_rank()}")


@dataclass
class WandbConfig:
    url: Optional[str]
    entity: Optional[str]
    project: str
    name: str


def setup_wandb(
    wandb_config: WandbConfig,
    save_dir: Path,
    dump: dict[str, Any],
) -> bool:
    rank = dist.get_rank()
    if rank == 0:
        dump["world_size"] = get_world_size()
        wandb.login(host=wandb_config.url)
        wandb.init(
            entity=wandb_config.entity,
            project=wandb_config.project,
            name=wandb_config.name,
            dir=save_dir,
            config=dump,
        )
        return True
    else:
        return False
