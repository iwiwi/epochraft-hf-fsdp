from __future__ import annotations

import argparse
from typing import Type, TypeVar

from omegaconf import OmegaConf


T = TypeVar("T")


def load_config_from_cli(config_cls: Type[T]) -> T:
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, nargs="+")
    parser.add_argument("-m", "--modify", type=str, nargs="*")
    args = parser.parse_args()

    config = OmegaConf.merge(
        OmegaConf.structured(config_cls),
        *[OmegaConf.load(path) for path in args.config_path],
        OmegaConf.from_cli(args.modify or []),
    )
    return OmegaConf.to_object(config)  # type: ignore
