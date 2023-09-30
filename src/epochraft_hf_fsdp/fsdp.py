from __future__ import annotations

import functools
import os
import tempfile
from functools import partial
from pathlib import Path
from typing import Type

import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import FullStateDictConfig  # type: ignore
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP  # type: ignore
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType  # type: ignore
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def get_rank() -> int:
    return int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", 0)))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", 0)))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", 1)))


def init_process_group() -> None:
    if os.environ.get("WORLD_SIZE", None):
        torch.distributed.init_process_group(rank=get_rank(), world_size=get_world_size())
    else:
        temp_file = tempfile.NamedTemporaryFile()
        torch.distributed.init_process_group(
            init_method=f"file://{temp_file.name}", rank=0, world_size=1
        )


def get_transformer_block_class(
    model: nn.Module, transformer_blocks_path: str = "model.layers"
) -> Type[nn.Module]:
    path = transformer_blocks_path.split(".")
    obj = model
    for name in path:
        obj = getattr(obj, name)
    obj = obj[0]  # type: ignore
    return type(obj)


def setup_fsdp(
    model: nn.Module,
    sharding_strategy: ShardingStrategy,
    transformer_block_class: Type[nn.Module],
) -> FSDP:
    local_rank = get_local_rank()

    model = FSDP(
        model,
        device_id=local_rank,
        sharding_strategy=sharding_strategy,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={transformer_block_class},
        ),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        ),
        forward_prefetch=True,
        limit_all_gathers=True,
    )

    return model


def apply_fsdp_checkpointing(model: FSDP, transformer_block_class: Type[nn.Module]) -> None:
    non_reentrant_wrapper = partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    def check_fn(submodule: nn.Module) -> bool:
        return isinstance(submodule, transformer_block_class)

    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )


def get_model_state_dict(model: FSDP) -> dict[str, torch.Tensor]:
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
    ):
        state_dict = model.state_dict()

    return state_dict


def save_model_state_dict(model: FSDP, path: Path) -> None:
    # TODO: explore SHARDED_STATE_DICT for efficiency
    state_dict = get_model_state_dict(model)
    if get_rank() == 0:
        torch.save(state_dict, path)


def save_optimizer_state_dict(model: FSDP, optimizer: torch.optim.Optimizer, path: Path) -> None:
    # TODO: explore SHARDED_STATE_DICT for efficiency

    state_dict = FSDP.full_optim_state_dict(model, optimizer)
    if get_rank() == 0:
        torch.save(state_dict, path)
