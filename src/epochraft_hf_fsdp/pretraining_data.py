from __future__ import annotations

import glob
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from logging import getLogger
from typing import Any, List, Optional, Union

from epochraft import CheckpointableDataset, Sample, braceexpand, interleave_datasets
from joblib import Memory
from transformers import PreTrainedTokenizerBase


logger = getLogger(__name__)
memory = Memory("./cache", verbose=1)


@dataclass
class DataSource:
    name: str

    # OmegaConf does not support Union :(
    url: Optional[str] = None
    urls: Optional[List[str]] = None

    weight: float = 1.0
    n_active_shards: int = 4
    n_standby_shards: int = 1


# Retrieving all files in a directory in S3 is very slow, so we cache the results
@memory.cache
def enumerate_jsonl_files_in_directory_s3(base_url: str) -> list[str]:
    cmd = ["aws", "s3", "ls", base_url]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    urls = [
        f"{base_url}{line.split()[-1]}"
        for line in result.stdout.splitlines()
        if line.endswith(".jsonl")
    ]
    return urls


def enumerate_jsonl_files_in_directory(base_url: str) -> list[str]:
    if base_url.startswith("s3://"):
        return enumerate_jsonl_files_in_directory_s3(base_url)  # type: ignore
    else:
        return glob.glob(f"{base_url}/**/*.jsonl", recursive=True)


def expand_urls(urls: Union[str, list[str]]) -> list[str]:
    if isinstance(urls, str):
        in_urls = [urls]
    else:
        in_urls = urls

    out_urls = []
    for in_url in in_urls:
        for expanded_url in braceexpand(in_url):
            if expanded_url.endswith("/"):
                out_urls.extend(enumerate_jsonl_files_in_directory(expanded_url))
            else:
                out_urls.append(expanded_url)

    return out_urls


def get_shards_and_stride(
    urls: list[str], dp_rank: int, dp_world: int
) -> tuple[list[str], int, int]:
    rank_to_urls: list[list[str]] = [[] for _ in range(dp_world)]
    url_to_ranks: defaultdict[str, list[int]] = defaultdict(list)

    for i in range(max(dp_world, len(urls))):
        rank = i % dp_world
        url = urls[i % len(urls)]
        rank_to_urls[rank].append(url)
        url_to_ranks[url].append(rank)

    my_urls = rank_to_urls[dp_rank]
    assert len(my_urls) > 0
    url = my_urls[0]
    my_stride_interval = len(url_to_ranks[url])
    my_stride_offset = url_to_ranks[url].index(dp_rank)

    return my_urls, my_stride_interval, my_stride_offset


def construct_dataset_from_source(
    source: DataSource,
    dp_rank: int,
    dp_world: int,
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
    repeat: bool,
    shuffle: bool,
) -> CheckpointableDataset:
    logger.info(f"Preparing dataset: {source.name}")

    if not source.url and not source.urls:
        raise ValueError("Either url or urls must be specified")
    if source.url and source.urls:
        raise ValueError("Only one of url and urls must be specified")

    if source.url:
        global_urls = [source.url]
    else:
        assert source.urls
        global_urls = source.urls
    global_urls = sum((expand_urls(url) for url in global_urls), [])

    my_urls, my_stride_interval, my_stride_offset = get_shards_and_stride(
        global_urls, dp_rank, dp_world
    )
    logger.info(
        f"Shard assignment: {source.name}, shards={len(global_urls)}, "
        f"world={dp_world}, rank={dp_rank} -> "
        f"shards={len(my_urls)}, interval={my_stride_interval}, offset={my_stride_offset}"
    )

    def _add_name_fn(sample: dict[str, Any]) -> dict[str, Any]:
        sample["source"] = source.name
        return sample

    if my_urls == ["dummy"]:
        logger.info("Using dummy dataset")
        dataset = (
            CheckpointableDataset.from_sequence(
                [
                    {"input_ids": list(range(100))},
                ],
                repeat=repeat,
                shuffle=shuffle,
            )
            .concat_chunk(seq_len)
            .map(_add_name_fn)
        )
    else:
        dataset = (
            CheckpointableDataset.from_files(
                my_urls,
                repeat=repeat,
                shuffle_shards=shuffle,
                format="jsonl",
                n_active_shards=source.n_active_shards,
                n_standby_shards=source.n_standby_shards,
                n_prefetch_samples=max(10, my_stride_offset * 4),
            )
            .stride(my_stride_interval, my_stride_offset)
            .tokenize(tokenizer, max_workers=1, executor_type="thread")
            .ensure_bos_eos(tokenizer)
            .concat_chunk(seq_len)
            .map(_add_name_fn)
        )

    if shuffle:
        dataset = dataset.shuffle(100)

    def _add_labels(sample: Sample) -> Sample:
        sample["labels"] = sample["input_ids"]
        return sample

    dataset = dataset.map(_add_labels)

    return dataset


def construct_training_dataset(
    sources: list[DataSource],
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
    dp_rank: int,
    dp_world: int,
    micro_batch_size: int,
) -> CheckpointableDataset:
    datasets = []
    weights = []
    for source in sources:
        datasets.append(
            construct_dataset_from_source(
                source, dp_rank, dp_world, tokenizer, seq_len, repeat=True, shuffle=True
            )
        )
        weights.append(source.weight)

    return interleave_datasets(datasets, weights).batch(micro_batch_size).shuffle(100)


def construct_val_dataset(
    sources: list[DataSource],
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
    dp_rank: int,
    dp_world: int,
    micro_batch_size: int,
    global_val_samples: int,
) -> list[tuple[str, CheckpointableDataset]]:
    global_val_batches = global_val_samples // micro_batch_size

    if global_val_batches >= dp_world:
        assert global_val_batches % dp_world == 0
        stride_interval = dp_world
        stride_offset = dp_rank
    else:
        assert dp_world % global_val_batches == 0
        stride_interval = global_val_batches
        stride_offset = dp_rank % global_val_batches

    datasets = []
    for source in sources:
        # dp_rank=0, dp_world=1
        dataset = construct_dataset_from_source(
            source,
            0,
            1,
            tokenizer,
            seq_len,
            repeat=True,
            shuffle=False,
        )
        dataset = (
            dataset.batch(micro_batch_size)
            .take(global_val_batches)
            .stride(stride_interval, stride_offset)
        )
        datasets.append((source.name, dataset))

    return datasets
