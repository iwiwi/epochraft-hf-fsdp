# epochraft-hf-fsdp

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/iwiwi/epochraft-hf-fsdp)
[![Checks status](https://github.com/iwiwi/epochraft-hf-fsdp/actions/workflows/checks.yml/badge.svg?branch=main)](https://github.com/iwiwi/epochraft-hf-fsdp/actions)

Example of using Epochraft to train HuggingFace transformers models with PyTorch FSDP 


## Quick start

```bash
pip install -e .
python train.py gpt2_testrun.yaml  # 1 GPU
torchrun --nproc-per-node=8 train.py gpt2_testrun.yaml  # 8 GPUs
```


## Development

```bash
pip install -e .[development]
mypy .; black .; flake8 .; isort .
```
