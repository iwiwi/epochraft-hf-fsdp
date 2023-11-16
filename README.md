# epochraft-hf-fsdp

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/iwiwi/epochraft-hf-fsdp)
[![Checks status](https://github.com/iwiwi/epochraft-hf-fsdp/actions/workflows/checks.yml/badge.svg?branch=main)](https://github.com/iwiwi/epochraft-hf-fsdp/actions)

Simple example of using Epochraft to train HuggingFace transformers models with PyTorch FSDP.


🌟 **News**: We are thrilled to announce the release of two new models:  [Japanese Stable LM Gamma 7B](https://huggingface.co/stabilityai/japanese-stablelm-base-gamma-7b) and [Japanese StableLM 3B](https://huggingface.co/stabilityai/japanese-stablelm-3b-4e1t-base), both trained using our codebase. 


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
