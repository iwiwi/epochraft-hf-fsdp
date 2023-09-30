# epochraft-hf-fsdp
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
