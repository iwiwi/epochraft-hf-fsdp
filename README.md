# epochraft-hf-fsdp
Example of using Epochraft to train HuggingFace transformers models with PyTorch FSDP 


## Quick start

```
pip install -e .
python train.py config.yaml
```


## Development

```
pip install -e .[development]
mypy .; black .; flake8 .; isort .
```
