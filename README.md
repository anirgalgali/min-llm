# min-llm

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A minimal and modular toolbox for implementing transformer-based language models.

## Overview

**min-llm** is a **from scratch** implementation of transformer-based language models in 
PyTorch. I built this library primarily to understand how LLMs work form the bottom-up by 
implementing the core model architecture (embeddings, self-attention, feed-forward layers 
and normalization) and the training utilities from first principles.

**Key features**
 - Decoder-only (GPT) and encoder-only (BERT) transformer models. 
 - Modern LLM components : supports Rotary positional embeddings (RoPE), gated MLP 
   architectures (SwiGLU, GeGLU, ...), pre and post-norm architectures (Layer/RMSNorm)
 - Advanced Sampling : Temperature controlled softmax and Top-p (nucleus) sampling
 - Training infrastructure : Mixed precision, Experiment tracking / MLOps with wandb

## Installation

Clone the repository and install in your preferred virtual environment

```bash
git clone https://github.com/anirgalgali/min-llm.git
cd min-llm
pip install -e .
```

For development (including testing dependencies):
```bash
pip install -e ".[dev]"
```
**Requirements:**

- Python 3.11
- PyTorch >= 2.0
- transformers >= 4.0.0 (for tokenization and datasets)
- numpy >= 2.0.0 (for data handling)
- einops >= 0.6.0 (for easy-to-read einsums)
- wandb >= 0.12.0 (for training MLOps)
- See `pyproject.toml` for complete dependency list

## Quick Start

### Training on TinyStories
```bash
python scripts/train_tinystories.py --num_iters 20000 --batch_size 64 --weight_decay 0.1 --lr 0.001
```

This will save checkpoints to `experiments/tinystories/<run_id>/checkpoints/`.

### Generating Text
```python
import torch
from transformers import AutoTokenizer
from mini_llm.models.causal_lm import TransformerLM
from mini_llm.decoding import TextDecoder
from mini_llm.config import DecoderLMConfig

# Load trained model
config = DecoderLMConfig()  # Uses default config from training
model = TransformerLM(config)
checkpoint = torch.load("experiments/tinystories/<run_id>/checkpoints/best_model.pt")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Generate text
tokenizer = AutoTokenizer.from_pretrained("gpt2")
decoder = TextDecoder(temperature=0.8, top_p=0.9, tokenizer=tokenizer)
output = decoder.generate(model, prompt="Once upon a time", max_tokens=100)
print(output)
```

## Features

### Model Architectures
- **Decoder-only Transformer** (GPT-style): Causal language modeling
- **Encoder-only Transformer** (BERT-style): Bidirectional encoding for classification tasks

### Modern Components
- **Positional Encodings**: Rotary Position Embeddings (RoPE), learned absolute embeddings
- **Normalization**: RMSNorm, LayerNorm
- **Activations**: SwiGLU feed-forward layers, GELU activations
- **Attention**: Multi-head self-attention with configurable heads and optional causal masking

### Text Generation
- Top-p (nucleus) sampling
- Temperature-based sampling

### Training Infrastructure
- Configurable dtype support (FP32/FP16/BF16) per layer
- Gradient clipping
- Checkpoint saving and resumption
- Cosine learning rate schedule with linear warmup
- Weights & Biases integration for experiment tracking

## Project Structure
```
min-llm/
├── min_llm/                    # Core library
│   ├── models/
│   │   ├── causal_llm.py      # Decoder-only transformer (GPT-style)
│   │   └── distilbert.py      # Encoder-only transformer (BERT-style)
│   ├── layers.py              # Building blocks: embeddings, linear layers, normalization, MLP,...
│   ├── blocks.py              # Transformer blocks: TrnasformerBlock, Decoder, Encoder
│   ├── functional.py          # Low-level operations: softmax, dot-product attention, etc.
│   ├── config.py              # Configuration dataclasses
│   ├── train.py               # Training loop and Trainer class
│   ├── decoding.py            # Text generation with sampling strategies
│   ├── data.py                # Dataset loading and processing
│   ├── tokenize_dataset.py    # Dataset tokenization utilities
│   └── utils.py               # Helper functions (to map state dict to/from HuggingFace)
├── scripts/                    # Training scripts
│   └── train_tinystories.py   # TinyStories training script
├── experiments/                # Training runs and checkpoints
│   └── <dataset>/<run_id>/    # Organized by dataset and W&B run ID
├── data/                       # Raw and processed datasets
├── notebooks/                  # Jupyter notebooks for demos and analysis
└── tests/                      # Unit tests
```


 

