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
**Requirements**

Python 3.11
PyTorch >= 2.0
transformers >= 4.0.0 (for tokenization and datasets)
einops >= 0.6.0 (for easy-to-read einsums)
wandb >= 0.12.0 (for training MLOps)
See `pyproject.toml` for complete dependency list


 

