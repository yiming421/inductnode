# GILT: Graph In-context Learning Transformer

Official implementation of **GILT: An LLM-Free, Tuning-Free Graph Foundational Model for In-Context Learning** (Under review).

## Overview

GILT is a novel Graph Foundational Model (GFM) that achieves **LLM-free** and **tuning-free** in-context learning on graphs. Unlike existing approaches that rely on Large Language Models or require costly per-graph tuning, GILT reframes few-shot graph learning as a token-based reasoning problem, enabling direct inference on new tasks without any parameter updates.

### Key Features

- **LLM-Free**: Works directly with numerical features without text dependency
- **Tuning-Free**: Adapts to new tasks via in-context learning without gradient updates
- **Multi-Task**: Unified framework for node, link, and graph classification
- **Efficient**: Orders of magnitude faster than tuning-based or LLM-based methods
- **Strong Performance**: State-of-the-art few-shot results across diverse benchmarks

## Architecture

GILT consists of two main components:

1. **Graph-Native Tokenization**: Converts heterogeneous graphs into unified token representations
   - PCA-based feature alignment for arbitrary dimensions
   - Deep linear GCN encoder for structure extraction
   - Prototypical token formulation with asymmetric design

2. **In-Context Reasoning**: Transformer-based reasoning over contextual tokens
   - Two-stage attention mechanism (context refinement + information gathering)
   - Prototypical prediction head for tuning-free classification
   - Causal masking to prevent query leakage

## Installation

```bash
# Create conda environment
conda env create -f env.yml
conda activate gnn
```

## Quick Start

### Training

```bash
# Activate conda environment
conda activate gnn

# Train GILT from scratch
python train.py \
  --enable_nc true \
  --enable_lp true \
  --enable_gc true \
  --epochs 50 \
  --gpu 0
```

### Few-Shot Evaluation

GILT requires no tuning for new tasks. Simply provide a few labeled examples at inference time.

#### Download Pre-trained Checkpoint

```bash
# Install huggingface-hub if not already installed
pip install huggingface-hub

# Download the pre-trained checkpoint
from huggingface_hub import hf_hub_download
checkpoint_path = hf_hub_download(
    repo_id="fdsajkshf/gilt-checkpoint",
    filename="gilt_model.pt",
    cache_dir="./checkpoints"
)
```

Or download directly:
```bash
wget https://huggingface.co/fdsajkshf/gilt-checkpoint/resolve/main/gilt_model.pt -O checkpoints/gilt_model.pt
```

#### Evaluation with Pre-trained Model

```bash
# Multi-task evaluation (all tasks)
python train.py \
  --use_pretrained_model true \
  --load_checkpoint checkpoints/gilt_model.pt \
  --enable_nc true \
  --enable_lp true \
  --enable_gc true \
  --gpu 0
```

The model performs in-context learning without any parameter updates, directly inferring from few-shot examples.

## Project Structure

```
.
├── train.py                      # Main training script with joint NC/LP/GC training
├── env.yml                       # Conda environment specification
├── README.md                     # This file
│
├── src/                          # Source code
│   ├── model.py                  # GILT architecture (GNN, Predictor, Transformer)
│   ├── config.py                 # Configuration and command-line arguments
│   ├── checkpoint_utils.py       # Checkpoint loading/saving utilities
│   │
│   ├── data_nc.py                # Node classification data loading
│   ├── data_lp.py                # Link prediction data loading
│   ├── data_gc.py                # Graph classification data loading
│   ├── data_utils.py             # Common data utilities
│   ├── data_minibatch.py         # Mini-batch data loader
│   ├── dataset_*.py              # Dataset-specific loaders
│   │
│   ├── engine_nc.py              # Node classification training/evaluation
│   ├── engine_lp.py              # Link prediction training/evaluation
│   ├── engine_gc.py              # Graph classification training/evaluation
│   │
│   ├── gpu_utils.py              # GPU management utilities
│   ├── logger.py                 # Training logger
│   └── utils.py                  # General utility functions
│
├── legacy/                       # Legacy code and experiments
```

## Key Arguments

```bash
--model              # GNN backbone: PureGCN_v1, GCN, UnifiedGNN (default: PureGCN_v1)
--hidden             # Hidden dimension (default: 256)
--num_layers         # GNN layers (default: 4)
--transformer_layers # Transformer layers in ICL module (default: 3)
--epochs             # Training epochs (default: 50)
--enable_nc          # Enable node classification task
--enable_lp          # Enable link prediction task
--enable_gc          # Enable graph classification task
```

See `src/config.py` for the complete list of arguments.