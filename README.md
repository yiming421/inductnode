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

#### Evaluation with a Pre-trained Model

If a checkpoint is available, pass its path with `--load_checkpoint`.

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
│   ├── config.py                 # Command-line configuration
│   ├── model.py                  # Main GNN/PFN model components
│   ├── checkpoint_utils.py       # Checkpoint loading/saving utilities
│   ├── data_*.py                 # Data loading and preprocessing modules
│   ├── engine_*.py               # Task training/evaluation engines
│   ├── graphpfn/                 # GraphPFN components
│   └── taglas_lite/              # Lightweight TAGLAS integration
│
├── tests/                        # Regression and correctness tests
├── legacy/                       # Deprecated single-task/experimental code
└── scripts/                      # Auxiliary debugging and data-preparation scripts
```

## Key Arguments

```bash
--model              # GNN backbone: PureGCN_v1, GCN, UnifiedGNN, GraphGPS, FAGCN, SIGN (default: PureGCN_v1)
--hidden             # Hidden dimension (default: 128)
--num_layers         # GNN layers (default: 4)
--transformer_layers # Transformer layers in ICL module (default: 3)
--epochs             # Training epochs (default: 50)
--enable_nc          # Enable node classification task
--enable_lp          # Enable link prediction task
--enable_gc          # Enable graph classification task
```

See `src/config.py` for the complete list of arguments.
