# Legacy Code Archive

This directory contains deprecated code that has been superseded by the joint training pipeline (`scripts/joint_training.py`).

## Status: DEPRECATED - DO NOT USE

These files are kept for historical reference only. They are **not maintained** and may not work with the current codebase.

---

## Legacy Training Scripts

### Single-Task Scripts (Replaced by Joint Training)

- **pfn.py** - Original node classification training script
  - Superseded by: `scripts/joint_training.py` with node classification task enabled
  - Last used: Pre-joint training implementation

- **link_prediction.py** - Original link prediction training script
  - Superseded by: `scripts/joint_training.py` with link prediction task enabled
  - Last used: Pre-joint training implementation

- **graph_classification.py** - Original graph classification training script
  - Superseded by: `scripts/joint_training.py` with graph classification task enabled
  - Last used: Pre-joint training implementation

- **pfn_parallel_ddp.py** - Early DDP implementation for node classification
  - Superseded by: Integrated DDP in `joint_training.py`
  - Last used: Early distributed training experiments

- **multipfn.py** - Multi-task experiment script
  - Superseded by: `scripts/joint_training.py` with multi-task support
  - Last used: Multi-task prototype phase

---

## Legacy Data Loaders

### Experimental Memory Optimization (Not Used)

- **data_unified.py** - Experimental unified data loader
  - Goal: Eliminate duplicate dataset loading between tasks
  - Status: Prototype - not integrated into production pipeline
  - Issues: Complex memory management, compatibility concerns

- **data_graph_cache_aware.py** - Cache-aware graph loading
  - Goal: Skip embedding loading when PCA cache exists
  - Status: Performance experiment - superseded by better caching strategy
  - Issues: Added complexity without sufficient benefit

---

## DDP Infrastructure (Unused)

- **ddp_utils.py** - DDP setup and cleanup utilities
- **ddp_monitor.py** - DDP process monitoring
- **ddp_gpu_monitor.py** - GPU memory monitoring for DDP
  - Status: Not used in joint_training.py (single-GPU only)
  - Only used by: legacy/link_prediction.py
  - Reason for deprecation: DDP implementation had issues and was never successfully used

## Utility Scripts

- **prepare_dataset_for_upload.py** - Dataset preparation utility for uploads
  - Status: Legacy utility, not part of main training pipeline
  - Use case: Preparing datasets for sharing/uploading

## Test/Minimal Scripts

- **minimum.py** - Minimal test script for basic functionality
- **mlp_minimum.py** - MLP baseline test script

---

## Migration to Joint Training

The joint training pipeline (`scripts/joint_training.py`) provides:

- **Unified multi-task training** - Node classification, link prediction, and graph classification in one script
- **Better resource management** - Integrated DDP, GPU monitoring, and memory optimization
- **Comprehensive logging** - Advanced training logger with configurable verbosity
- **Checkpoint management** - Robust checkpoint saving/loading across all tasks
- **Sweep support** - W&B sweep integration for hyperparameter optimization

### How to Use Modern Pipeline

```bash
# Node classification + Link prediction
python scripts/joint_training.py \
  --enable_nc --enable_lp \
  --nc_train_dataset ogbn-arxiv,CS \
  --lp_train_dataset ogbl-collab

# All three tasks
python scripts/joint_training.py \
  --enable_nc --enable_lp --enable_gc \
  --nc_train_dataset ogbn-arxiv \
  --lp_train_dataset ogbl-collab \
  --gc_train_dataset MUTAG,PROTEINS
```

---

## If You Need These Files

If you have a specific reason to reference or use these legacy scripts:

1. **Check if functionality exists in joint_training.py** - Most features have been ported
2. **Consult git history** - Use `git log --follow <filename>` to see evolution
3. **Ask maintainers** - There may be a modern equivalent you're not aware of

---

**Last Updated**: 2025-10-11
**Deprecated Since**: Joint training pipeline implementation
