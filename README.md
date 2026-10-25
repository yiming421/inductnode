# GILT

Implementation of **GILT: LLM-Free, Tuning-Free Graph In-Context Learning**, a shared model for node classification, link prediction, and graph classification.

## Main results

Results reported in the paper (%; mean ± standard deviation).

### Node classification — Table 2 (Accuracy)

| Model | Cora 1-shot | Cora 5-shot | Citeseer 1-shot | Citeseer 5-shot | Pubmed 1-shot | Pubmed 5-shot | WikiCS 1-shot | WikiCS 5-shot | Avg. 1-shot | Avg. 5-shot |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP | 23.77 ± 2.26 | 43.00 ± 1.78 | 26.57 ± 4.65 | 40.80 ± 1.93 | 43.60 ± 9.63 | 57.93 ± 10.47 | 26.05 ± 1.97 | 52.49 ± 1.83 | 30.00 | 48.56 |
| GCN | 38.57 ± 10.55 | 70.20 ± 1.73 | 27.57 ± 2.00 | 50.87 ± 3.49 | 52.90 ± 3.55 | 73.87 ± 7.52 | 40.70 ± 1.09 | 64.31 ± 3.75 | 39.94 | 64.81 |
| GAT | 41.33 ± 12.89 | 69.33 ± 5.51 | 30.13 ± 1.68 | 47.47 ± 3.94 | 50.50 ± 2.61 | 72.03 ± 8.48 | 36.67 ± 3.30 | 67.25 ± 3.70 | 39.66 | 64.02 |
| DGI | 36.87 ± 6.52 | 66.87 ± 4.43 | 27.90 ± 7.73 | 54.00 ± 3.52 | 52.27 ± 11.86 | 70.57 ± 2.75 | 33.58 ± 11.37 | 64.93 ± 0.19 | 37.65 | 64.09 |
| GraphCL | 36.77 ± 7.61 | 67.23 ± 3.87 | 28.37 ± 7.06 | 52.97 ± 3.71 | 51.90 ± 12.70 | 70.53 ± 2.25 | 33.51 ± 11.21 | 65.12 ± 0.34 | 37.64 | 63.96 |
| RiemannGFM | 36.63 ± 6.96 | 66.43 ± 2.76 | 26.40 ± 6.61 | 53.00 ± 2.18 | 38.33 ± 14.67 | 59.87 ± 4.35 | 41.76 ± 7.82 | 63.41 ± 1.86 | 35.78 | 60.68 |
| GCOPE | 45.03 ± 2.16 | 57.30 ± 1.82 | 30.43 ± 2.46 | 49.73 ± 3.99 | 38.20 ± 7.90 | 63.77 ± 3.31 | 25.33 ± 5.54 | 43.79 ± 1.32 | 34.75 | 53.65 |
| MDGFM | 43.40 ± 5.34 | 62.60 ± 4.16 | 38.47 ± 11.45 | 65.13 ± 1.36 | 46.07 ± 0.49 | 66.40 ± 5.84 | 29.07 ± 3.69 | 56.38 ± 3.98 | 39.25 | 62.63 |
| SCGFM | 47.74 ± 10.42 | 66.88 ± 3.94 | 27.64 ± 8.42 | 43.18 ± 5.50 | 38.70 ± 5.31 | 54.20 ± 10.70 | 33.81 ± 7.90 | 56.84 ± 1.42 | 36.97 | 55.28 |
| OFA | 18.43 ± 4.15 | 18.47 ± 2.05 | 19.76 ± 1.76 | 24.21 ± 4.14 | 36.63 ± 3.43 | 45.27 ± 2.96 | 15.93 ± 5.44 | 24.09 ± 2.32 | 22.69 | 28.01 |
| GraphAny | 35.37 ± 5.87 | 67.53 ± 3.01 | 34.40 ± 4.61 | 58.57 ± 1.03 | 54.67 ± 10.40 | 72.23 ± 1.89 | 47.45 ± 7.67 | 50.66 ± 3.74 | 42.97 | 62.25 |
| GILT-InTask | 55.51 ± 3.97 | 72.02 ± 2.38 | 44.56 ± 4.45 | 65.72 ± 2.05 | 61.45 ± 6.42 | 71.70 ± 1.52 | 48.08 ± 3.57 | 64.28 ± 2.96 | 52.40 | 68.43 |
| **GILT** | **56.42 ± 4.46** | **72.50 ± 2.02** | **43.04 ± 4.84** | **62.66 ± 1.84** | **60.03 ± 5.90** | **71.89 ± 1.72** | **50.14 ± 3.25** | **66.33 ± 2.22** | **52.41** | **68.34** |

### Link prediction — Table 3a (Hits@K)

| Model | Cora | Citeseer | Pubmed | ogbl-collab |
|---|---:|---:|---:|---:|
| GCN | 66.79 ± 1.65 | 67.08 ± 2.94 | 53.02 ± 1.39 | 44.75 ± 1.07 |
| SAGE | 55.02 ± 4.03 | 57.01 ± 3.74 | 39.66 ± 0.72 | 48.10 ± 0.81 |
| SEAL | 81.71 ± 1.30 | 83.89 ± 2.15 | 75.54 ± 1.32 | 64.74 ± 0.43 |
| MaskGAE | 82.48 ± 0.76 | 86.11 ± 2.26 | 80.38 ± 0.82 | 65.84 ± 0.47 |
| DGI | 78.65 ± 1.09 | 74.46 ± 1.14 | 56.38 ± 1.76 | 35.11 ± 1.62 |
| GraphCL | 76.99 ± 1.79 | 73.63 ± 0.40 | 51.75 ± 0.98 | 38.87 ± 2.08 |
| AnyGraph | 59.58 ± 13.26 | 61.80 ± 14.40 | 51.39 ± 12.25 | 50.05 ± 13.30 |
| UniLP | 52.91 ± 4.62 | 50.40 ± 7.49 | 54.44 ± 1.34 | 58.00 ± 3.10 |
| GILT-InTask | 86.36 ± 2.42 | 87.88 ± 0.98 | 72.93 ± 1.33 | 65.34 ± 0.51 |
| **GILT** | **85.34 ± 1.75** | **86.70 ± 1.13** | **73.37 ± 1.19** | **66.83 ± 0.29** |

GILT uses 5-shot support. Link prediction defaults to 1:1 positive/negative support (5 positive and 5 negative examples). GCN, SAGE, SEAL, and MaskGAE are fully supervised references. K = 100 for Cora/Citeseer/Pubmed and 50 for ogbl-collab.

### Graph classification — Table 3b (5-shot)

| Model | ogbg-molhiv | ogbg-molpcba | PROTEINS |
|---|---:|---:|---:|
| GCN | 54.82 ± 9.34 | 56.74 ± 1.16 | 57.49 ± 10.36 |
| GAT | 54.94 ± 8.79 | 56.56 ± 0.88 | 57.85 ± 7.39 |
| DGI | 55.05 ± 9.15 | 57.88 ± 0.88 | 59.01 ± 9.45 |
| GraphCL | 54.60 ± 9.05 | 58.46 ± 0.59 | 59.28 ± 9.40 |
| OFA | 52.99 ± 8.65 | 57.50 ± 0.78 | 39.28 ± 1.08 |
| GFT | 56.74 ± 8.48 | 58.67 ± 0.64 | 55.87 ± 7.37 |
| GILT-InTask | 56.63 ± 9.21 | 59.54 ± 0.37 | 64.13 ± 4.77 |
| **GILT** | **59.58 ± 8.66** | **58.70 ± 0.52** | **66.46 ± 1.78** |

## Reproduction

Install the environment and download the released checkpoint:

```bash
conda env create -f env.yml
conda activate gilt
python scripts/download_checkpoint.py
```

Evaluate GILT on the main benchmarks:

```bash
# Node classification: 1-shot
python scripts/reproduce.py --setting nc1 --checkpoint checkpoints/gilt_model.pt --runs 10

# Node classification and link prediction: 5-shot
python scripts/reproduce.py --setting nc5-lp5 --checkpoint checkpoints/gilt_model.pt --runs 10

# Graph classification: 5-shot
python scripts/evaluate_gc.py --dataset hiv --checkpoint checkpoints/gilt_model.pt --contexts 50
python scripts/evaluate_gc.py --dataset pcba --checkpoint checkpoints/gilt_model.pt --contexts 5
python scripts/evaluate_gc.py --dataset PROTEINS --checkpoint checkpoints/gilt_model.pt --contexts 5
```

Pretrain the shared GILT model:

```bash
python train.py --epochs 50 --runs 1 --seed 42 --gpu 0 \
  --context_sampling_plan decay --context_bounds '(5,20)(5,20)(5,20)' \
  --checkpoint_name gilt_model
```

## Project structure

```text
.
├── train.py                     # Shared NC/LP/GC training
├── env.yml                      # Dependencies
├── src/
│   ├── model.py                 # GILT encoder and in-context predictor
│   ├── node_label.py            # MPLP structural features
│   ├── data_*.py                # Data loading and preprocessing
│   ├── engine_*.py              # Training and evaluation
│   ├── checkpoint_utils.py      # Checkpoint I/O
│   ├── config.py                # Configuration
│   └── utils.py                 # Shared utilities
├── scripts/
│   ├── download_checkpoint.py   # Released checkpoint
│   ├── reproduce.py             # Main NC/LP evaluations
│   ├── evaluate_gc.py           # Main GC evaluations
│   └── protocol.py              # Evaluation seeds
└── tests/                       # Core regression tests
```
