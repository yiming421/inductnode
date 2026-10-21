# GILT

Minimal implementation of the shared GILT architecture and the frozen-checkpoint evaluation protocols. This tree contains source code, environment definitions and small regression tests. Datasets, checkpoints, predictions, result tables, logs and experimental variants are not distributed in the Git tree.

## Architecture

Numerical node features are aligned to 128 dimensions by PCA and zero padding, then standardized columnwise. A shared six-layer linear PureGCN encoder feeds a two-layer, four-head Transformer. Support tokens combine sample embeddings and class prototypes; query tokens combine sample embeddings and zero label slots. Support self-attention and query-to-support attention keep queries independent. Node and graph classification use the refined label half and class prototypes. Link prediction combines the refined edge representation with MPLP structural features.

The released numerical path uses **max graph pooling**, pre-normalized Transformer blocks, and encoder normalization before propagation in layers after the first. These implementation details differ from the manuscript's mean-pooling/normalization description. They are preserved so the released weights remain usable. Required TTA, validation-based NC postprocessing and LP gating remain part of evaluation.

## Environment

Linux with an NVIDIA GPU is the reference environment for NC/LP and source pretraining. Graph-table inference uses four CPU threads; PROTEINS reference feature preprocessing uses an NVIDIA A100 with the pinned CUDA/PyTorch environment.

```bash
conda env create -f env.yml
conda activate gilt
python -m pytest -q tests
python scripts/download_checkpoint.py
```

The downloader pins a checkpoint revision and verifies its SHA256 before loading. It writes a weights-only-compatible checkpoint under `checkpoints/`, with the same model tensors and no machine paths or author metadata. A local copy of the pinned original release can be supplied with `--source`. Nothing is uploaded by these commands.

PyG and OGB acquire requested raw datasets on first use. Put NC/LP caches in `dataset/`, or set `GILT_DATA_ROOT`. Use `--ogb-root` and `--tu-root` for the graph-table script. Downloads and processed datasets can be large: check space on both the data and temporary volumes before running. Full PCBA evaluation also requires substantial RAM; allow at least 64 GB RAM and 20 GB free storage for its data/preparation. Exact results depend on library versions and device-dependent PCA.

## Main evaluations

Every command defaults to one run/context. The explicit counts below reproduce the ordered context repetitions used for the corresponding paper entries; they do not retrain multiple models. Seeds are defined in `scripts/protocol.py`. Evaluation prints metrics to the terminal and does not create prediction or score artifacts.

```bash
# Table 2: NC 1-shot, 10 runs; three support resamples per dataset/run.
python scripts/reproduce.py --setting nc1 --checkpoint checkpoints/gilt_model.pt --runs 10

# Tables 2 and 3a: NC 5-shot and LP 5-shot, 10 runs.
python scripts/reproduce.py --setting nc5-lp5 --checkpoint checkpoints/gilt_model.pt --runs 10

# Table 3b: separate graph-table protocols, frozen weights and TTA5.
python scripts/evaluate_gc.py --dataset hiv --checkpoint checkpoints/gilt_model.pt --contexts 50
python scripts/evaluate_gc.py --dataset pcba --checkpoint checkpoints/gilt_model.pt --contexts 5
python scripts/evaluate_gc.py --dataset PROTEINS --checkpoint checkpoints/gilt_model.pt --contexts 5
```

NC uses Cora, Citeseer, Pubmed and WikiCS; LP uses Cora, Citeseer, Pubmed and ogbl-collab. The 5-shot NC/LP runner retains the original NC → LP → HIV preparation and evaluation order to preserve RNG consumption; its incidental HIV score is not the graph-table protocol. NC reports accuracy, LP reports Hits@100 on the citation graphs and Hits@50 on ogbl-collab.

HIV and PCBA use official OGB splits, five training positives and five training negatives per assay, fixed numerical features, and the original plus five projected views. Six logits are averaged within each context; metrics are then summarized across contexts with sample standard deviation. Missing PCBA labels and single-class assays are excluded from scoring. Both ROC-AUC and official PCBA AP are printed.

PROTEINS uses the original node-label one-hot features, a stratified 60/20/20 split with split seed 42, five support graphs per class, and context seeds 42–46. **The manuscript's PROTEINS column is labeled AUC, but its reported values correspond to accuracy in the underlying evaluation.** The script prints both accuracy and ROC-AUC explicitly. It does not relabel one metric as the other. Its reference preprocessing uses CUDA PCA on an NVIDIA A100. Regenerating features on that reference environment matched the original features exactly; an H20 produced different PCA signs and near-zero components, changing downstream scores. Use the reference hardware for numerical comparison with the paper. Other GPUs and `--preprocess-device cpu` are portability options, not exact numerical replacements. Prepared feature artifacts are not required or bundled.

## Source pretraining

```bash
python train.py --epochs 50 --runs 1 --seed 42 --gpu 0 \
  --context_sampling_plan decay --context_bounds '(5,20)(5,20)(5,20)' \
  --checkpoint_name gilt_model
```

Defaults specify the 16 NC/LP source graphs and seven molecular source datasets, six encoder layers, two Transformer layers, dimension 128, four heads, dropout 0.2, learning rate 1e-5 and weight decay 1e-4. The source validation metric selects the checkpoint; target-test metrics are not used for selection. OGB source pretraining retains the released split policy: official train and test indices are combined for source training and official validation is held out. The downstream target datasets are separately specified. This is not a molecule-disjoint pretraining corpus; exact raw-SMILES overlap exists between the original source pool and molecular targets.

Training writes a local ignored checkpoint. Retraining is stochastic; the frozen released checkpoint is the reference for the paper's main inference results. Use `python train.py --help` for batch-size, device and dataset controls. Keep dataset order, context seeds, feature preprocessing and TTA settings fixed when comparing results.
