"""
Data processing utilities for GraphPFN.

GraphPFN requires different data processing than standard models:
- PCA projection WITHOUT padding (foundational model handles arbitrary dimensions)
- No fixed hidden dimension requirement
- Parameter-free GNN accepts whatever PCA outputs
"""

import time
import torch
import torch.nn.functional as F
from .data_utils import (
    select_k_shot_context,
    apply_incremental_pca_cpu,
    generate_orthogonal_noise_features,
)


def process_data_graphpfn(
    data,
    split_idx,
    context_num,
    pca_target_dim=128,
    normalize_data=False,
    use_full_pca=False,
    pca_device='gpu',
    incremental_pca_batch_size=10000,
    rank=0,
    process_test_only=False,
    use_orthogonal_noise=False,
):
    """
    Process data for GraphPFN foundational model.

    Key differences from standard processing:
    - NO padding to fixed hidden dimension
    - PCA projects to pca_target_dim (can be any dimension)
    - Parameter-free GNN handles the output dimension

    Args:
        data: Graph dataset
        split_idx: Train/valid/test split indices
        context_num: Number of context nodes to sample
        pca_target_dim: Target dimension for PCA (no padding afterward!)
        normalize_data: Whether to normalize features (L2 normalization)
        use_full_pca: Whether to use full PCA or truncated
        pca_device: Device for PCA computation ('cpu' or 'gpu')
        incremental_pca_batch_size: Batch size for incremental PCA
        rank: Process rank for distributed training
        process_test_only: Whether to process only test split (for zero-shot eval)
        use_orthogonal_noise: Whether to replace features with orthogonal noise

    Returns:
        None (modifies data in-place)
    """
    device = data.x.device
    split_idx['train'] = split_idx['train'].to(device)
    split_idx['valid'] = split_idx['valid'].to(device)
    split_idx['test'] = split_idx['test'].to(device)

    # Context sampling
    # Only sample new context if it doesn't exist (to preserve manually set contexts during refresh)
    if not hasattr(data, 'context_sample') or data.context_sample is None:
        if len(split_idx['train']) > 0:
            context_source_split = split_idx['train']
        else:
            context_source_split = split_idx['test']
        data.context_sample = select_k_shot_context(data, context_num, context_source_split)
        data.context_sample = data.context_sample.to(device)
    else:
        # Context already set (e.g., during refresh), just ensure it's on the right device
        data.context_sample = data.context_sample.to(device)

    st = time.time()

    # Check if features have already been processed (skip PCA during context refresh)
    # We detect this by checking the _graphpfn_processed flag
    already_processed = hasattr(data, '_graphpfn_processed') and data._graphpfn_processed

    if already_processed and not use_orthogonal_noise:
        # Features already processed, just update context (no PCA recomputation)
        if rank == 0:
            print(f"[GraphPFN] Features already processed, skipping PCA")
            print(f"[GraphPFN] Processing time: {time.time()-st:.2f}s")
        return

    # Determine which nodes to process
    if process_test_only:
        context_indices = data.context_sample
        test_indices = split_idx['test']
        process_indices = torch.unique(torch.cat([context_indices, test_indices]))
        if rank == 0:
            print(f"[GraphPFN Test-Only] Processing {len(process_indices)} nodes")
    else:
        process_indices = None

    # Extract subset if in test-only mode
    if process_indices is not None:
        input_features = data.x[process_indices]
    else:
        input_features = data.x

    # Orthogonal Noise Ablation
    if use_orthogonal_noise:
        num_nodes = data.x.size(0)
        if rank == 0:
            print(f"[GraphPFN] Replacing features with orthogonal noise ({num_nodes} x {pca_target_dim})")

        data.x = generate_orthogonal_noise_features(
            num_nodes=num_nodes,
            target_dim=pca_target_dim,
            seed=42,
            device=device,
            dtype=input_features.dtype,
            rank=rank
        )

        # Apply normalization if requested
        if normalize_data:
            data.x = F.normalize(data.x, p=2, dim=1)
            if rank == 0:
                print(f"[GraphPFN] Applied normalization to orthogonal noise")

        if rank == 0:
            print(f"[GraphPFN] Processing time: {time.time()-st:.2f}s")
        return

    # Apply PCA (NO PADDING!)
    original_dim = input_features.size(1)

    if rank == 0:
        print(f"[GraphPFN] Original features: {original_dim} dim")
        print(f"[GraphPFN] PCA target: {pca_target_dim} dim (NO PADDING)")

    # Determine actual PCA dimension
    if original_dim <= pca_target_dim:
        # Not enough features for requested PCA dimension
        actual_pca_dim = original_dim
        if rank == 0:
            print(f"[GraphPFN] Using all {actual_pca_dim} features (less than target {pca_target_dim})")
    else:
        actual_pca_dim = pca_target_dim

    # Apply PCA
    if pca_device == 'cpu':
        if rank == 0:
            print(f"[GraphPFN] Applying CPU Incremental PCA...")
        data.x_pca = apply_incremental_pca_cpu(
            input_features,
            actual_pca_dim,
            batch_size=incremental_pca_batch_size,
            sign_normalize=False,
            rank=rank
        ).to(device)
    else:
        if rank == 0:
            print(f"[GraphPFN] Applying GPU PCA...")
        # Center the data
        mean = input_features.mean(dim=0, keepdim=True)
        centered = input_features - mean

        # PCA using SVD
        if use_full_pca:
            U, S, V = torch.pca_lowrank(centered, q=min(actual_pca_dim, centered.size(1)))
        else:
            U, S, V = torch.pca_lowrank(centered, q=actual_pca_dim)

        data.x_pca = U[:, :actual_pca_dim] * S[:actual_pca_dim]

    # Final feature tensor (NO PADDING!)
    data.x = data.x_pca

    if rank == 0:
        print(f"[GraphPFN] Final features: {data.x.shape[1]} dim (no padding)")

    # Scatter back to full array if in test-only mode
    if process_indices is not None:
        # Get original full dataset size before processing
        original_num_nodes = data.y.size(0)  # Use y as reference for full size
        full_processed = torch.zeros(
            original_num_nodes,
            data.x.size(1),
            device=device,
            dtype=data.x.dtype
        )
        full_processed[process_indices] = data.x
        data.x = full_processed
        if rank == 0:
            print(f"[GraphPFN Test-Only] Scattered back to full array ({original_num_nodes} nodes)")

    # Normalize if requested
    if normalize_data:
        data.x = F.normalize(data.x, p=2, dim=1)
        if rank == 0:
            print(f"[GraphPFN] Applied L2 normalization")

    # Mark as processed to avoid re-running PCA during context refresh
    data._graphpfn_processed = True

    if rank == 0:
        print(f"[GraphPFN] Processing time: {time.time()-st:.2f}s")
