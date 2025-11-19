"""
Multi-dataset training engine with temperature-based sampling.

This module provides a separate training pipeline for graph classification
that uses temperature-controlled random sampling across multiple datasets.

SIMPLIFIED VERSION: Only supports task-filtered mode (no FUG, no full-batch training).
"""

import time
import torch
import numpy as np

from .multi_dataset_sampler import MultiDatasetBatchSampler
from .data_gc import create_data_loaders
from .engine_gc import (
    train_graph_classification_single_task,
    get_dataset_metric,
    evaluate_graph_classification_full_batch
)

# Global caches that persist across epochs
_global_data_loaders_cache = {}
_global_iterators_cache = {}


def train_graph_classification_multi_dataset_sampling(
    model, predictor, train_processed_data_list, all_splits, optimizer,
    temperature, pooling_method='mean', device='cuda', batch_size=32,
    clip_grad=1.0, orthogonal_push=False, normalize_class_h=False,
    identity_projection=None, context_k=None, args=None
):
    """
    Train graph classification using temperature-based multi-dataset sampling.

    Uses two-stage sampling WITH REPLACEMENT:
    1. Sample a dataset based on temperature-weighted probabilities
    2. Uniformly sample a task within that dataset
    3. Train one batch from that (dataset, task) pair

    NOTE: Simplified version - only supports task-filtered mode.
    Does NOT support FUG embeddings or full-batch training.

    Args:
        model: GNN model
        predictor: PFN predictor
        train_processed_data_list: List of dataset info dicts
        all_splits: List of split indices for each dataset
        optimizer: Optimizer
        temperature: Temperature parameter for dataset sampling (0 <= T <= 1)
        pooling_method: Graph pooling method ('mean', 'max', 'sum')
        device: Device for computation
        batch_size: Batch size for data loaders
        clip_grad: Gradient clipping value
        orthogonal_push: Whether to use orthogonal push loss
        normalize_class_h: Whether to normalize class embeddings
        identity_projection: Identity projection for node features
        context_k: Number of context examples per class
        args: Training arguments

    Returns:
        float: Average loss across all batches in the epoch
    """
    model.train()
    predictor.train()

    # ALWAYS print this to confirm the function is being called
    print(f"\n{'='*70}")
    print(f"🔥 USING MULTI-DATASET SAMPLING WITH TEMPERATURE = {temperature:.3f}")
    print(f"{'='*70}\n")

    # Get verbose flag
    verbose_sampling = getattr(args, 'verbose_sampling', False) if args else False

    # Compute dataset sizes (num_graphs * num_tasks) and create sampler
    dataset_sizes = []
    num_tasks_per_dataset = []

    for dataset_info, splits in zip(train_processed_data_list, all_splits):
        num_graphs = len(splits['train'])
        num_tasks = dataset_info.get('num_tasks', 1)
        dataset_sizes.append(num_graphs * num_tasks)
        num_tasks_per_dataset.append(num_tasks)

    # Compute sampling probabilities
    from .multi_dataset_sampler import compute_dataset_sampling_probs
    dataset_probs = compute_dataset_sampling_probs(dataset_sizes, temperature)

    # Calculate total batches per epoch
    total_batches = 0
    for dataset_info, splits, num_tasks in zip(train_processed_data_list, all_splits, num_tasks_per_dataset):
        num_graphs = len(splits['train'])
        batches_per_task = int(np.ceil(num_graphs / batch_size))
        total_batches += num_tasks * batches_per_task

    if verbose_sampling:
        print(f"\n{'='*60}")
        print(f"Multi-Dataset Sampling (WITH REPLACEMENT)")
        print(f"{'='*60}")
        print(f"Temperature: {temperature:.3f}")
        print(f"Dataset sizes: {dataset_sizes}")
        print(f"Sampling probabilities: {[f'{p:.4f}' for p in dataset_probs]}")
        print(f"Total batches per epoch: {total_batches}")
        print(f"{'='*60}\n")

    # Use global caches that persist across epochs
    global _global_data_loaders_cache, _global_iterators_cache

    def get_data_loader(dataset_idx, task_idx):
        """Get or create data loader for (dataset, task) pair."""
        key = (dataset_idx, task_idx)

        if key not in _global_data_loaders_cache:
            dataset_info = train_processed_data_list[dataset_idx]
            splits = all_splits[dataset_idx]

            # Get task-filtered splits (use precomputed if available)
            if 'task_filtered_splits' in dataset_info:
                # Use precomputed task-filtered splits (much faster!)
                task_filtered_splits = dataset_info['task_filtered_splits']
                task_splits = task_filtered_splits.get(task_idx, splits)
            else:
                # Fallback: compute on-demand (shouldn't happen if setup is correct)
                from .data_gc import create_task_filtered_datasets
                task_filtered_splits = create_task_filtered_datasets(
                    dataset_info['dataset'],
                    splits
                )
                task_splits = task_filtered_splits.get(task_idx, splits)

            # Check if any embedding mapping is present to use index tracking (FUG, TSGFM, TAGDataset)
            # This is EXACTLY the same check as in train.py line 1798
            use_index_tracking = ('fug_mapping' in dataset_info or
                                'tsgfm_mapping' in dataset_info or
                                'tag_mapping' in dataset_info)

            # Create data loader
            loaders = create_data_loaders(
                dataset_info['dataset'],
                task_splits,
                batch_size=batch_size,
                shuffle=True,
                task_idx=task_idx,
                use_index_tracking=use_index_tracking
            )

            _global_data_loaders_cache[key] = loaders['train']

        return _global_data_loaders_cache[key]

    def get_infinite_iterator(dataset_idx, task_idx):
        """Get or create infinite iterator for (dataset, task) pair."""
        key = (dataset_idx, task_idx)

        if key not in _global_iterators_cache:
            loader = get_data_loader(dataset_idx, task_idx)

            def infinite_loader():
                while True:
                    for batch in loader:
                        yield batch

            _global_iterators_cache[key] = infinite_loader()

        return _global_iterators_cache[key]

    # Training loop with random sampling
    print(f"Starting training loop for {total_batches} batches...")
    epoch_loss = 0.0
    num_batches = 0
    dataset_sample_counts = [0] * len(train_processed_data_list)

    for batch_idx in range(total_batches):
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}/{total_batches}...")

        # Stage 1: Sample dataset with temperature-weighted probabilities
        dataset_idx = np.random.choice(len(dataset_probs), p=dataset_probs)

        # Stage 2: Uniformly sample task within selected dataset
        num_tasks = num_tasks_per_dataset[dataset_idx]
        task_idx = np.random.randint(0, num_tasks)

        # Get batch from infinite iterator
        iterator = get_infinite_iterator(dataset_idx, task_idx)
        batch = next(iterator)

        # Train on this batch using existing single-task training logic
        # NOTE: We train on ONE batch, so we need to extract the batch training logic
        dataset_info = train_processed_data_list[dataset_idx]

        optimizer.zero_grad()

        # Import needed functions
        from .engine_gc import (
            _get_node_embedding_table,
            _safe_lookup_node_embeddings,
            apply_feature_dropout_if_enabled,
            pool_graph_embeddings,
            refresh_gc_context_if_needed
        )
        from .data_utils import batch_edge_dropout
        from torch_sparse import SparseTensor
        import torch.nn.functional as F

        # Refresh context if needed
        if args and hasattr(args, 'context_batch_refresh_interval') and args.context_batch_refresh_interval > 0:
            if batch_idx % args.context_batch_refresh_interval == 0:
                refresh_gc_context_if_needed(dataset_info, batch_idx, 0, args, device)

        batch_data = batch.to(device)

        # Convert node indices to embeddings
        node_emb_table = _get_node_embedding_table(dataset_info['dataset'], task_idx, device, dataset_info)
        if node_emb_table is not None:
            batch_data.x = _safe_lookup_node_embeddings(
                node_emb_table, batch_data.x,
                context=f"train_ds{dataset_idx}_task{task_idx}",
                batch_data=batch_data,
                dataset_info=dataset_info
            )
        else:
            raise ValueError("Expected node embedding table under unified setting")

        # Apply identity projection if needed
        needs_proj = dataset_info.get('needs_identity_projection', False)
        has_proj = identity_projection is not None

        if needs_proj and has_proj:
            x_input = identity_projection(batch_data.x)
        else:
            x_input = batch_data.x

        # Apply feature dropout
        x_input = apply_feature_dropout_if_enabled(x_input, args, rank=0, training=model.training)

        # Apply edge dropout
        if args and hasattr(args, 'edge_dropout_enabled') and args.edge_dropout_enabled and hasattr(args, 'edge_dropout_rate'):
            batch_data = batch_edge_dropout(batch_data, args.edge_dropout_rate, training=model.training)

        # Create adjacency SparseTensor
        batch_data.adj_t = SparseTensor.from_edge_index(
            batch_data.edge_index,
            sparse_sizes=(batch_data.num_nodes, batch_data.num_nodes)
        ).to_symmetric().coalesce()

        # GNN forward pass
        node_embeddings = model(x_input, batch_data.adj_t, batch_data.batch)

        # Pool to graph embeddings
        target_embeddings = pool_graph_embeddings(node_embeddings, batch_data.batch, pooling_method)

        # Get labels
        batch_labels = batch_data.y
        batch_size_actual = target_embeddings.size(0)

        # Skip single-sample batches
        if batch_size_actual == 1:
            continue

        # Check if multi-task dataset (following engine_gc.py line 1063-1082)
        sample_graph = dataset_info['dataset'][0]
        is_multitask = sample_graph.y.numel() > 1

        if is_multitask:
            # Multi-task format: labels are flattened [batch_size * num_tasks]
            num_tasks = sample_graph.y.numel()
            if batch_labels.dim() == 1 and len(batch_labels) == batch_size_actual * num_tasks:
                # Reshape from flattened format back to [batch_size, num_tasks]
                batch_labels = batch_labels.view(batch_size_actual, num_tasks)

            # Extract labels for current task
            batch_labels = batch_labels[:, task_idx]
        else:
            # Single-task format: labels might be squeezed to 1D
            if batch_labels.dim() > 1:
                batch_labels = batch_labels.squeeze()

        # Ensure labels are long integers
        if batch_labels.dtype != torch.long:
            batch_labels = batch_labels.to(torch.long)

        # Create context embeddings using the existing helper function (engine_gc.py line 1085)
        from .engine_gc import _create_context_embeddings_computed

        context_embeddings, context_labels = _create_context_embeddings_computed(
            model, dataset_info['context_graphs'], dataset_info['dataset'], task_idx,
            pooling_method, device, identity_projection, dataset_info
        )

        # Prepare PFN data structure (following engine_gc.py pattern)
        from .engine_gc import prepare_pfn_data_structure
        from .utils import process_node_features

        pfn_data = prepare_pfn_data_structure(
            context_embeddings, context_labels,
            dataset_info['num_classes'], device
        )

        # Process context embeddings to create class prototypes
        class_h = process_node_features(
            context_embeddings, pfn_data,
            degree_normalize=False,
            attention_pool_module=None,
            mlp_module=None,
            normalize=normalize_class_h
        )

        # PFN prediction (following exact signature in engine_gc.py line 1104)
        scores_raw, refined_class_h = predictor(pfn_data, context_embeddings, target_embeddings, context_labels, class_h)

        # Compute loss (following engine_gc.py line 1115)
        scores_log = F.log_softmax(scores_raw, dim=1)
        loss = F.nll_loss(scores_log, batch_labels)

        # Backward and optimize
        loss.backward()

        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)

        optimizer.step()

        # Track statistics
        epoch_loss += loss.item()
        num_batches += 1
        dataset_sample_counts[dataset_idx] += 1

        # Verbose logging
        if verbose_sampling and (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{total_batches}: Dataset {dataset_idx}, Task {task_idx}, Loss {loss.item():.4f}")

    # Compute average loss
    avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

    # Print sampling statistics
    if verbose_sampling:
        print(f"\n{'='*60}")
        print(f"Multi-Dataset Sampling Statistics")
        print(f"{'='*60}")
        print(f"Total batches sampled: {num_batches}")
        print(f"Dataset sample counts: {dataset_sample_counts}")
        actual_freqs = np.array(dataset_sample_counts) / num_batches
        print(f"Expected probabilities: {[f'{p:.4f}' for p in dataset_probs]}")
        print(f"Actual frequencies:     {[f'{f:.4f}' for f in actual_freqs]}")
        print(f"{'='*60}\n")

    return avg_loss
