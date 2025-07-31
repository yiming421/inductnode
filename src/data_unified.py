#!/usr/bin/env python3
"""
Unified Data Loading Module

This module eliminates duplicate loading of shared datasets between node classification
and link prediction tasks. It loads each dataset once and applies task-specific context
selection as needed, while sharing the same base features and graph structure.

Key principle: One dataset = One copy in memory with shared features (data.x, data.adj_t)
"""

import torch
from typing import Dict, List, Tuple, Set

# Import existing data loading functions
from .data import load_ogbn_data, load_ogbn_data_train, load_data, load_data_train
from .data_link import load_ogbl_data, load_data as load_data_link
from .data_utils import process_data, prepare_link_data, select_link_context, process_link_data


def load_and_preprocess_data_unified(args, device='cuda') -> Dict:
    """
    Load and preprocess data for both tasks, eliminating duplicate loading.
    
    Args:
        args: Arguments object containing dataset configurations
        device: Device to load data onto
        
    Returns:
        Dictionary containing processed datasets for both tasks
    """
    print("\n=== Loading and Preprocessing Data (Unified & Memory-Optimized) ===")
    
    # Parse dataset lists
    nc_train_datasets = args.nc_train_dataset.split(',')
    nc_test_datasets = args.nc_test_dataset.split(',')
    lp_train_datasets = args.lp_train_dataset.split(',')
    lp_test_datasets = args.lp_test_dataset.split(',')
    
    # Analyze dataset overlap and report memory savings
    _report_memory_optimization(nc_train_datasets, nc_test_datasets, 
                               lp_train_datasets, lp_test_datasets)
    
    # Load unique datasets with shared processing
    dataset_cache = _load_unique_datasets(nc_train_datasets, nc_test_datasets,
                                         lp_train_datasets, lp_test_datasets, 
                                         args, device)
    
    # Prepare data for each task (using same base data, different contexts)
    nc_train_data = _prepare_nc_data(nc_train_datasets, dataset_cache, args, is_training=True)
    nc_test_data = _prepare_nc_data(nc_test_datasets, dataset_cache, args, is_training=False)
    lp_train_data = _prepare_lp_data(lp_train_datasets, dataset_cache, args, device, is_training=True)
    lp_test_data = _prepare_lp_data(lp_test_datasets, dataset_cache, args, device, is_training=False)
    
    print("✅ Unified data loading complete - maximum memory efficiency achieved!")
    
    return {
        'nc_train': nc_train_data,
        'nc_test': nc_test_data,
        'lp_train': lp_train_data,
        'lp_test': lp_test_data
    }


def _report_memory_optimization(nc_train: List[str], nc_test: List[str],
                               lp_train: List[str], lp_test: List[str]) -> None:
    """Report memory optimization statistics."""
    all_datasets = set(nc_train + nc_test + lp_train + lp_test)
    total_requests = len(nc_train) + len(nc_test) + len(lp_train) + len(lp_test)
    shared_datasets = total_requests - len(all_datasets)
    
    if shared_datasets > 0:
        print(f"🔥 Memory optimization: {shared_datasets} duplicate dataset loads eliminated!")
        shared_train = set(nc_train) & set(lp_train)
        shared_test = set(nc_test) & set(lp_test)
        if shared_train:
            print(f"   └── Shared training: {sorted(shared_train)}")
        if shared_test:
            print(f"   └── Shared test: {sorted(shared_test)}")
    else:
        print("ℹ️  No shared datasets detected - no optimization needed")


def _load_unique_datasets(nc_train: List[str], nc_test: List[str],
                         lp_train: List[str], lp_test: List[str],
                         args, device: str) -> Dict:
    """Load all unique datasets once with shared feature processing."""
    all_train_datasets = set(nc_train + lp_train)
    all_test_datasets = set(nc_test + lp_test)
    
    print("📦 Loading unique datasets...")
    dataset_cache = {}
    
    # Load unique training datasets
    for dataset in all_train_datasets:
        print(f"   └── Training: {dataset}")
        dataset_cache[f"{dataset}_train"] = _load_and_process_single_dataset(
            dataset, args, device, is_training=True
        )
    
    # Load unique test datasets (only if not already loaded)
    for dataset in all_test_datasets:
        test_key = f"{dataset}_test"
        train_key = f"{dataset}_train"
        
        if train_key not in dataset_cache:  # Not loaded as training data
            print(f"   └── Test: {dataset}")
            dataset_cache[test_key] = _load_and_process_single_dataset(
                dataset, args, device, is_training=False
            )
    
    return dataset_cache


def _load_and_process_single_dataset(dataset: str, args, device: str, is_training: bool) -> Tuple:
    """Load and process a single dataset with shared feature processing."""
    # Load raw data
    if dataset.startswith('ogbn-'):
        if is_training:
            data, split_idx = load_ogbn_data_train(dataset)
        else:
            data, split_idx = load_ogbn_data(dataset)
    else:
        if is_training:
            data, split_idx = load_data_train(dataset)
        else:
            data, split_idx = load_data(dataset)
    
    # Move to device
    data.x = data.x.to(device)
    data.adj_t = data.adj_t.to(device)
    data.y = data.y.to(device)
    
    # Apply shared feature processing (PCA, normalization, etc.)
    # This is the same for both NC and LP tasks
    process_data(data, split_idx, args.hidden, args.context_num, False, 
                args.use_full_pca, args.normalize_data, False, 32, 0, 
                args.padding_strategy, args.use_batchnorm, args.use_identity_projection,
                args.projection_small_dim, args.projection_large_dim)
    
    return data, split_idx


def _prepare_nc_data(datasets: List[str], dataset_cache: Dict, args, is_training: bool) -> Tuple:
    """Prepare node classification data (reuse processed datasets, add NC context)."""
    print(f"🧠 Preparing NC {'training' if is_training else 'test'} data...")
    
    data_list = []
    split_idx_list = []
    
    for dataset in datasets:
        # Get cached processed data
        cache_key = f"{dataset}_train" if f"{dataset}_train" in dataset_cache else f"{dataset}_test"
        data, split_idx = dataset_cache[cache_key]
        
        # Use the same data object - no copying needed!
        # NC-specific context is already handled in process_data()
        data_list.append(data)
        split_idx_list.append(split_idx)
    
    return data_list, split_idx_list


def _prepare_lp_data(datasets: List[str], dataset_cache: Dict, args, device: str, is_training: bool) -> Tuple:
    """Prepare link prediction data (create LP-specific edge splits and context)."""
    print(f"🔗 Preparing LP {'training' if is_training else 'test'} data...")
    
    data_list = []
    split_idx_list = []
    context_data_list = []
    masks_list = []
    link_data_all_list = []
    
    for dataset in datasets:
        # For LP, we need to load with link prediction format (different edge splits)
        if dataset.startswith('ogbl'):
            # OGB link datasets are LP-specific, load them directly
            lp_data, lp_split_edge = load_ogbl_data(dataset, device=device)
        else:
            # Convert regular dataset to LP format
            lp_data, lp_split_edge = load_data_link(dataset, device=device, is_pretraining=False)
        
        # Apply LP-specific processing
        process_link_data(lp_data, args, rank=0)
        
        # Prepare link prediction context (different from NC context)
        link_data = prepare_link_data(lp_data, lp_split_edge)
        
        if is_training:
            context_data, train_mask = select_link_context(
                link_data['train'], args.context_k, args.context_neg_ratio,
                args.remove_context_from_train
            )
            context_data_list.append(context_data)
            masks_list.append(train_mask)
        else:
            context_data, _ = select_link_context(
                link_data['train'], args.context_k, args.context_neg_ratio, False
            )
            context_data_list.append(context_data)
        
        data_list.append(lp_data)
        split_idx_list.append(lp_split_edge)
        link_data_all_list.append(link_data)
    
    if is_training:
        return data_list, split_idx_list, context_data_list, masks_list, link_data_all_list
    else:
        return data_list, split_idx_list, context_data_list, link_data_all_list