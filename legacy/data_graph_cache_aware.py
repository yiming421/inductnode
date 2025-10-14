"""
Cache-aware graph dataset loading to avoid loading massive embeddings when PCA cache exists.
This module provides memory-efficient alternatives that skip embedding loading when possible.
"""

import os
import torch
from pathlib import Path
from ogb.graphproppred import PygGraphPropPredDataset


def check_pca_cache_availability(dataset_names, hidden_dim, pca_cache_dir="./pca_cache"):
    """
    Check if PCA cache exists for all requested datasets.
    
    Args:
        dataset_names (list): List of dataset names
        hidden_dim (int): Target hidden dimension for PCA
        pca_cache_dir (str): Directory where PCA cache is stored
        
    Returns:
        dict: {dataset_name: bool} indicating cache availability
    """
    cache_status = {}
    cache_path = Path(pca_cache_dir)
    
    for dataset_name in dataset_names:
        cache_file = cache_path / f"{dataset_name}_dim{hidden_dim}.pt"
        cache_status[dataset_name] = cache_file.exists()
        
        if cache_status[dataset_name]:
            try:
                # Verify cache file is valid by attempting to load metadata
                cache_data = torch.load(cache_file, map_location='cpu')
                if 'processed_features' not in cache_data:
                    cache_status[dataset_name] = False
                    print(f"Warning: Invalid cache file for {dataset_name}, will reload")
            except Exception as e:
                cache_status[dataset_name] = False
                print(f"Warning: Corrupted cache file for {dataset_name}: {e}")
    
    return cache_status


def load_ogb_fug_dataset_minimal(name, ogb_root='./dataset/ogb', fug_root='./fug', skip_embeddings=True):
    """
    Load OGB+FUG dataset without the massive embedding tensor when cache exists.
    
    Args:
        name (str): Dataset name
        ogb_root (str): OGB dataset root
        fug_root (str): FUG embedding root  
        skip_embeddings (bool): If True, don't load the 48GB embedding tensor
        
    Returns:
        (dataset, fug_mapping_minimal) or None if failed
    """
    from .data_graph_fug_simple import load_ogb_fug_dataset
    
    # Dataset name mapping (same as original)
    ogb_names = {
        'bace': 'ogbg-molbace',
        'bbbp': 'ogbg-molbbbp', 
        'hiv': 'ogbg-molhiv',
        'chemhiv': 'ogbg-molhiv',
        'pcba': 'ogbg-molpcba',
        'chempcba': 'ogbg-molpcba',
        'molpcba': 'ogbg-molpcba',
        'tox21': 'ogbg-moltox21',
    }
    
    if name not in ogb_names:
        print(f"[FUG-Minimal] Unknown dataset: {name}")
        return None
        
    full_ogb_name = ogb_names[name]
    
    # Load OGB dataset (this is lightweight - just graph structure)
    try:
        print(f"[FUG-Minimal] Loading OGB dataset structure '{full_ogb_name}'...")
        dataset = PygGraphPropPredDataset(name=full_ogb_name, root=ogb_root)
        print(f"[FUG-Minimal] Loaded {len(dataset)} graphs (structure only)")
    except Exception as e:
        print(f"[FUG-Minimal] Failed to load OGB dataset: {e}")
        return None
    
    if skip_embeddings:
        # Create minimal FUG mapping without loading the huge embedding tensor
        embedding_file = os.path.join(fug_root, name, f'{full_ogb_name}_node_embeddings.pt')
        if not os.path.exists(embedding_file):
            print(f"[FUG-Minimal] Embedding file not found: {embedding_file}")
            return None
        
        # Create node index mapping without loading embeddings
        node_idx = 0
        sample_graph = dataset[0]
        is_multitask = sample_graph.y.numel() > 1
        
        if is_multitask:
            print(f"[FUG-Minimal] Multi-task dataset detected, adding task_mask for {sample_graph.y.numel()} tasks")
        
        # Create external mapping (same logic as original but without embeddings)
        node_index_mapping = {}
        total_nodes = 0
        for i in range(len(dataset)):
            graph = dataset[i]
            n_nodes = graph.num_nodes
            
            # Store the node index range for this graph
            node_index_mapping[i] = torch.arange(node_idx, node_idx + n_nodes, dtype=torch.long)
            node_idx += n_nodes
            total_nodes += n_nodes
            
            # Add task_mask for multi-task datasets
            if is_multitask:
                if graph.y.dtype.is_floating_point:
                    graph.task_mask = (~torch.isnan(graph.y)).float()
                else:
                    graph.task_mask = (graph.y != -1).float()
        
        # Create minimal FUG mapping (without the huge embedding tensor)
        fug_mapping_minimal = {
            'node_index_mapping': node_index_mapping,
            'node_embs': None,  # Will be loaded from cache instead
            'uses_fug_embeddings': True,
            'name': name,
            'is_multitask': is_multitask,
            'embedding_file': embedding_file,  # Store path for potential lazy loading
            'total_nodes': total_nodes,
            'cache_mode': True
        }
        
        print(f"[FUG-Minimal] Created minimal mapping for '{name}' with {total_nodes} nodes (embeddings skipped)")
        return dataset, fug_mapping_minimal
    
    else:
        # Fall back to original loading
        print(f"[FUG-Minimal] Cache miss - falling back to full loading")
        return load_ogb_fug_dataset(name, ogb_root, fug_root)


def load_dataset_cache_aware(name, root='./dataset', embedding_family='ST', hidden_dim=None, pca_cache_dir="./pca_cache"):
    """
    Cache-aware version of load_dataset that skips embedding loading when PCA cache exists.
    
    Args:
        name (str): Dataset name
        root (str): Dataset root directory
        embedding_family (str): Embedding family
        hidden_dim (int): Target hidden dimension for cache checking
        pca_cache_dir (str): PCA cache directory
        
    Returns:
        Same as original load_dataset but with potential memory savings
    """
    from .data_graph import load_dataset
    
    # Check if PCA cache exists for this dataset
    use_cache_mode = False
    if hidden_dim is not None:
        cache_status = check_pca_cache_availability([name], hidden_dim, pca_cache_dir)
        use_cache_mode = cache_status.get(name, False)
    
    if use_cache_mode and os.environ.get('USE_FUG_EMB', '0') == '1':        
        # Use minimal FUG loading (skip 48GB embeddings)
        fug_root = os.environ.get('FUG_EMB_ROOT', './fug')
        ogb_root = os.environ.get('OGB_ROOT', './dataset/ogb')
        
        result = load_ogb_fug_dataset_minimal(name, ogb_root=ogb_root, fug_root=fug_root, skip_embeddings=True)
        if result is not None:
            dataset, fug_mapping = result
            print(f"[Cache-Aware] Successfully loaded {name} in cache mode")
            dataset.name = name
            return dataset, fug_mapping
        else:
            print(f"[Cache-Aware] Minimal loading failed for {name}, falling back to full loading")
    
    # Fall back to original loading
    print(f"⚠️  [Cache-Aware] Using full loading for {name}")
    return load_dataset(name, root, embedding_family)


def load_all_graph_datasets_cache_aware(dataset_names, device='cuda', pretraining_mode=False, context_k=32, 
                                       embedding_family='ST', hidden_dim=None, pca_cache_dir="./pca_cache"):
    """
    Cache-aware version of load_all_graph_datasets with massive memory savings.
    
    Args:
        Same as original load_all_graph_datasets plus:
        hidden_dim (int): Target hidden dimension for cache checking
        pca_cache_dir (str): PCA cache directory
        
    Returns:
        Same as original but with potential 95% memory reduction
    """
    from .data_graph import create_dataset_splits, prepare_graph_data_for_pfn
    
    import time
    overall_start = time.time()
    
    # Check cache availability for all datasets
    if hidden_dim is not None:
        cache_status = check_pca_cache_availability(dataset_names, hidden_dim, pca_cache_dir)
        cache_hits = sum(cache_status.values())
        print(f"📊 Cache Status: {cache_hits}/{len(dataset_names)} datasets have PCA cache")
        for name, has_cache in cache_status.items():
            status = "✅ CACHED" if has_cache else "❌ CACHE MISS"
            print(f"  {name}: {status}")
    else:
        cache_status = {name: False for name in dataset_names}
    
    datasets = []
    processed_data_list = []
    
    for name in dataset_names:
        print(f"\nLoading dataset: {name}")
        
        # Time dataset loading
        load_start = time.time()
        
        # Use cache-aware loading
        result = load_dataset_cache_aware(name, embedding_family=embedding_family, 
                                        hidden_dim=hidden_dim, pca_cache_dir=pca_cache_dir)
        
        load_time = time.time() - load_start
        
        # Handle both single dataset and (dataset, fug_mapping) returns
        if result is None:
            print(f"Skipping {name} due to loading error")
            continue
        elif isinstance(result, tuple) and len(result) == 2:
            # FUG dataset with external mapping
            dataset, fug_mapping = result
            print(f"  FUG dataset loaded with external mapping")
        else:
            # Regular dataset
            dataset = result
            fug_mapping = None
        print(f"  Dataset loading: {load_time:.2f}s")
        
        # Time split creation
        split_start = time.time()
        split_idx = create_dataset_splits(dataset, name, pretraining_mode=pretraining_mode)
        split_time = time.time() - split_start
        print(f"  Split creation: {split_time:.2f}s")
        
        # Time context preparation
        prep_start = time.time()
        print(f"  Starting context preparation with k={context_k}...")
        
        processed_data = prepare_graph_data_for_pfn(
            dataset, split_idx, device=device, context_k=context_k, 
        )
        
        # Add FUG mapping to processed data if available
        if fug_mapping is not None:
            processed_data['fug_mapping'] = fug_mapping
            print(f"  FUG mapping attached to processed data")
        
        prep_time = time.time() - prep_start
        print(f"  Context preparation: {prep_time:.2f}s")
        
        datasets.append(dataset)
        processed_data_list.append(processed_data)
        
        total_dataset_time = time.time() - load_start
        print(f"Successfully processed {name} with {len(dataset)} graphs (total: {total_dataset_time:.2f}s)")
    
    overall_time = time.time() - overall_start
    total_cache_hits = sum(1 for name in dataset_names if cache_status.get(name, False))
    print(f"\n🎯 All datasets loaded in {overall_time:.2f}s ({total_cache_hits} used cache)")
    
    return datasets, processed_data_list
