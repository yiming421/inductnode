"""
Graph classification data loading and preprocessing utilities.
Handles loading of standard graph classification benchmarks and prepares them for PFN-based models.
Supports both TUDataset and TSGFM datasets with text features.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.loader import DataLoader
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import random


def load_tsgfm_dataset(name, root='./dataset'):
    """
    Load a TSGFM graph classification dataset with text features.
    Handles batched format with lookup tables for node text features.
    
    Args:
        name (str): Dataset name (e.g., 'bace', 'bbbp', 'chemhiv', etc.)
        root (str): Root directory for storing datasets
        
    Returns:
        dataset: PyTorch Geometric dataset for graph classification
    """
    try:
        dataset_path = os.path.join(root, name, 'processed', 'geometric_data_processed.pt')
        
        if not os.path.exists(dataset_path):
            print(f"Dataset file not found: {dataset_path}")
            return None
            
        # Load the processed dataset
        raw_data = torch.load(dataset_path, map_location='cpu')
        
        # Handle tuple format (data_obj, metadata)
        if isinstance(raw_data, tuple):
            data_obj = raw_data[0]
            metadata = raw_data[1] if len(raw_data) > 1 else None
        else:
            # Handle case where it's just the data object
            data_obj = raw_data
            metadata = None
        
        # For graph classification, we expect batched format with lookup tables
        if hasattr(data_obj, 'x') and hasattr(data_obj, 'node_embs') and metadata is not None:
            print(f"Loading {name} as graph classification dataset (batched format)")
            return load_graph_classification_tsgfm(data_obj, metadata, name)
        else:
            print(f"Error: {name} does not have expected graph classification format")
            print(f"Expected: batched format with x indices, node_embs lookup table, and metadata")
            return None
        
    except Exception as e:
        print(f"Failed to load TSGFM dataset {name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_graph_classification_tsgfm(data_obj, metadata, name):
    """Load graph classification TSGFM dataset (batched format with boundaries)"""
    
    # Extract graph boundaries from metadata
    node_boundaries = metadata['x']  # Indices where each graph's nodes start/end
    edge_boundaries = metadata['edge_index']  # Indices where each graph's edges start/end
    
    num_graphs = len(node_boundaries) - 1  # Last boundary is the final endpoint
    
    print(f"Extracting {num_graphs} graphs from batched format")
    
    # Extract node features using lookup table
    x_indices = data_obj.x  # Node type indices
    node_embs = data_obj.node_embs  # Node type embeddings lookup table
    node_text_feat = node_embs[x_indices]  # Reconstruct full node text features
    
    # Graph-level labels - handle both single-task and multi-task datasets
    graph_labels = data_obj.y
    
    # Standardize label format: always [num_graphs, num_tasks]
    if graph_labels.numel() % num_graphs == 0 and graph_labels.numel() != num_graphs:
        # Multi-task dataset
        num_tasks = graph_labels.numel() // num_graphs
        graph_labels = graph_labels.view(num_graphs, num_tasks)
        print(f"Multi-task dataset: {num_graphs} graphs, {num_tasks} tasks per graph")
        print(f"Reshaped labels from {data_obj.y.shape} to {graph_labels.shape}")
    elif graph_labels.numel() == num_graphs:
        # Single-task dataset - reshape to [num_graphs, 1]
        graph_labels = graph_labels.view(num_graphs, 1)
        num_tasks = 1
        print(f"Single-task dataset: {num_graphs} graphs, standardized to [num_graphs, 1] format")
    else:
        raise ValueError(f"Invalid label structure: {graph_labels.numel()} labels for {num_graphs} graphs")
    
    # Extract edge features using lookup table (if available)
    edge_text_feat = None
    if hasattr(data_obj, 'xe') and hasattr(data_obj, 'edge_embs'):
        xe_indices = data_obj.xe
        edge_embs = data_obj.edge_embs
        edge_text_feat = edge_embs[xe_indices]
    
    # Split the batched data into individual graphs
    graphs = []
    
    for i in range(num_graphs):
        # Node range for this graph
        node_start = node_boundaries[i].item()
        node_end = node_boundaries[i + 1].item()
        
        # Skip empty graphs
        if node_end <= node_start:
            print(f"Warning: Skipping empty graph {i}")
            continue
        
        # Edge range for this graph
        edge_start = edge_boundaries[i].item()
        edge_end = edge_boundaries[i + 1].item()
        
        # Extract nodes and edges for this graph
        graph_node_feat = node_text_feat[node_start:node_end]  # Shape: [num_nodes_in_graph, 384]
        graph_edges = data_obj.edge_index[:, edge_start:edge_end]  # Shape: [2, num_edges_in_graph]
        
        # Edge indices are already relative to each graph (starting from 0)
        # No adjustment needed - they're already in range [0, num_nodes_in_graph)
        
        # Get graph-level label(s) 
        graph_label = graph_labels[i]
        
        # Create task mask for multi-task datasets (1 for valid labels, 0 for NaN)
        if graph_label.dim() == 1 and len(graph_label) > 1:  # Multi-task
            task_mask = (~torch.isnan(graph_label)).float()
            # Replace NaN with 0 for computational stability and ensure integer labels for classification
            graph_label = torch.where(torch.isnan(graph_label), torch.zeros_like(graph_label), graph_label).long()
        else:  # Single-task
            task_mask = torch.ones_like(graph_label)
        
        # Create individual graph
        graph = Data(
            x=graph_node_feat,  # Use reconstructed text features
            edge_index=graph_edges,
            y=graph_label,
            task_mask=task_mask,  # Add task mask for NaN handling
            num_nodes=graph_node_feat.shape[0]
        )
        
        # Convert edge_index to SparseTensor format for GNN compatibility
        from torch_sparse import SparseTensor
        graph.adj_t = SparseTensor.from_edge_index(
            graph_edges, 
            sparse_sizes=(graph_node_feat.shape[0], graph_node_feat.shape[0])
        ).to_symmetric().coalesce()
        
        # Add edge features if available
        if edge_text_feat is not None:
            graph.edge_attr = edge_text_feat[edge_start:edge_end]
        
        graphs.append(graph)
    
    # Create dataset class
    class GraphClassificationDataset:
        def __init__(self, graphs, name):
            self.graphs = graphs
            self.name = name
            
            if len(graphs) > 0:
                sample = graphs[0]
                self.num_node_features = sample.x.shape[1]
                
                # All datasets now have labels in standardized format [num_tasks]
                sample_y = sample.y
                if sample_y.dim() == 1:  # Standardized format: [num_tasks]
                    self.num_tasks = sample_y.shape[0]
                    
                    if self.num_tasks == 1:
                        # Single-task: determine number of classes
                        all_labels = [graph.y[0].item() for graph in graphs]
                        unique_labels = sorted(set(label for label in all_labels if not np.isnan(label)))
                        self.num_classes = len(unique_labels)
                        print(f"Single-task graph classification: {len(graphs)} graphs, "
                              f"{self.num_classes} classes, {self.num_node_features} text features per node")
                    else:
                        # Multi-task: typically binary classification per task
                        self.num_classes = 2  # Binary per task
                        print(f"Multi-task graph classification: {len(graphs)} graphs, "
                              f"{self.num_tasks} tasks, {self.num_classes} classes per task, "
                              f"{self.num_node_features} text features per node")
                else:
                    raise ValueError(f"Unexpected label dimensionality: {sample_y.dim()}, expected 1D tensor")
            else:
                raise ValueError(f"Empty graph list for {name}")
        
        def __len__(self):
            return len(self.graphs)
        
        def __getitem__(self, idx):
            return self.graphs[idx]
        
        def __setitem__(self, idx, value):
            """Support item assignment for dataset processing."""
            self.graphs[idx] = value
    
    return GraphClassificationDataset(graphs, name)


def load_tu_dataset(name, root='../dataset/TU'):
    """
    Load a TUDataset (graph classification benchmark).
    
    Args:
        name (str): Dataset name (e.g., 'MUTAG', 'PROTEINS', 'DD', etc.)
        root (str): Root directory for storing datasets
        
    Returns:
        dataset: PyTorch Geometric dataset
    """
    try:
        # Ensure the directory exists
        os.makedirs(root, exist_ok=True)
        
        # Load the dataset
        dataset = TUDataset(root=root, name=name, transform=NormalizeFeatures())
        
        print(f"Loaded {name}: {len(dataset)} graphs, {dataset.num_classes} classes, "
              f"{dataset.num_node_features} node features")
        
        return dataset
    except Exception as e:
        print(f"Failed to load TU dataset {name}: {e}")
        return None


def load_dataset(name, root='./dataset'):
    """
    Load a graph classification dataset. Tries TSGFM format first, then TU format.
    
    Args:
        name (str): Dataset name
        root (str): Root directory for datasets
        
    Returns:
        dataset: PyTorch Geometric dataset or None if loading fails
    """
    # List of known TSGFM graph classification datasets (batched format)
    tsgfm_graph_datasets = ['bace', 'bbbp', 'chemhiv', 'chempcba', 'chemblpre', 'muv', 'tox21', 'toxcast']
    
    # Try TSGFM format first for known datasets
    if name.lower() in tsgfm_graph_datasets:
        print(f"Attempting to load {name} as TSGFM dataset...")
        dataset = load_tsgfm_dataset(name, root)
        if dataset is not None:
            return dataset
        
        print(f"Failed to load {name} as TSGFM dataset, trying TU format...")
    
    # Try TU format
    print(f"Attempting to load {name} as TU dataset...")
    dataset = load_tu_dataset(name, os.path.join(root, 'TU'))
    
    return dataset


def load_precomputed_splits(dataset_name, root='./dataset'):
    """
    Load pre-computed splits from TSGFM datasets.
    
    Args:
        dataset_name (str): Name of the dataset
        root (str): Root directory for datasets
        
    Returns:
        dict: Contains 'train', 'val', 'test' indices or None if not found
    """
    split_path = os.path.join(root, dataset_name, 'e2e_graph_splits.pt')
    
    if not os.path.exists(split_path):
        print(f"No pre-computed splits found for {dataset_name} at {split_path}")
        return None
    
    try:
        splits = torch.load(split_path, map_location='cpu')
        split_key = f'{dataset_name}_e2e_graph'
        
        if split_key not in splits:
            print(f"Split key {split_key} not found in splits file")
            return None
        
        dataset_splits = splits[split_key]
        
        # Convert to numpy arrays and rename 'valid' to 'val' for consistency
        split_idx = {
            'train': np.array(dataset_splits['train']),
            'val': np.array(dataset_splits['valid']),
            'test': np.array(dataset_splits['test'])
        }
        
        print(f"Loaded pre-computed splits for {dataset_name} - "
              f"Train: {len(split_idx['train'])}, Val: {len(split_idx['val'])}, Test: {len(split_idx['test'])}")
        
        return split_idx
        
    except Exception as e:
        print(f"Error loading pre-computed splits for {dataset_name}: {e}")
        return None


def create_scaffold_splits(dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42, pretraining_mode=False):
    """
    Create scaffold splits for molecular datasets (fallback when no pre-computed splits).
    TODO: Implement scaffold splitting using RDKit.
    
    For now, falls back to random splitting with warning.
    """
    print(f"WARNING: Scaffold splitting not yet implemented, using random splits for {dataset.name}")
    return create_random_splits(dataset, train_ratio, val_ratio, test_ratio, seed, pretraining_mode)


def create_random_splits(dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42, pretraining_mode=False):
    """
    Create random train/validation/test splits for graph classification.
    Used as fallback when pre-computed splits are not available.
    
    Args:
        dataset: PyTorch Geometric dataset
        train_ratio (float): Proportion of data for training
        val_ratio (float): Proportion of data for validation  
        test_ratio (float): Proportion of data for testing
        seed (int): Random seed for reproducibility
        pretraining_mode (bool): If True, only create train/val splits
        
    Returns:
        dict: Contains 'train', 'val' (and 'test' if not pretraining_mode) indices
    """
    if pretraining_mode:
        assert abs(train_ratio + val_ratio - 1.0) < 1e-6, "For pretraining mode, train_ratio + val_ratio must sum to 1.0"
    else:
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    # Create random permutation of indices
    indices = np.random.permutation(len(dataset))
    
    # Calculate split points
    train_end = int(len(dataset) * train_ratio)
    
    if pretraining_mode:
        # Only train/val splits
        val_end = len(dataset)  # Use all remaining data for validation
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        
        print(f"Pretraining split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}")
        
        return {
            'train': torch.tensor(train_idx),
            'val': torch.tensor(val_idx)
        }
    else:
        # Traditional train/val/test splits
        val_end = int(len(dataset) * (train_ratio + val_ratio))
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        print(f"Random split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
        
        return {
            'train': torch.tensor(train_idx),
            'val': torch.tensor(val_idx), 
            'test': torch.tensor(test_idx)
        }


def create_dataset_splits(dataset, dataset_name, root='./dataset', train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42, pretraining_mode=False):
    """
    Create dataset splits with priority order:
    1. Pre-computed splits (TSGFM)
    2. Scaffold splits (TODO)  
    3. Random splits (fallback)
    
    Args:
        dataset: PyTorch Geometric dataset
        dataset_name (str): Name of the dataset
        root (str): Root directory for datasets
        train_ratio (float): Proportion for training (if creating new splits)
        val_ratio (float): Proportion for validation (if creating new splits)
        test_ratio (float): Proportion for testing (if creating new splits)
        seed (int): Random seed for reproducibility
        pretraining_mode (bool): If True, only create train/val splits (no test) with optimized ratios
        
    Returns:
        dict: Contains 'train', 'val' (and 'test' if not pretraining_mode) indices
    """
    # Optimize ratios for pretraining mode (maximize training data)
    if pretraining_mode:
        # For pretraining, use 80/20 split (no test set)
        train_ratio = 0.8
        val_ratio = 0.2
        test_ratio = 0.0
        print(f"Pretraining mode: using {train_ratio:.1%} train / {val_ratio:.1%} val split (no test)")
    
    # 1. Try to load pre-computed splits first
    split_idx = load_precomputed_splits(dataset_name, root)
    if split_idx is not None:
        # If pretraining mode and splits have test set, merge test into train
        if pretraining_mode and 'test' in split_idx:
            print(f"Merging test set into training for pretraining")
            # Convert to torch tensors if they're numpy arrays
            train_tensor = torch.tensor(split_idx['train']) if isinstance(split_idx['train'], np.ndarray) else split_idx['train']
            test_tensor = torch.tensor(split_idx['test']) if isinstance(split_idx['test'], np.ndarray) else split_idx['test']
            val_tensor = torch.tensor(split_idx['val']) if isinstance(split_idx['val'], np.ndarray) else split_idx['val']
            
            combined_train = torch.cat([train_tensor, test_tensor])
            split_idx = {
                'train': combined_train,
                'val': val_tensor
            }
        return split_idx
    
    # 2. Fall back to scaffold splits for molecular datasets
    molecular_datasets = ['bace', 'bbbp', 'chemhiv', 'chempcba', 'muv', 'tox21', 'toxcast', 'chemblpre']
    if dataset_name.lower() in molecular_datasets:
        return create_scaffold_splits(dataset, train_ratio, val_ratio, test_ratio, seed, pretraining_mode)
    
    # 3. Fall back to random splits for other datasets
    print(f"Using random splits for {dataset_name}")
    return create_random_splits(dataset, train_ratio, val_ratio, test_ratio, seed, pretraining_mode)


def prepare_graph_data_for_pfn(dataset, split_idx, context_k=32, device='cuda', context_only_mode=False):
    """
    Prepare graph data for PFN-based graph classification.
    This function organizes graphs by class and prepares context samples.
    
    Args:
        dataset: PyTorch Geometric dataset
        split_idx (dict): Dictionary with 'train', 'val', 'test' indices
        context_k (int): Number of context graphs per class
        device (str): Device to load data onto
        context_only_mode (bool): If True, sample minimal context data for memory efficiency
        
    Returns:
        dict: Processed data containing context and target information
    """
    # Check if this is a multi-task dataset
    sample_graph = dataset[0]
    is_multitask = hasattr(sample_graph, 'task_mask') and sample_graph.y.numel() > 1
    
    if is_multitask:
        # Smart incremental context sampling for multi-task datasets
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        start_time = time.time()
        num_tasks = sample_graph.y.numel()
        print(f"Starting smart context sampling for {num_tasks} tasks...")
        
        
        # Single-pass multi-task sampling (much faster for large datasets)
        sampling_start = time.time()
        task_contexts = {}  # Task-specific contexts (indices only)
        
        # Initialize reservoir for all task/class combinations
        reservoirs = {}
        counts = {}
        for task_idx in range(num_tasks):
            task_contexts[task_idx] = {0: [], 1: []}
            reservoirs[task_idx] = {0: [], 1: []}
            counts[task_idx] = {0: 0, 1: 0}
        
        # Hybrid CPU-GPU approach - GPU for filtering, CPU for sampling
        print(f"  Hybrid CPU-GPU sampling for {num_tasks} tasks across {len(split_idx['train'])} graphs...")
        
        train_indices = split_idx['train']
        
        # For reservoir sampling, CPU-only is often faster due to random number generation
        # But we can use smart batching and early termination
        
        # Track completion status for early termination
        completed_tasks = set()
        total_combinations = num_tasks * 2  # tasks × classes
        
        for idx in train_indices:
            # Early termination: stop when all reservoirs are full
            if len(completed_tasks) >= total_combinations:
                print(f"  Early termination: all reservoirs full after {idx - train_indices[0] + 1} graphs")
                break
                
            graph = dataset[idx]
            valid_tasks = torch.where(graph.task_mask > 0)[0]
            
            # Process all valid tasks for this graph efficiently
            for task_idx in valid_tasks:
                task_idx_int = task_idx.item()
                class_label = graph.y[task_idx].item()
                
                if class_label in [0, 1]:
                    reservoir = reservoirs[task_idx_int][class_label]
                    count = counts[task_idx_int][class_label]
                    
                    # Check if this (task, class) combination is already complete
                    task_class_key = (task_idx_int, class_label)
                    if task_class_key in completed_tasks:
                        continue
                    
                    count += 1
                    if len(reservoir) < context_k:
                        reservoir.append(idx)
                        # Mark as complete when reservoir is full
                        if len(reservoir) == context_k:
                            completed_tasks.add(task_class_key)
                    else:
                        # Reservoir sampling replacement
                        j = np.random.randint(0, count)
                        if j < context_k:
                            reservoir[j] = idx
                    
                    counts[task_idx_int][class_label] = count
        
        # Report efficiency
        total_processed = min(len(train_indices), idx - train_indices[0] + 1) if 'idx' in locals() else len(train_indices)
        efficiency = (len(train_indices) - total_processed) / len(train_indices) * 100 if total_processed < len(train_indices) else 0
        print(f"  Early termination saved {efficiency:.1f}% of processing ({len(train_indices) - total_processed} graphs skipped)")
        
        # Copy results and pad if needed
        for task_idx in range(num_tasks):
            for class_label in [0, 1]:
                reservoir = reservoirs[task_idx][class_label]
                
                # Fast padding with replacement if needed
                if len(reservoir) < context_k and len(reservoir) > 0:
                    # Extend efficiently instead of appending one by one
                    needed = context_k - len(reservoir)
                    padding = np.random.choice(reservoir, needed, replace=True)
                    reservoir.extend(padding.tolist())
                
                # Store indices only (no graph loading yet)
                task_contexts[task_idx][class_label] = reservoir
        
        sampling_time = time.time() - sampling_start
        
        # Step 3: GPU memory deduplication (only at the end)
        dedup_start = time.time()
        unique_indices = set()
        
        # Collect all unique graph indices across all tasks
        for task_data in task_contexts.values():
            for class_data in task_data.values():
                unique_indices.update(class_data)
        
        print(f"Deduplication: {len(unique_indices)} unique graphs out of {num_tasks * 2 * context_k} total samples")
        
        # Step 4: Load unique graphs to GPU/create graph objects only once
        if context_only_mode:
            # For context_only_mode, just keep the indices - no actual loading
            pass  # task_contexts already contains indices
        else:
            # Load each unique graph only once
            gpu_graph_cache = {}
            for idx in unique_indices:
                gpu_graph_cache[idx] = dataset[idx]  # Load once
            
            # Replace indices with actual graph objects using cache
            for task_idx in task_contexts:
                for class_label in task_contexts[task_idx]:
                    indices = task_contexts[task_idx][class_label]
                    task_contexts[task_idx][class_label] = [gpu_graph_cache[idx] for idx in indices]
        
        dedup_time = time.time() - dedup_start
        total_time = time.time() - start_time
        
        print(f"Context sampling completed in {total_time:.2f}s:")
        print(f"  - Lazy sampling: {sampling_time:.2f}s")
        print(f"  - Deduplication: {dedup_time:.2f}s")
        print(f"  - Memory efficiency: {(1 - len(unique_indices)/(num_tasks * 2 * context_k))*100:.1f}% reduction")
        
        # Return unified task-aware context structure
        final_context_graphs = task_contexts  # task_idx -> {0: [...], 1: [...]}
        context_labels = {task_idx: {0: 0, 1: 1} for task_idx in range(num_tasks)}
        is_multitask_dataset = True
        
    else:
        # Single-task dataset: treat as task 0 with same deduplication approach
        import time
        start_time = time.time()
        
        print("Starting context sampling for single-task dataset...")
        
        # Lazy reservoir sampling for single-task dataset
        def reservoir_sample_for_class(class_label, k):
            """Reservoir sampling for specific class in single-task dataset"""
            reservoir = []
            count = 0
            
            for idx in split_idx['train']:
                graph = dataset[idx]
                if graph.y.numel() == 1:
                    label = graph.y.item()
                else:
                    label = graph.y[0].item()
                
                if label == class_label:
                    count += 1
                    if len(reservoir) < k:
                        reservoir.append(idx)
                    else:
                        # Replace with probability k/count
                        j = np.random.randint(0, count)
                        if j < k:
                            reservoir[j] = idx
            return reservoir
        
        # Binary classification datasets always have classes [0, 1]
        available_classes = {0, 1}
        
        # Direct sampling per class
        task_contexts = {0: {}}  # Single task = task 0
        unique_indices = set()
        
        for class_label in available_classes:
            selected_indices = reservoir_sample_for_class(class_label, context_k)
            
            # Pad with replacement if needed
            if len(selected_indices) < context_k and len(selected_indices) > 0:
                while len(selected_indices) < context_k:
                    selected_indices.append(np.random.choice(selected_indices))
            
            # Store indices and collect for deduplication
            task_contexts[0][class_label] = selected_indices
            unique_indices.update(selected_indices)
        
        # Apply deduplication (load unique graphs only once)
        if not context_only_mode:
            gpu_graph_cache = {}
            for idx in unique_indices:
                gpu_graph_cache[idx] = dataset[idx]
            
            # Replace indices with graph objects
            for class_label in task_contexts[0]:
                indices = task_contexts[0][class_label]
                task_contexts[0][class_label] = [gpu_graph_cache[idx] for idx in indices]
        
        total_time = time.time() - start_time
        total_samples = sum(len(indices) for indices in task_contexts[0].values())
        print(f"Single-task context sampling completed in {total_time:.2f}s")
        print(f"  - Classes found: {list(available_classes)}")
        print(f"  - Unique graphs: {len(unique_indices)} out of {total_samples} samples")
        print(f"  - Memory efficiency: {(1 - len(unique_indices)/max(1, total_samples))*100:.1f}% reduction")
        
        # Return unified structure (single task = task 0)
        final_context_graphs = task_contexts  # {0: {class: [...]}}
        context_labels = {0: {class_label: class_label for class_label in available_classes}}
        is_multitask_dataset = False
    
    return {
        'context_graphs': final_context_graphs,  # task_idx -> {class: [graphs]}
        'context_labels': context_labels,        # task_idx -> {class: class_label}
        'split_idx': split_idx,
        'dataset': dataset,
        'num_classes': dataset.num_classes,
        'num_features': dataset.num_node_features,
        'is_multitask': is_multitask_dataset,
        'num_tasks': num_tasks if is_multitask_dataset else 1
    }


def create_graph_batch(graphs, device='cuda'):
    """
    Create a batch from a list of graphs.
    
    Args:
        graphs (list): List of PyTorch Geometric Data objects
        device (str): Device to load batch onto
        
    Returns:
        batch: Batched graph data
    """
    batch = Batch.from_data_list(graphs).to(device)
    return batch


def graph_pooling(node_embeddings, batch, pooling_method='mean'):
    """
    Perform graph-level pooling to aggregate node embeddings into graph embeddings.
    
    Args:
        node_embeddings (torch.Tensor): Node embeddings [num_nodes, hidden_dim]
        batch (torch.Tensor): Batch assignment for each node
        pooling_method (str): Pooling method ('mean', 'max', 'sum')
        
    Returns:
        torch.Tensor: Graph-level embeddings [num_graphs, hidden_dim]
    """
    if pooling_method == 'mean':
        from torch_geometric.nn import global_mean_pool
        return global_mean_pool(node_embeddings, batch)
    elif pooling_method == 'max':
        from torch_geometric.nn import global_max_pool
        return global_max_pool(node_embeddings, batch)
    elif pooling_method == 'sum':
        from torch_geometric.nn import global_add_pool
        return global_add_pool(node_embeddings, batch)
    else:
        raise ValueError(f"Unsupported pooling method: {pooling_method}")


def load_all_graph_datasets(dataset_names, device='cuda', pretraining_mode=False, 
                          context_only_mode=False, context_k=32):
    """
    Load multiple graph classification datasets with memory-efficient context sampling.
    
    Args:
        dataset_names (list): List of dataset names to load
        device (str): Device to load data onto
        pretraining_mode (bool): If True, optimize splits for pretraining (train/val only)
        context_only_mode (bool): If True, sample minimal context data from train split for memory efficiency
        context_k (int): Number of context graphs per class to sample
        
    Returns:
        tuple: (datasets, processed_data_list)
    """
    import time
    overall_start = time.time()
    
    datasets = []
    processed_data_list = []
    
    for name in dataset_names:
        print(f"\nLoading dataset: {name}")
        
        # Time dataset loading
        load_start = time.time()
        dataset = load_dataset(name)  # Use the unified loading function
        load_time = time.time() - load_start
        
        if dataset is None:
            print(f"Skipping {name} due to loading error")
            continue
        print(f"  Dataset loading: {load_time:.2f}s")
        
        # Time split creation
        split_start = time.time()
        split_idx = create_dataset_splits(dataset, name, pretraining_mode=pretraining_mode)
        split_time = time.time() - split_start
        print(f"  Split creation: {split_time:.2f}s")
        
        # Time context preparation
        prep_start = time.time()
        processed_data = prepare_graph_data_for_pfn(
            dataset, split_idx, device=device, context_k=context_k, 
            context_only_mode=context_only_mode
        )
        prep_time = time.time() - prep_start
        print(f"  Context preparation: {prep_time:.2f}s")
        
        datasets.append(dataset)
        processed_data_list.append(processed_data)
        
        total_dataset_time = time.time() - load_start
        print(f"Successfully processed {name} with {len(dataset)} graphs (total: {total_dataset_time:.2f}s)")
    
    overall_time = time.time() - overall_start
    print(f"\nAll datasets loaded in {overall_time:.2f}s")
    
    return datasets, processed_data_list




def create_task_filtered_datasets(dataset, split_idx):
    """
    Create task-specific filtered datasets for multi-task learning.
    Each task gets its own dataset containing only graphs with valid labels for that task.
    
    Args:
        dataset: PyTorch Geometric dataset
        split_idx (dict): Dictionary with split indices
        
    Returns:
        dict: For single-task: {0: {split: [indices]}}
              For multi-task: {task_idx: {split: [indices with valid labels for task_idx]}}
    """
    # Check if multi-task dataset
    sample_graph = dataset[0] if len(dataset) > 0 else None
    is_multitask = (sample_graph is not None and 
                   hasattr(sample_graph, 'task_mask') and 
                   sample_graph.y.numel() > 1)
    
    if not is_multitask:
        # Single task: return original splits wrapped in task 0
        return {0: split_idx}
    
    # Multi-task: filter each split by task validity
    num_tasks = sample_graph.y.numel()
    task_filtered_splits = {}
    
    print(f"Prefiltering dataset for {num_tasks} tasks...")
    
    for task_idx in range(num_tasks):
        task_filtered_splits[task_idx] = {}
        
        for split_name, indices in split_idx.items():
            valid_indices = []
            
            for idx in indices:
                graph = dataset[idx]
                if hasattr(graph, 'task_mask') and graph.task_mask[task_idx]:
                    valid_indices.append(idx)
            
            task_filtered_splits[task_idx][split_name] = valid_indices
            
        # Log filtering results
        total_samples = sum(len(indices) for indices in split_idx.values())
        valid_samples = sum(len(indices) for indices in task_filtered_splits[task_idx].values())
        print(f"  Task {task_idx}: {valid_samples}/{total_samples} samples ({valid_samples/total_samples*100:.1f}%)")
    
    return task_filtered_splits


def create_data_loaders(dataset, split_idx, batch_size=128, shuffle=True, task_idx=None):
    """
    Create PyTorch data loaders for train/validation/test splits.
    
    Args:
        dataset: PyTorch Geometric dataset
        split_idx (dict): Dictionary with split indices (can be task-filtered)
        batch_size (int): Batch size for data loaders
        shuffle (bool): Whether to shuffle training data
        task_idx (int, optional): Task index for logging purposes
        
    Returns:
        dict: Dictionary with 'train', 'val', 'test' data loaders
    """
    loaders = {}
    
    for split_name, indices in split_idx.items():
        if len(indices) == 0:
            # Handle empty splits gracefully
            print(f"Warning: Empty {split_name} split for task {task_idx}")
            loaders[split_name] = DataLoader([], batch_size=batch_size)
            continue
            
        subset = [dataset[i] for i in indices]
        shuffle_data = shuffle if split_name == 'train' else False
        
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle_data,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=False  # Disabled because data is already on GPU
        )
        loaders[split_name] = loader
        
    # Create status message with available splits
    status_parts = []
    for split_name in ['train', 'val', 'test']:
        if split_name in loaders:
            status_parts.append(f"{split_name.capitalize()}: {len(loaders[split_name])} batches")
    
    print(f"Created data loaders - {', '.join(status_parts)}")
    
    return loaders


def process_graph_features(dataset, hidden_dim, device='cuda', 
                         use_identity_projection=False, projection_small_dim=128, projection_large_dim=256,
                         use_full_pca=False, sign_normalize=False, normalize_data=False,
                         padding_strategy='zero', use_batchnorm=False):
    """
    Process graph features to match the required hidden dimension using PCA and padding.
    Custom implementation for graph classification.
    
    Args:

        dataset: PyTorch Geometric dataset
        hidden_dim (int): Target hidden dimension
        device (str): Device for computations
        use_identity_projection (bool): Whether to use identity projection
        projection_small_dim (int): Small dimension for identity projection
        projection_large_dim (int): Large dimension for identity projection
        use_full_pca (bool): Use full SVD instead of lowrank PCA
        sign_normalize (bool): Normalize eigenvector signs
        normalize_data (bool): Apply normalization to final features
        padding_strategy (str): Strategy for padding ('zero', 'random', 'repeat')
        use_batchnorm (bool): Use BatchNorm-style normalization
        
    Returns:
        dict: Processing information and flags
    """
    import time
    
    # Check if dataset has node features
    sample = dataset[0] if len(dataset) > 0 else None
    if sample is None:
        raise ValueError("Empty dataset")
    
    # Determine original dimension
    if hasattr(sample, 'x') and sample.x is not None:
        original_dim = sample.x.shape[1]
    else:
        raise ValueError("Dataset has no node features")
    
    print(f"Processing graph features: {original_dim}D -> {hidden_dim}D")
    st = time.time()
    
    # Collect all node features from all graphs for PCA
    collection_start = time.time()
    all_node_features = []
    node_counts = []
    
    for graph in dataset:
        all_node_features.append(graph.x.to(device))
        node_counts.append(graph.x.size(0))
    
    # Stack all features
    stacked_features = torch.cat(all_node_features, dim=0)
    collection_time = time.time() - collection_start
    print(f"  Feature collection: {collection_time:.2f}s")
    
    if use_identity_projection:
        # Identity projection pathway: PCA to small_dim, then project to large_dim
        print(f"Using identity projection: {original_dim}D -> PCA to {projection_small_dim}D -> Project to {projection_large_dim}D")
        
        if original_dim >= projection_small_dim:
            # Apply PCA to small dimension
            pca_start = time.time()
            if use_full_pca:
                U, S, V = torch.svd(stacked_features)
                U = U[:, :projection_small_dim]
                S = S[:projection_small_dim]
            else:
                U, S, V = torch.pca_lowrank(stacked_features, q=projection_small_dim)
            pca_time = time.time() - pca_start
            print(f"  PCA computation: {pca_time:.2f}s")
            
            # Sign normalization
            if sign_normalize:
                for i in range(projection_small_dim):
                    feature_vector = U[:, i] * S[i]
                    max_idx = torch.argmax(torch.abs(feature_vector))
                    if feature_vector[max_idx] < 0:
                        U[:, i] = -U[:, i]
            
            processed_features = torch.mm(U, torch.diag(S))
        else:
            # For small datasets, use all dimensions + padding if needed
            processed_features = stacked_features
            if processed_features.size(1) < projection_small_dim:
                pad_size = projection_small_dim - processed_features.size(1)
                padding = torch.zeros(processed_features.size(0), pad_size, device=device)
                processed_features = torch.cat([processed_features, padding], dim=1)
        
        target_dim = projection_large_dim
        needs_identity_projection = True
        
    else:
        # Standard PCA pathway
        if original_dim >= hidden_dim:
            # Apply PCA to target dimension
            pca_start = time.time()
            if use_full_pca:
                U, S, V = torch.svd(stacked_features)
                U = U[:, :hidden_dim]
                S = S[:hidden_dim]
            else:
                U, S, V = torch.pca_lowrank(stacked_features, q=hidden_dim)
            pca_time = time.time() - pca_start
            print(f"  PCA computation: {pca_time:.2f}s")
            
            # Sign normalization
            if sign_normalize:
                for i in range(hidden_dim):
                    feature_vector = U[:, i] * S[i]
                    max_idx = torch.argmax(torch.abs(feature_vector))
                    if feature_vector[max_idx] < 0:
                        U[:, i] = -U[:, i]
            
            processed_features = torch.mm(U, torch.diag(S))
            
        else:
            # Not enough features, apply PCA to all available dimensions then pad
            pca_dim = min(original_dim, stacked_features.size(1))
            
            if use_full_pca:
                U, S, V = torch.svd(stacked_features)
                U = U[:, :pca_dim]
                S = S[:pca_dim]
            else:
                U, S, V = torch.pca_lowrank(stacked_features, q=pca_dim)
            
            # Sign normalization
            if sign_normalize:
                for i in range(pca_dim):
                    feature_vector = U[:, i] * S[i]
                    max_idx = torch.argmax(torch.abs(feature_vector))
                    if feature_vector[max_idx] < 0:
                        U[:, i] = -U[:, i]
            
            pca_features = torch.mm(U, torch.diag(S))
            
            # Apply padding to reach target dimension
            if pca_features.size(1) < hidden_dim:
                padding_size = hidden_dim - pca_features.size(1)
                
                if padding_strategy == 'zero':
                    padding = torch.zeros(pca_features.size(0), padding_size, device=device)
                    processed_features = torch.cat([pca_features, padding], dim=1)
                    
                elif padding_strategy == 'random':
                    # Random padding from same distribution as real features
                    real_std = pca_features.std(dim=0, keepdim=True)
                    real_mean = pca_features.mean(dim=0, keepdim=True)
                    random_padding = torch.randn(pca_features.size(0), padding_size, device=device) * real_std.mean() + real_mean.mean()
                    processed_features = torch.cat([pca_features, random_padding], dim=1)
                    
                elif padding_strategy == 'repeat':
                    # Feature repetition
                    processed_features = pca_features
                    while processed_features.size(1) < hidden_dim:
                        remaining = hidden_dim - processed_features.size(1)
                        repeat_size = min(remaining, pca_features.size(1))
                        processed_features = torch.cat([processed_features, pca_features[:, :repeat_size]], dim=1)
                        
                else:
                    raise ValueError(f"Unknown padding strategy: {padding_strategy}")
                    
                print(f"Applied {padding_strategy} padding ({pca_features.size(1)} -> {hidden_dim})")
            else:
                processed_features = pca_features[:, :hidden_dim]
        
        target_dim = hidden_dim
        needs_identity_projection = False
    
    # Apply normalization if requested
    if normalize_data:
        if use_batchnorm:
            # BatchNorm-style: normalize each feature across batch
            batch_mean = processed_features.mean(dim=0, keepdim=True)
            batch_std = processed_features.std(dim=0, keepdim=True, unbiased=False)
            processed_features = (processed_features - batch_mean) / (batch_std + 1e-5)
            print("Applied BatchNorm-style normalization")
        else:
            # LayerNorm-style: L2 normalization per sample across features
            processed_features = F.normalize(processed_features, p=2, dim=1)
            print("Applied LayerNorm-style normalization")
    
    # Split the processed features back to individual graphs
    current_idx = 0
    for i, graph in enumerate(dataset):
        num_nodes = node_counts[i]
        graph.x = processed_features[current_idx:current_idx + num_nodes]
        current_idx += num_nodes
    
    processing_info = {
        'original_dim': original_dim,
        'processed_dim': processed_features.size(1),
        'target_dim': target_dim,
        'needs_identity_projection': needs_identity_projection,
        'projection_target_dim': projection_large_dim if needs_identity_projection else None
    }
    
    print(f"Graph feature processing completed in {time.time()-st:.2f}s: {original_dim}D -> {processed_features.size(1)}D")
    
    return processing_info

def select_graph_context(processed_data, context_k=32):
    """
    Select context graphs for each class in the processed data.
    
    Args:
        processed_data (dict): Processed data from prepare_graph_data_for_pfn
        context_k (int): Number of context graphs per class
        
    Returns:
        dict: Context data with selected graphs and labels
    """
    context_graphs = processed_data['context_graphs']
    context_labels = processed_data['context_labels']
    
    # Ensure we have exactly context_k graphs per class
    selected_context = {}
    selected_labels = {}
    
    for class_label, graphs in context_graphs.items():
        if len(graphs) >= context_k:
            selected_indices = np.random.choice(len(graphs), context_k, replace=False)
            selected_graphs = [graphs[i] for i in selected_indices]
        else:
            # Sample with replacement if needed
            selected_indices = np.random.choice(len(graphs), context_k, replace=True)
            selected_graphs = [graphs[i] for i in selected_indices]
        
        selected_context[class_label] = selected_graphs
        selected_labels[class_label] = class_label
    
    return {
        'context_graphs': selected_context,
        'context_labels': selected_labels,
        'num_classes': processed_data['num_classes']
    }
