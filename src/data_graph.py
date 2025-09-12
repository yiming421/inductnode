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
from torch.utils.data import Subset
import numpy as np
import os
import random
import psutil, gc
from torch_sparse import SparseTensor
import time
import math  # (possibly needed later)
import hashlib
from pathlib import Path
# FUG-specific loaders moved to separate module
try:
    from .data_graph_fug_simple import load_ogb_fug_dataset
except ImportError:
    # Fallback if relative import path differs when executed as script
    try:
        from data_graph_fug_simple import load_ogb_fug_dataset
    except Exception:
        load_ogb_fug_dataset = lambda *a, **k: None


def load_gpse_embeddings(dataset_name, gpse_base_path="/home/maweishuo/GPSE/datasets"):
    """
    Load pre-computed GPSE embeddings for a dataset.
    
    Args:
        dataset_name: Name of the dataset (should match GPSE naming)
        gpse_base_path: Base path to GPSE datasets directory
        
    Returns:
        tuple: (node_embeddings_tensor, slices_tensor) or (None, None) if not found
    """
    # Map inductnode dataset names to GPSE dataset names
    dataset_mapping = {
        'bace': 'OGB-ogbg-molbace',
        'bbbp': 'OGB-ogbg-molbbbp', 
        'chemhiv': 'OGB-ogbg-molhiv',
        'tox21': 'OGB-ogbg-moltox21',
        'toxcast': 'OGB-ogbg-moltoxcast',
    }
    
    gpse_dataset_name = dataset_mapping.get(dataset_name)
    if not gpse_dataset_name:
        print(f"Warning: No GPSE embeddings available for dataset '{dataset_name}'")
        print(f"Available datasets: {list(dataset_mapping.keys())}")
        return None, None
    
    data_path = os.path.join(gpse_base_path, gpse_dataset_name, "pe_stats_GPSE", "1.0", "data.pt")
    slices_path = os.path.join(gpse_base_path, gpse_dataset_name, "pe_stats_GPSE", "1.0", "slices.pt")
    
    if not os.path.exists(data_path) or not os.path.exists(slices_path):
        print(f"Warning: GPSE files not found for {gpse_dataset_name}")
        print(f"  Expected: {data_path}")
        print(f"  Expected: {slices_path}")
        return None, None
    
    try:
        node_embeddings = torch.load(data_path, map_location='cpu')
        slices = torch.load(slices_path, map_location='cpu')
        
        print(f"Loaded GPSE embeddings for {gpse_dataset_name}")
        print(f"  Node embeddings shape: {node_embeddings.shape}")
        print(f"  Number of graphs: {len(slices) - 1}")
        print(f"  Embedding dimension: {node_embeddings.shape[1]}")
        
        return node_embeddings, slices
        
    except Exception as e:
        print(f"Error loading GPSE embeddings: {e}")
        return None, None


def create_gpse_node_lookup_table(dataset, gpse_embeddings, gpse_slices):
    """
    Create a node lookup table from GPSE embeddings that matches the dataset's node indexing.
    
    Args:
        dataset: The original dataset 
        gpse_embeddings: Concatenated GPSE node embeddings [total_nodes, embedding_dim]
        gpse_slices: Graph boundary slices [num_graphs + 1]
        
    Returns:
        torch.Tensor: Node lookup table [num_unique_nodes, embedding_dim]
    """
    print("Creating GPSE node lookup table...")
    
    # Verify that we have embeddings for all graphs in the dataset
    num_dataset_graphs = len(dataset)
    num_gpse_graphs = len(gpse_slices) - 1
    
    if num_dataset_graphs != num_gpse_graphs:
        print(f"Warning: Dataset has {num_dataset_graphs} graphs but GPSE has {num_gpse_graphs} graphs")
        # Take the minimum to avoid index errors
        num_graphs = min(num_dataset_graphs, num_gpse_graphs)
        print(f"Using first {num_graphs} graphs from both")
    else:
        num_graphs = num_dataset_graphs
    
    # Collect all node embeddings from GPSE data
    all_node_embeddings = []
    
    for graph_idx in range(num_graphs):
        start_idx = gpse_slices[graph_idx]
        end_idx = gpse_slices[graph_idx + 1]
        graph_embeddings = gpse_embeddings[start_idx:end_idx]  # [num_nodes_in_graph, embedding_dim]
        all_node_embeddings.append(graph_embeddings)
    
    # Create a single lookup table by concatenating all unique node embeddings
    # For graph classification, we can simply concatenate all node embeddings
    node_lookup_table = torch.cat(all_node_embeddings, dim=0)  # [total_nodes, embedding_dim]
    
    print(f"Created GPSE node lookup table: {node_lookup_table.shape}")
    
    return node_lookup_table


def update_dataset_with_gpse_embeddings(dataset, dataset_name, gpse_path):
    """
    Update dataset to use GPSE embeddings instead of original node embeddings.
    
    Args:
        dataset: The dataset to update
        dataset_name: Name of the dataset for GPSE lookup
        gpse_path: Path to GPSE datasets directory
        
    Returns:
        bool: True if successfully updated, False otherwise
    """
    # Load GPSE embeddings
    gpse_embeddings, gpse_slices = load_gpse_embeddings(dataset_name, gpse_path)
    
    if gpse_embeddings is None or gpse_slices is None:
        print(f"Could not load GPSE embeddings for {dataset_name}, using original embeddings")
        return False
    
    # Create new node lookup table from GPSE embeddings
    gpse_node_table = create_gpse_node_lookup_table(dataset, gpse_embeddings, gpse_slices)
    
    # Update the dataset's node embedding table
    dataset.node_embs = gpse_node_table
    dataset.num_node_features = gpse_node_table.shape[1]
    
    print(f"Successfully updated dataset with GPSE embeddings!")
    print(f"  New embedding dimension: {dataset.num_node_features}")
    
    return True

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

    st_time = time.time()
    
    # Extract graph boundaries from metadata
    node_boundaries = metadata['x']  # Indices where each graph's nodes start/end
    edge_boundaries = metadata['edge_index']  # Indices where each graph's edges start/end
    
    num_graphs = len(node_boundaries) - 1  # Last boundary is the final endpoint
    
    print(f"Extracting {num_graphs} graphs from batched format")
    
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
    
    # Store embedding lookup tables and indices (NO eager reconstruction!)
    x_indices = data_obj.x  # Node type indices
    node_embs = data_obj.node_embs  # Node embedding lookup table
    
    # Edge embedding info (if available)
    xe_indices = None
    edge_embs = None
    if hasattr(data_obj, 'xe') and hasattr(data_obj, 'edge_embs'):
        xe_indices = data_obj.xe
        edge_embs = data_obj.edge_embs
    
    # Split the batched data into individual graphs
    graphs = []

    print(f'Time to extract graph boundaries: {time.time() - st_time:.2f} seconds')

    st_time = time.time()
    
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
        
        # Extract node indices and edges for this graph (NO feature reconstruction!)
        graph_node_indices = x_indices[node_start:node_end]  # Node type indices for this graph
        graph_edges = data_obj.edge_index[:, edge_start:edge_end]  # Shape: [2, num_edges_in_graph]
        
        # Get graph-level label(s) and add batch dimension to match OGB format [1, num_tasks]
        graph_label = graph_labels[i].unsqueeze(0)  # [num_tasks] -> [1, num_tasks]
        
        # Create task mask for multi-task datasets (1 for valid labels, 0 for NaN)
        if graph_label.dim() == 2 and graph_label.size(1) > 1:  # Multi-task: [1, num_tasks]
            task_mask = (~torch.isnan(graph_label)).float()
            graph_label = torch.where(torch.isnan(graph_label), torch.zeros_like(graph_label), graph_label).long()
        else:  # Single-task: [1, 1]
            task_mask = torch.ones_like(graph_label)
        
        # Extract edge indices for this graph (if available)
        graph_edge_indices = None
        if xe_indices is not None:
            graph_edge_indices = xe_indices[edge_start:edge_end]
        
        # --- Self-loop fallback for empty-edge graphs ---
        num_nodes = graph_node_indices.shape[0]
        num_edges = graph_edges.size(1)
        if num_edges == 0 and num_nodes > 0:
            # Create identity self-loops to form an "idle" adjacency so later GNN layers do not crash
            idle_edge_index = torch.arange(num_nodes, device=graph_node_indices.device)
            idle_edge_index = idle_edge_index.unsqueeze(0).repeat(2, 1)  # [2, num_nodes]
            original_num_edges = 0
            edge_index_for_storage = idle_edge_index
            idle_flag = True
            # Build SparseTensor identity directly
            adj_t = SparseTensor.eye(num_nodes, device=graph_node_indices.device)
        else:
            original_num_edges = num_edges
            edge_index_for_storage = graph_edges
            idle_flag = False
            adj_t = SparseTensor.from_edge_index(
                graph_edges,
                sparse_sizes=(num_nodes, num_nodes)
            ).to_symmetric().coalesce()
        
        # Create individual graph with INDICES not features
        graph = Data(
            x=graph_node_indices,
            edge_index=edge_index_for_storage,
            y=graph_label,
            task_mask=task_mask,
            num_nodes=num_nodes,
            original_num_edges=original_num_edges,
            idle_edges=idle_flag
        )
        
        # Store edge indices if available (edge type indices, not features)
        if graph_edge_indices is not None:
            graph.edge_attr = graph_edge_indices
        
        # Attach adjacency tensor
        graph.adj_t = adj_t
        
        graphs.append(graph)

    print(f'Time to create individual graphs: {time.time() - st_time:.2f} seconds')

    st_time = time.time()
    
    # Create dataset class
    class GraphClassificationDataset:
        def __init__(self, graphs, name, node_embs, edge_embs=None):
            self.graphs = graphs
            self.name = name
            
            # Store embedding lookup tables for lazy computation
            self.node_embs = node_embs  # [num_node_types, embedding_dim]
            self.edge_embs = edge_embs  # [num_edge_types, embedding_dim] or None
            
            if len(graphs) > 0:
                sample = graphs[0]
                # Use embedding dimension instead of indices dimension
                self.num_node_features = node_embs.shape[1]
                
                # All datasets now have labels in standardized format [num_tasks]
                sample_y = sample.y
                self.num_tasks = sample_y.shape[1]
                
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
                raise ValueError(f"Empty graph list for {name}")
        
        def __len__(self):
            return len(self.graphs)
        
        def __getitem__(self, idx):
            return self.graphs[idx]
        
        def __setitem__(self, idx, value):
            """Support item assignment for dataset processing."""
            self.graphs[idx] = value
    
    dataset = GraphClassificationDataset(graphs, name, node_embs, edge_embs)
    print(f"Time to create dataset class: {time.time() - st_time:.2f} seconds")
    return dataset


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


def _create_tag_dataset(data, slices, name):
    """Helper to create TAGDataset from data and slices - simple PyG InMemoryDataset style."""
    class TAGInMemoryDataset:
        def __init__(self, data, slices, name):
            self.data = data
            self.slices = slices
            self.name = name
            
            # Standard PyG dataset properties - handle None values safely
            if hasattr(data, 'y') and data.y is not None:
                self.num_classes = len(torch.unique(data.y))
            else:
                self.num_classes = 2  # Default binary
                
            if hasattr(data, 'x') and data.x is not None:
                self.num_node_features = data.x.shape[1]
            else:
                self.num_node_features = 0
        
        def __len__(self):
            # TAGDataset uses different slice keys than standard PyG
            if 'edge_index' in self.slices:
                return len(self.slices['edge_index']) - 1
            elif 'x' in self.slices:
                return len(self.slices['x']) - 1
            else:
                return 0
        
        def __getitem__(self, idx):
            from torch_geometric.data import Data
            graph_data = {}
            
            # TAGDataset has special handling for mapping features
            if hasattr(self.data, 'keys'):
                data_keys = self.data.keys()
            else:
                data_keys = []
            
            for key in data_keys:
                if key in self.slices:
                    start = self.slices[key][idx].item()
                    end = self.slices[key][idx + 1].item()
                    attr_value = getattr(self.data, key)
                    if attr_value is not None:
                        graph_data[key] = attr_value[start:end]
                else:
                    attr_value = getattr(self.data, key)
                    if attr_value is not None:
                        graph_data[key] = attr_value
            
            # TAGDataset might not have 'y' directly - check for labels via mapping
            if 'y' not in graph_data:
                # Look for label mapping in TAGDataset format
                if hasattr(self.data, 'label_map') and self.data.label_map is not None:
                    # Create a simple binary label for now (can be enhanced later)
                    graph_data['y'] = torch.tensor([0], dtype=torch.long)  # Default label
            
            return Data(**graph_data)
    
    return TAGInMemoryDataset(data, slices, name)


def load_tag_dataset(name, root='./TAGDataset', embedding_family='ST'):
    """
    Load a TAGDataset using TAGLAS Lite with full pipeline integration.
    
    Args:
        name (str): Dataset name (e.g., 'bace', 'bbbp', 'hiv')
        root (str): Root directory for TAG datasets
        embedding_family (str): Text embedding family ('ST' for Sentence Transformer, 'e5' for E5 embeddings)
        
    Returns:
        dataset: Pipeline-compatible dataset or None if loading fails
    """
    try:
        from src.taglas_lite import (
            load_tag_dataset_with_pipeline_integration,
            convert_tagdataset_to_tsgfm_format
        )
        
        print(f"Loading TAGDataset {name} with TAGLAS Lite ({embedding_family} embeddings)...")
        
        # Load using TAGLAS Lite with full pipeline integration
        tag_pipeline_dataset = load_tag_dataset_with_pipeline_integration(name, root, embedding_family)
        
        # Convert to TSGFM-compatible format for existing pipeline code
        compatible_dataset = convert_tagdataset_to_tsgfm_format(tag_pipeline_dataset)
        
        print(f"Successfully loaded TAGDataset {name}: {len(compatible_dataset)} graphs")
        print(f"  Node features: {compatible_dataset.num_node_features}D")
        print(f"  Classes: {compatible_dataset.num_classes}")
        print(f"  Text features: {compatible_dataset.has_text_features}")
        print(f"  Embedding family: {embedding_family}")
        
        return compatible_dataset
        
    except Exception as e:
        print(f"Failed to load TAGDataset {name} with TAGLAS Lite: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_dataset(name, root='./dataset', embedding_family='ST'):
    """
    Load a graph classification dataset. Extended with optional FUG+OGB path.
    Priority:
      0. FUG OGB embeddings (if env USE_FUG_EMB=1 and embeddings available)
      1. TAGDataset (if TAG_DATASET_ROOT set)
      2. TSGFM batched format
      3. TU datasets
    """
    # --- FUG OGB path (simplified for unified embedding setting) ---
    if os.environ.get('USE_FUG_EMB', '0') == '1':
        fug_root = os.environ.get('FUG_EMB_ROOT', './fug')
        ogb_root = os.environ.get('OGB_ROOT', './dataset/ogb')
        
        print(f"[Loader] Loading FUG dataset '{name}' (unified embedding setting)")
        
        # Simple unified loading - just one function needed
        from .data_graph_fug_simple import load_ogb_fug_dataset
        result = load_ogb_fug_dataset(name, ogb_root=ogb_root, fug_root=fug_root)
        if result is not None:
            dataset, fug_mapping = result  # Unpack pristine dataset and external mapping
            print(f"[Loader] Successfully loaded FUG dataset '{name}'")
            dataset.name = name
            # Return both dataset and FUG mapping - let caller handle the mapping
            return dataset, fug_mapping
        else:
            print(f"[Loader] FUG loading failed for '{name}'. Continuing...")

    # Check if we should use TAG dataset format
    tag_dataset_root = os.environ.get('TAG_DATASET_ROOT')
    if tag_dataset_root and os.path.isdir(tag_dataset_root):
        print(f"TAG dataset mode enabled - attempting to load {name} as TAGDataset...")
        dataset = load_tag_dataset(name, tag_dataset_root, embedding_family)
        if dataset is not None:
            dataset.name = name
            return dataset
        print(f"Failed to load {name} as TAGDataset, falling back to standard formats...")

    # List of known TSGFM graph classification datasets (batched format)
    tsgfm_graph_datasets = ['bace', 'bbbp', 'chemhiv', 'chempcba', 'chemblpre', 'muv', 'tox21', 'toxcast']
    
    # Try TSGFM format first for known datasets
    if name.lower() in tsgfm_graph_datasets:
        print(f"Attempting to load {name} as TSGFM dataset...")
        dataset = load_tsgfm_dataset(name, root)
        if dataset is not None:
            dataset.name = name  # Set dataset name for consistency
            return dataset
        print(f"Failed to load {name} as TSGFM dataset, trying TU format...")

    # Try TU format
    print(f"Attempting to load {name} as TU dataset...")
    dataset = load_tu_dataset(name, os.path.join(root, 'TU'))

    return dataset

def load_precomputed_splits(dataset_name, root='./dataset'):
    """
    Load pre-computed splits from TSGFM datasets.
    Supports use_tag_dataset mode by checking for TAG_DATASET_ROOT environment variable.
    
    Args:
        dataset_name (str): Name of the dataset
        root (str): Root directory for datasets
        
    Returns:
        dict: Contains 'train', 'val', 'test' indices or None if not found
    """
    # Check if we should use TAG dataset root instead
    tag_dataset_root = os.environ.get('TAG_DATASET_ROOT')
    if tag_dataset_root and os.path.isdir(tag_dataset_root):
        root = tag_dataset_root
    
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
    0. OGB official splits (when using OGB FUG datasets)
    1. Pre-computed splits (TSGFM)
    2. Scaffold splits (TODO)  
    3. Random splits (fallback)
    
    Args:
        dataset: PyTorch Geometric dataset
        dataset_name (str): Name of the dataset
        root (str): Root directory for datasets
        train_ratio (float): Proportion for training (if creating new splits)
        val_ratio (float): Proportion of data for validation (if creating new splits)
        test_ratio (float): Proportion of data for testing (if creating new splits)
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
    
    # 0. Check if this is an OGB dataset and use official OGB splits
    use_ogb_fug = os.environ.get('USE_FUG_EMB', '0') == '1'
    if use_ogb_fug and hasattr(dataset, 'get_idx_split'):
        try:
            ogb_splits = dataset.get_idx_split()
            split_idx = {
                'train': ogb_splits['train'].numpy() if hasattr(ogb_splits['train'], 'numpy') else ogb_splits['train'],
                'val': ogb_splits['valid'].numpy() if hasattr(ogb_splits['valid'], 'numpy') else ogb_splits['valid'],
                'test': ogb_splits['test'].numpy() if hasattr(ogb_splits['test'], 'numpy') else ogb_splits['test']
            }
            
            # Handle pretraining mode for OGB datasets
            if pretraining_mode and 'test' in split_idx:
                print(f"[OGB] Merging test set into training for pretraining")
                train_tensor = torch.tensor(split_idx['train']) if isinstance(split_idx['train'], np.ndarray) else split_idx['train']
                test_tensor = torch.tensor(split_idx['test']) if isinstance(split_idx['test'], np.ndarray) else split_idx['test']
                val_tensor = torch.tensor(split_idx['val']) if isinstance(split_idx['val'], np.ndarray) else split_idx['val']
                
                combined_train = torch.cat([train_tensor, test_tensor])
                split_idx = {
                    'train': combined_train,
                    'val': val_tensor
                }
            
            print(f"[OGB] Using official OGB splits for {dataset_name} - "
                  f"Train: {len(split_idx['train'])}, Val: {len(split_idx['val'])}" + 
                  (f", Test: {len(split_idx['test'])}" if 'test' in split_idx else ""))
            return split_idx
        except Exception as e:
            print(f"[OGB] Failed to get official OGB splits for {dataset_name}: {e}")
            # Fall through to other splitting methods
    
    # 1. Try to load pre-computed splits
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


def prepare_graph_data_for_pfn(dataset, split_idx, context_k=32, device='cuda'):
    """
    Prepare graph data for PFN-based graph classification by sampling context graphs.
    
    Args:
        dataset: PyTorch Geometric dataset containing graph data
        split_idx (dict): Train/val/test split indices
        context_k (int): Number of context samples per class
        device (str): Device to place tensors on
        
    Returns:
        dict: Processed data containing context and target information
    """
    # Check if this is a multi-task dataset
    sample_graph = dataset[0]
    is_multitask = sample_graph.y.numel() > 1
    
    if is_multitask:
        # Smart incremental context sampling for multi-task datasets
        import time
        
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
        
        # Optimized vectorized reservoir sampling approach
        print(f"  Vectorized sampling for {num_tasks} tasks across {len(split_idx['train'])} graphs...")
        
        train_indices = split_idx['train']
        
        # --- STAGE 1: Pre-fetch all labels and masks ---
        train_indices_tensor = torch.as_tensor(train_indices, dtype=torch.long)
        num_train_samples = len(train_indices_tensor)

        # Pre-allocate tensors to store all labels and masks
        train_labels = torch.empty((num_train_samples, num_tasks), dtype=torch.long)
        train_masks = torch.empty((num_train_samples, num_tasks), dtype=torch.bool)

        for i, idx in enumerate(train_indices_tensor):
            graph = dataset[idx.item()]
            
            # Get/create the task mask
            if not hasattr(graph, 'task_mask'):
                if graph.y.dtype.is_floating_point:
                    mask = ~torch.isnan(graph.y)
                else:
                    mask = (graph.y != -1)
            else:
                mask = graph.task_mask
                
            train_masks[i] = mask.view(-1).bool()
            # Store labels, filling invalid ones with a placeholder like -1
            train_labels[i] = torch.where(mask.view(-1).bool(), graph.y.view(-1), -1)

        print("Pre-fetching complete.")

        # --- STAGE 2: Populate reservoirs using vectorized lookups ---
        # Initialize the data structures for the results
        reservoirs = [[[] for _ in range(2)] for _ in range(num_tasks)] # For classes [0, 1]

        # Loop through each task and class to populate its reservoir
        for task_idx in range(num_tasks):
            for class_label in [0, 1]:
                # Vectorized check for valid samples for this specific (task, class) pair
                # 1. Find samples that are valid for this task.
                mask_is_valid = train_masks[:, task_idx]
                # 2. Find samples that have the correct class label for this task.
                label_is_correct = train_labels[:, task_idx] == class_label
                
                # Combine masks to get the final set of candidate indices (relative to train_indices)
                final_mask = mask_is_valid & label_is_correct
                relative_indices = torch.where(final_mask)[0]
                
                # Map back to the original dataset indices
                candidate_indices = train_indices_tensor[relative_indices].numpy()
                
                # Perform reservoir sampling in one shot
                if len(candidate_indices) == 0:
                    continue # No samples found for this combination
                
                if len(candidate_indices) <= context_k:
                    # If we have fewer candidates than the reservoir size, take all of them
                    reservoirs[task_idx][class_label] = candidate_indices.tolist()
                else:
                    # If we have more, take a random sample without replacement
                    sampled_indices = np.random.choice(
                        candidate_indices, 
                        size=context_k, 
                        replace=False
                    )
                    reservoirs[task_idx][class_label] = sampled_indices.tolist()
        
        sampling_time = time.time() - sampling_start
        
        # Step 2.5: Transfer reservoirs to task_contexts
        for task_idx in range(num_tasks):
            for class_label in [0, 1]:
                task_contexts[task_idx][class_label] = reservoirs[task_idx][class_label]
        
        # Step 3: GPU memory deduplication (only at the end)
        dedup_start = time.time()
        unique_indices = set()
        
        # Collect all unique graph indices across all tasks
        for task_data in task_contexts.values():
            for class_data in task_data.values():
                unique_indices.update(class_data)
        
        print(f"Deduplication: {len(unique_indices)} unique graphs out of {num_tasks * 2 * context_k} total samples")
        
        # Step 4: Load unique graphs to GPU/create graph objects only once
        # Load each unique graph only once
        gpu_graph_cache = {}
        for idx in unique_indices:
            idx_int = idx.item() if torch.is_tensor(idx) else idx
            gpu_graph_cache[idx] = dataset[idx_int]  # Load once

        # Replace indices with actual graph objects using cache, but preserve indices for FUG
        for task_idx in task_contexts:
            for class_label in task_contexts[task_idx]:
                indices = task_contexts[task_idx][class_label]
                graphs = [gpu_graph_cache[idx] for idx in indices]
                task_contexts[task_idx][class_label] = {
                    'graphs': graphs,
                    'indices': indices
                }
        
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
                idx_int = idx.item() if torch.is_tensor(idx) else idx
                
                graph = dataset[idx_int]
            
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
        gpu_graph_cache = {}
        
        for idx in unique_indices:
            idx_int = idx.item() if torch.is_tensor(idx) else idx
            gpu_graph_cache[idx] = dataset[idx_int]
        
        # Replace indices with graph objects, but preserve indices for FUG
        for class_label in task_contexts[0]:
            indices = task_contexts[0][class_label]
            graphs = [gpu_graph_cache[idx] for idx in indices]
            # Store both graphs and indices for FUG external mapping
            task_contexts[0][class_label] = {
                'graphs': graphs,
                'indices': indices
            }
        
        total_time = time.time() - start_time
        total_samples = sum(len(context_data['indices']) for context_data in task_contexts[0].values())
        print(f"Single-task context sampling completed in {total_time:.2f}s")
        print(f"  - Classes found: {list(available_classes)}")
        print(f"  - Unique graphs: {len(unique_indices)} out of {total_samples} samples")
        print(f"  - Memory efficiency: {(1 - len(unique_indices)/max(1, total_samples))*100:.1f}% reduction")
        
        # Return unified structure (single task = task 0)
        final_context_graphs = task_contexts  # {0: {class: [...]}}
        context_labels = {0: {class_label: class_label for class_label in available_classes}}
        is_multitask_dataset = False

    ret = {
        'context_graphs': final_context_graphs,  # task_idx -> {class: [graphs]}
        'context_labels': context_labels,
        'split_idx': split_idx,
        'dataset': dataset,
        'num_classes': dataset.num_classes,
        'num_features': dataset.num_node_features,
        'is_multitask': is_multitask_dataset,
        'num_tasks': num_tasks if is_multitask_dataset else 1,
        'name': dataset.name
    }
    
    return ret


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


def load_all_graph_datasets(dataset_names, device='cuda', pretraining_mode=False, context_k=32, embedding_family='ST'):
    """
    Load multiple graph classification datasets with memory-efficient context sampling.
    
    Args:
        dataset_names (list): List of dataset names to load
        device (str): Device to load data onto
        pretraining_mode (bool): If True, optimize splits for pretraining (train/val only)
        context_k (int): Number of context graphs per class to sample
        embedding_family (str): Text embedding family ('ST' for Sentence Transformer, 'e5' for E5 embeddings)
        
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
        result = load_dataset(name, embedding_family=embedding_family)  # Use the unified loading function
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
    print(f"\nAll datasets loaded in {overall_time:.2f}s")
    
    return datasets, processed_data_list

def create_task_filtered_datasets(dataset, split_idx, filter_split=None):
    """
    Create task-specific filtered datasets for multi-task learning.
    Each task gets its own dataset containing only graphs with valid labels for that task.
    
    Args:
        dataset: PyTorch Geometric dataset
        split_idx (dict): Dictionary with split indices
        filter_split (str, optional): If specified, only filter this split ('train', 'val', 'test').
                                     Other splits will be returned as-is.
        
    Returns:
        dict: For single-task: {0: {split: [indices]}}
              For multi-task: {task_idx: {split: [indices with valid labels for task_idx]}}
    """
    # Check if multi-task dataset
    sample_graph = dataset[0] if len(dataset) > 0 else None
    is_multitask = (sample_graph is not None and sample_graph.y.numel() > 1)
    
    if not is_multitask:
        # Single task: return original splits wrapped in task 0
        print(f"Single-task dataset detected!")
        return {0: split_idx}
    
    # Validate filter_split parameter
    if filter_split is not None and filter_split not in ['train', 'val', 'test']:
        raise ValueError(f"Invalid filter_split '{filter_split}'. Must be one of: 'train', 'val', 'test'")
    if filter_split is not None and filter_split not in split_idx:
        raise ValueError(f"Filter split '{filter_split}' not found in split_idx. Available splits: {list(split_idx.keys())}")
    
    # Multi-task: filter each split by task validity
    num_tasks = sample_graph.y.numel()
    task_filtered_splits = {}
    
    if filter_split is not None:
        print(f"Prefiltering dataset for {num_tasks} tasks (filtering only '{filter_split}' split)...")
    else:
        print(f"Prefiltering dataset for {num_tasks} tasks...")
    
    # --- STAGE 1: Pre-compute the task mask for the ENTIRE dataset ---
    # This loop runs only ONCE over the dataset.
    print("Pre-computing task masks for the entire dataset...")
    all_task_masks = torch.empty((len(dataset), num_tasks), dtype=torch.bool)

    for i, graph in enumerate(dataset):
        # Create task_mask on-the-fly if missing
        if not hasattr(graph, 'task_mask'):
            if graph.y.dtype.is_floating_point:
                # Check for NaN for floating point labels
                mask = ~torch.isnan(graph.y)
            else:
                # Check for -1 for integer labels
                mask = (graph.y != -1)
        else:
            mask = graph.task_mask

        # Ensure mask has the correct shape and store it
        all_task_masks[i] = mask.view(-1).bool()

    print("Pre-computation complete.")

    # --- STAGE 2: Perform efficient, vectorized filtering ---
    task_filtered_splits = {}

    for task_idx in range(num_tasks):
        task_filtered_splits[task_idx] = {}
        
        # Get the boolean validity mask for the current task
        # This is a tensor of shape [len(dataset)]
        task_validity_mask = all_task_masks[:, task_idx]

        for split_name, indices in split_idx.items():
            if filter_split is None or split_name == filter_split:
                # Convert split indices to a tensor
                split_indices_tensor = torch.as_tensor(indices, dtype=torch.long)
                
                # Use the split indices to select the relevant masks
                mask_for_this_split = task_validity_mask[split_indices_tensor]
                
                # Apply the boolean mask to get the final valid indices
                valid_indices = split_indices_tensor[mask_for_this_split]
                
                task_filtered_splits[task_idx][split_name] = valid_indices.tolist()
            else:
                # Don't filter this split, use original indices
                task_filtered_splits[task_idx][split_name] = indices.tolist() if hasattr(indices, 'tolist') else indices
            
        # Log filtering results
        if filter_split is None:
            # Show total filtering stats
            total_samples = sum(len(indices) for indices in split_idx.values())
            valid_samples = sum(len(indices) for indices in task_filtered_splits[task_idx].values())
            print(f"  Task {task_idx}: {valid_samples}/{total_samples} samples ({valid_samples/total_samples*100:.1f}%)")
        else:
            # Show filtering stats only for the filtered split
            original_count = len(split_idx[filter_split])
            filtered_count = len(task_filtered_splits[task_idx][filter_split])
            print(f"  Task {task_idx} ({filter_split}): {filtered_count}/{original_count} samples ({filtered_count/original_count*100:.1f}%)")
    
    return task_filtered_splits


class IndexTrackingDataLoader:
    """
    Custom DataLoader that preserves original dataset indices for FUG external mapping.
    Zero-overhead approach that only adds metadata to batch objects.
    """
    def __init__(self, dataset, indices, batch_size=128, shuffle=True, num_workers=0, pin_memory=True):
        self.dataset = dataset
        self.indices = indices  # The subset indices we want to sample from
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # Create the underlying DataLoader with Subset
        self.subset = Subset(dataset, indices)
        self.loader = DataLoader(
            self.subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    def __iter__(self):
        """
        Iterator that yields batches with original dataset indices attached.
        """
        # We need to track which subset indices are in each batch
        # PyTorch DataLoader samples from subset indices, so we need to map back
        
        # Get the sampler indices
        if self.shuffle:
            # Shuffled indices from the subset
            perm = torch.randperm(len(self.indices))
            sampled_indices = [self.indices[perm[i]] for i in range(len(self.indices))]
        else:
            # Sequential indices from the subset  
            sampled_indices = list(self.indices)
        
        # Group into batches and yield with original indices
        for i in range(0, len(sampled_indices), self.batch_size):
            batch_subset_indices = sampled_indices[i:i + self.batch_size]
            
            # Get the actual graphs
            graphs = [self.dataset[idx] for idx in batch_subset_indices]
            
            # Create batch
            batch = Batch.from_data_list(graphs)
            
            # Add original dataset indices to batch (this is the key!)
            batch.original_graph_indices = torch.tensor(batch_subset_indices, dtype=torch.long)
            
            yield batch
    
    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size


def create_data_loaders(dataset, split_idx, batch_size=128, shuffle=True, task_idx=None, use_index_tracking=False):
    """
    Create PyTorch data loaders for train/validation/test splits.
    
    Args:
        dataset: The dataset to create loaders for
        split_idx: Dictionary with split indices
        batch_size: Batch size
        shuffle: Whether to shuffle training data
        task_idx: Task index (for compatibility)
        use_index_tracking: Whether to use index tracking DataLoader (for FUG)
    """
    loaders = {}
    
    num_workers = 0

    for split_name, indices in split_idx.items():        
        # DEBUG: Check subset graphs before DataLoader creation
        subset = Subset(dataset, indices)
        
        shuffle_data = shuffle if split_name == 'train' else False
        use_pin_memory = torch.cuda.is_available()
        
        if use_index_tracking:
            loader = IndexTrackingDataLoader(
                dataset=dataset,
                indices=indices,
                batch_size=batch_size,
                shuffle=shuffle_data,
                num_workers=num_workers,
                pin_memory=use_pin_memory
            )
        else:
            # Use standard DataLoader
            loader = DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=shuffle_data,
                num_workers=num_workers,
                pin_memory=use_pin_memory,
            )
        loaders[split_name] = loader
    
    return loaders

def _save_pca_cache(dataset_name, target_dim, processed_features, cache_dir="./pca_cache"):
    """
    Save PCA-transformed embeddings to cache.
    
    Args:
        dataset_name: Name of the dataset
        target_dim: Target dimensionality
        processed_features: PCA-transformed embeddings
        cache_dir: Directory to store cache files
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    cache_file = cache_path / f"{dataset_name}_dim{target_dim}.pt"
    
    cache_data = {
        'processed_features': processed_features.cpu(),  # Always store on CPU
        'dataset_name': dataset_name,
        'target_dim': target_dim,
        'timestamp': time.time()
    }
    
    torch.save(cache_data, cache_file)
    print(f"PCA cache saved: {cache_file}")


def _load_pca_cache(dataset_name, target_dim, cache_dir="./pca_cache"):
    """
    Load PCA-transformed embeddings from cache.
    
    Args:
        dataset_name: Name of the dataset
        target_dim: Target dimensionality
        cache_dir: Directory containing cache files
        
    Returns:
        torch.Tensor or None: processed_features if found, None otherwise
    """
    cache_path = Path(cache_dir)
    cache_file = cache_path / f"{dataset_name}_dim{target_dim}.pt"
    
    if not cache_file.exists():
        return None
    
    try:
        cache_data = torch.load(cache_file, map_location='cpu')
        print(f"PCA cache hit: {cache_file}")
        return cache_data['processed_features']
    except Exception as e:
        print(f"Error loading PCA cache {cache_file}: {e}")
        return None


def _apply_pca_unified(data_tensor, target_dim, use_full_pca, sign_normalize, pca_device, incremental_pca_batch_size, pca_sample_threshold=float('inf'), use_pca_cache=False, pca_cache_dir="./pca_cache", dataset_name=None, target_cuda_device=None):
    """
    Unified PCA application with support for sampling large datasets on GPU.
    """

    
    pca_start = time.time()
    n_samples = data_tensor.shape[0]
    print(f"Applying PCA on {n_samples:,} samples with target dimension {target_dim}, threshold {pca_sample_threshold:,}")
    
    # Check for cached results first
    if use_pca_cache and dataset_name:
        cached_result = _load_pca_cache(dataset_name, target_dim, pca_cache_dir)
        if cached_result is not None:
            return cached_result.to(data_tensor.device)
    
    # Cache miss - force CPU computation for accurate caching
    cache_miss = use_pca_cache and dataset_name
    if cache_miss:
        pca_device = 'cpu'
    
    # Handle device for GPU operations - keep data on CPU, only move samples/batches as needed
    original_device = data_tensor.device
    target_device = None
    if pca_device == 'gpu':
        if torch.cuda.is_available():
            if target_cuda_device is not None:
                target_device = target_cuda_device
            else:
                target_device = torch.cuda.current_device()
        else:
            print(f"WARNING: GPU requested but CUDA not available, falling back to CPU")
            pca_device = 'cpu'
    
    if pca_device == 'cpu':
        # Use CPU Incremental PCA
        from .data_utils import apply_incremental_pca_cpu
        processed_features = apply_incremental_pca_cpu(
            data_tensor, target_dim, incremental_pca_batch_size, sign_normalize, rank=0
        )
            
        # Move back to original device if needed
        processed_features = processed_features.to(data_tensor.device)
        
    else:
        # GPU PCA - unified path for both sampled and full data
        use_sampling = n_samples > pca_sample_threshold
        
        # Step 1: Determine what data to use for PCA fitting
        if use_sampling:
            print(f"  Large dataset ({n_samples:,} nodes) - using sampled PCA with {pca_sample_threshold:,} samples")
            sample_indices = torch.randperm(n_samples, device=original_device)[:pca_sample_threshold]
            pca_data = data_tensor[sample_indices]
        else:
            print(f"  Small dataset ({n_samples:,} nodes) - using full PCA")
            pca_data = data_tensor
        
        # Transfer PCA data to GPU
        if target_device is not None:
            pca_data = pca_data.to(target_device)
        
        # Step 2: Fit PCA on the chosen data
        if use_full_pca:
            U, S, V = torch.svd(pca_data)
            U = U[:, :target_dim]
            S = S[:target_dim]
        else:
            U, S, V = torch.pca_lowrank(pca_data, q=target_dim)
        
        # Step 3: Apply transformation to ALL data (always in batches for consistency)
        batch_size = 500000
        num_batches = (n_samples + batch_size - 1) // batch_size
        
        # Use pinned memory for faster transfers
        processed_features = torch.zeros(n_samples, target_dim, device=original_device).pin_memory()
        data_tensor_pinned = data_tensor.pin_memory()
        
        # V_gpu is small, transfer once
        V_gpu = V[:, :target_dim].to(target_device)
        
        # Create streams for overlapping operations
        s = torch.cuda.Stream()
        
        with torch.cuda.stream(s):
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                
                # Get batch from pinned memory
                data_batch_cpu = data_tensor_pinned[start_idx:end_idx]
                
                # Async transfer to GPU
                data_batch_gpu = data_batch_cpu.to(target_device, non_blocking=True)
                
                # Apply transformation (without S scaling)
                batch_transformed_gpu = data_batch_gpu @ V_gpu
                
                # Async transfer result back to CPU
                processed_features[start_idx:end_idx].copy_(batch_transformed_gpu, non_blocking=True)
            
            # Synchronize once at the end
            torch.cuda.synchronize()
        
        # Sign normalization (compute on a sample for efficiency)
        if sign_normalize:
            sign_sample_size = min(10000, n_samples)
            sign_indices = torch.randperm(n_samples, device=original_device)[:sign_sample_size]
            sample_transformed = processed_features[sign_indices]
            
            for i in range(target_dim):
                feature_vector = sample_transformed[:, i]
                if feature_vector.abs().argmax() < len(feature_vector) // 2:
                    processed_features[:, i] = -processed_features[:, i]
    
    pca_time = time.time() - pca_start
    method = 'cpu' if pca_device == 'cpu' else ('sampled-gpu' if n_samples > pca_sample_threshold else 'gpu')
    print(f"  PCA computation ({method}): {pca_time:.2f}s")
    
    # Save to cache if this was a cache miss
    if cache_miss:
        _save_pca_cache(dataset_name, target_dim, processed_features, pca_cache_dir)
    
    # Transfer result back to original device if we moved it for PCA
    if target_device is not None and original_device != data_tensor.device:
        processed_features = processed_features.to(original_device)
    
    return processed_features

def process_graph_features(dataset, hidden_dim, device='cuda', 
                         use_identity_projection=False, projection_small_dim=128, projection_large_dim=256,
                         use_full_pca=False, sign_normalize=False, normalize_data=False,
                         padding_strategy='zero', use_batchnorm=False,
                         use_gpse=False, gpse_path='/home/maweishuo/GPSE/datasets', dataset_name=None,
                         pca_device='gpu', incremental_pca_batch_size=10000, pca_sample_threshold=500000,
                         processed_data=None, pcba_context_only_pca=False,
                         use_pca_cache=False, pca_cache_dir="./pca_cache"):
    """
    Process graph features using PCA directly on embedding table for maximum memory efficiency.
    
    Args:
        dataset: PyTorch Geometric dataset with node_embs embedding table
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
        use_gpse (bool): Whether to use pre-computed GPSE embeddings
        gpse_path (str): Path to GPSE datasets directory
        dataset_name (str): Name of dataset for GPSE lookup
        pca_sample_threshold (int): Sample this many nodes for PCA if dataset is too large
        processed_data (dict): Processed data with potential FUG mapping
        pcba_context_only_pca (bool): Use only context + test graphs for PCA fitting
        
    Returns:
        dict: Processing information and flags
    """
    import time
    
    # PCA threshold is now strictly controlled by args parameter
    
    # GPSE Integration: Replace dataset embeddings with GPSE embeddings if requested
    if use_gpse and dataset_name is not None:
        print(f"GPSE embeddings requested for dataset: {dataset_name}")
        success = update_dataset_with_gpse_embeddings(dataset, dataset_name, gpse_path)
        if success:
            print("Successfully loaded GPSE embeddings - proceeding with GPSE-based processing")
        else:
            print("Failed to load GPSE embeddings - falling back to original embeddings")
    elif use_gpse:
        print("Warning: GPSE requested but dataset_name not provided - using original embeddings")
    
    # Check if dataset has node features
    sample = dataset[0] if len(dataset) > 0 else None
    if sample is None:
        raise ValueError("Empty dataset")
    
    # Check for FUG mapping first (external embeddings)
    fug_mapping = None
    node_embs = None
    if processed_data is not None and 'fug_mapping' in processed_data:
        fug_mapping = processed_data['fug_mapping']
        
        # Handle cache mode for FUG datasets
        if fug_mapping.get('cache_mode', False) and use_pca_cache and dataset_name:
            print(f"🚀 Cache mode enabled for {dataset_name} - attempting to load cached PCA features")
            cached_result = _load_pca_cache(dataset_name, hidden_dim, pca_cache_dir)
            if cached_result is not None:
                print(f"✅ Successfully loaded cached PCA features for {dataset_name}")
                
                # CRITICAL FIX: Set the cached features as the node embedding table
                fug_mapping['node_embs'] = cached_result
                fug_mapping['cache_mode'] = False  # Disable cache mode now that we have embeddings
                print(f"Set cached features as node_embs: {cached_result.shape}")
                
                # Continue with normal processing flow - don't return early
                node_embs = cached_result
                original_dim = cached_result.shape[1]
                print(f"Cache mode: Using cached embeddings {node_embs.shape} as normal flow")
            else:
                print(f"❌ Cache miss for {dataset_name} in cache mode - this shouldn't happen!")
                print(f"   Cache was detected during loading but file is missing/corrupted")
                # For safety, fall back to loading embeddings if cache is missing
                if fug_mapping.get('embedding_file'):
                    print(f"🔄 Loading embeddings as fallback: {fug_mapping['embedding_file']}")
                    try:
                        node_embs = torch.load(fug_mapping['embedding_file'], map_location='cpu')
                        fug_mapping['node_embs'] = node_embs
                        fug_mapping['cache_mode'] = False  # Disable cache mode
                        print(f"✅ Loaded fallback embeddings: {node_embs.shape}")
                    except Exception as e:
                        raise RuntimeError(f"Failed to load fallback embeddings: {e}")
        
        # Normal FUG mode (embeddings already loaded or set from cache)
        if fug_mapping.get('node_embs') is not None:
            if 'node_embs' not in locals():  # Only set if not already set by cache mode
                node_embs = fug_mapping['node_embs']
                original_dim = node_embs.shape[1]
                print(f"Using FUG external embeddings: {node_embs.shape}")
    # Fall back to dataset.node_embs for unified setting
    elif hasattr(dataset, 'node_embs') and dataset.node_embs is not None:
        node_embs = dataset.node_embs
        original_dim = node_embs.shape[1]
        print(f"Using dataset node_embs: {node_embs.shape}")
    else:
        raise ValueError("Expected dataset.node_embs under unified setting or FUG mapping in processed_data")
    
    print(f"Processing graph features: {original_dim}D -> {hidden_dim}D")
    st = time.time()
    
    # Verify we have node embeddings available
    if node_embs is None:
        raise ValueError("No node embeddings available - check FUG mapping or dataset.node_embs")
    
    # Helper function to get node embeddings for a graph
    def get_graph_node_embeddings(graph_idx):
        """Get node embeddings for a specific graph, handling both FUG and regular datasets."""
        if fug_mapping is not None:
            # FUG dataset: use external node index mapping
            node_indices = fug_mapping['node_index_mapping'][graph_idx]
            return node_embs[node_indices]
        else:
            # Regular dataset: use graph.x as indices into dataset.node_embs
            graph = dataset[graph_idx]
            return node_embs[graph.x]
    
    # Standard unified processing: apply PCA to node embeddings
    print("Unified setting: applying PCA to node_embs")
    
    # Apply PCA/projection to node embeddings
    # PCBA-specific: optionally use only context + test graphs for PCA
    if (dataset_name and dataset_name.lower() == 'pcba' and 
        pcba_context_only_pca and processed_data is not None):
        print("PCBA mode: Using context + test graphs for PCA fitting")
        
        # Collect context graph node embeddings using preserved indices (MUCH faster!)
        context_node_embs = []
        context_graphs = processed_data['context_graphs']
        for task_idx in context_graphs:
            for class_label in context_graphs[task_idx]:
                context_data = context_graphs[task_idx][class_label]
                
                # Use preserved indices instead of expensive linear search
                if isinstance(context_data, dict) and 'indices' in context_data:
                    # New format with preserved indices
                    indices = context_data['indices']
                    for graph_idx in indices:
                        context_node_embs.append(get_graph_node_embeddings(graph_idx))
                else:
                    # Fallback: old format - this will be slow but shouldn't happen
                    print("[WARNING] Using slow fallback for context graph indexing!")
                    for graph in context_data:
                        # Find the graph index in the dataset (slow!)
                        graph_idx = None
                        for i, dataset_graph in enumerate(dataset):
                            if dataset_graph is graph:  # Identity check
                                graph_idx = i
                                break
                        if graph_idx is not None:
                            context_node_embs.append(get_graph_node_embeddings(graph_idx))
        
        # Collect test graph node embeddings
        test_node_embs = []
        test_split = processed_data['split_idx']['test']
        for test_idx in test_split:
            test_idx_int = test_idx.item() if torch.is_tensor(test_idx) else test_idx
            test_node_embs.append(get_graph_node_embeddings(test_idx_int))
        
        # Concatenate all embeddings for PCA
        stacked_features = torch.cat(context_node_embs + test_node_embs, dim=0)
        print(f"PCBA PCA fitted on {stacked_features.size(0)} nodes from context + test graphs")
    else:
        # Standard: use all embeddings
        stacked_features = node_embs
    
    if use_identity_projection:
        # Identity projection pathway: PCA to small_dim, then project to large_dim
        print(f"Using identity projection: {original_dim}D -> PCA to {projection_small_dim}D -> Project to {projection_large_dim}D")

        # Determine maximum PCA dimensions available (limited by min(n_samples, n_features))
        max_pca_dim = min(original_dim, stacked_features.size(0))
        
        if max_pca_dim >= projection_small_dim:
            # Apply PCA to small dimension
            processed_features = _apply_pca_unified(
                stacked_features, projection_small_dim, use_full_pca, sign_normalize,
                pca_device, incremental_pca_batch_size, pca_sample_threshold,
                use_pca_cache, pca_cache_dir, dataset_name, device
            )
        else:
            # Not enough samples for full PCA to projection_small_dim, apply PCA to all available dimensions then pad
            pca_dim = max_pca_dim
            
            pca_features = _apply_pca_unified(
                stacked_features, pca_dim, use_full_pca, sign_normalize,
                pca_device, incremental_pca_batch_size, pca_sample_threshold,
                use_pca_cache, pca_cache_dir, dataset_name, device
            )
            
            # Apply padding to reach projection_small_dim
            if pca_features.size(1) < projection_small_dim:
                padding_size = projection_small_dim - pca_features.size(1)
                # Keep padding on the same device as pca_features
                padding = torch.zeros(pca_features.size(0), padding_size, device=pca_features.device)
                processed_features = torch.cat([pca_features, padding], dim=1)
                print(f"Applied zero padding ({pca_features.size(1)} -> {projection_small_dim})")
            else:
                processed_features = pca_features[:, :projection_small_dim]
        
        target_dim = projection_large_dim
        needs_identity_projection = True
        
    else:
        # Standard PCA pathway

        original_dim = min(original_dim, stacked_features.size(0))

        if original_dim >= hidden_dim:
            # Apply PCA to target dimension
            processed_features = _apply_pca_unified(
                stacked_features, hidden_dim, use_full_pca, sign_normalize,
                pca_device, incremental_pca_batch_size, pca_sample_threshold,
                use_pca_cache, pca_cache_dir, dataset_name, device
            )
            
        else:
            # Not enough features, apply PCA to all available dimensions then pad
            pca_dim = min(original_dim, stacked_features.size(1))
            
            pca_features = _apply_pca_unified(
                stacked_features, pca_dim, use_full_pca, sign_normalize,
                pca_device, incremental_pca_batch_size, pca_sample_threshold,
                use_pca_cache, pca_cache_dir, dataset_name, device
            )
            
            # Apply padding to reach target dimension
            if pca_features.size(1) < hidden_dim:
                padding_size = hidden_dim - pca_features.size(1)
                
                if padding_strategy == 'zero':
                    # Keep padding on the same device as pca_features
                    padding = torch.zeros(pca_features.size(0), padding_size, device=pca_features.device)
                    processed_features = torch.cat([pca_features, padding], dim=1)
                    
                elif padding_strategy == 'random':
                    # Random padding from same distribution as real features
                    # Keep padding on the same device as pca_features
                    real_std = pca_features.std(dim=0, keepdim=True)
                    real_mean = pca_features.mean(dim=0, keepdim=True)
                    random_padding = torch.randn(pca_features.size(0), padding_size, device=pca_features.device) * real_std.mean() + real_mean.mean()
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
    
    # Update the embedding table with PCA-transformed embeddings
    if fug_mapping is not None:
        # FUG dataset: update the external mapping
        fug_mapping['node_embs'] = processed_features
        print(f"  Updated FUG mapping embeddings: {processed_features.shape} - all graphs now use reduced embeddings!")
    else:
        # Regular dataset: update dataset.node_embs
        dataset.node_embs = processed_features
        print(f"  Updated embedding table: {processed_features.shape} - all graphs now use reduced embeddings!")
    
    processing_info = {
        'original_dim': original_dim,
        'processed_dim': processed_features.size(1),
        'target_dim': target_dim,
        'needs_identity_projection': needs_identity_projection,
        'projection_target_dim': projection_large_dim if needs_identity_projection else None
    }
    
    print(f"Graph feature processing completed in {time.time()-st:.2f}s: {original_dim}D -> {processed_features.size(1)}D (optimized: processed embedding table directly!)")
    
    return processing_info

def select_graph_context(processed_data, context_k=32):
    """
    Select context graphs for each class in the processed data.
    
    Args:
        processed_data (dict): Processed data from prepare_graph_data_for_pfn
        context_k (int): Number of context graphs per class to select
        
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
