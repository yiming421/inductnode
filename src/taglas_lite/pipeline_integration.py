"""
TAGLAS Lite - Pipeline Integration Functions
Handles conversion between TAGLAS format and InductNode pipeline format
"""

import torch
import numpy as np
from typing import Optional, Dict, Any
from torch_geometric.data import Data

from .data.dataset import SimpleTAGDataset


class TAGPipelineDataset:
    """
    Wrapper that makes TAGDataset compatible with InductNode pipeline.
    Handles text feature integration and provides standard dataset interface.
    """
    
    def __init__(self, name: str, root: str, text_features: Optional[Dict] = None):
        """
        Args:
            name: Dataset name (e.g., 'bace', 'bbbp', 'hiv')
            root: Root directory containing TAGDatasets
            text_features: Dict with 'node_features', 'edge_features', 'label_features'
        """
        self.name = name
        self.root = root
        self.text_features = text_features or {}
        
        # Load the TAGLAS dataset
        print(f"Loading TAGDataset {name} with TAGLAS Lite...")
        self.tag_dataset = SimpleTAGDataset(name, root)
        print(f"Successfully loaded {len(self.tag_dataset)} graphs")
        
        # Set standard properties
        self.num_classes = self.tag_dataset.num_classes
        
        # Determine node feature dimension from text features
        if 'node_features' in self.text_features:
            self.num_node_features = self.text_features['node_features'].shape[1]
            print(f"Using text features with {self.num_node_features} dimensions")
        else:
            self.num_node_features = 0
            print("No text features provided")
        
        # Additional pipeline-compatible properties
        self.has_text_features = len(self.text_features) > 0
    
    def get_predefined_splits(self):
        """Get predefined train/val/test splits from TAGLAS side_data if available."""
        if self.tag_dataset.side_data is not None and hasattr(self.tag_dataset.side_data, 'graph_split'):
            splits = self.tag_dataset.side_data.graph_split
            
            # Check available keys
            available_keys = list(splits.keys())
            print(f"  Available split keys: {available_keys}")
            
            # Handle different validation key names
            val_key = None
            if 'valid' in splits:
                val_key = 'valid'
            elif 'val' in splits:
                val_key = 'val'
            else:
                print(f"  Warning: No validation split found in keys: {available_keys}")
                return None
            
            return {
                'train': torch.tensor(splits['train'], dtype=torch.long),
                'valid': torch.tensor(splits[val_key], dtype=torch.long),  # Save as 'valid' for TSGFM compatibility
                'test': torch.tensor(splits['test'], dtype=torch.long)
            }
        return None

    def __len__(self):
        return len(self.tag_dataset)
    
    def __getitem__(self, idx):
        """
        Get a graph with proper pipeline-compatible format.
        Converts TAGData to standard PyG Data with integrated text features.
        """
        # Get the original TAGData
        tag_data = self.tag_dataset[idx]
        
        # Convert to standard PyG Data format for pipeline compatibility
        graph_data = {}
        
        # Copy basic graph structure
        if hasattr(tag_data, 'edge_index') and tag_data.edge_index is not None:
            graph_data['edge_index'] = tag_data.edge_index
            
            # Create adj_t from edge_index if needed (for PyG compatibility)
            try:
                from torch_geometric.utils import to_undirected, degree
                from torch_sparse import SparseTensor
                
                edge_index = tag_data.edge_index
                num_nodes = tag_data.node_map.shape[0]
                
                # Handle empty edge_index (graphs with no edges)
                if edge_index.numel() == 0:
                    # Create empty sparse tensor for graphs with no edges
                    adj_t = SparseTensor(row=torch.tensor([], dtype=torch.long), 
                                       col=torch.tensor([], dtype=torch.long), 
                                       sparse_sizes=(num_nodes, num_nodes))
                    graph_data['adj_t'] = adj_t
                else:
                    # Normal case: validate edge indices and create adj_t
                    max_edge_idx = edge_index.max().item()
                    assert(num_nodes >= max_edge_idx + 1), \
                        f"Number of nodes {num_nodes} is less than max edge index {max_edge_idx + 1}"
                    
                    # Create SparseTensor for adj_t (transpose of adjacency matrix)
                    row, col = edge_index[0], edge_index[1]
                    adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
                    graph_data['adj_t'] = adj_t
                
            except Exception as e:
                print(f"Warning: Could not create adj_t: {e}")
                # Skip adj_t if creation fails
        
        # Handle node features with text feature integration
        if 'node_features' in self.text_features and hasattr(tag_data, 'node_map'):
            # Use node_map directly - it contains the correct indices for this graph's atoms
            node_indices = tag_data.node_map.clone().detach().long()
            graph_data['x'] = node_indices  # Direct indices into embedding table
        elif hasattr(tag_data, 'x') and tag_data.x is not None:
            graph_data['x'] = tag_data.x
        
        # Handle edge features if available
        if 'edge_features' in self.text_features and hasattr(tag_data, 'edge_map'):
            edge_features = self._map_features_to_edges(
                self.text_features['edge_features'],
                tag_data.edge_map
            )
            graph_data['edge_attr'] = edge_features
        elif hasattr(tag_data, 'edge_attr') and tag_data.edge_attr is not None:
            graph_data['edge_attr'] = tag_data.edge_attr
        
        # Handle labels - convert to binary classification format
        if hasattr(tag_data, 'label_map') and tag_data.label_map is not None:
            # For graph classification, create binary label
            # TAGDataset uses different labeling - convert to 0/1
            label = self._extract_graph_label(tag_data)
            graph_data['y'] = label
        else:
            # Default to label 0 if no label found
            graph_data['y'] = torch.tensor([0], dtype=torch.long)
        
        # Copy other attributes that might be useful
        for attr in ['smiles', 'cum_label_map']:
            if hasattr(tag_data, attr):
                graph_data[attr] = getattr(tag_data, attr)
        
        return Data(**graph_data)
    
    def _map_features_to_nodes(self, node_features, node_map, expected_num_nodes):
        """
        Map text features to nodes using node_map.
        Returns the node indices (for embedding table lookup) rather than the actual features.
        Ensures the returned tensor has exactly expected_num_nodes elements.
        """
        try:
            if isinstance(node_map, torch.Tensor):
                node_indices = node_map.clone().detach()
            else:
                node_indices = torch.tensor(node_map)
            
            # Ensure indices are long integers for proper indexing
            node_indices = node_indices.long()
            
            # Ensure we have exactly the expected number of nodes
            current_num_nodes = node_indices.shape[0]
            
            if current_num_nodes < expected_num_nodes:
                # Pad with zeros (will map to first embedding) if we have fewer nodes
                padding = torch.zeros(expected_num_nodes - current_num_nodes, dtype=torch.long)
                node_indices = torch.cat([node_indices, padding], dim=0)
                print(f"  Padded node indices from {current_num_nodes} to {expected_num_nodes}")
            elif current_num_nodes > expected_num_nodes:
                # Truncate if we have too many nodes
                node_indices = node_indices[:expected_num_nodes]
                print(f"  Truncated node indices from {current_num_nodes} to {expected_num_nodes}")
            
            return node_indices  # [expected_num_nodes] - indices into node_embs table
            
        except Exception as e:
            print(f"Warning: Could not map node features: {e}")
            # Return dummy indices if mapping fails
            return torch.zeros(expected_num_nodes, dtype=torch.long)
    
    def _map_features_to_edges(self, edge_features, edge_map):
        """Map text features to edges using edge_map"""
        try:
            if isinstance(edge_map, torch.Tensor):
                edge_indices = edge_map.cpu().numpy()
            else:
                edge_indices = np.array(edge_map)
            
            # Get features for these edge indices
            mapped_features = edge_features[edge_indices]  # [num_edges, 768]
            return torch.tensor(mapped_features, dtype=torch.float32).clone().detach()
        except Exception as e:
            print(f"Warning: Could not map edge features: {e}")
            # Return dummy features if mapping fails
            num_edges = len(edge_map) if hasattr(edge_map, '__len__') else 1
            return torch.zeros((num_edges, edge_features.shape[1]), dtype=torch.float32)
    
    def _extract_graph_label(self, tag_data):
        """Extract graph-level label from TAGData"""
        try:
            # TAGDataset stores labels in label_map
            if hasattr(tag_data, 'label_map') and tag_data.label_map is not None:
                label_map = tag_data.label_map
                
                if isinstance(label_map, torch.Tensor):
                    if label_map.numel() > 0:
                        # Handle different tensor shapes
                        if label_map.dim() == 0:
                            # Scalar tensor
                            raw_label = label_map.item()
                            return torch.tensor([raw_label], dtype=torch.long)
                        elif label_map.dim() == 1:
                            # 1D tensor - could be multi-task labels
                            if label_map.numel() == 1:
                                # Single task
                                raw_label = label_map[0].item()
                                return torch.tensor([raw_label], dtype=torch.long)
                            else:
                                # Multi-task - return all labels as a 1D tensor
                                # Convert to float and handle -1 missing values  
                                labels = label_map.clone().detach().float()
                                # Convert -1 missing values to NaN for TSGFM compatibility
                                labels = torch.where(labels == -1, torch.tensor(float('nan')), labels)
                                return labels
                        elif label_map.dim() == 2:
                            # 2D tensor - could be [1, N] format (wrapped multi-task) or other formats
                            if label_map.shape[0] == 1:
                                # Format: [1, N] - extract the N tasks
                                task_labels = label_map[0]  # Get the N tasks
                                if task_labels.numel() == 1:
                                    # Single task wrapped: [1, 1]
                                    return torch.tensor([task_labels[0].item()], dtype=torch.long)
                                else:
                                    # Multi-task wrapped: [1, N] where N > 1
                                    labels = task_labels.clone().detach().float()
                                    # Convert -1 missing values to NaN for TSGFM compatibility
                                    labels = torch.where(labels == -1, torch.tensor(float('nan')), labels)
                                    return labels
                            else:
                                # Other 2D format - flatten and take first row
                                first_row = label_map[0].clone().detach()
                                if first_row.numel() == 1:
                                    return torch.tensor([first_row.item()], dtype=torch.long)
                                else:
                                    if first_row.dtype.is_floating_point:
                                        first_row = torch.where(torch.isnan(first_row), torch.tensor(-1.0), first_row)
                                        first_row = first_row.long()
                                    return first_row
                        else:
                            # Multi-dimensional tensor (3D+) - flatten and use first element
                            flattened = label_map.flatten()
                            if flattened.numel() > 0:
                                raw_label = flattened[0].item()
                                return torch.tensor([raw_label], dtype=torch.long)
                
                # Handle other formats (list, numpy array, etc.)
                if hasattr(label_map, '__iter__') and len(label_map) > 0:
                    first_label = label_map[0]
                    if isinstance(first_label, (int, float)):
                        return torch.tensor([int(first_label)], dtype=torch.long)
                    elif hasattr(first_label, '__iter__') and len(first_label) > 0:
                        # Multi-task case
                        labels = torch.tensor(first_label, dtype=torch.float32)
                        labels = torch.where(torch.isnan(labels), torch.tensor(-1.0), labels)
                        return labels.long()
            
            # Default to single task with label 0
            return torch.tensor([0], dtype=torch.long)
            
        except Exception as e:
            print(f"Warning: Could not extract label: {e}")
            # Return single task default label
            return torch.tensor([0], dtype=torch.long)


def load_tag_dataset_with_pipeline_integration(name: str, root: str, embedding_family: str = 'ST') -> TAGPipelineDataset:
    """
    Load TAGDataset with full pipeline integration.
    
    Args:
        name: Dataset name (e.g., 'bace', 'bbbp', 'hiv')  
        root: Root directory containing TAGDataset
        embedding_family: Text embedding family ('ST' for Sentence Transformer, 'e5' for E5 embeddings)
        
    Returns:
        TAGPipelineDataset: Pipeline-compatible dataset with text features
    """
    import os
    
    # Load text features from task/{embedding_family}/ directory  
    text_features = {}
    task_path = os.path.join(root, 'chembl', name, 'task', embedding_family)
    
    if os.path.exists(task_path):
        print(f"Loading {embedding_family} text features from {task_path}...")
        
        feature_files = {
            'node_features': 'node_features.pt',
            'edge_features': 'edge_features.pt',
            'label_features': 'label_features.pt'
        }
        
        for feature_name, filename in feature_files.items():
            filepath = os.path.join(task_path, filename)
            if os.path.exists(filepath):
                try:
                    features = torch.load(filepath, map_location='cpu')
                    text_features[feature_name] = features
                    print(f"  {feature_name}: {features.shape}")
                except Exception as e:
                    print(f"  Failed to load {filename}: {e}")
    else:
        print(f"Warning: {embedding_family} text features not found at {task_path}")
        # Try fallback to ST if e5 not found
        if embedding_family == 'e5':
            fallback_path = os.path.join(root, 'chembl', name, 'task', 'ST')
            if os.path.exists(fallback_path):
                print(f"Falling back to ST embeddings at {fallback_path}...")
                return load_tag_dataset_with_pipeline_integration(name, root, 'ST')
    
    # Create pipeline-compatible dataset
    return TAGPipelineDataset(name, root, text_features)


def convert_tagdataset_to_tsgfm_format(tag_dataset: TAGPipelineDataset):
    """
    Convert TAGPipelineDataset to TSGFM-compatible format for existing pipeline code.
    This creates the same interface that TSGFM datasets provide.
    """
    
    class TSGFMCompatibleDataset:
        """TSGFM-compatible wrapper for TAGPipelineDataset using TSGFM's pre-processing approach"""
        
        def __init__(self, tag_pipeline_dataset):
            self.tag_dataset = tag_pipeline_dataset
            self.name = tag_pipeline_dataset.name
            self.num_classes = tag_pipeline_dataset.num_classes
            self.num_node_features = tag_pipeline_dataset.num_node_features
            self.has_text_features = tag_pipeline_dataset.has_text_features
            self.text_features = tag_pipeline_dataset.text_features
            
            # Create node_embs embedding table that pipeline expects
            if 'node_features' in tag_pipeline_dataset.text_features:
                self.node_embs = tag_pipeline_dataset.text_features['node_features']
                print(f"  Created node_embs table: {self.node_embs.shape}")
            else:
                self.node_embs = None
                print("  Warning: No node_embs table created")
            
            # Detect multi-task dataset by checking sample labels (before pre-processing)
            self._detect_multitask_properties()
            
            # 🚀 TSGFM APPROACH: Pre-convert ALL graphs during initialization
            print(f"  ⚡ Pre-processing all {len(tag_pipeline_dataset)} graphs (TSGFM approach)...")
            import time
            start_time = time.time()
            
            self.graphs = []  # Store pre-converted graphs like TSGFM
            for i in range(len(tag_pipeline_dataset)):
                converted_graph = self._convert_graph(tag_pipeline_dataset[i], i)
                self.graphs.append(converted_graph)
                
                # Progress indicator for large datasets
                if (i + 1) % 10000 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    remaining = (len(tag_pipeline_dataset) - i - 1) / rate
                    print(f"    Progress: {i+1}/{len(tag_pipeline_dataset)} graphs ({rate:.0f} graphs/s, ~{remaining:.1f}s remaining)")
            
            conversion_time = time.time() - start_time
            print(f"  ✅ Pre-processing completed in {conversion_time:.2f}s ({len(self.graphs)/conversion_time:.0f} graphs/s)")
            print(f"  🎯 Future graph access will be instant (same as TSGFM)!")
            
            # Save predefined splits from TAGDataset if available
            self._save_tagdataset_splits()
        
        def _detect_multitask_properties(self):
            """Detect if this is a multi-task dataset and set appropriate properties."""
            if len(self.tag_dataset) > 0:
                # Check a few sample graphs to determine if multi-task
                sample_sizes = []
                for i in range(min(10, len(self.tag_dataset))):  # Check first 10 graphs
                    try:
                        sample_graph = self.tag_dataset[i]
                        if hasattr(sample_graph, 'y') and sample_graph.y is not None:
                            y_shape = sample_graph.y.shape if hasattr(sample_graph.y, 'shape') else torch.tensor(sample_graph.y).shape
                            sample_sizes.append(y_shape)
                    except:
                        continue
                
                if sample_sizes:
                    # Check if we have consistent multi-dimensional labels
                    most_common_shape = max(set(sample_sizes), key=sample_sizes.count)
                    
                    # Handle different label formats:
                    # Single task: [1] or scalar
                    # Multi-task: [N] where N > 1, or [1, N] where N > 1
                    if len(most_common_shape) == 1 and most_common_shape[0] > 1:
                        # Format: [N] where N > 1 (direct multi-task)
                        self.num_tasks = most_common_shape[0]
                        print(f"  Detected multi-task dataset: {self.num_tasks} tasks per graph (format: [{self.num_tasks}])")
                        self._add_task_masks_to_graphs()
                    elif len(most_common_shape) == 2 and most_common_shape[1] > 1:
                        # Format: [1, N] where N > 1 (wrapped multi-task)
                        self.num_tasks = most_common_shape[1]
                        print(f"  Detected multi-task dataset: {self.num_tasks} tasks per graph (format: [1, {self.num_tasks}])")
                        self._add_task_masks_to_graphs()
                    else:
                        # Single task
                        self.num_tasks = 1
                        print(f"  Detected single-task dataset (shape: {most_common_shape})")
                else:
                    self.num_tasks = 1
            else:
                self.num_tasks = 1
        
        def _add_task_masks_to_graphs(self):
            """Add task masks to multi-task graphs for NaN handling."""
            # This will be done lazily when graphs are accessed
            pass
        
        def _save_tagdataset_splits(self):
            """Save TAGDataset predefined splits in TSGFM format for pipeline compatibility."""
            predefined_splits = self.tag_dataset.get_predefined_splits()
            if predefined_splits is not None:
                # Create the expected split file path
                import os
                tag_dataset_root = os.environ.get('TAG_DATASET_ROOT', './TAGDataset')
                split_dir = os.path.join(tag_dataset_root, self.name)
                os.makedirs(split_dir, exist_ok=True)
                split_path = os.path.join(split_dir, 'e2e_graph_splits.pt')
                
                # Format splits in TSGFM expected format
                split_key = f'{self.name}_e2e_graph'
                splits_data = {
                    split_key: predefined_splits
                }
                
                # Save the splits file
                torch.save(splits_data, split_path)
                print(f"  Saved predefined splits to {split_path}")
                print(f"    Train: {len(predefined_splits['train'])}, Val: {len(predefined_splits['valid'])}, Test: {len(predefined_splits['test'])}")
            else:
                print("  No predefined splits found in TAGDataset")
        
        def __len__(self):
            return len(self.graphs)
        
        def _convert_graph(self, raw_graph, graph_idx):
            """Convert a single TAG graph to PyG format with all processing applied"""
            
            # Add task mask for multi-task datasets
            if hasattr(self, 'num_tasks') and self.num_tasks > 1:
                if hasattr(raw_graph, 'y') and raw_graph.y is not None:
                    # Create task mask (1 for valid labels, 0 for missing/NaN)
                    if hasattr(raw_graph.y, 'numel') and raw_graph.y.numel() > 1:
                        # Multi-task labels - create mask for non-NaN values (TSGFM compatible)
                        if raw_graph.y.dtype.is_floating_point:
                            task_mask = (~torch.isnan(raw_graph.y)).float()
                        else:
                            # Convert integer labels to float with NaN for missing values first
                            float_labels = raw_graph.y.float()
                            float_labels = torch.where(raw_graph.y == -1, torch.tensor(float('nan')), float_labels)
                            raw_graph.y = float_labels  # Update to float format
                            task_mask = (~torch.isnan(float_labels)).float()
                        
                        # Add task_mask attribute to the graph
                        raw_graph.task_mask = task_mask
            
            return raw_graph
        
        def __getitem__(self, idx):
            """TSGFM-style instant graph access using pre-processed graphs"""
            return self.graphs[idx]
        
        def __iter__(self):
            for i in range(len(self)):
                yield self[i]
    
    return TSGFMCompatibleDataset(tag_dataset)