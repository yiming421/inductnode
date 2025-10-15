#!/usr/bin/env python3
"""
Joint Training Script for Node Classification and Link Prediction
"""

import os
import sys
import time
import copy
import random
import torch
import wandb
import signal
import psutil
import numpy as np
from contextlib import contextmanager, nullcontext
from sknetwork.ranking import PageRank
from torch_geometric.utils import to_scipy_sparse_matrix

# Add the project root to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Core imports - reuse from existing scripts
from src.model import PureGCN_v1, PFNPredictorNodeCls, GCN, IdentityProjection, UnifiedGNN
from src.data_nc import load_all_data, load_all_data_train
from src.data_lp import load_all_data_link
from src.data_gc import load_all_graph_datasets, process_graph_features, create_data_loaders, create_task_filtered_datasets
from src.data_utils import process_data, prepare_link_data, select_link_context, process_link_data
from src.data_minibatch import MiniBatchNCLoader, compute_nc_loss_with_loader
from src.engine_nc import train_all, test_all, test_all_induct  # Node classification engines
from src.engine_lp import train_link_prediction, evaluate_link_prediction  # Link prediction engines
from src.engine_gc import (
    train_graph_classification_single_task,
    evaluate_graph_classification_single_task,
    aggregate_task_metrics,
    format_metric_results
)
from src.gpu_utils import parse_gpu_spec, setup_cuda_visible_devices, validate_gpu_availability, print_gpu_info
from transformers import get_cosine_schedule_with_warmup

# Logging and monitoring
from src.logger import TrainingLogger, LogLevel

from src.config import parse_joint_training_args
from src.checkpoint_utils import load_checkpoint_config, override_args_from_checkpoint, load_checkpoint_states, save_checkpoint


# Memory and time tracking utilities for Link Prediction
class LinkPredictionTracker:
    """Comprehensive memory and time tracker for link prediction operations."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset all tracking metrics."""
        self.timing_data = {
            'training': [],
            'evaluation': [],
            'data_preparation': [],
            'context_selection': [],
            'forward_pass': [],
            'loss_computation': [],
            'backward_pass': []
        }
        self.memory_data = {
            'gpu_peak': [],
            'gpu_allocated': [],
            'gpu_cached': [],
            'cpu_memory': [],
            'cpu_percent': []
        }
        self.operation_counts = {
            'training_steps': 0,
            'evaluation_steps': 0,
            'datasets_processed': 0
        }
    
    def get_memory_stats(self):
        """Get current memory statistics."""
        stats = {}
        
        # GPU memory
        if torch.cuda.is_available():
            stats['gpu_allocated'] = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            stats['gpu_cached'] = torch.cuda.memory_reserved(self.device) / 1024**3  # GB
            stats['gpu_peak'] = torch.cuda.max_memory_allocated(self.device) / 1024**3  # GB
        else:
            stats['gpu_allocated'] = 0
            stats['gpu_cached'] = 0  
            stats['gpu_peak'] = 0
        
        # CPU memory
        process = psutil.Process()
        stats['cpu_memory'] = process.memory_info().rss / 1024**3  # GB
        stats['cpu_percent'] = process.cpu_percent()
        
        return stats
    
    def record_memory(self):
        """Record current memory usage."""
        stats = self.get_memory_stats()
        self.memory_data['gpu_allocated'].append(stats['gpu_allocated'])
        self.memory_data['gpu_cached'].append(stats['gpu_cached'])
        self.memory_data['gpu_peak'].append(stats['gpu_peak'])
        self.memory_data['cpu_memory'].append(stats['cpu_memory'])
        self.memory_data['cpu_percent'].append(stats['cpu_percent'])
    
    @contextmanager
    def time_operation(self, operation_type):
        """Context manager to time operations."""
        start_time = time.time()
        self.record_memory()
        
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            if operation_type in self.timing_data:
                self.timing_data[operation_type].append(duration)
            
            self.record_memory()
    
    def get_summary_stats(self):
        """Get summary statistics for all tracked metrics."""
        summary = {}
        
        # Timing summaries
        for op_type, times in self.timing_data.items():
            if times:
                summary[f'time_{op_type}_avg'] = sum(times) / len(times)
                summary[f'time_{op_type}_total'] = sum(times)
                summary[f'time_{op_type}_max'] = max(times)
                summary[f'time_{op_type}_count'] = len(times)
            else:
                summary[f'time_{op_type}_avg'] = 0
                summary[f'time_{op_type}_total'] = 0
                summary[f'time_{op_type}_max'] = 0
                summary[f'time_{op_type}_count'] = 0
        
        # Memory summaries
        for mem_type, values in self.memory_data.items():
            if values:
                summary[f'memory_{mem_type}_avg'] = sum(values) / len(values)
                summary[f'memory_{mem_type}_peak'] = max(values)
                summary[f'memory_{mem_type}_final'] = values[-1]
            else:
                summary[f'memory_{mem_type}_avg'] = 0
                summary[f'memory_{mem_type}_peak'] = 0
                summary[f'memory_{mem_type}_final'] = 0
        
        # Operation counts
        summary.update(self.operation_counts)
        
        return summary
    
    
    def print_summary(self, epoch=None):
        """Print a human-readable summary of tracking statistics."""
        summary = self.get_summary_stats()
        
        epoch_str = f" (Epoch {epoch})" if epoch is not None else ""
        print(f"\n--- Link Prediction Performance Summary{epoch_str} ---")
        
        print("Timing Statistics:")
        for op_type in ['training', 'evaluation', 'data_preparation', 'context_selection']:
            avg_time = summary.get(f'time_{op_type}_avg', 0)
            total_time = summary.get(f'time_{op_type}_total', 0)
            count = summary.get(f'time_{op_type}_count', 0)
            if count > 0:
                print(f"  {op_type.replace('_', ' ').title()}: {avg_time:.3f}s avg, {total_time:.2f}s total ({count} ops)")
        
        print("Memory Statistics:")
        print(f"  GPU Peak: {summary.get('memory_gpu_peak_peak', 0):.2f} GB")
        print(f"  GPU Current: {summary.get('memory_gpu_allocated_final', 0):.2f} GB")
        print(f"  CPU Peak: {summary.get('memory_cpu_memory_peak', 0):.2f} GB")
        
        print("Operation Counts:")
        print(f"  Training Steps: {summary.get('training_steps', 0)}")
        print(f"  Evaluation Steps: {summary.get('evaluation_steps', 0)}")
        print(f"  Datasets Processed: {summary.get('datasets_processed', 0)}")
        print("------------------------------------------------")


# Memory and time tracking utilities for Graph Classification
class GraphClassificationTracker:
    """Comprehensive memory and time tracker for graph classification operations."""
    
    def __init__(self, device='cuda'):
        self.device = device
        self.reset()
    
    def reset(self):
        """Reset all tracking metrics."""
        self.timing_data = {
            'dataset_processing': [],
            'pca_computation': [],
            'feature_projection': [],
            'data_loading': [],
            'batch_processing': [],
            'training': [],
            'evaluation': [],
            'forward_pass': [],
            'backward_pass': [],
            'loss_computation': [],
            'task_processing': []
        }
        self.memory_data = {
            'gpu_peak': [],
            'gpu_allocated': [],
            'gpu_cached': [],
            'cpu_memory': [],
            'cpu_percent': [],
            'dataset_sizes': [],
            'batch_sizes': []
        }
        self.operation_counts = {
            'datasets_processed': 0,
            'tasks_processed': 0,
            'batches_processed': 0,
            'training_steps': 0,
            'evaluation_steps': 0,
            'pca_operations': 0,
            'feature_projections': 0
        }
        self.dataset_info = {}
    
    def get_memory_stats(self):
        """Get current memory statistics with additional CPU details."""
        stats = {}
        
        # GPU memory
        if torch.cuda.is_available():
            stats['gpu_allocated'] = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            stats['gpu_cached'] = torch.cuda.memory_reserved(self.device) / 1024**3  # GB
            stats['gpu_peak'] = torch.cuda.max_memory_allocated(self.device) / 1024**3  # GB
        else:
            stats['gpu_allocated'] = 0
            stats['gpu_cached'] = 0  
            stats['gpu_peak'] = 0
        
        # CPU memory with more details
        process = psutil.Process()
        memory_info = process.memory_info()
        stats['cpu_memory'] = memory_info.rss / 1024**3  # GB (Resident Set Size)
        stats['cpu_memory_vms'] = memory_info.vms / 1024**3  # GB (Virtual Memory Size)
        stats['cpu_percent'] = process.cpu_percent()
        
        # System-wide memory info
        system_memory = psutil.virtual_memory()
        stats['system_memory_total'] = system_memory.total / 1024**3  # GB
        stats['system_memory_available'] = system_memory.available / 1024**3  # GB
        stats['system_memory_percent'] = system_memory.percent
        
        return stats
    
    def record_memory(self):
        """Record current memory usage with enhanced CPU tracking."""
        stats = self.get_memory_stats()
        self.memory_data['gpu_allocated'].append(stats['gpu_allocated'])
        self.memory_data['gpu_cached'].append(stats['gpu_cached'])
        self.memory_data['gpu_peak'].append(stats['gpu_peak'])
        self.memory_data['cpu_memory'].append(stats['cpu_memory'])
        self.memory_data['cpu_percent'].append(stats['cpu_percent'])
    
    def log_memory(self, operation_name):
        """Log memory usage with operation name for detailed tracking."""
        stats = self.get_memory_stats()
        self.record_memory()
        
        # Log significant memory usage
        if stats['cpu_memory'] > 4.0:  # Log when CPU memory > 4GB
            print(f"[GC-MEMORY] {operation_name}: CPU={stats['cpu_memory']:.2f}GB, "
                  f"GPU={stats['gpu_allocated']:.2f}GB, System={stats['system_memory_percent']:.1f}%")
        
        # Check for memory spikes
        self.log_memory_spike_warning()
    
    def record_dataset_info(self, dataset_name, num_graphs, num_features, avg_nodes_per_graph=None):
        """Record dataset-specific information for memory analysis."""
        self.dataset_info[dataset_name] = {
            'num_graphs': num_graphs,
            'num_features': num_features,
            'avg_nodes_per_graph': avg_nodes_per_graph,
            'memory_at_load': self.get_memory_stats()['cpu_memory']
        }
        self.memory_data['dataset_sizes'].append(num_graphs * num_features)
    
    def record_batch_info(self, batch_size, num_nodes_in_batch=None):
        """Record batch processing information."""
        self.memory_data['batch_sizes'].append(batch_size)
        if num_nodes_in_batch:
            # This helps track memory usage patterns with batch complexity
            pass
    
    @contextmanager
    def time_operation(self, operation_type, log_memory=True):
        """Context manager to time operations with enhanced memory tracking."""
        start_time = time.time()
        if log_memory:
            start_memory = self.get_memory_stats()
            self.record_memory()
        
        try:
            yield
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            if operation_type in self.timing_data:
                self.timing_data[operation_type].append(duration)
            
            if log_memory:
                end_memory = self.get_memory_stats()
                self.record_memory()
                
                # Calculate memory delta for this operation
                memory_delta = end_memory['cpu_memory'] - start_memory['cpu_memory']
                if abs(memory_delta) > 0.1:  # Only log significant memory changes (>100MB)
                    print(f"[GC-MEMORY] {operation_type}: {memory_delta:+.2f} GB CPU memory change (Duration: {duration:.2f}s)")
    
    def log_memory_spike_warning(self, threshold_gb=8.0):
        """Check for memory spikes and log warnings."""
        current_stats = self.get_memory_stats()
        if current_stats['cpu_memory'] > threshold_gb:
            print(f"[GC-MEMORY-WARNING] High CPU memory usage: {current_stats['cpu_memory']:.2f} GB "
                  f"(System: {current_stats['system_memory_percent']:.1f}% used)")
        
        if current_stats['gpu_allocated'] > 10.0:  # 10GB GPU threshold
            print(f"[GC-MEMORY-WARNING] High GPU memory usage: {current_stats['gpu_allocated']:.2f} GB")
    
    def get_summary_stats(self):
        """Get summary statistics for all tracked metrics with enhanced CPU analysis."""
        summary = {}
        
        # Timing summaries
        for op_type, times in self.timing_data.items():
            if times:
                summary[f'time_{op_type}_avg'] = sum(times) / len(times)
                summary[f'time_{op_type}_total'] = sum(times)
                summary[f'time_{op_type}_max'] = max(times)
                summary[f'time_{op_type}_count'] = len(times)
            else:
                summary[f'time_{op_type}_avg'] = 0
                summary[f'time_{op_type}_total'] = 0
                summary[f'time_{op_type}_max'] = 0
                summary[f'time_{op_type}_count'] = 0
        
        # Memory summaries with enhanced CPU tracking
        for mem_type, values in self.memory_data.items():
            if values:
                summary[f'memory_{mem_type}_avg'] = sum(values) / len(values)
                summary[f'memory_{mem_type}_peak'] = max(values)
                summary[f'memory_{mem_type}_final'] = values[-1]
                if len(values) > 1:
                    summary[f'memory_{mem_type}_delta'] = values[-1] - values[0]
            else:
                summary[f'memory_{mem_type}_avg'] = 0
                summary[f'memory_{mem_type}_peak'] = 0
                summary[f'memory_{mem_type}_final'] = 0
                summary[f'memory_{mem_type}_delta'] = 0
        
        # Operation counts
        summary.update(self.operation_counts)
        
        # Dataset analysis
        if self.dataset_info:
            total_graphs = sum(info['num_graphs'] for info in self.dataset_info.values())
            total_features = sum(info['num_features'] for info in self.dataset_info.values())
            summary['total_graphs_processed'] = total_graphs
            summary['total_features_processed'] = total_features
        
        return summary
    
    
    def print_detailed_summary(self, epoch=None):
        """Print a comprehensive summary of tracking statistics."""
        summary = self.get_summary_stats()
        current_stats = self.get_memory_stats()
        
        epoch_str = f" (Epoch {epoch})" if epoch is not None else ""
        print(f"\n=== Graph Classification Memory & Performance Summary{epoch_str} ===")
        
        print("🕒 Timing Statistics:")
        critical_ops = ['dataset_processing', 'pca_computation', 'training', 'evaluation', 'batch_processing']
        for op_type in critical_ops:
            avg_time = summary.get(f'time_{op_type}_avg', 0)
            total_time = summary.get(f'time_{op_type}_total', 0)
            max_time = summary.get(f'time_{op_type}_max', 0)
            count = summary.get(f'time_{op_type}_count', 0)
            if count > 0:
                print(f"  {op_type.replace('_', ' ').title()}: "
                      f"avg={avg_time:.3f}s, total={total_time:.2f}s, max={max_time:.3f}s ({count} ops)")
        
        print("\n💾 CPU Memory Statistics:")
        print(f"  Current Usage: {current_stats['cpu_memory']:.2f} GB")
        print(f"  Peak Usage: {summary.get('memory_cpu_memory_peak', 0):.2f} GB")
        print(f"  Average Usage: {summary.get('memory_cpu_memory_avg', 0):.2f} GB")
        print(f"  Total Change: {summary.get('memory_cpu_memory_delta', 0):+.2f} GB")
        print(f"  System Memory: {current_stats['system_memory_percent']:.1f}% used "
              f"({current_stats['system_memory_available']:.1f} GB available)")
        
        print("\n🖥️  GPU Memory Statistics:")
        print(f"  Current Allocated: {current_stats['gpu_allocated']:.2f} GB")
        print(f"  Peak Allocated: {summary.get('memory_gpu_peak_peak', 0):.2f} GB")
        print(f"  Cached: {current_stats['gpu_cached']:.2f} GB")
        
        print("\n📊 Operation Counts:")
        print(f"  Datasets: {summary.get('datasets_processed', 0)}")
        print(f"  Tasks: {summary.get('tasks_processed', 0)}")
        print(f"  Batches: {summary.get('batches_processed', 0)}")
        print(f"  PCA Operations: {summary.get('pca_operations', 0)}")
        print(f"  Training Steps: {summary.get('training_steps', 0)}")
        
        if self.dataset_info:
            print("\n🗃️  Dataset Information:")
            for name, info in self.dataset_info.items():
                print(f"  {name}: {info['num_graphs']} graphs, {info['num_features']} features, "
                      f"memory at load: {info['memory_at_load']:.2f} GB")
        
        print("=" * 70)
    
    def print_summary(self, epoch=None):
        """Alias for print_detailed_summary for compatibility."""
        self.print_detailed_summary(epoch)


# Global tracker instances
lp_tracker = None
gc_tracker = None


def parse_context_overrides(override_string):
    """
    Parse "dataset1:shots1,dataset2:shots2" into dict.
    
    Args:
        override_string (str): String in format "dataset1:shots1,dataset2:shots2"
        
    Returns:
        dict: Mapping from dataset name to context shots
    """
    if not override_string:
        return {}
    
    overrides = {}
    for pair in override_string.split(','):
        if ':' in pair:
            dataset, shots = pair.strip().split(':')
            overrides[dataset.strip()] = int(shots.strip())
    return overrides


def parse_context_bounds(bounds_string):
    """
    Parse context bounds string like "(10,30)(64,192)(8,24)" into bounds for NC/LP/GC.
    
    Args:
        bounds_string (str): String in format "(lower,upper)(lower,upper)(lower,upper)"
        
    Returns:
        dict: Dictionary with 'nc', 'lp', 'gc' keys, each containing (lower, upper) tuple
    """
    import re
    
    if not bounds_string:
        # Default bounds
        return {
            'nc': (10, 30),
            'lp': (64, 192), 
            'gc': (8, 24)
        }
    
    # Extract all (lower,upper) pairs using regex
    pattern = r'\((\d+),(\d+)\)'
    matches = re.findall(pattern, bounds_string)
    
    if len(matches) != 3:
        raise ValueError(f"Expected 3 bound pairs for NC/LP/GC, got {len(matches)}: {bounds_string}")
    
    # Convert to integers and create bounds dict
    bounds = {
        'nc': (int(matches[0][0]), int(matches[0][1])),
        'lp': (int(matches[1][0]), int(matches[1][1])), 
        'gc': (int(matches[2][0]), int(matches[2][1]))
    }
    
    # Validate bounds
    for task_type, (lower, upper) in bounds.items():
        if lower >= upper:
            raise ValueError(f"Invalid bounds for {task_type.upper()}: lower ({lower}) must be < upper ({upper})")
        if lower <= 0:
            raise ValueError(f"Invalid bounds for {task_type.upper()}: lower bound ({lower}) must be > 0")
    
    return bounds


def filter_candidates_with_pagerank(data, candidate_indices, max_candidates):
    """
    Efficiently filter candidate nodes using PageRank on the original graph.
    
    Args:
        data: Graph data object with adjacency information
        candidate_indices: Tensor of candidate node indices
        max_candidates (int): Maximum number of candidates to keep
        
    Returns:
        torch.Tensor: Filtered candidate indices based on PageRank scores
    """
    if len(candidate_indices) <= max_candidates:
        return candidate_indices
    
    device = candidate_indices.device
    
    # Convert to SciPy sparse matrix
    row, col, _ = data.adj_t.coo()  # torch_sparse.SparseTensor.coo() returns (row, col, values)
    edge_index = torch.stack([row, col], dim=0)
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=data.x.shape[0])
    
    # Compute PageRank using scikit-network
    pagerank = PageRank()
    pagerank.fit(adj)
    scores = pagerank.scores_
    
    # Extract scores for candidates using direct numpy indexing
    candidate_nodes = candidate_indices.cpu().numpy()
    candidate_scores = scores[candidate_nodes]
    
    # Use simple argsort for sorting (descending order)
    top_k_indices = np.argsort(candidate_scores)[-max_candidates:][::-1]
    top_candidates = candidate_nodes[top_k_indices]
    
    # Convert back to tensor
    filtered_candidates = torch.tensor(top_candidates, dtype=torch.long, device=device)
    
    print(f"[PageRank] Filtered {len(candidate_indices)} candidates to {len(filtered_candidates)} using PageRank")
    return filtered_candidates


def sample_context_with_kmedoids(data, k_shot, train_indices, use_kmedoids=False, random_state=None):
    """
    Sample context nodes using K-Medoids clustering for better representativeness.
    
    Args:
        data: Graph data object with node features and labels
        k_shot (int): Number of context samples per class
        train_indices: Training node indices
        use_kmedoids (bool): Whether to use K-Medoids clustering
        random_state (int): Random seed
        
    Returns:
        context_sample (torch.Tensor): Indices of selected context nodes
    """
    if not use_kmedoids:
        # Fallback to original random sampling
        from src.data_utils import select_k_shot_context
        return select_k_shot_context(data, k_shot, train_indices)
    
    try:
        from sklearn_extra.cluster import KMedoids
    except ImportError:
        print("[K-Medoids] scikit-learn-extra not available, falling back to random sampling")
        from src.data_utils import select_k_shot_context
        return select_k_shot_context(data, k_shot, train_indices)
    
    device = data.x.device
    context_samples = []
    
    # Get unique classes
    unique_classes = data.y.unique()
    
    for class_label in unique_classes:
        # Get training nodes for this class
        class_mask = data.y == class_label
        class_train_mask = torch.zeros_like(data.y, dtype=torch.bool)
        class_train_mask[train_indices] = True
        
        # Find intersection: nodes that are both in this class and in training set
        class_train_nodes = torch.where(class_mask & class_train_mask)[0]
        
        if len(class_train_nodes) == 0:
            continue
            
        if len(class_train_nodes) <= k_shot:
            # If we have fewer nodes than k_shot, take all of them
            context_samples.append(class_train_nodes)
        else:
            # Check if candidate pool is too large (>10x k_shot)
            if len(class_train_nodes) > 10 * k_shot:
                # Use PageRank to pre-filter candidates
                max_candidates = 10 * k_shot
                class_train_nodes = filter_candidates_with_pagerank(data, class_train_nodes, max_candidates)
            
            # Use K-Medoids clustering to find representative nodes for this class
            class_features = data.x[class_train_nodes].detach().cpu().numpy()
            
            # Apply K-Medoids clustering
            n_clusters = min(k_shot, len(class_train_nodes))
            kmedoids = KMedoids(
                n_clusters=n_clusters,
                metric='cosine',  # Use cosine distance for node features
                init='k-medoids++',  # Smart initialization
                max_iter=100,
                random_state=random_state
            )
            
            kmedoids.fit(class_features)
            medoid_indices_in_class = kmedoids.medoid_indices_
            
            # Map medoid indices back to original node indices
            selected_nodes = class_train_nodes[medoid_indices_in_class]
            context_samples.append(selected_nodes)
    
    if not context_samples:
        # Fallback if no samples found
        from src.data_utils import select_k_shot_context
        return select_k_shot_context(data, k_shot, train_indices)
    
    # Concatenate all context samples
    context_sample = torch.cat(context_samples, dim=0)
    
    print(f"[K-Medoids] Sampled {len(context_sample)} context nodes using clustering (target: {k_shot} per class)")
    
    return context_sample


def resolve_context_shots(dataset_name, task_type, args, epoch=None):
    """
    Resolve context shots for a specific dataset and task using the configured sampling plan.
    Falls back to global defaults if no override specified.
    
    Args:
        dataset_name (str): Name of the dataset
        task_type (str): Task type ('nc', 'lp', 'gc')
        args: Parsed command line arguments
        epoch (int, optional): Current training epoch (needed for decay plan)
        
    Returns:
        int: Number of context shots to use for this dataset
    """
    # Get override mapping for this task type
    override_attr = f'{task_type}_context_overrides'
    override_string = getattr(args, override_attr, None)
    override_map = parse_context_overrides(override_string)
    
    # Return override if exists (overrides always use original fixed behavior)
    if dataset_name in override_map:
        print(f"[Context Override] {task_type.upper()} dataset '{dataset_name}': using {override_map[dataset_name]} context shots")
        return override_map[dataset_name]
    
    # Check sampling plan
    sampling_plan = getattr(args, 'context_sampling_plan', 'ori')
    
    if sampling_plan == 'ori':
        # Original behavior: use global defaults
        defaults = {
            'nc': args.context_num,
            'lp': args.context_k, 
            'gc': args.context_graph_num
        }
        return defaults[task_type]
        
    elif sampling_plan == 'random':
        # Random sampling within bounds
        import random
        bounds = parse_context_bounds(getattr(args, 'context_bounds', None))
        lower, upper = bounds[task_type]
        context_shots = random.randint(lower, upper)
        print(f"[Context Random] {task_type.upper()} dataset '{dataset_name}': using {context_shots} context shots (range: {lower}-{upper})")
        return context_shots
        
    elif sampling_plan == 'decay':
        # Gradual decay from upper to lower bound over training
        if epoch is None:
            # During evaluation, fall back to original fixed behavior instead of upper bound
            defaults = {
                'nc': args.context_num,
                'lp': args.context_k,
                'gc': args.context_graph_num
            }
            context_shots = defaults[task_type]
            print(f"[Context Evaluation] {task_type.upper()} dataset '{dataset_name}': using {context_shots} context shots (evaluation mode, ignoring decay)")
            return context_shots
            
        bounds = parse_context_bounds(getattr(args, 'context_bounds', None))
        lower, upper = bounds[task_type]
        
        # Linear decay from upper to lower over training epochs
        total_epochs = getattr(args, 'epochs', 100)
        progress = min(1.0, epoch / max(1, total_epochs - 1))  # 0 to 1
        
        # Interpolate: start at upper, end at lower
        context_shots = int(upper - progress * (upper - lower))
        context_shots = max(lower, min(upper, context_shots))  # Clamp to bounds
        
        print(f"[Context Decay] {task_type.upper()} dataset '{dataset_name}': using {context_shots} context shots (epoch {epoch}, progress {progress:.3f})")
        return context_shots
    
    else:
        raise ValueError(f"Unknown context sampling plan: {sampling_plan}")
    
    # Fallback to global defaults (should not reach here)
    defaults = {
        'nc': args.context_num,
        'lp': args.context_k, 
        'gc': args.context_graph_num
    }
    return defaults[task_type]


def refresh_nc_contexts(nc_data, args=None, epoch=None):
    """Refresh node classification context samples"""
    if nc_data[0] is None:
        return
    
    nc_data_list, nc_split_idx_list, nc_external_embeddings = nc_data
    print("  🔄 Refreshing NC contexts...")
    
    for data, split_idx in zip(nc_data_list, nc_split_idx_list):
        if hasattr(data, 'context_sample'):
            # Determine context size based on sampling plan
            if args is not None:
                # Use new sampling strategy
                context_shots = resolve_context_shots(data.name, 'nc', args, epoch)
                current_context_size = context_shots
            else:
                # Fallback to original behavior: get current context size
                current_context_size = len(data.context_sample) // len(data.y.unique())
            
            # Check if K-Medoids sampling is enabled
            use_kmedoids = getattr(args, 'use_kmedoids_sampling', False) if args is not None else False
            
            if use_kmedoids:
                # Use K-Medoids clustering for context sampling
                new_context_sample = sample_context_with_kmedoids(
                    data, current_context_size, split_idx['train'], 
                    use_kmedoids=True, random_state=epoch
                )
            else:
                # Use original random sampling
                from src.data_utils import select_k_shot_context
                new_context_sample = select_k_shot_context(data, current_context_size, split_idx['train'])
            
            data.context_sample = new_context_sample.to(data.context_sample.device)
            sampling_method = "K-Medoids clustering" if use_kmedoids else "random sampling"
            print(f"    ✓ Refreshed {data.name}: {len(new_context_sample)} context samples ({current_context_size} per class, {sampling_method})")


def refresh_lp_contexts(lp_data, args, epoch=None):
    """Refresh link prediction context samples"""
    if lp_data[0] is None:
        return
    
    lp_data_list, lp_split_idx_list, lp_context_data, lp_masks, lp_link_data_all = lp_data
    print("  🔄 Refreshing LP contexts...")
    
    # Refresh context for each dataset
    for i, (data, split_idx) in enumerate(zip(lp_data_list, lp_split_idx_list)):
        context_shots = resolve_context_shots(data.name, 'lp', args, epoch)
        
        # Regenerate context
        link_data = lp_link_data_all[i]
        if 'train' in link_data and link_data['train']['edge_pairs'].size(0) > 0:
            new_context_data, new_train_mask = select_link_context(
                link_data['train'], context_shots, args.context_neg_ratio,
                args.remove_context_from_train
            )
            
            # Update stored context data
            lp_context_data[i] = new_context_data
            lp_masks[i] = new_train_mask
            print(f"    ✓ Refreshed {data.name}: {context_shots} context shots")


def refresh_gc_contexts(gc_data, args, epoch=None):
    """Refresh graph classification context samples"""
    if len(gc_data[0]) == 0:
        return
    
    gc_data_list, gc_processed_data_list = gc_data
    print("  🔄 Refreshing GC contexts...")
    
    # For graph classification, we need to regenerate task-filtered datasets
    # This is more complex as it involves resampling from the original datasets
    for dataset_info in gc_processed_data_list:
        if 'dataset' in dataset_info:
            dataset_name = dataset_info['dataset'].name if hasattr(dataset_info['dataset'], 'name') else 'GC dataset'
            
            # Get dynamic context shots for this dataset
            context_shots = resolve_context_shots(dataset_name, 'gc', args, epoch)
            
            # Regenerate task-filtered splits with new context size
            # Note: This requires modifying create_task_filtered_datasets to accept context_k parameter
            # For now, we'll regenerate with the current logic and note the context change
            task_filtered_splits = create_task_filtered_datasets(
                dataset_info['dataset'], 
                dataset_info['split_idx']
            )
            dataset_info['task_filtered_splits'] = task_filtered_splits
            
            print(f"    ✓ Refreshed {dataset_name}: context samples regenerated (target: {context_shots} context shots)")


def refresh_contexts_if_needed(epoch, args, data_dict):
    """Refresh contexts for all tasks if needed based on interval"""
    
    # Check if refresh is enabled and if it's time to refresh
    if args.context_refresh_interval <= 0 or epoch % args.context_refresh_interval != 0:
        return
    
    print(f"\n🔄 Refreshing contexts at epoch {epoch} (interval: {args.context_refresh_interval})")
    print(f"   Sampling plan: {getattr(args, 'context_sampling_plan', 'ori')}")
    
    # Set different seed for each refresh to ensure diversity
    refresh_seed = args.seed + epoch // args.context_refresh_interval
    torch.manual_seed(refresh_seed)
    
    # Refresh each task with epoch information
    if getattr(args, 'enable_nc', True) and data_dict['nc_train'][0] is not None:
        refresh_nc_contexts(data_dict['nc_train'], args, epoch)
        
    if getattr(args, 'enable_lp', True) and data_dict['lp_train'][0] is not None:
        refresh_lp_contexts(data_dict['lp_train'], args, epoch)
        
    if getattr(args, 'enable_gc', True) and len(data_dict['gc_train'][0]) > 0:
        refresh_gc_contexts(data_dict['gc_train'], args, epoch)
    
    print("  ✅ Context refresh completed\n")


def refresh_contexts_if_needed_batch(batch_idx, epoch, args, data_dict, task_type='all'):
    """
    Simple batch-level context refresh. Much more efficient than epoch-level.
    
    Args:
        batch_idx (int): Current batch index within the task
        epoch (int): Current epoch (for decay/random seed)
        args: Command line arguments
        data_dict: Data dictionary with contexts
        task_type (str): Which task to refresh ('nc', 'lp', 'gc', 'all')
    """
    # Check if batch refresh is enabled and it's time to refresh
    if args.context_batch_refresh_interval <= 0 or batch_idx % args.context_batch_refresh_interval != 0:
        return
    
    # Skip if batch_idx is 0 (first batch of each task - no need to refresh immediately)
    if batch_idx == 0:
        return
    
    print(f"🔄 Batch refresh at batch {batch_idx} (interval: {args.context_batch_refresh_interval})")
    
    # Use batch index + epoch for seed diversity
    refresh_seed = args.seed + epoch * 1000 + batch_idx
    torch.manual_seed(refresh_seed)
    
    # Refresh specific task or all tasks
    if task_type in ['nc', 'all'] and getattr(args, 'enable_nc', True) and data_dict['nc_train'][0] is not None:
        refresh_nc_contexts(data_dict['nc_train'], args, epoch)
        
    if task_type in ['lp', 'all'] and getattr(args, 'enable_lp', True) and data_dict['lp_train'][0] is not None:
        refresh_lp_contexts(data_dict['lp_train'], args, epoch)
        
    if task_type in ['gc', 'all'] and getattr(args, 'enable_gc', True) and len(data_dict['gc_train'][0]) > 0:
        refresh_gc_contexts(data_dict['gc_train'], args, epoch)


def setup_graph_dataset_environment(args):
    """Configure graph dataset embeddings (FUG, TSGFM, or TAGDataset) if requested."""
    if not args.enable_gc:
        # Clear all environment variables if graph classification is disabled
        os.environ.pop('USE_FUG_EMB', None)
        os.environ.pop('TAG_DATASET_ROOT', None)
        return
        
    # Priority: Original Features > FUG > TAGDataset > TSGFM
    if hasattr(args, 'use_original_features') and args.use_original_features:
        print(f"Enabling ORIGINAL FEATURES for graph classification")
        print(f"OGB root: {args.ogb_root}, TU root: ./dataset/TU")
        os.environ['USE_ORIGINAL_FEATURES'] = '1'
        os.environ['OGB_ROOT'] = args.ogb_root
        os.environ['TU_ROOT'] = './dataset/TU'
        # Clear other embedding settings
        os.environ.pop('USE_FUG_EMB', None)
        os.environ.pop('TAG_DATASET_ROOT', None)

    elif hasattr(args, 'use_ogb_fug') and args.use_ogb_fug:
        print(f"Enabling OGB FUG embeddings for graph classification")
        print(f"FUG root: {args.fug_root}, OGB root: {args.ogb_root}")
        os.environ['USE_FUG_EMB'] = '1'
        os.environ['FUG_EMB_ROOT'] = args.fug_root
        os.environ['OGB_ROOT'] = args.ogb_root
        # Clear other embedding settings
        os.environ.pop('TAG_DATASET_ROOT', None)
        os.environ.pop('USE_ORIGINAL_FEATURES', None)

    elif hasattr(args, 'use_tag_dataset') and args.use_tag_dataset:
        print(f"Enabling TAGDataset embeddings for graph classification")
        print(f"TAGDataset root: {args.tag_dataset_root}, Embedding family: {args.embedding_family}")
        os.environ['TAG_DATASET_ROOT'] = args.tag_dataset_root
        os.environ['EMBEDDING_FAMILY'] = args.embedding_family
        # Clear FUG settings
        os.environ.pop('USE_FUG_EMB', None)
        
    elif hasattr(args, 'use_tsgfm') and args.use_tsgfm:
        print(f"Using TSGFM datasets for graph classification")
        # Clear other embedding settings
        os.environ.pop('USE_FUG_EMB', None)
        os.environ.pop('TAG_DATASET_ROOT', None)
        
    else:
        print(f"Using standard TU datasets for graph classification")
        # Clear all special embedding settings
        os.environ.pop('USE_FUG_EMB', None)
        os.environ.pop('TAG_DATASET_ROOT', None)


def process_datasets_for_models(datasets, processed_data_list, args, device, test_datasets=False):
    """
    Process datasets to handle feature dimensions and create any necessary dummy features.
    Adapted from graph_classification.py for joint training.
    """
    processed_datasets = []
    final_num_features = args.hidden
    
    for dataset, dataset_info in zip(datasets, processed_data_list):
        # Check if this is a TSGFM dataset - if so, disable PCA caching
        is_tsgfm_dataset = hasattr(args, 'use_tsgfm') and args.use_tsgfm and not (hasattr(args, 'use_ogb_fug') and args.use_ogb_fug) and not (hasattr(args, 'use_tag_dataset') and args.use_tag_dataset)
        use_pca_cache_for_dataset = args.use_pca_cache and not is_tsgfm_dataset
        
        if is_tsgfm_dataset:
            print(f"TSGFM dataset detected ({getattr(dataset, 'name', 'unknown')}): PCA caching disabled")
        
        # Process features using PCA and padding
        if test_datasets:
            processing_info = process_graph_features(
                dataset, args.hidden, device, 
                args.use_identity_projection, args.projection_small_dim, args.projection_large_dim,
                args.use_full_pca, False, args.normalize_data,
                args.padding_strategy, args.use_batchnorm,
                pca_device=args.pca_device, incremental_pca_batch_size=args.incremental_pca_batch_size,
                pca_sample_threshold=500000,  # Default threshold for sampled PCA
                processed_data=dataset_info,  # Pass embedding mapping info (FUG/TSGFM/TAGDataset)
                pcba_context_only_pca=False,  # Use full optimization for test datasets
                use_pca_cache=use_pca_cache_for_dataset, pca_cache_dir=args.pca_cache_dir,
                dataset_name=getattr(dataset, 'name', None)
            )
        else:
            processing_info = process_graph_features(
                dataset, args.hidden, device, 
                args.use_identity_projection, args.projection_small_dim, args.projection_large_dim,
                args.use_full_pca, False, args.normalize_data,
                args.padding_strategy, args.use_batchnorm,
                pca_device=args.pca_device, incremental_pca_batch_size=args.incremental_pca_batch_size,
                pca_sample_threshold=500000,  # Default threshold for sampled PCA
                processed_data=dataset_info,  # Pass embedding mapping info (FUG/TSGFM/TAGDataset)
                pcba_context_only_pca=False,  # Use full optimization for training datasets
                use_pca_cache=use_pca_cache_for_dataset, pca_cache_dir=args.pca_cache_dir,
                dataset_name=getattr(dataset, 'name', None)
            )
        
        processed_datasets.append(dataset)

        # Update dataset info with processing information
        dataset_info.update(processing_info)
    
    return processed_datasets, processed_data_list, final_num_features

def create_unified_model(args, input_dim, device):
    """
    Create the unified model components for both tasks.
    Reuses model creation logic from existing scripts.
    """
    # Initialize optional components
    identity_projection = None

    hidden = args.projection_large_dim if args.use_identity_projection else args.hidden
        
    if args.use_identity_projection:
        identity_projection = IdentityProjection(args.projection_small_dim, args.projection_large_dim)
        identity_projection = identity_projection.to(device)
    
    # Create GNN backbone
    if args.model == 'PureGCN_v1':
        model = PureGCN_v1(hidden, args.num_layers, hidden, args.dp, args.norm,
                          args.res, args.relu, args.gnn_norm_affine,
                          activation=getattr(args, 'activation', 'relu'),
                          use_virtual_node=getattr(args, 'use_virtual_node', False))
    elif args.model == 'GCN':
        model = GCN(hidden, hidden, args.norm, args.relu, args.num_layers, args.dp,
                   args.multilayer, args.use_gin, args.res, args.gnn_norm_affine,
                   activation=getattr(args, 'activation', 'relu'))
    elif args.model == 'UnifiedGNN':
        model = UnifiedGNN(
            model_type=getattr(args, 'unified_model_type', 'gcn'),
            in_feats=hidden,
            h_feats=hidden,
            prop_step=getattr(args, 'num_layers', 2),  # Reuse num_layers as prop_step
            conv=getattr(args, 'conv_type', 'GCN'),
            multilayer=getattr(args, 'multilayer', False),
            norm=getattr(args, 'norm', False),
            relu=getattr(args, 'relu', False),
            dropout=getattr(args, 'dp', 0.2),
            residual=getattr(args, 'residual', 1.0),
            linear=getattr(args, 'linear', False),
            alpha=getattr(args, 'alpha', 0.5),
            exp=getattr(args, 'exp', False),
            res=getattr(args, 'res', False),
            supports_edge_weight=getattr(args, 'supports_edge_weight', False),
            no_parameters=getattr(args, 'no_parameters', False),
            input_norm=getattr(args, 'input_norm', False),
            activation=getattr(args, 'activation', 'relu')
        )
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")
    
    # Create unified predictor (same for both tasks)
    if args.predictor == 'PFN':
        predictor = PFNPredictorNodeCls(
            hidden, args.nhead, args.transformer_layers, args.mlp_layers,
            args.dp, args.norm, args.seperate, False, None, None, args.sim,
            args.padding, args.mlp_norm_affine, args.normalize_class_h,
            norm_type=getattr(args, 'transformer_norm_type', 'post'),
            use_moe=getattr(args, 'use_moe', False),
            moe_num_experts=getattr(args, 'moe_num_experts', 4),
            moe_top_k=getattr(args, 'moe_top_k', 2),
            moe_auxiliary_loss_weight=getattr(args, 'moe_auxiliary_loss_weight', 0.01),
            ffn_expansion_ratio=getattr(args, 'ffn_expansion_ratio', 4)
        )
    else:
        raise NotImplementedError(f"Predictor {args.predictor} not implemented")
    
    model = model.to(device)
    predictor = predictor.to(device)
    
    return model, predictor, identity_projection


def load_and_preprocess_data(args, device, skip_training_data=False, gc_tracker=None):
    """
    Load and preprocess data for enabled tasks.
    Returns processed datasets for node classification, link prediction, and graph classification.
    
    Args:
        skip_training_data: If True, skip loading training datasets (for pretrained model evaluation)
    """
    global lp_tracker
    
    print("\n=== Loading and Preprocessing Data ===")
    
    # === Node Classification Data ===
    nc_train_data_list, nc_train_split_idx_list = None, None
    nc_test_data_list, nc_test_split_idx_list = None, None
    # Ensure local initialization before any reference to avoid UnboundLocalError
    nc_train_external_embeddings = None
    nc_test_external_embeddings = None
    
    if args.enable_nc:
        print("Loading node classification datasets...")
        nc_train_datasets = args.nc_train_dataset.split(',')
        nc_test_datasets = args.nc_test_dataset.split(',')

        # Auto-fix: Change 'legacy' to 'small_valid' for large datasets to prevent infinite loops
        if args.split_rebalance_strategy == 'legacy':
            large_datasets = ['ogbn-products', 'ogbn-papers100M']
            if any(dataset in nc_train_datasets for dataset in large_datasets):
                print(f"⚠️  WARNING: Detected large dataset(s) {[d for d in large_datasets if d in nc_train_datasets]} with 'legacy' split strategy.")
                print(f"⚠️  Automatically changing split_rebalance_strategy from 'legacy' to 'small_valid' to prevent infinite training loops.")
                args.split_rebalance_strategy = 'smallest_for_valid'

        # 1) Load training/test data lists first
        if not skip_training_data:
            nc_train_data_list, nc_train_split_idx_list = load_all_data_train(
                nc_train_datasets,
                split_strategy=args.split_rebalance_strategy
            )
        else:
            print("  Skipping NC training data loading (using pretrained model)")

        nc_test_data_list, nc_test_split_idx_list = load_all_data(nc_test_datasets)

        # 2) Load GPSE/LapPE/RWSE embeddings (if enabled)
        if args.use_gpse or args.use_lappe or args.use_rwse:
            from src.data import attach_gpse_embeddings

            pe_types = []
            if args.use_gpse: pe_types.append('GPSE')
            if args.use_lappe: pe_types.append('LapPE')
            if args.use_rwse: pe_types.append('RWSE')
            pe_str = '+'.join(pe_types)

            print(f"\n📊 Loading {pe_str} embeddings...")
            if not skip_training_data:
                train_count = attach_gpse_embeddings(
                    nc_train_data_list,
                    nc_train_datasets,
                    gpse_dir=args.gpse_dir,
                    verbose=args.gpse_verbose,
                    use_gpse=args.use_gpse,
                    use_lappe=args.use_lappe,
                    use_rwse=args.use_rwse
                )
                print(f"  ✓ {pe_str} embeddings loaded for {train_count}/{len(nc_train_data_list)} training datasets")

            test_count = attach_gpse_embeddings(
                nc_test_data_list,
                nc_test_datasets,
                gpse_dir=args.gpse_dir,
                verbose=args.gpse_verbose,
                use_gpse=args.use_gpse,
                use_lappe=args.use_lappe,
                use_rwse=args.use_rwse
            )
            print(f"  ✓ {pe_str} embeddings loaded for {test_count}/{len(nc_test_data_list)} test datasets\n")

        # 3) Load external embeddings (if enabled) before processing datasets so they can be used
        if getattr(args, 'use_external_embeddings_nc', False):
            print("Loading external embeddings for node classification...")

            if not os.path.exists(args.fug_root):
                print(f"  WARNING: fug_root directory does not exist: {args.fug_root}")
                print(f"  Continuing without external embeddings...")
            else:
                # Training embeddings
                if not skip_training_data and nc_train_data_list is not None:
                    nc_train_external_embeddings = []
                    for data in nc_train_data_list:
                        dataset_name = getattr(data, 'name', 'unknown')
                        emb_file = os.path.join(args.fug_root, dataset_name, f"{dataset_name}.pt")
                        if os.path.exists(emb_file):
                            try:
                                external_embeddings = torch.load(emb_file, map_location='cpu')
                                print(f"  Loaded external embeddings for {dataset_name}: {external_embeddings.shape}")
                                if external_embeddings.size(0) != data.num_nodes:
                                    print(f"  WARNING: {dataset_name} embedding count {external_embeddings.size(0)} != dataset nodes {data.num_nodes}")
                                nc_train_external_embeddings.append(external_embeddings)
                            except Exception as e:
                                print(f"  Failed to load external embeddings for {dataset_name}: {e}")
                                nc_train_external_embeddings.append(None)
                        else:
                            print(f"  No external embeddings found for {dataset_name} (expected: {emb_file})")
                            nc_train_external_embeddings.append(None)
                # Test embeddings
                if nc_test_data_list is not None:
                    nc_test_external_embeddings = []
                    for data in nc_test_data_list:
                        dataset_name = getattr(data, 'name', 'unknown')
                        emb_file = os.path.join(args.fug_root, dataset_name, f"{dataset_name}.pt")
                        if os.path.exists(emb_file):
                            try:
                                external_embeddings = torch.load(emb_file, map_location='cpu')
                                print(f"  Loaded external embeddings for {dataset_name}: {external_embeddings.shape}")
                                if external_embeddings.size(0) != data.num_nodes:
                                    print(f"  WARNING: {dataset_name} embedding count {external_embeddings.size(0)} != dataset nodes {data.num_nodes}")
                                nc_test_external_embeddings.append(external_embeddings)
                            except Exception as e:
                                print(f"  Failed to load external embeddings for {dataset_name}: {e}")
                                nc_test_external_embeddings.append(None)
                        else:
                            print(f"  No external embeddings found for {dataset_name} (expected: {emb_file})")
                            nc_test_external_embeddings.append(None)

        # 3) Process training data (now embeddings are ready if enabled)
        if not skip_training_data:
            for i, (data, split_idx) in enumerate(zip(nc_train_data_list, nc_train_split_idx_list)):
                data.x = data.x.to(device)
                data.adj_t = data.adj_t.to(device)
                data.y = data.y.to(device)

                external_emb = nc_train_external_embeddings[i] if nc_train_external_embeddings else None

                process_data(
                    data, split_idx, args.hidden, args.context_num, False, args.use_full_pca,
                    args.normalize_data, False, 32, 0, args.padding_strategy,
                    args.use_batchnorm, args.use_identity_projection, args.projection_small_dim, args.projection_large_dim, args.pca_device,
                    args.incremental_pca_batch_size, external_emb
                )

                if getattr(args, 'use_kmedoids_sampling', False):
                    new_context_sample = sample_context_with_kmedoids(
                        data, args.context_num, split_idx['train'], use_kmedoids=True, random_state=args.seed
                    )
                    data.context_sample = new_context_sample.to(data.context_sample.device)

        # 4) Process test data
        for i, (data, split_idx) in enumerate(zip(nc_test_data_list, nc_test_split_idx_list)):
            data.x = data.x.to(device)
            data.adj_t = data.adj_t.to(device)
            data.y = data.y.to(device)

            external_emb = nc_test_external_embeddings[i] if nc_test_external_embeddings else None

            context_shots = resolve_context_shots(data.name, 'nc', args, epoch=None)
            process_data(
                data, split_idx, args.hidden, context_shots, False, args.use_full_pca,
                args.normalize_data, False, 32, 0, args.padding_strategy,
                args.use_batchnorm, args.use_identity_projection, args.projection_small_dim, args.projection_large_dim, args.pca_device,
                args.incremental_pca_batch_size, external_emb
            )

            if getattr(args, 'use_kmedoids_sampling', False):
                new_context_sample = sample_context_with_kmedoids(
                    data, context_shots, split_idx['train'], use_kmedoids=True, random_state=args.seed
                )
                data.context_sample = new_context_sample.to(data.context_sample.device)
    else:
        print("Node classification task disabled, skipping dataset loading...")

    # (External embedding section moved earlier within NC block.)

    # === Link Prediction Data ===
    lp_train_data_list, lp_train_split_idx_list = None, None
    lp_train_context_data, lp_train_masks, lp_train_link_data_all = [], [], []
    lp_test_data_list, lp_test_split_idx_list = None, None
    lp_test_context_data, lp_test_link_data_all = [], []
    
    if args.enable_lp:
        print("Loading link prediction datasets...")
        lp_train_datasets = args.lp_train_dataset.split(',')
        lp_test_datasets = args.lp_test_dataset.split(',')
        
        # Initialize link prediction tracker
        if lp_tracker is None:
            lp_tracker = LinkPredictionTracker(device=device)
            print(f"[LP_TRACKER] Initialized tracker in data loading phase")
        
        # Record memory before link prediction data loading
        lp_tracker.record_memory()
        before_lp_data = lp_tracker.get_memory_stats()
        print(f"[LP_TRACKER] Before LP Data Loading - GPU: {before_lp_data['gpu_allocated']:.2f}GB, CPU: {before_lp_data['cpu_memory']:.2f}GB")
        
        # Load training data for link prediction (skip if using pretrained model)
        if not skip_training_data:
            with lp_tracker.time_operation('data_preparation'):
                lp_train_data_list, lp_train_split_idx_list = load_all_data_link(lp_train_datasets, device='cpu')
                print(f"[MEMORY_FIX] Loaded {len(lp_train_data_list)} training datasets on CPU (was loading to GPU before!)")
        else:
            lp_train_data_list, lp_train_split_idx_list = [], []
            print("  Skipping LP training data loading (using pretrained model)")
        
        # Record memory after loading link prediction data
        lp_tracker.record_memory() 
        after_lp_data = lp_tracker.get_memory_stats()
        print(f"[LP_TRACKER] After LP Data Loading - GPU: {after_lp_data['gpu_allocated']:.2f}GB, CPU: {after_lp_data['cpu_memory']:.2f}GB")
        
        # Process link prediction training data (skip if using pretrained model)
        lp_train_context_data = []
        lp_train_masks = []
        lp_train_link_data_all = []
        
        if not skip_training_data:
            for i, (data, split_idx) in enumerate(zip(lp_train_data_list, lp_train_split_idx_list)):
                with lp_tracker.time_operation('data_preparation'):
                    # Move data.x to GPU temporarily for fast PCA computation
                    original_x_device = data.x.device
                    if data.x.device.type == 'cpu':
                        print(f"[MEMORY_FIX] Moving {data.name} features to GPU for fast PCA computation")
                        data.x = data.x.to(device)
                    
                    # Process link-specific data (PCA will run on GPU now)
                    process_link_data(data, args, rank=0)
                    
                    # Move ALL processed data back to CPU (not just data.x!)
                    data.x = data.x.to(original_x_device)
                    
                    # Clean up additional GPU attributes created by process_link_data
                    if hasattr(data, 'x_pca') and data.x_pca is not None:
                        data.x_pca = data.x_pca.to(original_x_device)
                    
                    if hasattr(data, 'context_sample') and data.context_sample is not None:
                        data.context_sample = data.context_sample.to(original_x_device)
                        print(f"[MEMORY_FIX] Moved {data.name} context_sample back to CPU")
                    
                    print(f"[MEMORY_FIX] Moved {data.name} processed features back to CPU")
                    
                    # Prepare link data and select context
                    link_data = prepare_link_data(data, split_idx)
                
                with lp_tracker.time_operation('context_selection'):
                    # Resolve context shots for this specific dataset
                    context_shots = resolve_context_shots(data.name, 'lp', args, epoch=None)
                    context_data, train_mask = select_link_context(link_data['train'], context_shots, args.context_neg_ratio,
                                                                   args.remove_context_from_train)
                
                    lp_train_context_data.append(context_data)
                    lp_train_masks.append(train_mask)
                    lp_train_link_data_all.append(link_data)
                    lp_tracker.operation_counts['datasets_processed'] += 1
        else:
            print("  Skipping LP training data processing (using pretrained model)")
        
        # Record memory after processing all link prediction training data
        lp_tracker.record_memory()
        after_lp_processing = lp_tracker.get_memory_stats()
        print(f"[LP_TRACKER] After LP Training Data Processing - GPU: {after_lp_processing['gpu_allocated']:.2f}GB, CPU: {after_lp_processing['cpu_memory']:.2f}GB")
        
        # Load test data for link prediction (keep on CPU to save GPU memory)
        with lp_tracker.time_operation('data_preparation'):
            lp_test_data_list, lp_test_split_idx_list = load_all_data_link(lp_test_datasets, device='cpu')
            print(f"[MEMORY_FIX] Loaded {len(lp_test_data_list)} test datasets on CPU (was loading to GPU before!)")
        
        # Process link prediction test data
        lp_test_context_data = []
        lp_test_link_data_all = []
        
        for i, (data, split_idx) in enumerate(zip(lp_test_data_list, lp_test_split_idx_list)):
            with lp_tracker.time_operation('data_preparation'):
                # Move data.x to GPU temporarily for fast PCA computation
                original_x_device = data.x.device
                if data.x.device.type == 'cpu':
                    print(f"[MEMORY_FIX] Moving {data.name} features to GPU for fast PCA computation")
                    data.x = data.x.to(device)
                
                # Process link-specific data (PCA will run on GPU now)
                process_link_data(data, args, rank=0)
                
                # Move ALL processed data back to CPU (not just data.x!)
                data.x = data.x.to(original_x_device)
                
                # Clean up additional GPU attributes created by process_link_data
                if hasattr(data, 'x_pca') and data.x_pca is not None:
                    data.x_pca = data.x_pca.to(original_x_device)
                    print(f"[MEMORY_FIX] Moved {data.name} x_pca back to CPU")
                
                if hasattr(data, 'context_sample') and data.context_sample is not None:
                    data.context_sample = data.context_sample.to(original_x_device)
                    print(f"[MEMORY_FIX] Moved {data.name} context_sample back to CPU")
                
                print(f"[MEMORY_FIX] Moved {data.name} processed features back to CPU")
                
                # Prepare link data and select context
                link_data = prepare_link_data(data, split_idx)
            
            with lp_tracker.time_operation('context_selection'):
                # Resolve context shots for this specific dataset
                context_shots = resolve_context_shots(data.name, 'lp', args, epoch=None)
                context_data, _ = select_link_context(link_data['train'], context_shots, args.context_neg_ratio, False)
            
            lp_test_context_data.append(context_data)
            lp_test_link_data_all.append(link_data)
            lp_tracker.operation_counts['datasets_processed'] += 1
    else:
        print("Link prediction task disabled, skipping dataset loading...")
    
    # === Graph Classification Data ===
    gc_train_data_list, gc_train_processed_data_list = [], []
    gc_test_data_list, gc_test_processed_data_list = [], []
    
    if args.enable_gc:
        print("Loading graph classification datasets...")
        
        if gc_tracker:
            gc_tracker.log_memory("gc_data_loading_start")
        
        # Setup graph dataset environment BEFORE loading datasets
        setup_graph_dataset_environment(args)
        
        if gc_tracker:
            gc_tracker.log_memory("gc_environment_setup_complete")
        
        gc_train_datasets = args.gc_train_dataset.split(',')
        gc_test_datasets = args.gc_test_dataset.split(',')
        
        # Load training data for graph classification (skip if using pretrained model)
        if not skip_training_data:
            if gc_tracker:
                gc_tracker.log_memory("gc_train_data_loading_start")

            # Use cache-aware loading ONLY for FUG embeddings (not for original features)
            use_cache_aware = (hasattr(args, 'use_ogb_fug') and args.use_ogb_fug) and \
                              not (hasattr(args, 'use_original_features') and args.use_original_features)

            if use_cache_aware:
                from src.data_graph_cache_aware import load_all_graph_datasets_cache_aware
                try:
                    gc_train_data_list, gc_train_processed_data_list = load_all_graph_datasets_cache_aware(
                        gc_train_datasets, device, pretraining_mode=True, context_k=args.context_graph_num,
                        hidden_dim=args.hidden, pca_cache_dir=args.pca_cache_dir
                    )
                    print(f"✅ Used cache-aware loading for training datasets")
                except ImportError:
                    print(f"⚠️  Cache-aware module not available, using standard loading")
                    gc_train_data_list, gc_train_processed_data_list = load_all_graph_datasets(
                        gc_train_datasets, device, pretraining_mode=True, context_k=args.context_graph_num
                    )
            else:
                # Standard loading for original features (no PCA cache needed)
                print(f"Using standard loading (original features, no PCA cache)")
                gc_train_data_list, gc_train_processed_data_list = load_all_graph_datasets(
                    gc_train_datasets, device, pretraining_mode=True, context_k=args.context_graph_num
                )
            
            if gc_tracker:
                gc_tracker.log_memory("gc_train_data_loaded_raw")
            
            # Process graph classification training data
            if len(gc_train_data_list) > 0:
                if gc_tracker:
                    gc_tracker.log_memory("gc_train_data_processing_start")
                    
                gc_train_data_list, gc_train_processed_data_list, _ = process_datasets_for_models(
                    gc_train_data_list, gc_train_processed_data_list, args, device
                )
                
                if gc_tracker:
                    gc_tracker.log_memory("gc_train_data_processed")
                
                # Precompute task-filtered splits once for efficiency
                print("Precomputing task-filtered splits for training datasets...")
                for dataset_info in gc_train_processed_data_list:
                    task_filtered_splits = create_task_filtered_datasets(
                        dataset_info['dataset'], 
                        dataset_info['split_idx']
                    )
                    dataset_info['task_filtered_splits'] = task_filtered_splits
        else:
            gc_train_data_list, gc_train_processed_data_list = [], []
            print("  Skipping GC training data loading (using pretrained model)")
        
        # Load test data for graph classification
        if gc_tracker:
            gc_tracker.log_memory("gc_test_data_loading_start")

        # Handle per-dataset context resolution for GC test datasets
        gc_test_data_list = []
        gc_test_processed_data_list = []

        # Use cache-aware loading ONLY for FUG embeddings (not for original features)
        use_cache_aware = (hasattr(args, 'use_ogb_fug') and args.use_ogb_fug) and \
                          not (hasattr(args, 'use_original_features') and args.use_original_features)

        if use_cache_aware:
            # Use cache-aware loading for test datasets
            from src.data_graph_cache_aware import load_all_graph_datasets_cache_aware, check_pca_cache_availability

            # Check cache status for all test datasets
            cache_status = check_pca_cache_availability(gc_test_datasets, args.hidden, args.pca_cache_dir)
            cache_hits = sum(cache_status.values())
            total_datasets = len(gc_test_datasets)
            print(f"📊 Test Dataset Cache Status: {cache_hits}/{total_datasets} datasets have PCA cache")

            # Estimate potential memory savings
            if cache_hits > 0:
                estimated_savings = cache_hits * 48  # Rough estimate: 48GB per dataset
                print(f"🎉 Estimated memory savings: ~{estimated_savings}GB!")
        else:
            print(f"Using standard loading for test datasets (original features, no PCA cache)")

        for dataset_name in gc_test_datasets:
            dataset_name = dataset_name.strip()
            if gc_tracker:
                gc_tracker.log_memory(f"gc_test_dataset_{dataset_name}_loading_start")

            # Resolve context shots for this specific dataset
            context_shots = resolve_context_shots(dataset_name, 'gc', args, epoch=None)

            # Use cache-aware loading only for FUG, standard loading for original features
            if use_cache_aware:
                try:
                    single_data_list, single_processed_list = load_all_graph_datasets_cache_aware(
                        [dataset_name], device, context_k=context_shots,
                        hidden_dim=args.hidden, pca_cache_dir=args.pca_cache_dir
                    )
                    cache_used = cache_status.get(dataset_name, False)
                    memory_status = " (🚀 Cache used - 48GB saved!)" if cache_used else " (Full loading)"
                    print(f"  {dataset_name}: Loaded{memory_status}")
                except ImportError:
                    print(f"⚠️  Cache-aware module not available for {dataset_name}, using standard loading")
                    single_data_list, single_processed_list = load_all_graph_datasets(
                        [dataset_name], device, context_k=context_shots
                    )
            else:
                # Standard loading for original features (no cache)
                single_data_list, single_processed_list = load_all_graph_datasets(
                    [dataset_name], device, context_k=context_shots
                )
                print(f"  {dataset_name}: Loaded (original features, no cache)")

            gc_test_data_list.extend(single_data_list)
            gc_test_processed_data_list.extend(single_processed_list)
            
            if gc_tracker:
                gc_tracker.log_memory(f"gc_test_dataset_{dataset_name}_loaded")
        
        if gc_tracker:
            gc_tracker.log_memory("gc_test_data_all_datasets_loaded")
        
        # Process graph classification test data
        if len(gc_test_data_list) > 0:
            if gc_tracker:
                gc_tracker.log_memory("gc_test_data_processing_start")
                
            gc_test_data_list, gc_test_processed_data_list, _ = process_datasets_for_models(
                gc_test_data_list, gc_test_processed_data_list, args, device, test_datasets=True
            )
            
            if gc_tracker:
                gc_tracker.log_memory("gc_test_data_processed")
            
            # Precompute task-filtered splits once for efficiency
            print("Precomputing task-filtered splits for test datasets...")
            if gc_tracker:
                gc_tracker.log_memory("gc_test_splits_filtering_start")
                
            for dataset_info in gc_test_processed_data_list:
                # For test datasets, only filter the test split (no need to filter train/val)
                task_filtered_splits_test = create_task_filtered_datasets(
                    dataset_info['dataset'], 
                    dataset_info['split_idx'],
                    "test"
                )
                # For test datasets, both references point to the same test-only filtered data
                dataset_info['task_filtered_splits'] = task_filtered_splits_test
                dataset_info['task_filtered_splits_test_only'] = task_filtered_splits_test
    else:
        print("Graph classification task disabled, skipping dataset loading...")

    # Create mini-batch loaders for NC training datasets
    nc_train_loaders = []
    if args.enable_nc and nc_train_data_list is not None:
        print("\n=== Creating Mini-Batch Loaders for NC Training ===")
        for i, (data, split_idx) in enumerate(zip(nc_train_data_list, nc_train_split_idx_list)):
            dataset_name = getattr(data, 'name', f'dataset_{i}')
            print(f"  [{i+1}/{len(nc_train_data_list)}] {dataset_name}:")
            loader = MiniBatchNCLoader(data, split_idx, args, device)
            nc_train_loaders.append(loader)

    data_dict = {
        'nc_train': (nc_train_data_list, nc_train_split_idx_list, nc_train_external_embeddings),
        'nc_test': (nc_test_data_list, nc_test_split_idx_list, nc_test_external_embeddings),
        'nc_train_loaders': nc_train_loaders,  # Add loaders to data_dict
        'lp_train': (lp_train_data_list, lp_train_split_idx_list, lp_train_context_data, lp_train_masks, lp_train_link_data_all),
        'lp_test': (lp_test_data_list, lp_test_split_idx_list, lp_test_context_data, lp_test_link_data_all),
        'gc_train': (gc_train_data_list, gc_train_processed_data_list),
        'gc_test': (gc_test_data_list, gc_test_processed_data_list)
    }

    if gc_tracker:
        gc_tracker.log_memory("all_data_loading_complete")

    return data_dict


def get_hierarchical_task_schedule(epoch, args):
    """
    Get which tasks should be active for the current epoch based on hierarchical training schedule.

    Args:
        epoch: Current training epoch (0-indexed)
        args: Arguments containing hierarchical_phases

    Returns:
        dict: {'nc': bool, 'lp': bool, 'gc': bool} indicating which tasks are active

    Example:
        hierarchical_phases = "lp,nc+lp,nc+lp+gc"
        epoch 0-14:   {'nc': False, 'lp': True, 'gc': False}
        epoch 15-29:  {'nc': True, 'lp': True, 'gc': False}
        epoch 30+:    {'nc': True, 'lp': True, 'gc': True}
    """
    if not args.use_hierarchical_training:
        # All tasks active if hierarchical training is disabled
        return {'nc': True, 'lp': True, 'gc': True}

    # Parse phase configuration
    phases = args.hierarchical_phases.split(',')

    # Fixed phase boundaries: epochs 15 and 30
    phase_boundaries = [15, 30]

    # Determine current phase
    current_phase = 0
    for boundary in phase_boundaries:
        if epoch >= boundary:
            current_phase += 1
        else:
            break

    # Handle edge case: more epochs than phases
    if current_phase >= len(phases):
        current_phase = len(phases) - 1

    # Parse active tasks for current phase
    phase_tasks = phases[current_phase].strip()
    active_tasks = {
        'nc': 'nc' in phase_tasks,
        'lp': 'lp' in phase_tasks,
        'gc': 'gc' in phase_tasks
    }

    return active_tasks


def joint_training_step(model, predictor, nc_data, lp_data, gc_data, optimizer, args, epoch,
                       identity_projection=None, nc_loaders=None):
    """
    Perform one joint training step combining all three tasks.

    This function calculates the loss for each task, combines them with weights,
    and performs a single backward pass and optimizer step.

    Supports hierarchical/phased training to reduce task conflict.

    Args:
        nc_loaders: List of MiniBatchNCLoader instances for NC datasets
    """
    global lp_tracker

    model.train()
    predictor.train()

    device = optimizer.param_groups[0]['params'][0].device
    total_nc_loss = torch.tensor(0.0, device=device)
    total_lp_loss = torch.tensor(0.0, device=device)
    total_gc_loss = torch.tensor(0.0, device=device)
    nc_count = 0
    lp_count = 0
    gc_count = 0

    # Unpack data
    nc_data_list, nc_split_idx_list, nc_external_embeddings = nc_data
    (lp_data_list, lp_split_idx_list, lp_context_data, lp_masks, lp_link_data_all) = lp_data
    gc_data_list, gc_processed_data_list = gc_data

    # Get hierarchical task schedule for this epoch
    active_tasks = get_hierarchical_task_schedule(epoch, args)

    # Log active tasks (only on rank 0 and first iteration to avoid spam)
    if epoch == 0 or (args.use_hierarchical_training and (epoch == 15 or epoch == 30)):
        active_task_names = [name.upper() for name, active in active_tasks.items() if active]
        print(f"\n{'='*60}")
        print(f"[HIERARCHICAL TRAINING] Epoch {epoch}: Active tasks = {', '.join(active_task_names)}")
        print(f"{'='*60}\n")

    # --- 1. Calculate Losses without Optimization ---

    # Node Classification Loss
    if active_tasks['nc'] and hasattr(args, 'enable_nc') and args.enable_nc and nc_data_list is not None and len(nc_data_list) > 0 and args.lambda_nc > 0:
        # Use mini-batch loaders if provided, otherwise fallback to train_all
        if nc_loaders is not None and len(nc_loaders) > 0:
            nc_loss_sum = 0.0
            for i, (data_loader, split_idx) in enumerate(zip(nc_loaders, nc_split_idx_list)):
                external_emb = nc_external_embeddings[i] if nc_external_embeddings else None
                nc_loss = compute_nc_loss_with_loader(
                    data_loader, split_idx, model, predictor, args, device,
                    identity_projection=identity_projection,
                    external_embeddings=external_emb,
                    optimizer=optimizer
                )
                nc_loss_sum += nc_loss  # nc_loss is already a scalar
            total_nc_loss = torch.tensor(nc_loss_sum, device=device)
            nc_count = len(nc_loaders)
        else:
            # Fallback to original full-batch training
            nc_loss = train_all(model, nc_data_list, nc_split_idx_list, optimizer=optimizer, pred=predictor,
                              batch_size=args.nc_batch_size, degree=False,
                              orthogonal_push=args.orthogonal_push, normalize_class_h=args.normalize_class_h,
                              clip_grad=args.clip_grad, rank=0, epoch=epoch,
                              identity_projection=identity_projection, lambda_=args.lambda_nc, args=args)
            if nc_loss is not None:
                total_nc_loss = nc_loss
                nc_count = len(nc_data_list)
    
    # Link Prediction Loss
    if active_tasks['lp'] and hasattr(args, 'enable_lp') and args.enable_lp and lp_data_list is not None and len(lp_data_list) > 0 and args.lambda_lp > 0:
        if lp_tracker is None:
            lp_tracker = LinkPredictionTracker(device=device)
        
        lp_loss_sum = 0.0
        for i, (data, split_idx) in enumerate(zip(lp_data_list, lp_split_idx_list)):
            link_data_all = lp_link_data_all[i]
            context_data = lp_context_data[i]
            train_mask = lp_masks[i]
            
            if 'train' in link_data_all and link_data_all['train']['edge_pairs'].size(0) > 0:
                with lp_tracker.time_operation('training'):
                    # Move only required data to GPU (data is on CPU by default now)
                    data.x = data.x.to(device)
                    data.adj_t = data.adj_t.to(device)
                    if hasattr(data, 'full_adj_t') and data.full_adj_t is not None:
                        data.full_adj_t = data.full_adj_t.to(device)
                    
                    # Record memory before forward pass
                    lp_tracker.record_memory()
                    
                    with lp_tracker.time_operation('forward_pass'):
                        lp_loss = train_link_prediction(
                            model, predictor, data, link_data_all['train'], context_data, train_mask,
                            optimizer=optimizer, batch_size=args.lp_batch_size, 
                            identity_projection=identity_projection, 
                            clip_grad=args.clip_grad, rank=0, orthogonal_push=args.orthogonal_push, 
                            normalize_class_h=args.normalize_class_h, epoch=epoch, 
                            mask_target_edges=args.mask_target_edges, degree=False, lambda_=args.lambda_lp,
                            args=args
                        )
                    
                    # Move all data back to CPU to free GPU memory
                    data.x = data.x.cpu()
                    data.adj_t = data.adj_t.cpu() 
                    if hasattr(data, 'full_adj_t') and data.full_adj_t is not None:
                        data.full_adj_t = data.full_adj_t.cpu()
                    
                    # Also move processed data attributes back to CPU
                    if hasattr(data, 'x_pca') and data.x_pca is not None:
                        data.x_pca = data.x_pca.cpu()
                    if hasattr(data, 'context_sample') and data.context_sample is not None:
                        data.context_sample = data.context_sample.cpu()
                    
                    # Record memory after cleanup
                    lp_tracker.record_memory()
                    
                    if lp_loss is not None:
                        lp_loss_sum += lp_loss
                        lp_count += 1
                        lp_tracker.operation_counts['training_steps'] += 1
        
        if lp_count > 0:
            total_lp_loss = lp_loss_sum / lp_count
    
    # Graph Classification Loss
    if active_tasks['gc'] and hasattr(args, 'enable_gc') and args.enable_gc and len(gc_data_list) > 0 and args.lambda_gc > 0:
        gc_tracker.log_memory("gc_section_start")
        gc_loss_sum = 0.0
        gc_dataset_count = 0
        
        for dataset_idx, dataset_info in enumerate(gc_processed_data_list):
            gc_tracker.log_memory(f"gc_dataset_{dataset_idx}_start")
            
            # Use precomputed task-filtered splits
            task_filtered_splits = dataset_info['task_filtered_splits']
            
            dataset_loss = 0.0
            dataset_tasks = 0
            
            gc_tracker.log_memory(f"gc_dataset_{dataset_idx}_splits_loaded")
            
            # Train on each task separately using prefiltered data
            for task_idx, task_splits in task_filtered_splits.items():
                gc_tracker.log_memory(f"gc_dataset_{dataset_idx}_task_{task_idx}_start")

                # Check if any embedding mapping is present to use index tracking (FUG, TSGFM, TAGDataset)
                use_index_tracking = ('fug_mapping' in dataset_info or
                                    'tsgfm_mapping' in dataset_info or
                                    'tag_mapping' in dataset_info)

                # Create task-specific data loaders
                task_data_loaders = create_data_loaders(
                    dataset_info['dataset'], 
                    task_splits,
                    batch_size=args.gc_batch_size,
                    shuffle=True,
                    task_idx=task_idx,
                    use_index_tracking=use_index_tracking
                )
                
                # Track memory before graph classification training
                gc_tracker.log_memory(f"gc_task_{task_idx}_training_start")
                
                # Train on this specific task
                task_loss = train_graph_classification_single_task(
                    model, predictor, dataset_info, task_data_loaders, optimizer, task_idx,
                    pooling_method=args.graph_pooling, device=device,
                    clip_grad=args.clip_grad, orthogonal_push=args.orthogonal_push,
                    normalize_class_h=args.normalize_class_h, identity_projection=identity_projection,
                    lambda_=args.lambda_gc
                )
                
                # Track memory after graph classification training
                gc_tracker.log_memory(f"gc_task_{task_idx}_training_complete")
                
                dataset_loss += task_loss
                dataset_tasks += 1
            
            # Track memory after processing all tasks for this dataset
            gc_tracker.log_memory(f"gc_dataset_{dataset_idx}_all_tasks_complete")
            
            # Average loss across tasks for this dataset
            if dataset_tasks > 0:
                avg_dataset_loss = dataset_loss / dataset_tasks
                gc_loss_sum += avg_dataset_loss
                gc_dataset_count += 1
                
            gc_tracker.log_memory(f"gc_dataset_{dataset_idx}_complete")
        
        if gc_dataset_count > 0:
            total_gc_loss = gc_loss_sum / gc_dataset_count  # Already scaled in train_graph_classification_single_task
            gc_count = gc_dataset_count
            
        gc_tracker.log_memory("gc_section_complete")
    
    # Combined loss
    combined_loss = total_nc_loss + total_lp_loss + total_gc_loss
    
    return {
        'nc_loss': total_nc_loss,
        'lp_loss': total_lp_loss,
        'gc_loss': total_gc_loss,
        'combined_loss': combined_loss,
        'nc_count': nc_count,
        'lp_count': lp_count,
        'gc_count': gc_count
    }


def evaluate_node_classification(model, predictor, nc_data, args, split='valid', identity_projection=None, nc_loaders=None):
    """
    Evaluate node classification task only.

    Returns:
        Dictionary with node classification metrics
    """
    import time

    eval_start_time = time.time()
    print(f"Starting node classification evaluation ({split} split)...")

    model.eval()
    predictor.eval()

    results = {}

    with torch.no_grad():
        nc_data_list, nc_split_idx_list, nc_external_embeddings = nc_data
        if len(nc_data_list) > 0:
            if split == 'test':
                print(f"  Processing {len(nc_data_list)} unseen test datasets...")
                datasets_start_time = time.time()

                # Use inductive evaluation for unseen datasets
                train_metrics, valid_metrics, test_metrics = test_all_induct(
                    model, predictor, nc_data_list, nc_split_idx_list, args.test_batch_size,
                    False, None, None, True, None, 0, identity_projection
                )

                datasets_time = time.time() - datasets_start_time
                print(f"  All {len(nc_data_list)} datasets completed in {datasets_time:.2f}s")

                results = {
                    'train': sum(train_metrics) / len(train_metrics) if train_metrics else 0.0,
                    'valid': sum(valid_metrics) / len(valid_metrics) if valid_metrics else 0.0,
                    'test': sum(test_metrics) / len(test_metrics) if test_metrics else 0.0,
                    'individual_test_metrics': test_metrics if test_metrics else [],
                    'evaluation_time': datasets_time
                }
            else:
                # Check if we should use mini-batch evaluation
                if nc_loaders is not None and len(nc_loaders) > 0:
                    # Use mini-batch evaluation
                    from src.data_minibatch import evaluate_with_loader
                    device = next(model.parameters()).device
                    eval_accs = []

                    for i, (loader, split_idx) in enumerate(zip(nc_loaders, nc_split_idx_list)):
                        if loader.is_minibatch():
                            # Mini-batch evaluation
                            eval_acc = evaluate_with_loader(
                                loader, split_idx, model, predictor, args, device,
                                eval_split=split, identity_projection=identity_projection
                            )
                            eval_accs.append(eval_acc)
                        else:
                            # Fall back to full-batch for small datasets
                            from src.engine_nc import test
                            _, eval_acc, _ = test(
                                model, predictor, loader.data, split_idx['train'], split_idx['valid'], split_idx['test'],
                                args.test_batch_size, False, None, None, True, None, 0, identity_projection
                            )
                            eval_accs.append(eval_acc)

                    results = {
                        split: sum(eval_accs) / len(eval_accs) if eval_accs else 0.0,
                        'individual_test_metrics': eval_accs
                    }
                else:
                    # Use transductive evaluation for seen datasets (original approach)
                    train_metrics, valid_metrics, test_metrics = test_all(
                        model, predictor, nc_data_list, nc_split_idx_list, args.test_batch_size,
                        False, None, None, True, None, 0, identity_projection
                    )
                    results = {
                        'train': train_metrics if isinstance(train_metrics, (int, float)) else sum(train_metrics) / len(train_metrics),
                        'valid': valid_metrics if isinstance(valid_metrics, (int, float)) else sum(valid_metrics) / len(valid_metrics),
                        'test': test_metrics if isinstance(test_metrics, (int, float)) else sum(test_metrics) / len(test_metrics),
                        'individual_test_metrics': test_metrics if isinstance(test_metrics, list) else [test_metrics]
                    }

    eval_total_time = time.time() - eval_start_time
    print(f"Node classification evaluation ({split} split) completed in {eval_total_time:.2f}s")

    return results


def evaluate_link_prediction_task(model, predictor, lp_data, args, split='valid', identity_projection=None):
    """
    Evaluate link prediction task only.
    
    Returns:
        Dictionary with link prediction metrics
    """
    global lp_tracker
    
    model.eval()
    predictor.eval()
    
    results = {}
    
    # Initialize tracker if needed
    if lp_tracker is None:
        device = next(model.parameters()).device
        lp_tracker = LinkPredictionTracker(device=device)
    
    with torch.no_grad():
        if split == 'test':
            lp_data_list, lp_split_idx_list, lp_context_data, lp_link_data_all = lp_data
        else:
            lp_data_list, lp_split_idx_list, lp_context_data, _, lp_link_data_all = lp_data
            
        if len(lp_data_list) > 0:
            lp_results_list = []
            
            for i, (data, split_idx) in enumerate(zip(lp_data_list, lp_split_idx_list)):
                link_data_all = lp_link_data_all[i]
                context_data = lp_context_data[i]
                
                split_key = split if split in link_data_all else 'test'
                if split_key in link_data_all and link_data_all[split_key]['edge_pairs'].size(0) > 0:
                    
                    with lp_tracker.time_operation('evaluation'):
                        # Move only required data to GPU (data is on CPU by default now)
                        data.x = data.x.to(next(model.parameters()).device)
                        data.adj_t = data.adj_t.to(next(model.parameters()).device)
                        if hasattr(data, 'full_adj_t') and data.full_adj_t is not None:
                            data.full_adj_t = data.full_adj_t.to(next(model.parameters()).device)
                        
                        # Record memory before evaluation
                        lp_tracker.record_memory()
                        
                        lp_results = evaluate_link_prediction(
                            model, predictor, data, link_data_all[split_key], context_data,
                            args.test_batch_size, None, None, None, identity_projection,
                            0, True, degree=False, k_values=[20, 50, 100],
                            use_full_adj_for_test=True, lp_metric=args.lp_metric
                        )
                        
                        # Move all data back to CPU to free GPU memory
                        data.x = data.x.cpu()
                        data.adj_t = data.adj_t.cpu()
                        if hasattr(data, 'full_adj_t') and data.full_adj_t is not None:
                            data.full_adj_t = data.full_adj_t.cpu()
                        
                        # Also move processed data attributes back to CPU
                        if hasattr(data, 'x_pca') and data.x_pca is not None:
                            data.x_pca = data.x_pca.cpu()
                        if hasattr(data, 'context_sample') and data.context_sample is not None:
                            data.context_sample = data.context_sample.cpu()

                        # Record memory after cleanup
                        lp_tracker.record_memory()
                        
                        lp_tracker.operation_counts['evaluation_steps'] += 1
                        
                    lp_results_list.append(lp_results.get('default_metric', 0.0))
                else:
                    lp_results_list.append(0.0)
            
            if lp_results_list:
                avg_result = sum(lp_results_list) / len(lp_results_list)
                results = {
                    'train': avg_result,  # For consistency with NC format
                    'valid': avg_result,
                    'test': avg_result,
                    'individual_test_metrics': lp_results_list
                }
    
    return results


def evaluate_graph_classification_task(model, predictor, gc_data, args, split='valid', identity_projection=None, gc_tracker=None):
    """
    Evaluate graph classification task only.
    
    Returns:
        Dictionary with graph classification metrics
    """
    import time
    
    eval_start_time = time.time()
    print(f"Starting graph classification evaluation ({split} split)...")
    
    model.eval()
    predictor.eval()
    
    if gc_tracker:
        gc_tracker.log_memory(f"eval_{split}_start")
    
    results = {}
    
    with torch.no_grad():
        gc_data_list, gc_processed_data_list = gc_data
        
        if gc_tracker:
            gc_tracker.log_memory(f"eval_{split}_data_loaded")
        
        if len(gc_data_list) > 0:
            all_dataset_results = []
            individual_results = []
            
            datasets_start_time = time.time()
            print(f"  Processing {len(gc_processed_data_list)} datasets...")
            
            for dataset_idx, dataset_info in enumerate(gc_processed_data_list):
                dataset_start_time = time.time()
                if gc_tracker:
                    gc_tracker.log_memory(f"eval_{split}_dataset_{dataset_idx}_start")
                    
                dataset_name = dataset_info['dataset'].name if hasattr(dataset_info['dataset'], 'name') else f'gc_dataset_{dataset_idx}'
                
                # Use precomputed task-filtered datasets
                if split == 'test':
                    task_filtered_splits = dataset_info['task_filtered_splits_test_only']
                else:
                    task_filtered_splits = dataset_info['task_filtered_splits']
                
                # Evaluate each task separately and aggregate results
                task_results = []
                
                tasks_start_time = time.time()
                print(f"    Dataset {dataset_idx} ({dataset_name}): Processing {len(task_filtered_splits)} tasks...")
                
                for task_idx, task_splits in task_filtered_splits.items():
                    task_start_time = time.time()
                    
                    # Check if any embedding mapping is present to use index tracking (FUG, TSGFM, TAGDataset)
                    setup_start_time = time.time()
                    use_index_tracking = ('fug_mapping' in dataset_info or 
                                        'tsgfm_mapping' in dataset_info or 
                                        'tag_mapping' in dataset_info)
                    
                    # Create task-specific data loaders for evaluation
                    dataloader_start_time = time.time()
                    if split == 'test':
                        # Only use test split for unseen datasets
                        test_only_splits = {'test': task_splits['test']}
                        task_eval_loaders = create_data_loaders(
                            dataset_info['dataset'], 
                            test_only_splits,
                            batch_size=args.gc_test_batch_size,
                            shuffle=False,
                            task_idx=task_idx,
                            use_index_tracking=use_index_tracking
                        )
                    else:
                        # Use all splits for seen datasets
                        task_eval_loaders = create_data_loaders(
                            dataset_info['dataset'], 
                            task_splits,
                            batch_size=args.gc_test_batch_size,
                            shuffle=False,
                            task_idx=task_idx,
                            use_index_tracking=use_index_tracking
                        )
                    
                    dataloader_time = time.time() - dataloader_start_time
                    
                    # Debug: Check profiling setting
                    profiling_enabled = getattr(args, 'enable_profiling', False)
                    
                    if gc_tracker:
                        gc_tracker.log_memory(f"eval_{split}_dataset_{dataset_idx}_task_{task_idx}_before_eval")
                    
                    # Per-task profiling if enabled
                    eval_start_time = time.time()
                    if profiling_enabled:
                        from torch.profiler import profile, record_function, ProfilerActivity
                        prof = profile(
                            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                            record_shapes=True,
                            profile_memory=True,
                            with_stack=True
                        )
                        prof.start()
                        with record_function(f"gc_eval_{dataset_name}_task_{task_idx}"):
                            # Evaluate this specific task
                            task_eval_results = evaluate_graph_classification_single_task(
                                model, predictor, dataset_info, task_eval_loaders, task_idx,
                                pooling_method=args.graph_pooling, device=model.parameters().__next__().device,
                                normalize_class_h=args.normalize_class_h, dataset_name=dataset_name, identity_projection=identity_projection
                            )
                        prof.stop()
                        eval_time = time.time() - eval_start_time
                        
                        if gc_tracker:
                            gc_tracker.log_memory(f"eval_{split}_dataset_{dataset_idx}_task_{task_idx}_after_profiled_eval")
                        
                        # Save per-task profiling results
                        profile_filename = f"gc_eval_{dataset_name}_task_{task_idx}.json"
                        prof.export_chrome_trace(profile_filename)
                        print(f"      [PROFILING] Task {task_idx}: evaluation: {eval_time:.3f}s, saved profiling to {profile_filename}")
                        
                        # Print top CPU functions for this task
                        print(f"      [PROFILING] Task {task_idx} Top CPU ops:")
                        cpu_table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=5)
                        for line in cpu_table.split('\n')[:7]:  # Header + top 5 rows
                            if line.strip():
                                print(f"        {line}")
                    else:
                        # Evaluate this specific task
                        task_eval_results = evaluate_graph_classification_single_task(
                            model, predictor, dataset_info, task_eval_loaders, task_idx,
                            pooling_method=args.graph_pooling, device=model.parameters().__next__().device,
                            normalize_class_h=args.normalize_class_h, dataset_name=dataset_name, identity_projection=identity_projection
                        )
                        eval_time = time.time() - eval_start_time
                        
                        if gc_tracker:
                            gc_tracker.log_memory(f"eval_{split}_dataset_{dataset_idx}_task_{task_idx}_after_eval")
                    
                    
                    if gc_tracker:
                        gc_tracker.log_memory(f"eval_{split}_dataset_{dataset_idx}_task_{task_idx}_complete")
                    
                    # Extract the appropriate split result
                    # Map 'valid' to 'val' for consistency with data loader keys
                    result_key = 'val' if split == 'valid' else split
                    split_result = task_eval_results.get(result_key, 0.0)
                    task_results.append(split_result)
                    
                    task_time = time.time() - task_start_time
                    overhead_time = task_time - eval_time - dataloader_time
                    
                    # Handle both numeric and dict results for display
                    if isinstance(split_result, dict):
                        # Extract primary metric for display
                        display_metric = split_result.get('ap', split_result.get('auc', 0.0))
                        print(f"      Task {task_idx}: {display_metric:.4f} (eval: {eval_time:.2f}s)")
                    else:
                        print(f"      Task {task_idx}: {split_result:.4f} (eval: {eval_time:.2f}s)")
                
                # Aggregate results across tasks for this dataset
                # Task accumulation completed for this dataset
                if gc_tracker:
                    gc_tracker.log_memory(f"eval_{split}_dataset_{dataset_idx}_tasks_complete")
                
                if task_results:
                    dataset_avg = aggregate_task_metrics(task_results)
                    # For averaging across datasets, extract primary metric if multiple metrics
                    if isinstance(dataset_avg, dict):
                        primary_metric = dataset_avg.get('ap', dataset_avg.get('auc', 0.0))
                        all_dataset_results.append(primary_metric)
                        individual_results.append(primary_metric)  # Store primary metric for averaging
                    else:
                        all_dataset_results.append(dataset_avg)
                        individual_results.append(dataset_avg)
                else:
                    all_dataset_results.append(0.0)
                    individual_results.append(0.0)
                    
                if gc_tracker:
                    gc_tracker.log_memory(f"eval_{split}_dataset_{dataset_idx}_complete")
                
                dataset_time = time.time() - dataset_start_time
                tasks_time = time.time() - tasks_start_time
                print(f"    Dataset {dataset_idx} ({dataset_name}): completed in {dataset_time:.2f}s (tasks: {tasks_time:.2f}s)")
            
            # Calculate overall average
            if all_dataset_results:
                avg_result = sum(all_dataset_results) / len(all_dataset_results)
                results = {
                    'train': avg_result,  # For consistency with other tasks
                    'valid': avg_result,
                    'test': avg_result,
                    'individual_test_metrics': individual_results
                }
            else:
                results = {
                    'train': 0.0,
                    'valid': 0.0,
                    'test': 0.0,
                    'individual_test_metrics': []
                }
            
            datasets_total_time = time.time() - datasets_start_time
            print(f"  All {len(gc_processed_data_list)} datasets completed in {datasets_total_time:.2f}s")
                
        if gc_tracker:
            gc_tracker.log_memory(f"eval_{split}_complete")
    
    eval_total_time = time.time() - eval_start_time
    print(f"Graph classification evaluation ({split} split) completed in {eval_total_time:.2f}s")
    
    return results


def joint_evaluation(model, predictor, nc_data, lp_data, gc_data, args, split='valid',
                    identity_projection=None, gc_tracker=None, nc_loaders=None):
    """
    Evaluate enabled tasks and return metrics.

    Args:
        nc_loaders: Optional list of MiniBatchNCLoader for mini-batch evaluation

    Returns:
        Dictionary with metrics for enabled tasks
    """

    results = {'nc_metrics': {}, 'lp_metrics': {}, 'gc_metrics': {}}

    # Evaluate node classification
    if hasattr(args, 'enable_nc') and args.enable_nc and nc_data is not None and nc_data[0] is not None:
            nc_results = evaluate_node_classification(model, predictor, nc_data, args, split, identity_projection, nc_loaders)
            results['nc_metrics'] = nc_results
    
    # Evaluate link prediction  
    if hasattr(args, 'enable_lp') and args.enable_lp and lp_data is not None and lp_data[0] is not None:
            lp_results = evaluate_link_prediction_task(model, predictor, lp_data, args, split, identity_projection)
            results['lp_metrics'] = lp_results
    
    # Evaluate graph classification
    if hasattr(args, 'enable_gc') and args.enable_gc and gc_data is not None and len(gc_data[0]) > 0:
            gc_results = evaluate_graph_classification_task(model, predictor, gc_data, args, split, identity_projection, gc_tracker)
            results['gc_metrics'] = gc_results
    
    return results


def run_joint_training(args, device='cuda:0'):
    """
    Main joint training function.

    Note: Random seeds are already set by the calling function (main()) before this function is called.
    This ensures deterministic behavior for both training and inference.
    """
    # Declare global trackers at the very beginning
    global lp_tracker, gc_tracker

    print(f"\n=== Starting Joint Training ===")
    print(f"Device: {device}")
    print(f"Enabled Tasks:")
    print(f"  Node Classification: {'✓' if getattr(args, 'enable_nc', True) else '✗'} (lambda: {args.lambda_nc})")
    print(f"  Link Prediction: {'✓' if getattr(args, 'enable_lp', True) else '✗'} (lambda: {args.lambda_lp})")
    print(f"  Graph Classification: {'✓' if getattr(args, 'enable_gc', True) else '✗'} (lambda: {args.lambda_gc})")
    
    # Initialize link prediction tracker early if link prediction is enabled
    if getattr(args, 'enable_lp', True) and lp_tracker is None:
        lp_tracker = LinkPredictionTracker(device=device)
        print(f"✓ Link Prediction Tracker initialized on {device}")
        # Record initial memory state
        lp_tracker.record_memory()
        initial_stats = lp_tracker.get_memory_stats()
        print(f"Initial Memory - GPU: {initial_stats['gpu_allocated']:.2f}GB, CPU: {initial_stats['cpu_memory']:.2f}GB")
    
    # Initialize graph classification tracker early if graph classification is enabled
    if getattr(args, 'enable_gc', True) and gc_tracker is None:
        gc_tracker = GraphClassificationTracker(device=device)
        print(f"✓ Graph Classification Tracker initialized on {device}")
        # Record initial memory state
        gc_tracker.record_memory()
        initial_stats = gc_tracker.get_memory_stats()
        print(f"[GC-MEMORY] Initial Memory - GPU: {initial_stats['gpu_allocated']:.2f}GB, "
              f"CPU: {initial_stats['cpu_memory']:.2f}GB, System: {initial_stats['system_memory_percent']:.1f}% used")
        gc_tracker.log_memory_spike_warning(threshold_gb=4.0)  # Lower threshold for initial warning
    
    # Initialize logging
    logger = TrainingLogger(
        rank=0,  # Single GPU training, rank is always 0
        world_size=1,
        log_level=LogLevel.INFO,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        analysis_interval=10000
    )
    
    # Initialize wandb
    wandb.init(project='inductnode-joint', config=args)
    
    # Check for checkpoint loading and override args if needed
    checkpoint = None
    if args.use_pretrained_model and args.load_checkpoint is not None:
        print(f"Loading checkpoint from: {args.load_checkpoint}")
        print("Extracting model configuration from checkpoint...")
        
        # Load checkpoint configuration
        checkpoint_info, checkpoint = load_checkpoint_config(args.load_checkpoint)
        
        # Override current args with checkpoint's configuration
        if 'args' in checkpoint:
            checkpoint_args = checkpoint['args']
            args = override_args_from_checkpoint(args, checkpoint_args, rank=0)
        else:
            print("Warning: No argument configuration found in checkpoint, using current arguments")
    
    # Load and preprocess data (skip training data if using pretrained model)
    with lp_tracker.time_operation('data_preparation') if lp_tracker else nullcontext():
        with gc_tracker.time_operation('dataset_processing') if gc_tracker else nullcontext():
            data_dict = load_and_preprocess_data(args, device, skip_training_data=args.use_pretrained_model, gc_tracker=gc_tracker)
    
    if lp_tracker:
        lp_tracker.record_memory()
        after_data_stats = lp_tracker.get_memory_stats()
    
    if gc_tracker:
        gc_tracker.record_memory()
        after_data_stats = gc_tracker.get_memory_stats()
        print(f"[GC-MEMORY] After Data Loading - CPU: {after_data_stats['cpu_memory']:.2f}GB, "
              f"GPU: {after_data_stats['gpu_allocated']:.2f}GB")
        gc_tracker.log_memory_spike_warning(threshold_gb=6.0)
        print(f"After Data Loading - GPU: {after_data_stats['gpu_allocated']:.2f}GB, CPU: {after_data_stats['cpu_memory']:.2f}GB")
    
    # Create unified model
    with lp_tracker.time_operation('data_preparation') if lp_tracker else nullcontext():
        model, predictor, identity_projection = create_unified_model(
            args, args.hidden, device)
    
    if lp_tracker:
        lp_tracker.record_memory()
        after_model_stats = lp_tracker.get_memory_stats()
        print(f"After Model Creation - GPU: {after_model_stats['gpu_allocated']:.2f}GB, CPU: {after_model_stats['cpu_memory']:.2f}GB")
    
    # Setup optimizer and scheduler
    parameters = []
    for module in [model, predictor, identity_projection]:
        if module is not None:
            parameters.extend(list(module.parameters()))
    
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    
    if args.schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.schedule == 'step':
        step = max(1, args.epochs // 5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.5)
    elif args.schedule == 'warmup':
        scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                  num_warmup_steps=args.epochs // 10, 
                                                  num_training_steps=args.epochs)
    else:
        scheduler = None
    
    print(f"Total parameters: {sum(p.numel() for p in parameters):,}")
    
    # Load checkpoint states if checkpoint was provided
    if checkpoint is not None:
        print("Loading model states from checkpoint...")
        best_epoch, best_valid, final_test = load_checkpoint_states(
            checkpoint, model, predictor, optimizer, identity_projection=identity_projection,
            scheduler=scheduler, rank=0
        )
        print(f"Checkpoint loaded - Best epoch: {best_epoch}, Best valid: {best_valid:.4f}, Final test: {final_test:.4f}")
        
        # If loading from checkpoint, skip training and go directly to final evaluation
        if args.use_pretrained_model:
            print("\n=== Skipping Training - Using Pretrained Model ===")

            # Ensure models are in evaluation mode for inference
            model.eval()
            predictor.eval()
            if identity_projection is not None:
                identity_projection.eval()

            # Final evaluation on test sets
            print("\n=== Final Test Evaluation (From Checkpoint) ===")
            
            # Initialize empty results
            nc_test_results = {}
            lp_test_results = {}
            gc_test_results = {}
            
            # Node Classification Test on unseen datasets
            if getattr(args, 'enable_nc', True) and data_dict['nc_test'][0] is not None:
                nc_test_results = evaluate_node_classification(
                    model, predictor, data_dict['nc_test'], args, 'test', identity_projection
                )
            
            # Link Prediction Test on unseen datasets
            if getattr(args, 'enable_lp', True) and data_dict['lp_test'][0] is not None:
                lp_test_results = evaluate_link_prediction_task(
                    model, predictor, data_dict['lp_test'], args, 'test', identity_projection
                )
            
            # Graph Classification Test on unseen datasets
            if getattr(args, 'enable_gc', True) and len(data_dict['gc_test'][0]) > 0:
                gc_test_results = evaluate_graph_classification_task(
                    model, predictor, data_dict['gc_test'], args, 'test', identity_projection, gc_tracker
                )
            
            # Print and log final results
            nc_test_metric = nc_test_results.get('test', 0.0)
            lp_test_metric = lp_test_results.get('test', 0.0)
            gc_test_metric = gc_test_results.get('test', 0.0)
            
            # Handle case where test metrics might be dictionaries (e.g., PCBA)
            if isinstance(nc_test_metric, dict):
                nc_test_metric = nc_test_metric.get('auc', nc_test_metric.get('ap', list(nc_test_metric.values())[0] if nc_test_metric else 0.0))
            if isinstance(lp_test_metric, dict):
                lp_test_metric = lp_test_metric.get('auc', lp_test_metric.get('ap', list(lp_test_metric.values())[0] if lp_test_metric else 0.0))
            if isinstance(gc_test_metric, dict):
                gc_test_metric = gc_test_metric.get('auc', gc_test_metric.get('ap', list(gc_test_metric.values())[0] if gc_test_metric else 0.0))
            
            final_results_msg = f"Node Classification Test: {nc_test_metric:.4f}\n"
            
            # Add LP results with individual dataset breakdown
            final_results_msg += f"Link Prediction Test: {lp_test_metric:.4f}\n"
            lp_individual = lp_test_results.get('individual_test_metrics', [])
            if lp_individual and hasattr(args, 'lp_test_dataset'):
                lp_test_datasets = args.lp_test_dataset.split(',')
                for dataset_name, metric in zip(lp_test_datasets, lp_individual):
                    final_results_msg += f"  {dataset_name.strip()}: {metric:.4f}\n"
            
            final_results_msg += f"Graph Classification Test: {gc_test_metric:.4f}"
            
            print(final_results_msg)
            logger.info(final_results_msg, LogLevel.INFO)
            
            # Final wandb log
            final_wandb_log = {
                'test/nc_metric': nc_test_metric,
                'test/lp_metric': lp_test_metric,
                'test/gc_metric': gc_test_metric,
                'test/combined_score': nc_test_metric + lp_test_metric + gc_test_metric,
                'loaded_from_checkpoint': True,
                'checkpoint_path': args.load_checkpoint
            }
            wandb.log(final_wandb_log)
            
            # Return results
            nc_individual = nc_test_results.get('individual_test_metrics', [])
            lp_individual = lp_test_results.get('individual_test_metrics', [])
            gc_individual = gc_test_results.get('individual_test_metrics', [])
            
            return nc_test_metric, lp_test_metric, gc_test_metric, nc_individual, lp_individual, gc_individual
    
    # Final memory checkpoint before training starts
    if lp_tracker:
        lp_tracker.record_memory()
        pre_training_stats = lp_tracker.get_memory_stats()
        print(f"\n[LP_TRACKER] === Memory Progression Summary ===")
        print(f"Pre-training GPU Memory: {pre_training_stats['gpu_allocated']:.2f}GB")
        print(f"Peak GPU so far: {pre_training_stats['gpu_peak']:.2f}GB")
        
        # Show timing breakdown so far
        timing_summary = lp_tracker.get_summary_stats()
        print(f"Data Preparation Time: {timing_summary.get('time_data_preparation_total', 0):.2f}s")
        print(f"Context Selection Time: {timing_summary.get('time_context_selection_total', 0):.2f}s")
        print(f"================================================\n")
    
    # Training loop
    best_valid_score = 0.0
    best_epoch = 0
    best_model_state = None
    
    for epoch in range(args.epochs):
        start_time = time.time()

        # Refresh contexts if needed
        refresh_contexts_if_needed(epoch, args, data_dict)

        # Joint training step
        train_results = joint_training_step(
            model, predictor, data_dict['nc_train'], data_dict['lp_train'], data_dict['gc_train'],
            optimizer, args, epoch, identity_projection,
            nc_loaders=data_dict.get('nc_train_loaders', None)
        )

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        # Every epoch: Validation on seen datasets (training data) for early stopping
        seen_valid_results = joint_evaluation(
            model, predictor, data_dict['nc_train'], data_dict['lp_train'], data_dict['gc_train'],
            args, 'valid', identity_projection, gc_tracker,
            nc_loaders=data_dict.get('nc_train_loaders', None)
        )

        # Compute combined validation score on seen datasets
        nc_valid_seen = seen_valid_results['nc_metrics'].get('valid', 0.0) if seen_valid_results['nc_metrics'] else 0.0
        lp_valid_seen = seen_valid_results['lp_metrics'].get('valid', 0.0) if seen_valid_results['lp_metrics'] else 0.0
        gc_valid_seen = seen_valid_results['gc_metrics'].get('valid', 0.0) if seen_valid_results['gc_metrics'] else 0.0
        combined_valid_seen = nc_valid_seen + lp_valid_seen + gc_valid_seen
        
        # Save best model based on seen validation performance
        if combined_valid_seen > best_valid_score:
            best_valid_score = combined_valid_seen
            best_epoch = epoch
            best_model_state = {
                'model': copy.deepcopy(model.state_dict()),
                'predictor': copy.deepcopy(predictor.state_dict()),
                'identity_projection': copy.deepcopy(identity_projection.state_dict()) if identity_projection is not None else None,
            }
        
        # Periodic evaluation on unseen datasets (controlled by eval_interval)
        unseen_metrics = {}
        if epoch % args.eval_interval == 0:
            # Evaluate on unseen test datasets
            nc_unseen_results = {}
            lp_unseen_results = {}
            gc_unseen_results = {}
            
            if getattr(args, 'enable_nc', True) and data_dict['nc_test'][0] is not None:
                nc_unseen_results = evaluate_node_classification(
                    model, predictor, data_dict['nc_test'], args, 'test', identity_projection
                )
            
            if getattr(args, 'enable_lp', True) and data_dict['lp_test'][0] is not None:
                lp_unseen_results = evaluate_link_prediction_task(
                    model, predictor, data_dict['lp_test'], args, 'test', identity_projection
                )
            
            if getattr(args, 'enable_gc', True) and len(data_dict['gc_test'][0]) > 0:
                gc_unseen_results = evaluate_graph_classification_task(
                    model, predictor, data_dict['gc_test'], args, 'test', identity_projection, gc_tracker
                )
            
            nc_test_unseen = nc_unseen_results.get('test', 0.0)
            lp_test_unseen = lp_unseen_results.get('test', 0.0)
            gc_test_unseen = gc_unseen_results.get('test', 0.0)
            combined_test_unseen = nc_test_unseen + lp_test_unseen + gc_test_unseen
            
            # Get individual dataset metrics
            nc_individual = nc_unseen_results.get('individual_test_metrics', [])
            lp_individual = lp_unseen_results.get('individual_test_metrics', [])
            gc_individual = gc_unseen_results.get('individual_test_metrics', [])
            
            # Get dataset names for logging
            nc_test_datasets = args.nc_test_dataset.split(',') if getattr(args, 'enable_nc', True) and hasattr(args, 'nc_test_dataset') else []
            lp_test_datasets = args.lp_test_dataset.split(',') if getattr(args, 'enable_lp', True) and hasattr(args, 'lp_test_dataset') else []
            gc_test_datasets = args.gc_test_dataset.split(',') if getattr(args, 'enable_gc', True) and hasattr(args, 'gc_test_dataset') else []
            
            unseen_metrics = {
                'test_unseen/nc_metric': nc_test_unseen,
                'test_unseen/lp_metric': lp_test_unseen,
                'test_unseen/gc_metric': gc_test_unseen,
                'test_unseen/combined_score': combined_test_unseen
            }
            
            # Add individual dataset metrics to wandb
            for i, (dataset_name, metric) in enumerate(zip(nc_test_datasets, nc_individual)):
                unseen_metrics[f'test_unseen/nc_{dataset_name.strip()}'] = metric
            
            for i, (dataset_name, metric) in enumerate(zip(lp_test_datasets, lp_individual)):
                unseen_metrics[f'test_unseen/lp_{dataset_name.strip()}'] = metric
            
            for i, (dataset_name, metric) in enumerate(zip(gc_test_datasets, gc_individual)):
                unseen_metrics[f'test_unseen/gc_{dataset_name.strip()}'] = metric
            
            # Print individual dataset performance
            print(f"\n--- Unseen Dataset Performance (Epoch {epoch}) ---")
            print(f"Node Classification (Average): {nc_test_unseen:.4f}")
            for i, (dataset_name, metric) in enumerate(zip(nc_test_datasets, nc_individual)):
                print(f"  {dataset_name.strip()}: {metric:.4f}")
            
            print(f"Link Prediction (Average): {lp_test_unseen:.4f}")
            for i, (dataset_name, metric) in enumerate(zip(lp_test_datasets, lp_individual)):
                print(f"  {dataset_name.strip()}: {metric:.4f}")
            
            print(f"Graph Classification (Average): {gc_test_unseen:.4f}")
            for i, (dataset_name, metric) in enumerate(zip(gc_test_datasets, gc_individual)):
                metric_str = format_metric_results(metric) if isinstance(metric, dict) else f"{metric:.4f}"
                print(f"  {dataset_name.strip()}: {metric_str}")
            print("-----------------------------------------------")
        
        # Logging
        if epoch % args.log_interval == 0:
            epoch_time = time.time() - start_time
            log_message = (f"Epoch {epoch:3d} | Time: {epoch_time:.2f}s | "
                         f"NC Loss: {train_results['nc_loss']:.4f} | "
                         f"LP Loss: {train_results['lp_loss']:.4f} | "
                         f"GC Loss: {train_results['gc_loss']:.4f} | "
                         f"Combined: {train_results['combined_loss']:.4f} | "
                         f"NC Valid (Seen): {nc_valid_seen:.4f} | LP Valid (Seen): {lp_valid_seen:.4f} | GC Valid (Seen): {gc_valid_seen:.4f}")
            
            if unseen_metrics:
                log_message += f" | NC Test (Unseen): {unseen_metrics['test_unseen/nc_metric']:.4f} | LP Test (Unseen): {unseen_metrics['test_unseen/lp_metric']:.4f} | GC Test (Unseen): {unseen_metrics['test_unseen/gc_metric']:.4f}"
            
            print(log_message)
            logger.info(log_message, LogLevel.INFO)
            
            # Log link prediction tracking stats
            if lp_tracker is not None and getattr(args, 'enable_lp', True):
                # Print detailed tracking summary every 10 epochs or at the first epoch
                if epoch % (args.log_interval * 10) == 0 or epoch == 0:
                    lp_tracker.print_summary(epoch)
                
                # Log to wandb
# lp_tracker.log_to_wandb() - REMOVED to reduce wandb bloat
            
            # Log graph classification tracking stats
            if gc_tracker is not None and getattr(args, 'enable_gc', True):
                # Print detailed tracking summary every 10 epochs or at the first epoch
                if epoch % (args.log_interval * 10) == 0 or epoch == 0:
                    gc_tracker.print_summary(epoch)
                
                # Log to wandb
# gc_tracker.log_to_wandb() - REMOVED to reduce wandb bloat
            
            # Log to wandb
            wandb_log = {
                'epoch': epoch,
                'train/nc_loss': train_results['nc_loss'],
                'train/lp_loss': train_results['lp_loss'],
                'train/gc_loss': train_results['gc_loss'],
                'train/combined_loss': train_results['combined_loss'],
                'valid_seen/nc_metric': nc_valid_seen,
                'valid_seen/lp_metric': lp_valid_seen,
                'valid_seen/gc_metric': gc_valid_seen,
                'valid_seen/combined_score': combined_valid_seen,
                'lr': optimizer.param_groups[0]['lr']
            }
            
            # Add unseen test metrics if available
            if unseen_metrics:
                wandb_log.update(unseen_metrics)
            
            wandb.log(wandb_log)
    
    # Load best model for final evaluation
    if best_model_state is not None:
        logger.info(f"Loading best model from epoch {best_epoch}", LogLevel.INFO)
        print(f"\nLoading best model from epoch {best_epoch}")
        model.load_state_dict(best_model_state['model'])
        predictor.load_state_dict(best_model_state['predictor'])
        if identity_projection is not None and best_model_state['identity_projection'] is not None:
            identity_projection.load_state_dict(best_model_state['identity_projection'])
    
    # Final evaluation on test sets
    print("\n=== Final Test Evaluation ===")
    
    # Initialize empty results
    nc_test_results = {}
    lp_test_results = {}
    gc_test_results = {}
    
    # Node Classification Test on unseen datasets
    if getattr(args, 'enable_nc', True) and data_dict['nc_test'][0] is not None:
        nc_test_results = evaluate_node_classification(
            model, predictor, data_dict['nc_test'], args, 'test', identity_projection
        )
    
    # Link Prediction Test on unseen datasets
    if getattr(args, 'enable_lp', True) and data_dict['lp_test'][0] is not None:
        lp_test_results = evaluate_link_prediction_task(
            model, predictor, data_dict['lp_test'], args, 'test', identity_projection
        )
    
    # Graph Classification Test on unseen datasets
    if getattr(args, 'enable_gc', True) and len(data_dict['gc_test'][0]) > 0:
        gc_test_results = evaluate_graph_classification_task(
            model, predictor, data_dict['gc_test'], args, 'test', identity_projection, gc_tracker
        )
    
    # Print and log final results
    nc_test_metric = nc_test_results.get('test', 0.0)
    lp_test_metric = lp_test_results.get('test', 0.0)
    gc_test_metric = gc_test_results.get('test', 0.0)
    
    # Handle case where test metrics might be dictionaries (e.g., PCBA)
    if isinstance(nc_test_metric, dict):
        nc_test_metric = nc_test_metric.get('auc', nc_test_metric.get('ap', list(nc_test_metric.values())[0] if nc_test_metric else 0.0))
    if isinstance(lp_test_metric, dict):
        lp_test_metric = lp_test_metric.get('auc', lp_test_metric.get('ap', list(lp_test_metric.values())[0] if lp_test_metric else 0.0))
    if isinstance(gc_test_metric, dict):
        gc_test_metric = gc_test_metric.get('auc', gc_test_metric.get('ap', list(gc_test_metric.values())[0] if gc_test_metric else 0.0))
    
    final_results_msg = (f"Node Classification Test: {nc_test_metric:.4f}\n"
                        f"Link Prediction Test: {lp_test_metric:.4f}\n"
                        f"Graph Classification Test: {gc_test_metric:.4f}")
    print(final_results_msg)
    logger.info(final_results_msg, LogLevel.INFO)
    
    # Final wandb log
    final_wandb_log = {
        'test/nc_metric': nc_test_metric,
        'test/lp_metric': lp_test_metric,
        'test/gc_metric': gc_test_metric,
        'test/combined_score': nc_test_metric + lp_test_metric + gc_test_metric,
        'best_epoch': best_epoch,
        'best_valid_score': best_valid_score
    }
    
    # Add final link prediction tracking summary
    if lp_tracker is not None and getattr(args, 'enable_lp', True):
        print("\n=== Final Link Prediction Performance Summary ===")
        lp_tracker.print_summary()
    
    # Add final graph classification tracking summary
    if gc_tracker is not None and getattr(args, 'enable_gc', True):
        print("\n=== Final Graph Classification Performance Summary ===")
        gc_tracker.print_summary()
# gc_tracker.log_to_wandb(prefix="final_gc_tracker") - REMOVED to reduce wandb bloat
        
# # Add final summary stats to wandb - REMOVED to reduce wandb bloat
        # summary_stats = gc_tracker.get_summary_stats()
        # for key, value in summary_stats.items():
        #     final_wandb_log[f'final_gc_summary/{key}'] = value
    
    wandb.log(final_wandb_log)
    
    # Save checkpoint after training completion
    if args.save_checkpoint and checkpoint is None:  # Only save if we didn't load from checkpoint
        print("\n=== Saving Final Checkpoint ===")
        
        # Prepare best metrics for checkpoint
        best_metrics = {
            'train_metric': 0.0,  # Not tracked in joint training currently
            'valid_metric': best_valid_score,
            'test_metric': nc_test_metric + lp_test_metric + gc_test_metric,
            'nc_test_metric': nc_test_metric,
            'lp_test_metric': lp_test_metric,
            'gc_test_metric': gc_test_metric,
            'best_valid': best_valid_score,
            'final_test': nc_test_metric + lp_test_metric + gc_test_metric
        }
        
        # Save the final checkpoint
        checkpoint_path = save_checkpoint(
            model, predictor, optimizer,  # Include optimizer state
            args, best_metrics, best_epoch,
            identity_projection=identity_projection, rank=0
        )

        # Update wandb log with checkpoint path (only if checkpoint was saved)
        if checkpoint_path is not None:
            wandb.log({'checkpoint_path': checkpoint_path})
            print(f"✅ Final checkpoint saved to: {checkpoint_path}")
        else:
            print(f"ℹ️  Final checkpoint not saved due to threshold requirement")
        
    elif args.save_checkpoint and checkpoint is not None:
        print("ℹ️  Skipping checkpoint save since model was loaded from checkpoint")
        print(f"   Original checkpoint: {args.load_checkpoint}")
    
    # Return both average metrics and individual dataset metrics
    nc_individual = nc_test_results.get('individual_test_metrics', [])
    lp_individual = lp_test_results.get('individual_test_metrics', [])
    gc_individual = gc_test_results.get('individual_test_metrics', [])
    
    return nc_test_metric, lp_test_metric, gc_test_metric, nc_individual, lp_individual, gc_individual


def set_all_random_seeds(seed):
    """Set all random seeds for reproducible results."""
    print(f"Setting all random seeds to: {seed}")

    # Python random module
    random.seed(seed)

    # NumPy random module
    np.random.seed(seed)

    # PyTorch random seeds
    torch.manual_seed(seed)

    # CUDA random seeds (if CUDA is available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Set CUDA deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print("✓ CUDA deterministic settings enabled")

    print("✓ All random seeds initialized for reproducible inference")

def main():
    """Main function."""
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Parse arguments
    args = parse_joint_training_args()

    # Set all random seeds for reproducible results
    set_all_random_seeds(args.seed)
    
    # Validate checkpoint arguments
    if args.use_pretrained_model and args.load_checkpoint is None:
        print("Error: Must provide --load_checkpoint when --use_pretrained_model is True.")
        sys.exit(1)
    
    if args.load_checkpoint is not None and not os.path.exists(args.load_checkpoint):
        print(f"Error: Checkpoint file does not exist: {args.load_checkpoint}")
        sys.exit(1)
    
    if args.load_checkpoint is not None:
        print(f"✓ Checkpoint validation passed: {args.load_checkpoint}")
        print(f"✓ Use pretrained model: {args.use_pretrained_model}")
    
    # GPU setup
    try:
        gpu_ids = parse_gpu_spec(args.gpu)
        validate_gpu_availability(gpu_ids)
        print_gpu_info(gpu_ids)
        setup_cuda_visible_devices(gpu_ids)
        
        device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')

        # Claim all available GPU memory if requested
        if args.claim_all_gpu_memory and torch.cuda.is_available():
            try:
                print(f"🔒 Claiming all available GPU memory on {device}...")
                # Get current memory usage
                allocated_before = torch.cuda.memory_allocated(device) / 1024**3
                reserved_before = torch.cuda.memory_reserved(device) / 1024**3

                # Get total GPU memory
                total_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3

                print(f"   Before claiming - Allocated: {allocated_before:.2f}GB, Reserved: {reserved_before:.2f}GB, Total: {total_memory:.2f}GB")

                # Reserve most available memory by creating large tensors
                free_memory = torch.cuda.mem_get_info(device)[0] / 1024**3  # Free memory in GB
                memory_to_claim = free_memory - 1.0  # Leave 1GB buffer for operations

                if memory_to_claim > 1.0:  # Only claim if more than 1GB available
                    print(f"   Free memory: {free_memory:.2f}GB, claiming: {memory_to_claim:.2f}GB")

                    # Create tensor to claim memory (using float32 which is 4 bytes per element)
                    elements_to_allocate = int(memory_to_claim * 1024**3 / 4)  # 4 bytes per float32
                    memory_tensor = torch.empty(elements_to_allocate, dtype=torch.float32, device=device)

                    # Check final memory usage
                    allocated_after = torch.cuda.memory_allocated(device) / 1024**3
                    reserved_after = torch.cuda.memory_reserved(device) / 1024**3

                    print(f"   ✅ Memory claimed - Allocated: {allocated_after:.2f}GB, Reserved: {reserved_after:.2f}GB")
                    print(f"   Successfully claimed {allocated_after - allocated_before:.2f}GB of GPU memory")

                    # Delete the tensor but keep the memory reserved
                    del memory_tensor
                    torch.cuda.empty_cache()

                    final_reserved = torch.cuda.memory_reserved(device) / 1024**3
                    print(f"   Final reserved memory: {final_reserved:.2f}GB")
                else:
                    print(f"   ⚠️  Only {memory_to_claim:.2f}GB available to claim, skipping memory claiming")

            except Exception as e:
                print(f"   ⚠️  Failed to claim GPU memory: {e}")
                print(f"   Continuing without memory claiming...")

    except (ValueError, RuntimeError) as e:
        print(f"GPU setup error: {e}")
        return
    
    # Run joint training
    print(f"\n🚀 Starting Joint Training for {args.runs} runs")
    
    all_nc_results = []
    all_lp_results = []
    all_gc_results = []
    all_nc_individual_results = []
    all_lp_individual_results = []
    all_gc_individual_results = []
    
    for run in range(args.runs):
        print(f"\n{'='*50}")
        print(f"Run {run + 1}/{args.runs}")
        print(f"{'='*50}")

        # Set different seed for each run with all generators
        run_seed = args.seed + run
        set_all_random_seeds(run_seed)

        nc_result, lp_result, gc_result, nc_individual, lp_individual, gc_individual = run_joint_training(args, device)
        all_nc_results.append(nc_result)
        all_lp_results.append(lp_result)
        all_gc_results.append(gc_result)
        all_nc_individual_results.append(nc_individual)
        all_lp_individual_results.append(lp_individual)
        all_gc_individual_results.append(gc_individual)

        # Force cleanup between runs
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    # Aggregate results
    avg_nc = sum(all_nc_results) / len(all_nc_results)
    avg_lp = sum(all_lp_results) / len(all_lp_results)
    avg_gc = sum(all_gc_results) / len(all_gc_results)
    
    # Aggregate individual dataset metrics across runs
    avg_nc_individual = []
    std_nc_individual = []
    avg_lp_individual = []
    std_lp_individual = []
    avg_gc_individual = []
    std_gc_individual = []

    if all_nc_individual_results and all_nc_individual_results[0]:
        num_nc_datasets = len(all_nc_individual_results[0])
        for dataset_idx in range(num_nc_datasets):
            dataset_metrics = [run_results[dataset_idx] for run_results in all_nc_individual_results if len(run_results) > dataset_idx]
            avg_nc_individual.append(sum(dataset_metrics) / len(dataset_metrics))
            std_nc_individual.append(torch.std(torch.tensor(dataset_metrics)).item() if len(dataset_metrics) > 1 else 0.0)

    if all_lp_individual_results and all_lp_individual_results[0]:
        num_lp_datasets = len(all_lp_individual_results[0])
        for dataset_idx in range(num_lp_datasets):
            dataset_metrics = [run_results[dataset_idx] for run_results in all_lp_individual_results if len(run_results) > dataset_idx]
            avg_lp_individual.append(sum(dataset_metrics) / len(dataset_metrics))
            std_lp_individual.append(torch.std(torch.tensor(dataset_metrics)).item() if len(dataset_metrics) > 1 else 0.0)

    if all_gc_individual_results and all_gc_individual_results[0]:
        num_gc_datasets = len(all_gc_individual_results[0])
        for dataset_idx in range(num_gc_datasets):
            dataset_metrics = [run_results[dataset_idx] for run_results in all_gc_individual_results if len(run_results) > dataset_idx]
            # Extract primary metric (AUC if available, otherwise AP, otherwise raw value) for PCBA
            primary_metrics = []
            for metric in dataset_metrics:
                if isinstance(metric, dict):
                    # Multiple metrics (e.g., PCBA with AUC and AP) - prioritize AUC
                    primary_metric = metric.get('auc', metric.get('ap', next(iter(metric.values()))))
                    primary_metrics.append(primary_metric)
                else:
                    primary_metrics.append(metric)
            avg_gc_individual.append(sum(primary_metrics) / len(primary_metrics))
            std_gc_individual.append(torch.std(torch.tensor(primary_metrics)).item() if len(primary_metrics) > 1 else 0.0)
    
    print(f"\n{'='*50}")
    print(f"Final Results (Average over {args.runs} runs)")
    print(f"{'='*50}")
    print(f"Node Classification: {avg_nc:.4f} ± {torch.std(torch.tensor(all_nc_results)):.4f}")
    print(f"Link Prediction: {avg_lp:.4f} ± {torch.std(torch.tensor(all_lp_results)):.4f}")
    print(f"Graph Classification: {avg_gc:.4f} ± {torch.std(torch.tensor(all_gc_results)):.4f}")
    
    # Final sweep metric (for hyperparameter optimization)
    sweep_metric = 0.0
    if getattr(args, 'enable_nc', True):
        sweep_metric += avg_nc
    if getattr(args, 'enable_lp', True):
        sweep_metric += avg_lp
    if getattr(args, 'enable_gc', True):
        sweep_metric += avg_gc
    print(f"Combined Score: {sweep_metric:.4f}")
    
    # Print individual dataset results
    if avg_nc_individual:
        print(f"\nNode Classification Individual Results:")
        nc_test_datasets = args.nc_test_dataset.split(',')
        for dataset_name, metric, std in zip(nc_test_datasets, avg_nc_individual, std_nc_individual):
            print(f"  {dataset_name.strip()}: {metric:.4f} ± {std:.4f}")

    if avg_lp_individual:
        print(f"\nLink Prediction Individual Results:")
        lp_test_datasets = args.lp_test_dataset.split(',')
        for dataset_name, metric, std in zip(lp_test_datasets, avg_lp_individual, std_lp_individual):
            print(f"  {dataset_name.strip()}: {metric:.4f} ± {std:.4f}")

    if avg_gc_individual:
        print(f"\nGraph Classification Individual Results:")
        gc_test_datasets = args.gc_test_dataset.split(',')
        for dataset_name, metric, std in zip(gc_test_datasets, avg_gc_individual, std_gc_individual):
            metric_str = format_metric_results(metric) if isinstance(metric, dict) else f"{metric:.4f}"
            print(f"  {dataset_name.strip()}: {metric_str} ± {std:.4f}")
    
    # Get dataset names
    nc_test_datasets = args.nc_test_dataset.split(',') if getattr(args, 'enable_nc', True) and hasattr(args, 'nc_test_dataset') else []
    lp_test_datasets = args.lp_test_dataset.split(',') if getattr(args, 'enable_lp', True) and hasattr(args, 'lp_test_dataset') else []
    gc_test_datasets = args.gc_test_dataset.split(',') if getattr(args, 'enable_gc', True) and hasattr(args, 'gc_test_dataset') else []
    
    # Log final aggregated results
    wandb.init(project='inductnode-joint-summary')
    final_log = {
        'final/avg_nc_metric': avg_nc,
        'final/avg_lp_metric': avg_lp,
        'final/avg_gc_metric': avg_gc,
        'final/combined_score': sweep_metric,
        'runs': args.runs
    }
    
    # Add individual dataset metrics
    for dataset_name, metric in zip(nc_test_datasets, avg_nc_individual):
        final_log[f'final/nc_{dataset_name.strip()}'] = metric
    
    for dataset_name, metric in zip(lp_test_datasets, avg_lp_individual):
        final_log[f'final/lp_{dataset_name.strip()}'] = metric
    
    for dataset_name, metric in zip(gc_test_datasets, avg_gc_individual):
        final_log[f'final/gc_{dataset_name.strip()}'] = metric
    
    wandb.log(final_log)
    
    print("\n🎉 Joint training completed successfully!")

if __name__ == '__main__':
    main()
