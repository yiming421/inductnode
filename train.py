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

# Add the project root to the Python path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Core imports - reuse from existing scripts
from src.model import PureGCN_v1, PFNPredictorNodeCls, GCN, IdentityProjection, UnifiedGNN
from src.model_graphpfn import ParameterFreeGCN
from src.graphpfn import GraphPFNConfig, GraphPFNPredictor
from src.data_nc import load_all_data, load_all_data_train
from src.data_lp import load_all_data_link
from src.data_gc import load_all_graph_datasets, process_graph_features, create_data_loaders, create_task_filtered_datasets
from src.data_utils import process_data, prepare_link_data, select_link_context, process_link_data
from src.data_utils_graphpfn import process_data_graphpfn
from src.data_minibatch import MiniBatchNCLoader, compute_nc_loss_with_loader
from src.engine_nc import train_all, test_all, test_all_induct  # Node classification engines
from src.engine_nc_graphpfn import train_all_graphpfn, test_all_graphpfn  # GraphPFN engines
from src.engine_lp import train_link_prediction, evaluate_link_prediction  # Link prediction engines
from src.engine_gc import (
    train_graph_classification_single_task,
    evaluate_graph_classification_single_task,
    aggregate_task_metrics,
    format_metric_results
)
from src.engine_graphcl import (
    train_graphcl,
    load_graphcl_datasets,
    prepare_graphcl_data,
    create_graphcl_projection_head
)
from src.gpu_utils import parse_gpu_spec, setup_cuda_visible_devices, validate_gpu_availability, print_gpu_info
from transformers import get_cosine_schedule_with_warmup

# Logging and monitoring
from src.logger import TrainingLogger, LogLevel

from src.config import parse_joint_training_args
from src.checkpoint_utils import load_checkpoint_config, override_args_from_checkpoint, load_checkpoint_states, save_checkpoint


def _select_primary_metric(metric_dict, override=None, prefer='auc'):
    if not isinstance(metric_dict, dict):
        return metric_dict
    if not metric_dict:
        return 0.0
    if override and override != 'auto':
        return metric_dict.get(override, metric_dict.get(prefer, metric_dict.get('ap', next(iter(metric_dict.values())))))
    if prefer == 'ap':
        return metric_dict.get('ap', metric_dict.get('auc', next(iter(metric_dict.values()))))
    return metric_dict.get('auc', metric_dict.get('ap', next(iter(metric_dict.values()))))


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
        # Random sampling within bounds (only during training)
        # During evaluation/test, use fixed default to ensure reproducibility
        if epoch is None:
            # During evaluation, fall back to original fixed behavior
            defaults = {
                'nc': args.context_num,
                'lp': args.context_k,
                'gc': args.context_graph_num
            }
            context_shots = defaults[task_type]
            print(f"[Context Evaluation] {task_type.upper()} dataset '{dataset_name}': using {context_shots} TRAINING context shots (evaluation mode, fixed)")
            return context_shots

        # During training, sample randomly
        import random
        bounds = parse_context_bounds(getattr(args, 'context_bounds', None))
        lower, upper = bounds[task_type]
        context_shots = random.randint(lower, upper)
        print(f"[Context Random] {task_type.upper()} dataset '{dataset_name}': using {context_shots} TRAINING context shots (range: {lower}-{upper})")
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
            
            # Use original random sampling
            from src.data_utils import select_k_shot_context
            new_context_sample = select_k_shot_context(data, current_context_size, split_idx['train'])
            
            data.context_sample = new_context_sample.to(data.context_sample.device)
            print(f"    ✓ Refreshed {data.name}: {len(new_context_sample)} context samples ({current_context_size} per class, random sampling)")

            # Reprocess data with GraphPFN if enabled
            if args is not None and getattr(args, 'use_graphpfn', False):
                process_data_graphpfn(
                    data,
                    split_idx,
                    context_num=current_context_size,
                    pca_target_dim=args.hidden,
                    normalize_data=args.normalize_data,
                    use_full_pca=args.use_full_pca,
                    pca_device=args.pca_device,
                    incremental_pca_batch_size=args.incremental_pca_batch_size,
                    rank=0,
                    process_test_only=False,
                    use_orthogonal_noise=args.use_orthogonal_noise,
                )
                print(f"    ✓ Reprocessed {data.name} with GraphPFN (PCA to dim {args.hidden})")


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
    if getattr(args, 'use_mlp_projection', False):
        final_num_features = args.mlp_projection_input_dim
    else:
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
                dataset_name=getattr(dataset, 'name', None),
                use_random_orthogonal=args.use_random_orthogonal,
                plot_tsne=args.plot_tsne, tsne_save_dir=args.tsne_save_dir,
                use_quantile_normalization=args.use_quantile_normalization,
                quantile_norm_before_padding=args.quantile_norm_before_padding,
                use_mlp_projection=getattr(args, 'use_mlp_projection', False)
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
                dataset_name=getattr(dataset, 'name', None),
                use_random_orthogonal=args.use_random_orthogonal,
                plot_tsne=args.plot_tsne, tsne_save_dir=args.tsne_save_dir,
                use_quantile_normalization=args.use_quantile_normalization,
                quantile_norm_before_padding=args.quantile_norm_before_padding,
                use_mlp_projection=getattr(args, 'use_mlp_projection', False)
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
    projector = None

    # GraphPFN is incompatible with identity projection - override if both are enabled
    if getattr(args, 'use_graphpfn', False) and args.use_identity_projection:
        print("⚠️  WARNING: Disabling identity_projection (incompatible with GraphPFN)")
        args.use_identity_projection = False

    # Determine the actual hidden dimension for GNN input
    # Priority: FUG embeddings > identity projection > raw features
    if getattr(args, 'use_external_embeddings_nc', False):
        # FUG embeddings mode: use MLP projector to map 1024 -> hidden
        hidden = args.hidden
        from src.model import MLP
        projector = MLP(
            in_channels=1024,  # FUG generates uniform 1024-dim embeddings
            hidden_channels=args.hidden,
            out_channels=args.hidden,
            num_layers=2,
            dropout=args.dp,
            norm=args.norm
        )
        projector = projector.to(device)
        print(f"Created FUG projector: 1024 -> {args.hidden} (MLP with 2 layers)")
    elif args.use_identity_projection:
        hidden = args.projection_large_dim
        identity_projection = IdentityProjection(args.projection_small_dim, args.projection_large_dim)
        identity_projection = identity_projection.to(device)
    else:
        hidden = args.hidden
    
    # Override input dim if using MLP projection (debug mode)
    if getattr(args, 'use_mlp_projection', False):
        gnn_input_dim = input_dim
        print(f"Model Input Override: Using MLP projection input dim {gnn_input_dim} -> {hidden}")
    else:
        gnn_input_dim = hidden

    # Create GNN backbone
    # Use ParameterFreeGCN when using GraphPFN (handles arbitrary dimensions)
    if getattr(args, 'use_graphpfn', False):
        print("\n=== Using ParameterFreeGCN for GraphPFN ===")
        model = ParameterFreeGCN(
            num_layers=args.num_layers,
            dropout=args.dp,
            use_norm=args.norm,
            use_residual=args.res,
        )
        print(f"ParameterFreeGCN: num_layers={args.num_layers}, dropout={args.dp}, "
              f"norm={args.norm}, residual={args.res}")
    elif args.model == 'PureGCN_v1':
        model = PureGCN_v1(gnn_input_dim, args.num_layers, hidden, args.dp, args.norm,
                          args.res, args.relu, args.gnn_norm_affine,
                          activation=getattr(args, 'activation', 'relu'),
                          use_virtual_node=getattr(args, 'use_virtual_node', False))
    elif args.model == 'GCN':
        model = GCN(gnn_input_dim, hidden, args.norm, args.relu, args.num_layers, args.dp,
                   args.multilayer, args.use_gin, args.res, args.gnn_norm_affine,
                   activation=getattr(args, 'activation', 'relu'))
    elif args.model == 'UnifiedGNN':
        model = UnifiedGNN(
            model_type=getattr(args, 'unified_model_type', 'gcn'),
            in_feats=gnn_input_dim,
            h_feats=hidden,
            prop_step=getattr(args, 'num_layers', 2),  # Reuse num_layers as prop_step
            conv=getattr(args, 'conv_type', 'GCN'),
            gin_aggr=getattr(args, 'gin_aggr', 'sum'),
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

    # Wrap with Dynamic Encoder if enabled
    if args.use_dynamic_encoder:
        from src.model import GNNWithDE

        # DE output dim should match GNN input dim (hidden)
        de_output_dim = hidden

        # DE dropout should match main model dropout
        de_dropout = args.dp

        # DE activation should match main model activation
        de_activation = getattr(args, 'activation', 'relu')

        print(f"\n{'='*70}")
        print(f"Wrapping model with Dynamic Encoder:")
        print(f"  - Sample size: {args.de_sample_size}")
        print(f"  - Hidden dim: {args.de_hidden_dim}")
        print(f"  - Output dim: {de_output_dim} (matches GNN input)")
        print(f"  - Activation: {de_activation} (matches main model)")
        print(f"  - Dropout: {de_dropout} (matches main model)")
        print(f"  - Lambda DE: {args.lambda_de}")
        print(f"  - Update sample every: {args.de_update_sample_every_n_steps} steps")
        print(f"{'='*70}\n")

        model = GNNWithDE(
            gnn_model=model,
            de_sample_size=args.de_sample_size,
            de_hidden_dim=args.de_hidden_dim,
            de_output_dim=de_output_dim,
            de_activation=de_activation,
            de_use_layernorm=True,  # Always use LayerNorm
            de_dropout=de_dropout,
            de_norm_affine=True,  # Always use affine
            lambda_de=args.lambda_de,
            update_sample_every_n_steps=args.de_update_sample_every_n_steps
        )

    # Create unified predictor (same for both tasks)
    if getattr(args, 'use_graphpfn', False):
        # Use GraphPFN (TabPFN-style dual attention transformer)
        print("\n=== Using GraphPFN Predictor ===")

        # Convert string 'none' to Python None for feature_positional_embedding
        feature_pos_emb = args.graphpfn_feature_positional_embedding
        if feature_pos_emb == 'none':
            feature_pos_emb = None

        graphpfn_config = GraphPFNConfig(
            emsize=args.graphpfn_emsize,
            nhead=args.graphpfn_nhead,
            nlayers=args.graphpfn_nlayers,
            nhid_factor=args.graphpfn_nhid_factor,
            features_per_group=args.graphpfn_features_per_group,
            dropout=args.graphpfn_dropout,
            attention_between_features=args.graphpfn_attention_between_features,
            feature_positional_embedding=feature_pos_emb,
            seed=args.graphpfn_seed,
            normalize_x=args.graphpfn_normalize_x,
            n_out=10,  # Fixed for foundational model (TabPFN default)
            fourier_feature_scale=args.graphpfn_fourier_scale,
        )
        # Foundational model: no task-specific parameters
        # Accepts arbitrary GNN output dimension (determined at forward time)
        # num_classes is passed at forward() time for logit slicing
        predictor = GraphPFNPredictor(
            config=graphpfn_config,
            cache_trainset_representation=args.graphpfn_cache_trainset,
        )
        print(f"GraphPFN Config: emsize={graphpfn_config.emsize}, nhead={graphpfn_config.nhead}, "
              f"nlayers={graphpfn_config.nlayers}, features_per_group={graphpfn_config.features_per_group}")
        print(f"GraphPFN PE mode: {graphpfn_config.feature_positional_embedding}")
    elif args.predictor == 'PFN':
        # Original PFN predictor
        predictor = PFNPredictorNodeCls(
            hidden, args.nhead, args.transformer_layers, args.mlp_layers,
            args.dp, args.norm, args.seperate, False, None, None, args.sim,
            args.padding, args.mlp_norm_affine, args.normalize_class_h,
            use_first_half_embedding=getattr(args, 'use_first_half_embedding', False),
            use_full_embedding=getattr(args, 'use_full_embedding', False),
            norm_type=getattr(args, 'transformer_norm_type', 'post'),
            ffn_expansion_ratio=getattr(args, 'ffn_expansion_ratio', 4),
            use_matching_network=getattr(args, 'use_matching_network', False),
            matching_network_projection=getattr(args, 'matching_network_projection', 'linear'),
            matching_network_temperature=getattr(args, 'matching_network_temperature', 0.1),
            matching_network_learnable_temp=getattr(args, 'matching_network_learnable_temp', True),
            # Task-specific ridge regression configuration
            nc_sim=args.nc_sim,  # Use explicit config values (have proper defaults)
            nc_ridge_alpha=args.nc_ridge_alpha,
            lp_sim=args.lp_sim,
            lp_ridge_alpha=args.lp_ridge_alpha,
            gc_sim=args.gc_sim,
            gc_ridge_alpha=args.gc_ridge_alpha,
            head_num_layers=getattr(args, 'head_num_layers', 2),
            nc_head_num_layers=getattr(args, 'nc_head_num_layers', None),
            lp_head_num_layers=getattr(args, 'lp_head_num_layers', None),
            lp_head_type=getattr(args, 'lp_head_type', 'standard'),
            mplp_signature_dim=getattr(args, 'mplp_signature_dim', 64),
            mplp_num_hops=getattr(args, 'mplp_num_hops', 2),
            mplp_feature_combine=getattr(args, 'mplp_feature_combine', 'hadamard'),
            mplp_prop_type=getattr(args, 'mplp_prop_type', 'combine'),
            mplp_signature_sampling=getattr(args, 'mplp_signature_sampling', 'torchhd'),
            mplp_use_subgraph=getattr(args, 'mplp_use_subgraph', True),
            mplp_use_degree=getattr(args, 'mplp_use_degree', 'none'),
            mplp_context_calibrate=getattr(args, 'mplp_context_calibrate', True),
            mplp_calib_train_static=getattr(args, 'mplp_calib_train_static', False),
            mplp_calib_shuffle_labels=getattr(args, 'mplp_calib_shuffle_labels', False),
            mplp_calib_w_min=getattr(args, 'mplp_calib_w_min', 0.1),
            mplp_calib_w_max=getattr(args, 'mplp_calib_w_max', 10.0),
            mplp_calib_reg=getattr(args, 'mplp_calib_reg', 1e-3),
            ncn_beta=getattr(args, 'ncn_beta', 1.0),
            ncn_cndeg=getattr(args, 'ncn_cndeg', -1),
            lp_concat_common_neighbors=getattr(args, 'lp_concat_common_neighbors', False),
            # NEW: Skip token formulation option
            skip_token_formulation=getattr(args, 'skip_token_formulation', False),
            lp_use_linear_predictor=getattr(args, 'lp_use_linear_predictor', False)
        )
    else:
        raise NotImplementedError(f"Predictor {args.predictor} not implemented")
    
    model = model.to(device)
    predictor = predictor.to(device)

    return model, predictor, identity_projection, projector


def load_and_preprocess_data(args, device, skip_training_data=False, gc_tracker=None):
    """
    Load and preprocess data for enabled tasks.
    Returns processed datasets for node classification, link prediction, and graph classification.
    
    Args:
        skip_training_data: If True, skip loading training datasets (for pretrained model evaluation)
    """
    global lp_tracker
    
    print("\n=== Loading and Preprocessing Data ===")

    def move_nc_tensors_to_cpu(data_obj, split_idx):
        """Move NC data tensors and split indices to CPU to keep GPU memory free until use."""
        if hasattr(data_obj, 'x'):
            data_obj.x = data_obj.x.cpu()
        if hasattr(data_obj, 'adj_t'):
            data_obj.adj_t = data_obj.adj_t.cpu()
        if hasattr(data_obj, 'y'):
            data_obj.y = data_obj.y.cpu()
        if hasattr(data_obj, 'context_sample'):
            data_obj.context_sample = data_obj.context_sample.cpu()
        if hasattr(data_obj, 'x_pca'):
            data_obj.x_pca = data_obj.x_pca.cpu()
        if hasattr(data_obj, 'x_original'):
            data_obj.x_original = data_obj.x_original.cpu()
        split_idx['train'] = split_idx['train'].cpu()
        split_idx['valid'] = split_idx['valid'].cpu()
        split_idx['test'] = split_idx['test'].cpu()
    
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
                split_strategy=args.split_rebalance_strategy,
                use_augmentation=args.use_random_projection_augmentation,
                num_augmentations=args.num_augmentations,
                augmentation_mode=args.augmentation_mode,
                augmentation_activation=args.augmentation_activation,
                augmentation_max_depth=args.augmentation_max_depth,
                augmentation_verbose=args.augmentation_verbose,
                augmentation_use_random_noise=args.augmentation_use_random_noise,
                augmentation_dropout_rate=args.augmentation_dropout_rate,
                augmentation_use_feature_mixing=args.augmentation_use_feature_mixing,
                augmentation_mix_ratio=args.augmentation_mix_ratio,
                augmentation_mix_alpha=args.augmentation_mix_alpha
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

                # Use GraphPFN processing if enabled, otherwise use standard processing
                if getattr(args, 'use_graphpfn', False):
                    process_data_graphpfn(
                        data,
                        split_idx,
                        context_num=args.context_num,
                        pca_target_dim=args.hidden,
                        normalize_data=args.normalize_data,
                        use_full_pca=args.use_full_pca,
                        pca_device=args.pca_device,
                        incremental_pca_batch_size=args.incremental_pca_batch_size,
                        rank=0,
                        process_test_only=False,
                        use_orthogonal_noise=args.use_orthogonal_noise,
                    )
                else:
                    process_data(
                        data, split_idx, args.hidden, args.context_num, False, args.use_full_pca,
                        args.normalize_data, False, 32, 0, args.padding_strategy,
                        args.use_batchnorm, args.use_identity_projection, args.projection_small_dim, args.projection_large_dim, args.pca_device,
                        args.incremental_pca_batch_size, external_emb, args.use_random_orthogonal,
                        args.use_sparse_random, args.sparse_random_density,
                        args.plot_tsne, args.tsne_save_dir, args.use_pca_whitening, args.whitening_epsilon,
                        args.use_quantile_normalization, args.quantile_norm_before_padding,
                        getattr(args, 'use_external_embeddings_nc', False),
                        args.use_dynamic_encoder,
                        process_test_only=False,  # Training datasets: process all nodes
                        use_orthogonal_noise=args.use_orthogonal_noise,
                        args=args
                    )

                # Move processed tensors back to CPU to free GPU memory until training uses them
                move_nc_tensors_to_cpu(data, split_idx)

        # 4) Process test data
        for i, (data, split_idx) in enumerate(zip(nc_test_data_list, nc_test_split_idx_list)):
            data.x = data.x.to(device)
            data.adj_t = data.adj_t.to(device)
            data.y = data.y.to(device)

            external_emb = nc_test_external_embeddings[i] if nc_test_external_embeddings else None

            context_shots = resolve_context_shots(data.name, 'nc', args, epoch=None)

            # Use GraphPFN processing if enabled, otherwise use standard processing
            if getattr(args, 'use_graphpfn', False):
                process_data_graphpfn(
                    data,
                    split_idx,
                    context_num=context_shots,
                    pca_target_dim=args.hidden,
                    normalize_data=args.normalize_data,
                    use_full_pca=args.use_full_pca,
                    pca_device=args.pca_device,
                    incremental_pca_batch_size=args.incremental_pca_batch_size,
                    rank=0,
                    process_test_only=False,  # Node classification: always process all nodes
                    use_orthogonal_noise=args.use_orthogonal_noise,
                )
            else:
                process_data(
                    data, split_idx, args.hidden, context_shots, False, args.use_full_pca,
                    args.normalize_data, False, 32, 0, args.padding_strategy,
                    args.use_batchnorm, args.use_identity_projection, args.projection_small_dim, args.projection_large_dim, args.pca_device,
                    args.incremental_pca_batch_size, external_emb, args.use_random_orthogonal,
                    args.use_sparse_random, args.sparse_random_density,
                    args.plot_tsne, args.tsne_save_dir, args.use_pca_whitening, args.whitening_epsilon,
                    args.use_quantile_normalization, args.quantile_norm_before_padding,
                    getattr(args, 'use_external_embeddings_nc', False),
                    args.use_dynamic_encoder,
                    process_test_only=False,  # Node classification: always process all nodes
                    use_orthogonal_noise=args.use_orthogonal_noise,
                    args=args
                )

            # Move processed tensors back to CPU to free GPU memory until evaluation
            move_nc_tensors_to_cpu(data, split_idx)
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
                    if getattr(args, 'gc_supervised_mlp', False):
                        dataset_info['gc_supervised_head'] = None
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
                if getattr(args, 'gc_supervised_mlp', False):
                    dataset_info['gc_supervised_head'] = None
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
        return {'nc': True, 'lp': True, 'gc': True, 'graphcl': True}

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
        'gc': 'gc' in phase_tasks,
        'graphcl': 'graphcl' in phase_tasks or 'gcl' in phase_tasks  # Support both names
    }

    return active_tasks


def joint_training_step(model, predictor, nc_data, lp_data, gc_data, optimizer, args, epoch,
                       identity_projection=None, projector=None, nc_loaders=None, optimizer_nc=None, optimizer_lp=None, optimizer_gc=None,
                       optimizer_graphcl=None, graphcl_projection_head=None, graphcl_data_loader=None):
    """
    Perform one joint training step combining all three tasks.

    This function calculates the loss for each task, combines them with weights,
    and performs a single backward pass and optimizer step.

    Supports hierarchical/phased training to reduce task conflict.

    Args:
        nc_loaders: List of MiniBatchNCLoader instances for NC datasets
        optimizer_nc/lp/gc: Separate optimizers for each task (if use_separate_optimizers=True)
    """
    global lp_tracker

    model.train()
    predictor.train()

    # Use task-specific optimizers if provided, otherwise use unified optimizer
    opt_nc = optimizer_nc if optimizer_nc is not None else optimizer
    opt_lp = optimizer_lp if optimizer_lp is not None else optimizer
    opt_gc = optimizer_gc if optimizer_gc is not None else optimizer

    # When using separate optimizers, disable lambda scaling (use LR instead)
    # Temporarily override lambda values to 1.0 for this training step
    if args.use_separate_optimizers:
        original_lambda_nc = args.lambda_nc
        original_lambda_lp = args.lambda_lp
        original_lambda_gc = args.lambda_gc
        args.lambda_nc = 1.0
        args.lambda_lp = 1.0
        args.lambda_gc = 1.0

    device = optimizer.param_groups[0]['params'][0].device
    total_nc_loss = torch.tensor(0.0, device=device)
    total_lp_loss = torch.tensor(0.0, device=device)
    total_gc_loss = torch.tensor(0.0, device=device)
    nc_count = 0
    lp_count = 0
    gc_count = 0
    lp_gate_sum = 0.0
    lp_gate_count = 0
    lp_gate_calib_sum = 0.0
    lp_gate_calib_count = 0
    lp_struct_loss_sum = 0.0
    lp_struct_loss_count = 0
    # Initialize loss breakdown variables
    nc_nll_loss = 0.0
    nc_de_loss = 0.0
    nc_contrastive_loss = 0.0

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
            nc_nll_sum = 0.0
            nc_de_sum = 0.0
            nc_contrastive_sum = 0.0

            # Track augmentation statistics for mini-batch path
            # Determine if augmentation was used by checking data_list size
            use_augmentation = args.use_random_projection_augmentation if hasattr(args, 'use_random_projection_augmentation') else False
            num_augmentations = args.num_augmentations if hasattr(args, 'num_augmentations') else 1
            include_original = args.augmentation_include_original if hasattr(args, 'augmentation_include_original') else True
            augmentation_mode = args.augmentation_mode if hasattr(args, 'augmentation_mode') else 'preprocessing'

            print(f"[DEBUG MINIBATCH] use_augmentation: {use_augmentation}, num_augmentations: {num_augmentations}, "
                  f"include_original: {include_original}, total_loaders: {len(nc_loaders)}")

            # Regenerate augmented data and recreate loaders at specified intervals
            augmentation_interval = getattr(args, 'augmentation_regenerate_interval', 1)
            
            if augmentation_mode == 'preprocessing':
                # Preprocessing mode: only regenerate at epoch 0 (one-time generation)
                should_regenerate = (epoch == 0)
            else:
                # Per-epoch mode: regenerate at intervals
                should_regenerate = (epoch >= 0 and epoch % augmentation_interval == 0)

            if use_augmentation and augmentation_mode in ('per_epoch', 'preprocessing') and should_regenerate:
                print(f"\n{'='*80}")
                print(f"[Augmentation Regeneration] Epoch {epoch}: Regenerating augmentations (interval={augmentation_interval})")
                print(f"{'='*80}\n")

                from src.data_utils import apply_random_projection_augmentation, process_data
                import gc

                # Extract original data (loaders start as originals; after first regen len may double if include_original)
                if include_original and len(nc_loaders) % 2 == 0 and epoch > 0:
                    num_original_graphs = len(nc_loaders) // 2
                else:
                    num_original_graphs = len(nc_loaders)
                original_loaders = nc_loaders[:num_original_graphs]

                # Get original data from loaders
                original_data_list = []
                original_split_idx_list = []
                for loader in original_loaders:
                    original_data_list.append(loader.data)
                    original_split_idx_list.append(loader.split_idx)

                # Get augmentation settings
                augmentation_activation = getattr(args, 'augmentation_activation', 'random')
                augmentation_max_depth = getattr(args, 'augmentation_max_depth', 1)
                augmentation_use_random_noise = getattr(args, 'augmentation_use_random_noise', False)
                augmentation_dropout_rate = getattr(args, 'augmentation_dropout_rate', 0.0)
                augmentation_use_feature_mixing = getattr(args, 'augmentation_use_feature_mixing', False)
                augmentation_mix_ratio = getattr(args, 'augmentation_mix_ratio', 0.3)
                augmentation_mix_alpha = getattr(args, 'augmentation_mix_alpha', 0.5)

                if augmentation_activation == 'random':
                    activation_pool_to_use = None
                else:
                    activation_pool_to_use = [augmentation_activation]

                # Create new augmented data (one augmentation per graph per refresh)
                new_augmented_data_list = []
                new_augmented_split_idx_list = []

                # Fixed seed pool (size = num_augmentations) for preprocessing mode
                seed_pool = list(range(1, num_augmentations + 1)) if num_augmentations > 0 else [1]

                for graph_idx, (data, split_idx) in enumerate(zip(original_data_list, original_split_idx_list)):
                    if augmentation_mode == 'preprocessing':
                        seed = seed_pool[epoch % len(seed_pool)] + graph_idx * 100
                    else:  # per_epoch
                        seed = epoch * 100000 + graph_idx * 100 + 42

                    # Clone data and restore x_original
                    data_for_aug = data.clone()
                    if hasattr(data, 'x_original'):
                        data_for_aug.x = data.x_original.clone()
                        if graph_idx == 0:
                            print(f"\n  [Aug] Graph 0: using x_original, mean={data_for_aug.x.mean().item():.6f}")
                    else:
                        if graph_idx == 0:
                            print(f"\n  [WARNING] x_original not available, using current data.x")

                    # Keep PCA on GPU when requested (match initial preprocessing behavior)
                    if args.pca_device != 'cpu' and device.type == 'cuda':
                        data_for_aug.x = data_for_aug.x.to(device)
                        if hasattr(data_for_aug, 'y'):
                            data_for_aug.y = data_for_aug.y.to(device)

                    # Apply augmentation to RAW features
                    data_aug = apply_random_projection_augmentation(
                        data_for_aug,
                        hidden_dim_range=None,
                        activation_pool=activation_pool_to_use,
                        seed=seed,
                        verbose=False,
                        rank=0,
                        use_random_noise=augmentation_use_random_noise,
                        max_depth=augmentation_max_depth,
                        dropout_rate=augmentation_dropout_rate,
                        use_feature_mixing=augmentation_use_feature_mixing,
                        mix_ratio=augmentation_mix_ratio,
                        mix_alpha=augmentation_mix_alpha
                    )

                    # Apply process_data to do PCA/padding
                    external_emb = nc_external_embeddings[graph_idx] if nc_external_embeddings else None

                    process_data(
                        data_aug, split_idx, args.hidden, args.context_num, False, args.use_full_pca,
                        args.normalize_data, False, 32, 0, args.padding_strategy,
                        args.use_batchnorm, args.use_identity_projection,
                        args.projection_small_dim, args.projection_large_dim, args.pca_device,
                        args.incremental_pca_batch_size, external_emb, args.use_random_orthogonal,
                        args.use_sparse_random, args.sparse_random_density,
                        False, './tsne_plots', args.use_pca_whitening, args.whitening_epsilon,
                        args.use_quantile_normalization, args.quantile_norm_before_padding,
                        getattr(args, 'use_external_embeddings_nc', False),
                        args.use_dynamic_encoder, False, args.use_orthogonal_noise,
                        args=args, is_augmented_data=True
                    )

                    # Attach original data's x_pca_original to augmented data for contrastive loss
                    if hasattr(data, 'x_pca_original') and data.x_pca_original is not None:
                        data_aug.x_pca_original = data.x_pca_original

                    new_augmented_data_list.append(data_aug)
                    new_augmented_split_idx_list.append(split_idx)

                # Recreate loaders for augmented data
                new_augmented_loaders = []
                for i, (data_aug, split_idx) in enumerate(zip(new_augmented_data_list, new_augmented_split_idx_list)):
                    loader = MiniBatchNCLoader(data_aug, split_idx, args, device)
                    new_augmented_loaders.append(loader)

                # Replace nc_loaders with original + new augmented (mutate in-place so caller's list updates across epochs)
                if include_original:
                    nc_loaders.clear()
                    nc_loaders.extend(list(original_loaders) + new_augmented_loaders)
                    nc_split_idx_list = original_split_idx_list * (1 + num_augmentations)
                else:
                    nc_loaders.clear()
                    nc_loaders.extend(new_augmented_loaders)
                    nc_split_idx_list = new_augmented_split_idx_list

            if use_augmentation and augmentation_mode in ('preprocessing', 'per_epoch') and include_original:
                # Original + regenerated augmented loaders
                num_original_graphs = len(nc_loaders) // (1 + num_augmentations)
                num_augmented_graphs = len(nc_loaders) - num_original_graphs
            elif use_augmentation and augmentation_mode in ('preprocessing', 'per_epoch') and not include_original:
                # Training only on regenerated augmented loaders
                num_original_graphs = 0
                num_augmented_graphs = len(nc_loaders)
            else:
                # per_step mode augments inside engine_nc/train; loader counts stay original
                num_original_graphs = len(nc_loaders)
                num_augmented_graphs = 0

            original_loss_sum = 0.0
            augmented_loss_sum = 0.0

            # Track per-dataset metrics
            nc_dataset_losses = {}  # {dataset_name: loss_value}

            for i, (data_loader, split_idx) in enumerate(zip(nc_loaders, nc_split_idx_list)):
                external_emb = nc_external_embeddings[i] if nc_external_embeddings else None
                nc_loss_result = compute_nc_loss_with_loader(
                    data_loader, split_idx, model, predictor, args, device,
                    identity_projection=identity_projection,
                    projector=projector,
                    external_embeddings=external_emb,
                    optimizer=opt_nc,
                    epoch=epoch
                )
                # Handle dict return
                if isinstance(nc_loss_result, dict):
                    loss_val = nc_loss_result['total']
                    nc_loss_sum += loss_val
                    nc_nll_sum += nc_loss_result['nll']
                    nc_de_sum += nc_loss_result['de']
                    nc_contrastive_sum += nc_loss_result.get('contrastive', 0)

                    # Track per-dataset loss
                    dataset_name = nc_loss_result.get('dataset_name', f'dataset_{i}')
                    nc_dataset_losses[dataset_name] = loss_val

                    # Track augmented vs original
                    is_augmented = i >= num_original_graphs
                    if is_augmented:
                        augmented_loss_sum += loss_val
                    else:
                        original_loss_sum += loss_val
                else:
                    # Fallback for backward compatibility
                    nc_loss_sum += nc_loss_result
                    nc_nll_sum += nc_loss_result
                    nc_de_sum += 0.0

                    is_augmented = i >= num_original_graphs
                    if is_augmented:
                        augmented_loss_sum += nc_loss_result
                    else:
                        original_loss_sum += nc_loss_result

            total_nc_loss = torch.tensor(nc_loss_sum / len(nc_loaders), device=device)
            nc_nll_loss = nc_nll_sum / len(nc_loaders)
            nc_de_loss = nc_de_sum / len(nc_loaders)
            nc_contrastive_loss = nc_contrastive_sum / len(nc_loaders)
            nc_count = len(nc_loaders)

            # Store augmentation statistics
            nc_original_loss = original_loss_sum / num_original_graphs if num_original_graphs > 0 else 0
            nc_augmented_loss = augmented_loss_sum / num_augmented_graphs if num_augmented_graphs > 0 else 0
            nc_num_original = num_original_graphs
            nc_num_augmented = num_augmented_graphs

            # Debug print to verify augmentation statistics
            if use_augmentation:
                print(f"[DEBUG MINIBATCH] Augmentation stats - num_original: {nc_num_original}, num_augmented: {nc_num_augmented}, "
                      f"original_loss: {nc_original_loss:.4f}, augmented_loss: {nc_augmented_loss:.4f}")
        else:
            # Fallback to original full-batch training
            # Use GraphPFN engine if enabled
            if getattr(args, 'use_graphpfn', False):
                nc_loss_result = train_all_graphpfn(
                    model, nc_data_list, nc_split_idx_list, optimizer=opt_nc,
                    graphpfn_predictor=predictor,
                    batch_size=args.nc_batch_size, projector=projector, rank=0, epoch=epoch,
                    identity_projection=identity_projection, lambda_=args.lambda_, args=args,
                    external_embeddings_list=None)
            else:
                nc_loss_result = train_all(model, nc_data_list, nc_split_idx_list, optimizer=opt_nc, pred=predictor,
                              batch_size=args.nc_batch_size, degree=False,
                              orthogonal_push=args.orthogonal_push, normalize_class_h=args.normalize_class_h,
                              clip_grad=args.clip_grad, projector=projector, rank=0, epoch=epoch,
                              identity_projection=identity_projection, lambda_=args.lambda_nc, args=args)
            if nc_loss_result is not None:
                # Handle dict return with DE loss breakdown
                if isinstance(nc_loss_result, dict):
                    total_nc_loss = nc_loss_result['total']
                    nc_nll_loss = nc_loss_result['nll']
                    nc_de_loss = nc_loss_result['de']
                    nc_contrastive_loss = nc_loss_result.get('contrastive', 0)
                    # Store augmentation statistics
                    nc_original_loss = nc_loss_result.get('original_total', 0)
                    nc_augmented_loss = nc_loss_result.get('augmented_total', 0)
                    nc_num_original = nc_loss_result.get('num_original', 0)
                    nc_num_augmented = nc_loss_result.get('num_augmented', 0)
                else:
                    total_nc_loss = nc_loss_result
                    nc_nll_loss = nc_loss_result
                    nc_de_loss = 0.0
                    nc_contrastive_loss = 0.0
                    nc_original_loss = 0
                    nc_augmented_loss = 0
                    nc_num_original = 0
                    nc_num_augmented = 0
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
                            optimizer=opt_lp, batch_size=args.lp_batch_size,
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
                        gate_val = getattr(getattr(predictor, 'lp_head', None), 'last_gate_mean_train', None)
                        if gate_val is not None:
                            lp_gate_sum += float(gate_val)
                            lp_gate_count += 1
                        calib_val = getattr(getattr(predictor, 'lp_head', None), 'last_gate_calib_ms_train', None)
                        if calib_val is not None:
                            lp_gate_calib_sum += float(calib_val)
                            lp_gate_calib_count += 1
                        struct_loss_val = getattr(getattr(predictor, 'lp_head', None), 'last_struct_loss_train', None)
                        if struct_loss_val is not None:
                            lp_struct_loss_sum += float(struct_loss_val)
                            lp_struct_loss_count += 1
        
        if lp_count > 0:
            total_lp_loss = lp_loss_sum / lp_count
    
    # Graph Classification Loss
    if active_tasks['gc'] and hasattr(args, 'enable_gc') and args.enable_gc and len(gc_data_list) > 0 and args.lambda_gc > 0:
        gc_tracker.log_memory("gc_section_start")
        gc_loss_sum = 0.0
        gc_dataset_count = 0

        # Check if multi-dataset sampling is enabled
        use_multi_dataset_sampling = getattr(args, 'multi_dataset_sampling', False)
        use_gc_vectorized = getattr(args, 'gc_multitask_vectorized', False)

        if use_multi_dataset_sampling:
            # NEW: Multi-dataset sampling with temperature
            from src.engine_gc_multi_dataset import train_graph_classification_multi_dataset_sampling

            # Extract splits from processed data
            all_splits = [dataset_info['split_idx'] for dataset_info in gc_processed_data_list]

            # Get temperature parameter
            temperature = getattr(args, 'sampling_temperature', 0.5)

            gc_tracker.log_memory("gc_multi_dataset_sampling_start")

            # Train using multi-dataset sampling
            total_gc_loss = train_graph_classification_multi_dataset_sampling(
                model, predictor, gc_processed_data_list, all_splits, opt_gc,
                temperature,
                pooling_method=args.graph_pooling,
                device=device,
                batch_size=args.gc_batch_size,
                clip_grad=args.clip_grad,
                orthogonal_push=args.orthogonal_push,
                normalize_class_h=args.normalize_class_h,
                identity_projection=identity_projection,
                context_k=getattr(args, 'context_k', None),
                args=args
            ) * args.lambda_gc  # Scale by lambda

            gc_count = len(gc_processed_data_list)

            gc_tracker.log_memory("gc_multi_dataset_sampling_complete")

        else:
            # ORIGINAL: Task-specific training (dataset by dataset, task by task)
            for dataset_idx, dataset_info in enumerate(gc_processed_data_list):
                gc_tracker.log_memory(f"gc_dataset_{dataset_idx}_start")

                # Use precomputed task-filtered splits
                task_filtered_splits = dataset_info['task_filtered_splits']

                dataset_loss = 0.0
                dataset_tasks = 0

                gc_tracker.log_memory(f"gc_dataset_{dataset_idx}_splits_loaded")

                # Vectorized multi-task path (e.g., PCBA) if enabled
                sample_graph = dataset_info['dataset'][0]
                is_multitask = sample_graph.y.numel() > 1
                if use_gc_vectorized and is_multitask:
                    # Create unfiltered data loaders once (no task filtering)
                    use_index_tracking = ('fug_mapping' in dataset_info or
                                          'tsgfm_mapping' in dataset_info or
                                          'tag_mapping' in dataset_info)
                    unfiltered_loaders = create_data_loaders(
                        dataset_info['dataset'],
                        dataset_info['split_idx'],
                        batch_size=args.gc_batch_size,
                        shuffle=True,
                        task_idx=None,
                        use_index_tracking=use_index_tracking
                    )
                    from src.engine_gc import train_graph_classification_multitask_vectorized
                    dataset_loss = train_graph_classification_multitask_vectorized(
                        model, predictor, dataset_info, unfiltered_loaders, opt_gc,
                        pooling_method=args.graph_pooling, device=device,
                        clip_grad=args.clip_grad, orthogonal_push=args.orthogonal_push,
                        normalize_class_h=args.normalize_class_h, identity_projection=identity_projection,
                        args=args, lambda_=args.lambda_gc
                    )
                    dataset_tasks = 1
                else:
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
                            model, predictor, dataset_info, task_data_loaders, opt_gc, task_idx,
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

    # GraphCL Loss (Contrastive Learning)
    total_graphcl_loss = torch.tensor(0.0, device=device)
    graphcl_count = 0
    if active_tasks.get('graphcl', False) and getattr(args, 'enable_graphcl', False) and graphcl_data_loader is not None and graphcl_projection_head is not None and args.lambda_graphcl > 0:
        opt_graphcl = optimizer_graphcl if optimizer_graphcl is not None else optimizer

        # Train GraphCL for one epoch on the data loader
        graphcl_results = train_graphcl(
            model, graphcl_projection_head, graphcl_data_loader, opt_graphcl, args,
            device=device, identity_projection=identity_projection, rank=0, lambda_=args.lambda_graphcl
        )
        total_graphcl_loss = torch.tensor(graphcl_results['loss'], device=device)  # Loss already includes lambda scaling
        graphcl_count = 1

    # Combined loss
    combined_loss = total_nc_loss + total_lp_loss + total_gc_loss + total_graphcl_loss

    # Restore original lambda values if we temporarily changed them
    if args.use_separate_optimizers:
        args.lambda_nc = original_lambda_nc
        args.lambda_lp = original_lambda_lp
        args.lambda_gc = original_lambda_gc

    result_dict = {
        'nc_loss': total_nc_loss,
        'nc_nll_loss': nc_nll_loss,
        'nc_de_loss': nc_de_loss,
        'nc_contrastive_loss': nc_contrastive_loss,
        'lp_loss': total_lp_loss,
        'gc_loss': total_gc_loss,
        'graphcl_loss': total_graphcl_loss,
        'combined_loss': combined_loss,
        'nc_count': nc_count,
        'lp_count': lp_count,
        'gc_count': gc_count,
        'graphcl_count': graphcl_count,
        # Augmentation statistics
        'nc_original_loss': nc_original_loss if 'nc_original_loss' in locals() else 0,
        'nc_augmented_loss': nc_augmented_loss if 'nc_augmented_loss' in locals() else 0,
        'nc_num_original': nc_num_original if 'nc_num_original' in locals() else 0,
        'nc_num_augmented': nc_num_augmented if 'nc_num_augmented' in locals() else 0,
        # Per-dataset metrics
        'nc_dataset_losses': nc_dataset_losses if 'nc_dataset_losses' in locals() else {},
        'lp_gate_mean_train': (lp_gate_sum / lp_gate_count) if lp_gate_count > 0 else None,
        'lp_gate_calib_ms_train': (lp_gate_calib_sum / lp_gate_calib_count) if lp_gate_calib_count > 0 else None,
        'lp_struct_only_loss_train': (lp_struct_loss_sum / lp_struct_loss_count) if lp_struct_loss_count > 0 else None,
        'lp_struct_score_mean_train': getattr(getattr(predictor, 'lp_head', None), 'last_struct_score_mean_train', None),
        'lp_struct_score_std_train': getattr(getattr(predictor, 'lp_head', None), 'last_struct_score_std_train', None),
        'lp_feat_score_mean_train': getattr(getattr(predictor, 'lp_head', None), 'last_feat_score_mean_train', None),
        'lp_feat_score_std_train': getattr(getattr(predictor, 'lp_head', None), 'last_feat_score_std_train', None),
        'lp_logit_mean_train': getattr(getattr(predictor, 'lp_head', None), 'last_logit_mean_train', None),
        'lp_logit_std_train': getattr(getattr(predictor, 'lp_head', None), 'last_logit_std_train', None)
    }

    # Debug print to verify return values
    print(f"[DEBUG RETURN] Returning from joint_training_step: nc_num_augmented={result_dict['nc_num_augmented']}, "
          f"nc_original_loss={result_dict['nc_original_loss']:.4f}, nc_augmented_loss={result_dict['nc_augmented_loss']:.4f}")

    return result_dict


def evaluate_node_classification(model, predictor, nc_data, args, split='valid', identity_projection=None, projector=None, nc_loaders=None, epoch=0):
    """
    Evaluate node classification task only.

    Args:
        epoch: Current epoch number (for Bank of Tags permutation refresh)

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

                # Check if TTA is enabled for final test evaluation
                use_tta = getattr(args, 'use_test_time_augmentation', False)
                tta_gate_by_valid = getattr(args, 'tta_gate_by_valid', True)
                num_context_samples = max(1, int(getattr(args, 'unseen_test_context_samples', 1)))
                if num_context_samples > 1:
                    print(f"  Averaging test metrics over {num_context_samples} random context resamples")

                def _average_metric_lists(metric_lists):
                    if not metric_lists:
                        return []
                    if not isinstance(metric_lists[0], (list, tuple)):
                        return sum(metric_lists) / len(metric_lists)
                    num_items = len(metric_lists[0])
                    avg = []
                    for i in range(num_items):
                        vals = [ml[i] for ml in metric_lists if len(ml) > i]
                        avg.append(sum(vals) / len(vals) if vals else 0.0)
                    return avg

                # Use inductive evaluation for unseen datasets
                def _run_eval_once():
                    # Use GraphPFN engine if enabled
                    if getattr(args, 'use_graphpfn', False):
                        from src.engine_nc_graphpfn import test_all_induct_graphpfn
                        return test_all_induct_graphpfn(
                            model, predictor, nc_data_list, nc_split_idx_list, args.test_batch_size,
                            projector, 0, identity_projection, None, args=args
                        )
                    if use_tta:
                        # Use TTA for final test evaluation (gating handled inside TTA when enabled)
                        from src.engine_nc import test_all_induct_with_tta
                        return test_all_induct_with_tta(
                            model, predictor, nc_data_list, nc_split_idx_list, args.test_batch_size,
                            False, None, None, True, projector, 0, identity_projection, None,
                            use_cs=args.use_cs, cs_num_iters=args.cs_num_iters, cs_alpha=args.cs_alpha,
                            args=args
                        )
                    # Standard evaluation without TTA
                    return test_all_induct(
                        model, predictor, nc_data_list, nc_split_idx_list, args.test_batch_size,
                        False, None, None, True, projector, 0, identity_projection, None,
                        use_cs=args.use_cs, cs_num_iters=args.cs_num_iters, cs_alpha=args.cs_alpha,
                        args=args, epoch=epoch
                    )

                if num_context_samples == 1:
                    train_metrics, valid_metrics, test_metrics = _run_eval_once()
                else:
                    from src.data_utils import select_k_shot_context
                    original_contexts = []
                    for data in nc_data_list:
                        ctx = data.context_sample.clone() if hasattr(data, 'context_sample') and data.context_sample is not None else None
                        original_contexts.append(ctx)

                    all_train_metrics = []
                    all_valid_metrics = []
                    all_test_metrics = []

                    base_seed = int(getattr(args, 'seed', 42))
                    for sample_idx in range(num_context_samples):
                        torch.manual_seed(base_seed + sample_idx)
                        for data, split_idx in zip(nc_data_list, nc_split_idx_list):
                            context_shots = resolve_context_shots(data.name, 'nc', args, epoch=None)
                            if len(split_idx['train']) > 0:
                                context_source_split = split_idx['train']
                            else:
                                context_source_split = split_idx['test']
                            context_source_split = context_source_split.to(data.y.device)
                            new_context = select_k_shot_context(data, context_shots, context_source_split)
                            data.context_sample = new_context.to(data.y.device)

                        train_m, valid_m, test_m = _run_eval_once()
                        all_train_metrics.append(train_m)
                        all_valid_metrics.append(valid_m)
                        all_test_metrics.append(test_m)

                    # Restore original contexts
                    for data, ctx in zip(nc_data_list, original_contexts):
                        if ctx is not None:
                            data.context_sample = ctx.to(data.y.device)

                    train_metrics = _average_metric_lists(all_train_metrics)
                    valid_metrics = _average_metric_lists(all_valid_metrics)
                    test_metrics = _average_metric_lists(all_test_metrics)

                datasets_time = time.time() - datasets_start_time
                if use_tta and tta_gate_by_valid:
                    tta_suffix = " (TTA gated)"
                else:
                    tta_suffix = " (with TTA)" if use_tta else ""
                print(f"  All {len(nc_data_list)} datasets completed in {datasets_time:.2f}s{tta_suffix}")

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
                    # Get device from model parameters, or predictor if model has no parameters (e.g., ParameterFreeGCN)
                    try:
                        device = next(model.parameters()).device
                    except StopIteration:
                        device = next(predictor.parameters()).device
                    eval_accs = []
                    dataset_accs = {}  # {dataset_name: accuracy}

                    for i, (loader, split_idx) in enumerate(zip(nc_loaders, nc_split_idx_list)):
                        if loader.is_minibatch():
                            # Mini-batch evaluation
                            eval_result = evaluate_with_loader(
                                loader, split_idx, model, predictor, args, device,
                                eval_split=split, identity_projection=identity_projection, projector=projector
                            )
                            # Handle dict return from updated evaluate_with_loader
                            if isinstance(eval_result, dict):
                                eval_acc = eval_result['accuracy']
                                dataset_name = eval_result.get('dataset_name', f'dataset_{i}')
                                dataset_accs[dataset_name] = eval_acc
                            else:
                                eval_acc = eval_result
                            eval_accs.append(eval_acc)
                        else:
                            data_obj = nc_data_list[i] if nc_data_list is not None and i < len(nc_data_list) else None
                            if data_obj is None and hasattr(loader, 'data'):
                                data_obj = loader.data
                            # Fall back to full-batch for small datasets
                            # Use GraphPFN engine if enabled
                            if getattr(args, 'use_graphpfn', False):
                                from src.engine_nc_graphpfn import test_graphpfn
                                _, eval_acc, _, _ = test_graphpfn(
                                    model, predictor, data_obj, split_idx['train'], split_idx['valid'], split_idx['test'],
                                    args.test_batch_size, projector, 0, identity_projection, None, args=args
                                )
                            else:
                                from src.engine_nc import test
                                _, eval_acc, _ = test(
                                    model, predictor, data_obj, split_idx['train'], split_idx['valid'], split_idx['test'],
                                    args.test_batch_size, False, None, None, True, projector, 0, identity_projection
                                )
                            eval_accs.append(eval_acc)

                    results = {
                        split: sum(eval_accs) / len(eval_accs) if eval_accs else 0.0,
                        'individual_test_metrics': eval_accs,
                        'dataset_accs': dataset_accs if 'dataset_accs' in locals() else {}
                    }
                else:
                    # Use transductive evaluation for seen datasets (original approach)
                    # Use GraphPFN engine if enabled
                    if getattr(args, 'use_graphpfn', False):
                        train_metrics, valid_metrics, test_metrics = test_all_graphpfn(
                            model, predictor, nc_data_list, nc_split_idx_list, args.test_batch_size,
                            projector, 0, identity_projection, None, args=args
                        )
                    else:
                        train_metrics, valid_metrics, test_metrics = test_all(
                            model, predictor, nc_data_list, nc_split_idx_list, args.test_batch_size,
                            False, None, None, True, projector, 0, identity_projection, None,
                            use_cs=args.use_cs, cs_num_iters=args.cs_num_iters, cs_alpha=args.cs_alpha
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

    num_context_samples = 1
    if split == 'test':
        num_context_samples = max(1, int(getattr(args, 'unseen_test_context_samples', 1)))
        if num_context_samples > 1:
            print(f"  Averaging LP test metrics over {num_context_samples} random context resamples")
    
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
            lp_gate_values = []
            lp_gate_calib_values = []
            lp_struct_values = []
            lp_feat_values = []
            lp_gate_ratio_values = []
            lp_feat_abs_values = []
            lp_gate_abs_struct_values = []
            lp_gate_by_dataset = {}
            lp_struct_by_dataset = {}
            lp_feat_by_dataset = {}
            
            for i, (data, split_idx) in enumerate(zip(lp_data_list, lp_split_idx_list)):
                link_data_all = lp_link_data_all[i]
                context_data = lp_context_data[i]
                dataset_name = getattr(data, 'name', f'dataset_{i}')
                
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

                        def _eval_with_context(context_edges):
                            lp_results = evaluate_link_prediction(
                                model, predictor, data, link_data_all[split_key], context_edges,
                                args.test_batch_size, None, None, None, identity_projection,
                                0, True, degree=False, k_values=[20, 50, 100],
                                use_full_adj_for_test=(split == 'test'), lp_metric=args.lp_metric,
                                lp_concat_common_neighbors=getattr(args, 'lp_concat_common_neighbors', False)
                            )
                            return (
                                lp_results.get('default_metric', 0.0),
                                lp_results.get('mplp_gate_mean', None),
                                lp_results.get('mplp_gate_calib_ms', None),
                                lp_results.get('mplp_feat_only_metric', None),
                                lp_results.get('mplp_gate_struct_abs_ratio', None),
                                lp_results.get('mplp_feat_abs_mean', None),
                                lp_results.get('mplp_gate_abs_struct_mean', None),
                                lp_results.get('mplp_struct_only_metric', None)
                            )

                        if num_context_samples == 1:
                            (lp_metric_value, lp_gate_value, lp_gate_calib, lp_feat_value, lp_gate_ratio,
                             lp_feat_abs_mean, lp_gate_abs_struct_mean, lp_struct_value) = _eval_with_context(context_data)
                        else:
                            context_source = link_data_all.get('train', None)
                            context_shots = resolve_context_shots(data.name, 'lp', args, epoch=None)
                            base_seed = int(getattr(args, 'seed', 42))
                            sample_metrics = []
                            sample_gates = []
                            sample_struct = []
                            sample_feat = []
                            sample_gate_ratio = []
                            sample_feat_abs = []
                            sample_gate_abs_struct = []
                            sample_calib = []
                            for sample_idx in range(num_context_samples):
                                torch.manual_seed(base_seed + i * 1000 + sample_idx)
                                if context_source is not None and context_source['edge_pairs'].size(0) > 0:
                                    context_data_sample, _ = select_link_context(
                                        context_source, context_shots, args.context_neg_ratio, False
                                    )
                                else:
                                    context_data_sample = context_data

                                (metric_val, gate_val, calib_val, feat_val, gate_ratio_val,
                                 feat_abs_val, gate_abs_struct_val, struct_val) = _eval_with_context(context_data_sample)
                                sample_metrics.append(metric_val)
                                if gate_val is not None:
                                    sample_gates.append(gate_val)
                                if calib_val is not None:
                                    sample_calib.append(calib_val)
                                if feat_val is not None:
                                    sample_feat.append(feat_val)
                                if gate_ratio_val is not None:
                                    sample_gate_ratio.append(gate_ratio_val)
                                if feat_abs_val is not None:
                                    sample_feat_abs.append(feat_abs_val)
                                if gate_abs_struct_val is not None:
                                    sample_gate_abs_struct.append(gate_abs_struct_val)
                                if struct_val is not None:
                                    sample_struct.append(struct_val)

                            lp_metric_value = sum(sample_metrics) / len(sample_metrics) if sample_metrics else 0.0
                            lp_gate_value = (sum(sample_gates) / len(sample_gates)) if sample_gates else None
                            lp_gate_calib = (sum(sample_calib) / len(sample_calib)) if sample_calib else None
                            lp_feat_value = (sum(sample_feat) / len(sample_feat)) if sample_feat else None
                            lp_gate_ratio = (sum(sample_gate_ratio) / len(sample_gate_ratio)) if sample_gate_ratio else None
                            lp_feat_abs_mean = (sum(sample_feat_abs) / len(sample_feat_abs)) if sample_feat_abs else None
                            lp_gate_abs_struct_mean = (sum(sample_gate_abs_struct) / len(sample_gate_abs_struct)) if sample_gate_abs_struct else None
                            lp_struct_value = (sum(sample_struct) / len(sample_struct)) if sample_struct else None
                        
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
                        
                    lp_results_list.append(lp_metric_value)
                    lp_gate_values.append(lp_gate_value)
                    lp_gate_calib_values.append(lp_gate_calib)
                    lp_struct_values.append(lp_struct_value)
                    lp_gate_ratio_values.append(lp_gate_ratio)
                    lp_feat_values.append(lp_feat_value)
                    lp_feat_abs_values.append(lp_feat_abs_mean)
                    lp_gate_abs_struct_values.append(lp_gate_abs_struct_mean)
                    lp_gate_by_dataset[dataset_name] = lp_gate_value
                    lp_struct_by_dataset[dataset_name] = lp_struct_value
                    lp_feat_by_dataset[dataset_name] = lp_feat_value
                else:
                    lp_results_list.append(0.0)
                    lp_gate_values.append(None)
                    lp_gate_calib_values.append(None)
                    lp_gate_ratio_values.append(None)
                    lp_feat_values.append(None)
                    lp_feat_abs_values.append(None)
                    lp_gate_abs_struct_values.append(None)
                    lp_struct_values.append(None)
            
            if lp_results_list:
                avg_result = sum(lp_results_list) / len(lp_results_list)
                gate_values = [v for v in lp_gate_values if v is not None]
                avg_gate = (sum(gate_values) / len(gate_values)) if gate_values else None
                gate_calib_values = [v for v in lp_gate_calib_values if v is not None]
                avg_gate_calib = (sum(gate_calib_values) / len(gate_calib_values)) if gate_calib_values else None
                struct_values = [v for v in lp_struct_values if v is not None]
                avg_struct = (sum(struct_values) / len(struct_values)) if struct_values else None
                feat_values = [v for v in lp_feat_values if v is not None]
                avg_feat = (sum(feat_values) / len(feat_values)) if feat_values else None
                gate_ratio_values = [v for v in lp_gate_ratio_values if v is not None]
                avg_gate_ratio = (sum(gate_ratio_values) / len(gate_ratio_values)) if gate_ratio_values else None
                feat_abs_values = [v for v in lp_feat_abs_values if v is not None]
                avg_feat_abs = (sum(feat_abs_values) / len(feat_abs_values)) if feat_abs_values else None
                gate_abs_struct_values = [v for v in lp_gate_abs_struct_values if v is not None]
                avg_gate_abs_struct = (sum(gate_abs_struct_values) / len(gate_abs_struct_values)) if gate_abs_struct_values else None
                results = {
                    'train': avg_result,  # For consistency with NC format
                    'valid': avg_result,
                    'test': avg_result,
                    'individual_test_metrics': lp_results_list,
                    'mplp_gate_mean': avg_gate,
                    'individual_gate_means': lp_gate_values,
                    'mplp_gate_mean_by_dataset': lp_gate_by_dataset,
                    'mplp_struct_only_metric_by_dataset': lp_struct_by_dataset,
                    'mplp_feat_only_metric_by_dataset': lp_feat_by_dataset,
                    'mplp_gate_calib_ms': avg_gate_calib,
                    'individual_gate_calib_ms': lp_gate_calib_values,
                    'mplp_feat_only_metric': avg_feat,
                    'individual_feat_only_metrics': lp_feat_values,
                    'mplp_gate_struct_abs_ratio': avg_gate_ratio,
                    'individual_gate_struct_abs_ratio': lp_gate_ratio_values,
                    'mplp_feat_abs_mean': avg_feat_abs,
                    'individual_feat_abs_mean': lp_feat_abs_values,
                    'mplp_gate_abs_struct_mean': avg_gate_abs_struct,
                    'individual_gate_abs_struct_mean': lp_gate_abs_struct_values,
                    'mplp_struct_only_metric': avg_struct,
                    'individual_struct_only_metrics': lp_struct_values
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

                sample_graph = dataset_info['dataset'][0]
                is_multitask = sample_graph.y.numel() > 1
                if is_multitask and getattr(args, 'gc_multitask_vectorized', False):
                    print(f"    Dataset {dataset_idx} ({dataset_name}): Vectorized multi-task evaluation")
                    from src.engine_gc import evaluate_graph_classification_multitask_vectorized
                    use_index_tracking = ('fug_mapping' in dataset_info or 
                                          'tsgfm_mapping' in dataset_info or 
                                          'tag_mapping' in dataset_info)

                    if split == 'test':
                        test_only_splits = {'test': dataset_info['split_idx']['test']}
                        shuffle_train_eval = False
                        task_eval_loaders = create_data_loaders(
                            dataset_info['dataset'],
                            test_only_splits,
                            batch_size=args.gc_test_batch_size,
                            shuffle=shuffle_train_eval,
                            task_idx=None,
                            use_index_tracking=use_index_tracking
                        )
                    else:
                        shuffle_train_eval = (
                            split == 'train'
                            and getattr(args, 'gc_train_eval_max_batches', 0) > 0
                            and getattr(args, 'gc_train_eval_shuffle', True)
                        )
                        task_eval_loaders = create_data_loaders(
                            dataset_info['dataset'],
                            dataset_info['split_idx'],
                            batch_size=args.gc_test_batch_size,
                            shuffle=shuffle_train_eval,
                            task_idx=None,
                            use_index_tracking=use_index_tracking
                        )

                    task_eval_results = evaluate_graph_classification_multitask_vectorized(
                        model, predictor, dataset_info, task_eval_loaders,
                        pooling_method=args.graph_pooling, device=model.parameters().__next__().device,
                        normalize_class_h=args.normalize_class_h, dataset_name=dataset_name,
                        identity_projection=identity_projection, args=args
                    )

                    result_key = 'val' if split == 'valid' else split
                    split_result = task_eval_results.get(result_key, 0.0)
                    task_results.append(split_result)
                else:
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
                            shuffle_train_eval = False
                            task_eval_loaders = create_data_loaders(
                                dataset_info['dataset'], 
                                test_only_splits,
                                batch_size=args.gc_test_batch_size,
                                shuffle=shuffle_train_eval,
                                task_idx=task_idx,
                                use_index_tracking=use_index_tracking
                            )
                        else:
                            shuffle_train_eval = (
                                split == 'train'
                                and getattr(args, 'gc_train_eval_max_batches', 0) > 0
                                and getattr(args, 'gc_train_eval_shuffle', True)
                            )
                            # Use all splits for seen datasets
                            task_eval_loaders = create_data_loaders(
                                dataset_info['dataset'], 
                                task_splits,
                                batch_size=args.gc_test_batch_size,
                                shuffle=shuffle_train_eval,
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
                            # Extract primary metric for display (respect GC metric override)
                            display_metric = _select_primary_metric(
                                split_result,
                                override=getattr(args, 'gc_metric', 'auto'),
                                prefer='ap'
                            )
                            print(f"      Task {task_idx}: {display_metric:.4f} (eval: {eval_time:.2f}s)")
                        else:
                            print(f"      Task {task_idx}: {split_result:.4f} (eval: {eval_time:.2f}s)")
                # End per-task loop (non-vectorized)
                
                # Aggregate results across tasks for this dataset
                # Task accumulation completed for this dataset
                if gc_tracker:
                    gc_tracker.log_memory(f"eval_{split}_dataset_{dataset_idx}_tasks_complete")
                
                if task_results:
                    dataset_avg = aggregate_task_metrics(task_results)
                    # For averaging across datasets, extract primary metric if multiple metrics
                    if isinstance(dataset_avg, dict):
                        primary_metric = _select_primary_metric(
                            dataset_avg,
                            override=getattr(args, 'gc_metric', 'auto'),
                            prefer='ap'
                        )
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
                    identity_projection=None, projector=None, gc_tracker=None, nc_loaders=None, epoch=0):
    """
    Evaluate enabled tasks and return metrics.

    Args:
        nc_loaders: Optional list of MiniBatchNCLoader for mini-batch evaluation
        epoch: Current epoch number (for Bank of Tags permutation refresh)

    Returns:
        Dictionary with metrics for enabled tasks
    """

    results = {'nc_metrics': {}, 'lp_metrics': {}, 'gc_metrics': {}}

    # Evaluate node classification
    if hasattr(args, 'enable_nc') and args.enable_nc and nc_data is not None and nc_data[0] is not None:
            nc_results = evaluate_node_classification(model, predictor, nc_data, args, split, identity_projection, projector, nc_loaders, epoch=epoch)
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
    print(f"  GraphCL (Contrastive): {'✓' if getattr(args, 'enable_graphcl', False) else '✗'} (lambda: {getattr(args, 'lambda_graphcl', 0.0)})")
    
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
    wandb_kwargs = {'project': 'inductnode-joint', 'config': args}
    if args.wandb_run_name is not None:
        wandb_kwargs['name'] = args.wandb_run_name
    wandb.init(**wandb_kwargs)
    
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
        model_input_dim = args.mlp_projection_input_dim if getattr(args, 'use_mlp_projection', False) else args.hidden
        model, predictor, identity_projection, projector = create_unified_model(
            args, model_input_dim, device)

    # Attach supervised GC heads (if enabled) after model creation
    if getattr(args, 'gc_supervised_mlp', False) and getattr(args, 'enable_gc', True):
        from src.model import GraphClassificationMLPHead
        gc_heads = []
        gc_head_ids = set()
        head_by_name = {}

        def _gc_dataset_key(dataset_info):
            dataset = dataset_info.get('dataset', None)
            name = getattr(dataset, 'name', None)
            if isinstance(name, str) and name.strip():
                return name.strip()
            return None

        def _attach_head(dataset_info, fallback_key):
            out_dim = int(dataset_info.get('num_tasks', 1))
            key = _gc_dataset_key(dataset_info) or fallback_key
            head = head_by_name.get(key)
            if head is not None and getattr(head, 'out_dim', None) != out_dim:
                # Avoid shape mismatch if same name is used with different task counts.
                key = f"{key}:{out_dim}"
                head = None
            if head is None:
                head = GraphClassificationMLPHead(
                    in_dim=args.hidden,
                    out_dim=out_dim,
                    num_layers=max(1, getattr(args, 'head_num_layers', 2)),
                    dropout=args.dp,
                    norm=args.norm,
                    norm_affine=args.mlp_norm_affine
                ).to(device)
                head_by_name[key] = head
            dataset_info['gc_supervised_head'] = head
            if id(head) not in gc_head_ids:
                gc_heads.append(head)
                gc_head_ids.add(id(head))

        for idx, dataset_info in enumerate(data_dict.get('gc_train', ([], []))[1]):
            _attach_head(dataset_info, f"gc_train_{idx}")

        for idx, dataset_info in enumerate(data_dict.get('gc_test', ([], []))[1]):
            _attach_head(dataset_info, f"gc_test_{idx}")

        if gc_heads:
            print(f"Supervised GC MLP heads initialized (unique): {len(gc_heads)}")
    
    if lp_tracker:
        lp_tracker.record_memory()
        after_model_stats = lp_tracker.get_memory_stats()
        print(f"After Model Creation - GPU: {after_model_stats['gpu_allocated']:.2f}GB, CPU: {after_model_stats['cpu_memory']:.2f}GB")

    # Setup GraphCL if enabled
    graphcl_projection_head = None
    graphcl_data_loader = None
    if getattr(args, 'enable_graphcl', False):
        print(f"\n=== Setting up GraphCL ===")
        # Load GraphCL datasets
        dataset_names = [name.strip() for name in args.graphcl_dataset.split(',')]
        graphcl_datasets = load_graphcl_datasets(dataset_names, args, device=device, rank=0)

        # Prepare data loader
        if len(graphcl_datasets) > 0:
            graphcl_data_loader = prepare_graphcl_data(graphcl_datasets, args, device=device, rank=0)

            # Create projection head
            graphcl_projection_head = create_graphcl_projection_head(args, device=device)
            print(f"GraphCL projection head created: {args.hidden}D -> {args.graphcl_projection_dim}D")
            print(f"GraphCL datasets loaded: {len(graphcl_datasets)} datasets, {len(graphcl_data_loader.dataset)} graphs")
        else:
            print("Warning: No GraphCL datasets loaded, disabling GraphCL")
            args.enable_graphcl = False

    # Setup optimizer and scheduler
    # Use parameter groups for DE (lower lr due to ~50x larger gradients)
    de_lr_scale = getattr(args, 'de_lr_scale', 0.02)  # DE gets 2% of base lr by default

    gc_supervised_heads = []
    if getattr(args, 'gc_supervised_mlp', False) and getattr(args, 'enable_gc', True):
        for dataset_info in data_dict.get('gc_train', ([], []))[1]:
            head = dataset_info.get('gc_supervised_head')
            if head is not None:
                gc_supervised_heads.append(head)
        for dataset_info in data_dict.get('gc_test', ([], []))[1]:
            head = dataset_info.get('gc_supervised_head')
            if head is not None and head not in gc_supervised_heads:
                gc_supervised_heads.append(head)

    if args.use_dynamic_encoder and hasattr(model, 'de'):
        # Separate DE parameters from main model
        de_params = list(model.de.parameters())
        if hasattr(model, 'proj_layer_norm'):
            de_params.extend(list(model.proj_layer_norm.parameters()))
        de_param_ids = set(id(p) for p in de_params)

        # Collect non-DE parameters
        other_params = []
        for module in [model, predictor, identity_projection, projector, graphcl_projection_head]:
            if module is not None:
                for p in module.parameters():
                    if id(p) not in de_param_ids:
                        other_params.append(p)
        for head in gc_supervised_heads:
            other_params.extend(list(head.parameters()))

        # Parameter groups: DE gets lower lr
        param_groups = [
            {'params': other_params, 'lr': args.lr},
            {'params': de_params, 'lr': args.lr * de_lr_scale},
        ]
        parameters = param_groups
        print(f"DE learning rate: {args.lr * de_lr_scale:.2e} ({de_lr_scale:.0%} of base lr)")
    else:
        parameters = []
        for module in [model, predictor, identity_projection, projector, graphcl_projection_head]:
            if module is not None:
                parameters.extend(list(module.parameters()))
        for head in gc_supervised_heads:
            parameters.extend(list(head.parameters()))

    if args.use_separate_optimizers:
        # Separate optimizers mode: one optimizer per task
        lr_nc = args.lr_nc if args.lr_nc is not None else args.lr
        lr_lp = args.lr_lp if args.lr_lp is not None else args.lr
        lr_gc = args.lr_gc if args.lr_gc is not None else args.lr
        lr_graphcl = getattr(args, 'lr_graphcl', None) or args.lr

        print(f"Using separate optimizers - NC LR: {lr_nc:.2e}, LP LR: {lr_lp:.2e}, GC LR: {lr_gc:.2e}, GraphCL LR: {lr_graphcl:.2e}")

        if args.optimizer == 'adam':
            optimizer_nc = torch.optim.Adam(parameters, lr=lr_nc, weight_decay=args.weight_decay, eps=args.eps)
            optimizer_lp = torch.optim.Adam(parameters, lr=lr_lp, weight_decay=args.weight_decay, eps=args.eps)
            optimizer_gc = torch.optim.Adam(parameters, lr=lr_gc, weight_decay=args.weight_decay, eps=args.eps)
            optimizer_graphcl = torch.optim.Adam(parameters, lr=lr_graphcl, weight_decay=args.weight_decay, eps=args.eps) if getattr(args, 'enable_graphcl', False) else None
        elif args.optimizer == 'adamw':
            optimizer_nc = torch.optim.AdamW(parameters, lr=lr_nc, weight_decay=args.weight_decay, eps=args.eps)
            optimizer_lp = torch.optim.AdamW(parameters, lr=lr_lp, weight_decay=args.weight_decay, eps=args.eps)
            optimizer_gc = torch.optim.AdamW(parameters, lr=lr_gc, weight_decay=args.weight_decay, eps=args.eps)
            optimizer_graphcl = torch.optim.AdamW(parameters, lr=lr_graphcl, weight_decay=args.weight_decay, eps=args.eps) if getattr(args, 'enable_graphcl', False) else None

        def create_scheduler(opt):
            if args.schedule == 'cosine':
                return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
            elif args.schedule == 'step':
                step = max(1, args.epochs // 5)
                return torch.optim.lr_scheduler.StepLR(opt, step_size=step, gamma=0.5)
            elif args.schedule == 'warmup':
                return get_cosine_schedule_with_warmup(opt, num_warmup_steps=args.epochs // 10, num_training_steps=args.epochs)
            else:
                return None

        scheduler_nc = create_scheduler(optimizer_nc)
        scheduler_lp = create_scheduler(optimizer_lp)
        scheduler_gc = create_scheduler(optimizer_gc)
        scheduler_graphcl = create_scheduler(optimizer_graphcl) if optimizer_graphcl is not None else None

        # For compatibility, set primary optimizer to NC
        optimizer = optimizer_nc
        scheduler = scheduler_nc
    else:
        # Traditional unified optimizer
        print(f"Using unified optimizer with LR: {args.lr:.2e}")

        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)
        elif args.optimizer == 'adamw':
            optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)

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

        optimizer_nc = optimizer_lp = optimizer_gc = optimizer_graphcl = None
        scheduler_nc = scheduler_lp = scheduler_gc = scheduler_graphcl = None
    
    # Count total parameters (handle both list and param_groups)
    if isinstance(parameters, list) and len(parameters) > 0 and isinstance(parameters[0], dict):
        total_params = sum(p.numel() for group in parameters for p in group['params'])
    else:
        total_params = sum(p.numel() for p in parameters)
    print(f"Total parameters: {total_params:,}")
    
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
                    model, predictor, data_dict['nc_test'], args, 'test', identity_projection, projector
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
                gc_test_metric = _select_primary_metric(
                    gc_test_metric,
                    override=getattr(args, 'gc_metric', 'auto'),
                    prefer='auc'
                )
            
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

    # Store initial projector state for monitoring weight changes
    initial_projector_state = None
    if projector is not None:
        initial_projector_state = {name: param.data.clone()
                                   for name, param in projector.named_parameters()}
        print("Stored initial projector weights for change monitoring")

    for epoch in range(args.epochs):
        start_time = time.time()

        # Refresh contexts if needed
        refresh_contexts_if_needed(epoch, args, data_dict)

        # Joint training step
        train_results = joint_training_step(
            model, predictor, data_dict['nc_train'], data_dict['lp_train'], data_dict['gc_train'],
            optimizer, args, epoch, identity_projection, projector,
            nc_loaders=data_dict.get('nc_train_loaders', None),
            optimizer_nc=optimizer_nc if args.use_separate_optimizers else None,
            optimizer_lp=optimizer_lp if args.use_separate_optimizers else None,
            optimizer_gc=optimizer_gc if args.use_separate_optimizers else None,
            optimizer_graphcl=optimizer_graphcl if args.use_separate_optimizers else None,
            graphcl_projection_head=graphcl_projection_head,
            graphcl_data_loader=graphcl_data_loader
        )

        # Monitor projector gradients and weights (for FUG embeddings debugging)
        if projector is not None and epoch % 5 == 0:  # Every 5 epochs
            # Gradient monitoring
            grad_norms = []
            for name, param in projector.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    grad_norms.append(grad_norm)

            if grad_norms:
                avg_grad = sum(grad_norms) / len(grad_norms)
                max_grad = max(grad_norms)
                min_grad = min(grad_norms)
                print(f"[Epoch {epoch}] Projector Gradients - Avg: {avg_grad:.6e}, Max: {max_grad:.6e}, Min: {min_grad:.6e}")
                if avg_grad < 1e-6:
                    print(f"  ⚠️  WARNING: Very small projector gradients ({avg_grad:.6e}) - learning may be too slow!")
                if max_grad > 100:
                    print(f"  ⚠️  WARNING: Large projector gradients ({max_grad:.6e}) - possible instability!")

            # Weight change monitoring
            if initial_projector_state is not None:
                weight_changes = []
                for name, param in projector.named_parameters():
                    if name in initial_projector_state:
                        initial_weight = initial_projector_state[name]
                        current_weight = param.data
                        change = (current_weight - initial_weight).abs().mean().item()
                        weight_changes.append(change)

                if weight_changes:
                    avg_change = sum(weight_changes) / len(weight_changes)
                    max_change = max(weight_changes)
                    print(f"[Epoch {epoch}] Projector Weight Changes - Avg: {avg_change:.6e}, Max: {max_change:.6e}")
                    if avg_change < 1e-6:
                        print(f"  ❌ CRITICAL: Weights barely changed ({avg_change:.6e}) - projector NOT learning!")
                    elif avg_change < 1e-4:
                        print(f"  ⚠️  WARNING: Very small weight changes ({avg_change:.6e}) - slow learning!")
                    else:
                        print(f"  ✓ Weights are changing (learning in progress)")

        # Step scheduler(s)
        if args.use_separate_optimizers:
            if scheduler_nc is not None:
                scheduler_nc.step()
            if scheduler_lp is not None:
                scheduler_lp.step()
            if scheduler_gc is not None:
                scheduler_gc.step()
            if scheduler_graphcl is not None:
                scheduler_graphcl.step()
        else:
            if scheduler is not None:
                scheduler.step()

        # Every epoch: Validation on seen datasets (training data) for early stopping
        seen_valid_results = joint_evaluation(
            model, predictor, data_dict['nc_train'], data_dict['lp_train'], data_dict['gc_train'],
            args, 'valid', identity_projection, projector, gc_tracker,
            nc_loaders=data_dict.get('nc_train_loaders', None), epoch=epoch
        )

        # Compute combined validation score on seen datasets
        nc_valid_seen = seen_valid_results['nc_metrics'].get('valid', 0.0) if seen_valid_results['nc_metrics'] else 0.0
        lp_valid_seen = seen_valid_results['lp_metrics'].get('valid', 0.0) if seen_valid_results['lp_metrics'] else 0.0
        gc_valid_seen = seen_valid_results['gc_metrics'].get('valid', 0.0) if seen_valid_results['gc_metrics'] else 0.0
        combined_valid_seen = nc_valid_seen + lp_valid_seen + gc_valid_seen

        # Optional: Log GC train metrics for diagnostics
        gc_train_seen = None
        if (getattr(args, 'enable_gc', True) and getattr(args, 'gc_log_train_metrics', True)
            and data_dict['gc_train'] is not None and len(data_dict['gc_train'][0]) > 0):
            gc_train_results = evaluate_graph_classification_task(
                model, predictor, data_dict['gc_train'], args, 'train', identity_projection, gc_tracker
            )
            gc_train_seen = gc_train_results.get('train', 0.0)
        
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
                    model, predictor, data_dict['nc_test'], args, 'test', identity_projection, projector
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
            lp_gate_unseen = lp_unseen_results.get('mplp_gate_mean', None)
            if lp_gate_unseen is not None:
                unseen_metrics['test_unseen/lp_mplp_gate_mean'] = lp_gate_unseen
            lp_gate_unseen_by_dataset = lp_unseen_results.get('mplp_gate_mean_by_dataset', None)
            if isinstance(lp_gate_unseen_by_dataset, dict):
                for ds_name, gate_val in lp_gate_unseen_by_dataset.items():
                    if gate_val is not None:
                        unseen_metrics[f'test_unseen/lp_mplp_gate_mean/{ds_name}'] = gate_val
            lp_struct_unseen_by_dataset = lp_unseen_results.get('mplp_struct_only_metric_by_dataset', None)
            if isinstance(lp_struct_unseen_by_dataset, dict):
                for ds_name, struct_val in lp_struct_unseen_by_dataset.items():
                    if struct_val is not None:
                        unseen_metrics[f'test_unseen/lp_struct_only_metric/{ds_name}'] = struct_val
            lp_feat_unseen_by_dataset = lp_unseen_results.get('mplp_feat_only_metric_by_dataset', None)
            if isinstance(lp_feat_unseen_by_dataset, dict):
                for ds_name, feat_val in lp_feat_unseen_by_dataset.items():
                    if feat_val is not None:
                        unseen_metrics[f'test_unseen/lp_feat_only_metric/{ds_name}'] = feat_val
            lp_gate_calib_unseen = lp_unseen_results.get('mplp_gate_calib_ms', None)
            if lp_gate_calib_unseen is not None:
                unseen_metrics['test_unseen/lp_mplp_gate_calib_ms'] = lp_gate_calib_unseen
            lp_feat_only_unseen = lp_unseen_results.get('mplp_feat_only_metric', None)
            if lp_feat_only_unseen is not None:
                unseen_metrics['test_unseen/lp_feat_only_metric'] = lp_feat_only_unseen
            lp_gate_ratio_unseen = lp_unseen_results.get('mplp_gate_struct_abs_ratio', None)
            if lp_gate_ratio_unseen is not None:
                unseen_metrics['test_unseen/lp_gate_struct_abs_ratio'] = lp_gate_ratio_unseen
            lp_feat_abs_unseen = lp_unseen_results.get('mplp_feat_abs_mean', None)
            if lp_feat_abs_unseen is not None:
                unseen_metrics['test_unseen/lp_feat_abs_mean'] = lp_feat_abs_unseen
            lp_gate_abs_struct_unseen = lp_unseen_results.get('mplp_gate_abs_struct_mean', None)
            if lp_gate_abs_struct_unseen is not None:
                unseen_metrics['test_unseen/lp_gate_abs_struct_mean'] = lp_gate_abs_struct_unseen
            lp_struct_unseen = lp_unseen_results.get('mplp_struct_only_metric', None)
            if lp_struct_unseen is not None:
                unseen_metrics['test_unseen/lp_struct_only_metric'] = lp_struct_unseen
            lp_struct_mean_unseen = lp_unseen_results.get('mplp_struct_score_mean', None)
            lp_struct_std_unseen = lp_unseen_results.get('mplp_struct_score_std', None)
            lp_feat_mean_unseen = lp_unseen_results.get('mplp_feat_score_mean', None)
            lp_feat_std_unseen = lp_unseen_results.get('mplp_feat_score_std', None)
            lp_std_ratio_unseen = lp_unseen_results.get('mplp_struct_feat_std_ratio', None)
            lp_mean_ratio_unseen = lp_unseen_results.get('mplp_struct_feat_absmean_ratio', None)
            if lp_struct_mean_unseen is not None:
                unseen_metrics['test_unseen/lp_struct_score_mean'] = lp_struct_mean_unseen
            if lp_struct_std_unseen is not None:
                unseen_metrics['test_unseen/lp_struct_score_std'] = lp_struct_std_unseen
            if lp_feat_mean_unseen is not None:
                unseen_metrics['test_unseen/lp_feat_score_mean'] = lp_feat_mean_unseen
            if lp_feat_std_unseen is not None:
                unseen_metrics['test_unseen/lp_feat_score_std'] = lp_feat_std_unseen
            if lp_std_ratio_unseen is not None:
                unseen_metrics['test_unseen/lp_struct_feat_std_ratio'] = lp_std_ratio_unseen
            if lp_mean_ratio_unseen is not None:
                unseen_metrics['test_unseen/lp_struct_feat_absmean_ratio'] = lp_mean_ratio_unseen
            
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
                         f"GraphCL Loss: {train_results['graphcl_loss']:.4f} | "
                         f"Combined: {train_results['combined_loss']:.4f} | "
                         f"NC Valid (Seen): {nc_valid_seen:.4f} | LP Valid (Seen): {lp_valid_seen:.4f} | GC Valid (Seen): {gc_valid_seen:.4f}")

            if gc_train_seen is not None:
                gc_train_str = format_metric_results(gc_train_seen) if isinstance(gc_train_seen, dict) else f"{gc_train_seen:.4f}"
                log_message += f" | GC Train (Seen): {gc_train_str}"
            
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
                'train/nc_nll_loss': train_results.get('nc_nll_loss', train_results['nc_loss']),
                'train/nc_de_loss': train_results.get('nc_de_loss', 0.0),
                'train/nc_contrastive_loss': train_results.get('nc_contrastive_loss', 0.0),
                'train/lp_loss': train_results['lp_loss'],
                'train/gc_loss': train_results['gc_loss'],
                'train/combined_loss': train_results['combined_loss'],
                'valid_seen/nc_metric': nc_valid_seen,
                'valid_seen/lp_metric': lp_valid_seen,
                'valid_seen/gc_metric': gc_valid_seen,
                'valid_seen/combined_score': combined_valid_seen,
                'lr': optimizer.param_groups[0]['lr']
            }

            lp_gate_train = train_results.get('lp_gate_mean_train', None)
            if lp_gate_train is not None:
                wandb_log['train/lp_mplp_gate_mean'] = lp_gate_train
            lp_gate_calib_train = train_results.get('lp_gate_calib_ms_train', None)
            if lp_gate_calib_train is not None:
                wandb_log['train/lp_mplp_gate_calib_ms'] = lp_gate_calib_train
            lp_struct_loss_train = train_results.get('lp_struct_only_loss_train', None)
            if lp_struct_loss_train is not None:
                wandb_log['train/lp_struct_only_loss'] = lp_struct_loss_train

            lp_struct_mean_train = train_results.get('lp_struct_score_mean_train', None)
            lp_struct_std_train = train_results.get('lp_struct_score_std_train', None)
            lp_feat_mean_train = train_results.get('lp_feat_score_mean_train', None)
            lp_feat_std_train = train_results.get('lp_feat_score_std_train', None)
            lp_logit_mean_train = train_results.get('lp_logit_mean_train', None)
            lp_logit_std_train = train_results.get('lp_logit_std_train', None)
            if lp_struct_mean_train is not None:
                wandb_log['train/lp_struct_score_mean'] = lp_struct_mean_train
            if lp_struct_std_train is not None:
                wandb_log['train/lp_struct_score_std'] = lp_struct_std_train
            if lp_feat_mean_train is not None:
                wandb_log['train/lp_feat_score_mean'] = lp_feat_mean_train
            if lp_feat_std_train is not None:
                wandb_log['train/lp_feat_score_std'] = lp_feat_std_train
            if lp_logit_mean_train is not None:
                wandb_log['train/lp_logit_mean'] = lp_logit_mean_train
            if lp_logit_std_train is not None:
                wandb_log['train/lp_logit_std'] = lp_logit_std_train
            if lp_struct_std_train is not None and lp_feat_std_train is not None:
                wandb_log['train/lp_struct_feat_std_ratio'] = lp_struct_std_train / (lp_feat_std_train + 1e-8)
            if lp_struct_mean_train is not None and lp_feat_mean_train is not None:
                wandb_log['train/lp_struct_feat_absmean_ratio'] = abs(lp_struct_mean_train) / (abs(lp_feat_mean_train) + 1e-8)

            lp_gate_valid = seen_valid_results.get('lp_metrics', {}).get('mplp_gate_mean', None)
            if lp_gate_valid is not None:
                wandb_log['valid_seen/lp_mplp_gate_mean'] = lp_gate_valid
            lp_gate_valid_by_dataset = seen_valid_results.get('lp_metrics', {}).get('mplp_gate_mean_by_dataset', None)
            if isinstance(lp_gate_valid_by_dataset, dict):
                for ds_name, gate_val in lp_gate_valid_by_dataset.items():
                    if gate_val is not None:
                        wandb_log[f'valid_seen/lp_mplp_gate_mean/{ds_name}'] = gate_val
            lp_struct_valid_by_dataset = seen_valid_results.get('lp_metrics', {}).get('mplp_struct_only_metric_by_dataset', None)
            if isinstance(lp_struct_valid_by_dataset, dict):
                for ds_name, struct_val in lp_struct_valid_by_dataset.items():
                    if struct_val is not None:
                        wandb_log[f'valid_seen/lp_struct_only_metric/{ds_name}'] = struct_val
            lp_feat_valid_by_dataset = seen_valid_results.get('lp_metrics', {}).get('mplp_feat_only_metric_by_dataset', None)
            if isinstance(lp_feat_valid_by_dataset, dict):
                for ds_name, feat_val in lp_feat_valid_by_dataset.items():
                    if feat_val is not None:
                        wandb_log[f'valid_seen/lp_feat_only_metric/{ds_name}'] = feat_val
            lp_gate_calib_valid = seen_valid_results.get('lp_metrics', {}).get('mplp_gate_calib_ms', None)
            if lp_gate_calib_valid is not None:
                wandb_log['valid_seen/lp_mplp_gate_calib_ms'] = lp_gate_calib_valid
            lp_feat_only_valid = seen_valid_results.get('lp_metrics', {}).get('mplp_feat_only_metric', None)
            if lp_feat_only_valid is not None:
                wandb_log['valid_seen/lp_feat_only_metric'] = lp_feat_only_valid
            lp_gate_ratio_valid = seen_valid_results.get('lp_metrics', {}).get('mplp_gate_struct_abs_ratio', None)
            if lp_gate_ratio_valid is not None:
                wandb_log['valid_seen/lp_gate_struct_abs_ratio'] = lp_gate_ratio_valid
            lp_feat_abs_valid = seen_valid_results.get('lp_metrics', {}).get('mplp_feat_abs_mean', None)
            if lp_feat_abs_valid is not None:
                wandb_log['valid_seen/lp_feat_abs_mean'] = lp_feat_abs_valid
            lp_gate_abs_struct_valid = seen_valid_results.get('lp_metrics', {}).get('mplp_gate_abs_struct_mean', None)
            if lp_gate_abs_struct_valid is not None:
                wandb_log['valid_seen/lp_gate_abs_struct_mean'] = lp_gate_abs_struct_valid
            lp_struct_valid = seen_valid_results.get('lp_metrics', {}).get('mplp_struct_only_metric', None)
            if lp_struct_valid is not None:
                wandb_log['valid_seen/lp_struct_only_metric'] = lp_struct_valid
            lp_struct_mean_valid = seen_valid_results.get('lp_metrics', {}).get('mplp_struct_score_mean', None)
            lp_struct_std_valid = seen_valid_results.get('lp_metrics', {}).get('mplp_struct_score_std', None)
            lp_feat_mean_valid = seen_valid_results.get('lp_metrics', {}).get('mplp_feat_score_mean', None)
            lp_feat_std_valid = seen_valid_results.get('lp_metrics', {}).get('mplp_feat_score_std', None)
            lp_std_ratio_valid = seen_valid_results.get('lp_metrics', {}).get('mplp_struct_feat_std_ratio', None)
            lp_mean_ratio_valid = seen_valid_results.get('lp_metrics', {}).get('mplp_struct_feat_absmean_ratio', None)
            if lp_struct_mean_valid is not None:
                wandb_log['valid_seen/lp_struct_score_mean'] = lp_struct_mean_valid
            if lp_struct_std_valid is not None:
                wandb_log['valid_seen/lp_struct_score_std'] = lp_struct_std_valid
            if lp_feat_mean_valid is not None:
                wandb_log['valid_seen/lp_feat_score_mean'] = lp_feat_mean_valid
            if lp_feat_std_valid is not None:
                wandb_log['valid_seen/lp_feat_score_std'] = lp_feat_std_valid
            if lp_std_ratio_valid is not None:
                wandb_log['valid_seen/lp_struct_feat_std_ratio'] = lp_std_ratio_valid
            if lp_mean_ratio_valid is not None:
                wandb_log['valid_seen/lp_struct_feat_absmean_ratio'] = lp_mean_ratio_valid

            if gc_train_seen is not None:
                if isinstance(gc_train_seen, dict):
                    for metric_name, metric_val in gc_train_seen.items():
                        wandb_log[f'train/gc_metric_{metric_name}'] = metric_val
                else:
                    wandb_log['train/gc_metric'] = gc_train_seen

            # Optionally add per-dataset training loss and validation accuracy metrics
            if getattr(args, 'log_individual_datasets', False):
                # Add per-dataset training loss metrics
                nc_dataset_losses = train_results.get('nc_dataset_losses', {})
                for dataset_name, loss_val in nc_dataset_losses.items():
                    wandb_log[f'train/nc_loss_{dataset_name}'] = loss_val

                # Add per-dataset validation accuracy metrics
                nc_dataset_accs = seen_valid_results['nc_metrics'].get('dataset_accs', {})
                for dataset_name, acc_val in nc_dataset_accs.items():
                    wandb_log[f'valid_seen/nc_acc_{dataset_name}'] = acc_val

            # Add augmentation loss statistics if available
            if train_results.get('nc_num_augmented', 0) > 0:
                aug_original = train_results.get('nc_original_loss', 0)
                aug_augmented = train_results.get('nc_augmented_loss', 0)
                print(f"[DEBUG] Adding augmentation metrics to wandb: original={aug_original:.4f}, augmented={aug_augmented:.4f}")
                wandb_log.update({
                    'aug/train_loss_original': aug_original,
                    'aug/train_loss_augmented': aug_augmented,
                    'aug/train_loss_diff': aug_augmented - aug_original
                })
            else:
                print(f"[DEBUG] No augmentation metrics: nc_num_augmented={train_results.get('nc_num_augmented', 0)}")

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
            model, predictor, data_dict['nc_test'], args, 'test', identity_projection, projector
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
        gc_test_metric = _select_primary_metric(
            gc_test_metric,
            override=getattr(args, 'gc_metric', 'auto'),
            prefer='auc'
        )
    
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
            projector=projector,
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
                    primary_metric = _select_primary_metric(
                        metric,
                        override=getattr(args, 'gc_metric', 'auto'),
                        prefer='auc'
                    )
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
    wandb_summary_kwargs = {'project': 'inductnode-joint-summary'}
    if args.wandb_run_name is not None:
        wandb_summary_kwargs['name'] = f"{args.wandb_run_name}-summary"
    wandb.init(**wandb_summary_kwargs)
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
