#!/usr/bin/env python3
"""
Joint Training Script for Node Classification and Link Prediction
"""

import os
import sys
import time
import copy
import torch
import wandb
import traceback
import signal
import psutil
import gc
from contextlib import contextmanager, nullcontext

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Core imports - reuse from existing scripts
from src.model import PureGCN_v1, PureGCN, PFNPredictorNodeCls, GCN, IdentityProjection
from src.data import load_all_data, load_all_data_train
from src.data_link import load_all_data_link
from src.data_graph import load_all_graph_datasets, process_graph_features, create_data_loaders, create_task_filtered_datasets
from src.data_utils import process_data, prepare_link_data, select_link_context, process_link_data
from src.engine import train_all, test_all, test_all_induct  # Node classification engines
from src.engine_link_pred import train_link_prediction, evaluate_link_prediction  # Link prediction engines
from src.engine_graph import (
    train_and_evaluate_graph_classification, 
    train_graph_classification_single_task,
    evaluate_graph_classification_single_task,
    pool_graph_embeddings,
    create_context_embeddings,
    prepare_pfn_data_structure,
    get_dataset_metric,
    calculate_metric,
    aggregate_task_metrics,
    format_metric_results
)
from src.gpu_utils import parse_gpu_spec, setup_cuda_visible_devices, get_effective_world_size, validate_gpu_availability, print_gpu_info
from transformers import get_cosine_schedule_with_warmup

# Logging and monitoring
from src.logger import TrainingLogger, LogLevel

from src.config import parse_joint_training_args


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
    
    def log_to_wandb(self, prefix="lp_tracker"):
        """Log tracking statistics to wandb."""
        summary = self.get_summary_stats()
        wandb_log = {f"{prefix}/{key}": value for key, value in summary.items()}
        wandb.log(wandb_log)
    
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


# Global tracker instance
lp_tracker = None


def setup_fug_environment(args):
    """Configure OGB FUG embeddings if requested."""
    if args.enable_gc and hasattr(args, 'use_ogb_fug') and args.use_ogb_fug:
        print(f"Enabling OGB FUG embeddings for graph classification")
        print(f"FUG root: {args.fug_root}, OGB root: {args.ogb_root}")
        os.environ['USE_FUG_EMB'] = '1'
        os.environ['FUG_EMB_ROOT'] = args.fug_root
        os.environ['OGB_ROOT'] = args.ogb_root
    else:
        # Ensure FUG is disabled for non-FUG runs
        os.environ.pop('USE_FUG_EMB', None)


def process_datasets_for_models(datasets, processed_data_list, args, device, test_datasets=False):
    """
    Process datasets to handle feature dimensions and create any necessary dummy features.
    Adapted from graph_classification.py for joint training.
    """
    processed_datasets = []
    final_num_features = args.hidden
    
    for dataset, dataset_info in zip(datasets, processed_data_list):
        # Process features using PCA and padding
        if test_datasets:
            processing_info = process_graph_features(
                dataset, args.hidden, device, 
                args.use_identity_projection, args.projection_small_dim, args.projection_large_dim,
                args.use_full_pca, False, args.normalize_data,
                args.padding_strategy, args.use_batchnorm,
                pca_device=args.pca_device, incremental_pca_batch_size=args.incremental_pca_batch_size,
                pca_sample_threshold=500000,  # Default threshold for sampled PCA
                processed_data=dataset_info,  # Pass FUG mapping info
                pcba_context_only_pca=False,  # Use full optimization for test datasets
                use_pca_cache=args.use_pca_cache, pca_cache_dir=args.pca_cache_dir,
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
                processed_data=dataset_info,  # Pass FUG mapping info
                pcba_context_only_pca=False,  # Use full optimization for training datasets
                use_pca_cache=args.use_pca_cache, pca_cache_dir=args.pca_cache_dir,
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
                          args.res, args.relu, args.gnn_norm_affine)
    elif args.model == 'GCN':
        model = GCN(hidden, hidden, args.norm, args.relu, args.num_layers, args.dp,
                   args.multilayer, args.use_gin, args.res, args.gnn_norm_affine)
    else:
        raise NotImplementedError(f"Model {args.model} not implemented")
    
    # Create unified predictor (same for both tasks)
    if args.predictor == 'PFN':
        predictor = PFNPredictorNodeCls(
            hidden, args.nhead, args.transformer_layers, args.mlp_layers, 
            args.dp, args.norm, args.seperate, False, None, None, args.sim, 
            args.padding, args.mlp_norm_affine, args.normalize_class_h
        )
    else:
        raise NotImplementedError(f"Predictor {args.predictor} not implemented")
    
    model = model.to(device)
    predictor = predictor.to(device)
    
    return model, predictor, identity_projection


def load_and_preprocess_data(args, device):
    """
    Load and preprocess data for enabled tasks.
    Returns processed datasets for node classification, link prediction, and graph classification.
    """
    global lp_tracker
    
    print("\n=== Loading and Preprocessing Data ===")
    
    # === Node Classification Data ===
    nc_train_data_list, nc_train_split_idx_list = None, None
    nc_test_data_list, nc_test_split_idx_list = None, None
    
    if args.enable_nc:
        print("Loading node classification datasets...")
        nc_train_datasets = args.nc_train_dataset.split(',')
        nc_test_datasets = args.nc_test_dataset.split(',')
        
        # Load training data for node classification
        nc_train_data_list, nc_train_split_idx_list = load_all_data_train(nc_train_datasets)
        
        # Process node classification training data
        for data, split_idx in zip(nc_train_data_list, nc_train_split_idx_list):
            data.x = data.x.to(device)
            data.adj_t = data.adj_t.to(device)
            data.y = data.y.to(device)
            # Apply node classification specific preprocessing

            process_data(data, split_idx, args.hidden, args.context_num, False, args.use_full_pca, 
                        args.normalize_data, False, 32, 0, args.padding_strategy, 
                        args.use_batchnorm, args.use_identity_projection, args.projection_small_dim, args.projection_large_dim, args.pca_device,
                        args.incremental_pca_batch_size)

        # Load test data for node classification
        nc_test_data_list, nc_test_split_idx_list = load_all_data(nc_test_datasets)
        
        # Process node classification test data
        for data, split_idx in zip(nc_test_data_list, nc_test_split_idx_list):
            data.x = data.x.to(device)
            data.adj_t = data.adj_t.to(device)
            data.y = data.y.to(device)
            process_data(data, split_idx, args.hidden, args.context_num, False, args.use_full_pca, 
                        args.normalize_data, False, 32, 0, args.padding_strategy, 
                        args.use_batchnorm, args.use_identity_projection, args.projection_small_dim, args.projection_large_dim, args.pca_device,
                        args.incremental_pca_batch_size)
    else:
        print("Node classification task disabled, skipping dataset loading...")
    
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
        
        # Load training data for link prediction (keep on CPU to save GPU memory)
        with lp_tracker.time_operation('data_preparation'):
            lp_train_data_list, lp_train_split_idx_list = load_all_data_link(lp_train_datasets, device='cpu')
            print(f"[MEMORY_FIX] Loaded {len(lp_train_data_list)} training datasets on CPU (was loading to GPU before!)")
        
        # Record memory after loading link prediction data
        lp_tracker.record_memory() 
        after_lp_data = lp_tracker.get_memory_stats()
        print(f"[LP_TRACKER] After LP Data Loading - GPU: {after_lp_data['gpu_allocated']:.2f}GB, CPU: {after_lp_data['cpu_memory']:.2f}GB")
        
        # Process link prediction training data
        lp_train_context_data = []
        lp_train_masks = []
        lp_train_link_data_all = []
        
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
                
                # Clean GPU memory after PCA
                torch.cuda.empty_cache()
                
                # Prepare link data and select context
                link_data = prepare_link_data(data, split_idx)
            
            with lp_tracker.time_operation('context_selection'):
                context_data, train_mask = select_link_context(link_data['train'], args.context_k, args.context_neg_ratio,
                                                               args.remove_context_from_train)
            
            lp_train_context_data.append(context_data)
            lp_train_masks.append(train_mask)
            lp_train_link_data_all.append(link_data)
            lp_tracker.operation_counts['datasets_processed'] += 1
        
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
                
                # Clean GPU memory after PCA
                torch.cuda.empty_cache()
                
                # Prepare link data and select context
                link_data = prepare_link_data(data, split_idx)
            
            with lp_tracker.time_operation('context_selection'):
                context_data, _ = select_link_context(link_data['train'], args.context_k, args.context_neg_ratio, False)
            
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
        
        # Setup FUG environment BEFORE loading datasets
        setup_fug_environment(args)
        
        gc_train_datasets = args.gc_train_dataset.split(',')
        gc_test_datasets = args.gc_test_dataset.split(',')
        
        # Load training data for graph classification
        gc_train_data_list, gc_train_processed_data_list = load_all_graph_datasets(
            gc_train_datasets, device, pretraining_mode=True, context_k=args.context_graph_num
        )
        
        # Process graph classification training data
        if len(gc_train_data_list) > 0:
            gc_train_data_list, gc_train_processed_data_list, _ = process_datasets_for_models(
                gc_train_data_list, gc_train_processed_data_list, args, device
            )
            
            # Precompute task-filtered splits once for efficiency
            print("Precomputing task-filtered splits for training datasets...")
            for dataset_info in gc_train_processed_data_list:
                task_filtered_splits = create_task_filtered_datasets(
                    dataset_info['dataset'], 
                    dataset_info['split_idx']
                )
                dataset_info['task_filtered_splits'] = task_filtered_splits
        
        # Load test data for graph classification
        gc_test_data_list, gc_test_processed_data_list = load_all_graph_datasets(
            gc_test_datasets, device, context_k=args.context_graph_num
        )
        
        # Process graph classification test data
        if len(gc_test_data_list) > 0:
            gc_test_data_list, gc_test_processed_data_list, _ = process_datasets_for_models(
                gc_test_data_list, gc_test_processed_data_list, args, device, test_datasets=True
            )
            
            # Precompute task-filtered splits once for efficiency
            print("Precomputing task-filtered splits for test datasets...")
            for dataset_info in gc_test_processed_data_list:
                # Precompute both full splits and test-only splits
                task_filtered_splits_full = create_task_filtered_datasets(
                    dataset_info['dataset'], 
                    dataset_info['split_idx']
                )
                task_filtered_splits_test = create_task_filtered_datasets(
                    dataset_info['dataset'], 
                    dataset_info['split_idx'],
                    "test"
                )
                dataset_info['task_filtered_splits'] = task_filtered_splits_full
                dataset_info['task_filtered_splits_test_only'] = task_filtered_splits_test
    else:
        print("Graph classification task disabled, skipping dataset loading...")
    
    return {
        'nc_train': (nc_train_data_list, nc_train_split_idx_list),
        'nc_test': (nc_test_data_list, nc_test_split_idx_list),
        'lp_train': (lp_train_data_list, lp_train_split_idx_list, lp_train_context_data, lp_train_masks, lp_train_link_data_all),
        'lp_test': (lp_test_data_list, lp_test_split_idx_list, lp_test_context_data, lp_test_link_data_all),
        'gc_train': (gc_train_data_list, gc_train_processed_data_list),
        'gc_test': (gc_test_data_list, gc_test_processed_data_list)
    }


def joint_training_step(model, predictor, nc_data, lp_data, gc_data, optimizer, args, epoch, 
                       identity_projection=None):
    """
    Perform one joint training step combining all three tasks.
    
    This function calculates the loss for each task, combines them with weights,
    and performs a single backward pass and optimizer step.
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
    nc_data_list, nc_split_idx_list = nc_data
    (lp_data_list, lp_split_idx_list, lp_context_data, lp_masks, lp_link_data_all) = lp_data
    gc_data_list, gc_processed_data_list = gc_data
    
    # --- 1. Calculate Losses without Optimization ---
    
    # Node Classification Loss
    if hasattr(args, 'enable_nc') and args.enable_nc and nc_data_list is not None and len(nc_data_list) > 0 and args.lambda_nc > 0:
        nc_loss = train_all(model, nc_data_list, nc_split_idx_list, optimizer=optimizer, pred=predictor, 
                          batch_size=args.nc_batch_size, degree=False, 
                          orthogonal_push=args.orthogonal_push, normalize_class_h=args.normalize_class_h, 
                          clip_grad=args.clip_grad, rank=0, epoch=epoch, 
                          identity_projection=identity_projection, lambda_=args.lambda_nc)
        if nc_loss is not None:
            total_nc_loss = nc_loss
            nc_count = len(nc_data_list)
    
    # Link Prediction Loss
    if hasattr(args, 'enable_lp') and args.enable_lp and lp_data_list is not None and len(lp_data_list) > 0 and args.lambda_lp > 0:
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
                            mask_target_edges=args.mask_target_edges, degree=False, lambda_=args.lambda_lp
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
                    
                    # Force aggressive GPU cleanup
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # Record memory after cleanup
                    lp_tracker.record_memory()
                    
                    if lp_loss is not None:
                        lp_loss_sum += lp_loss
                        lp_count += 1
                        lp_tracker.operation_counts['training_steps'] += 1
        
        if lp_count > 0:
            total_lp_loss = lp_loss_sum / lp_count
    
    # Graph Classification Loss
    if hasattr(args, 'enable_gc') and args.enable_gc and len(gc_data_list) > 0 and args.lambda_gc > 0:
        gc_loss_sum = 0.0
        gc_dataset_count = 0
        
        for dataset_idx, dataset_info in enumerate(gc_processed_data_list):
            # Use precomputed task-filtered splits
            task_filtered_splits = dataset_info['task_filtered_splits']
            
            dataset_loss = 0.0
            dataset_tasks = 0
            
            # Train on each task separately using prefiltered data
            for task_idx, task_splits in task_filtered_splits.items():
                # Check if FUG mapping is present to use index tracking
                use_fug_tracking = 'fug_mapping' in dataset_info
                
                # Create task-specific data loaders
                task_data_loaders = create_data_loaders(
                    dataset_info['dataset'], 
                    task_splits,
                    batch_size=args.gc_batch_size,
                    shuffle=True,
                    task_idx=task_idx,
                    use_index_tracking=use_fug_tracking
                )
                
                # Train on this specific task
                task_loss = train_graph_classification_single_task(
                    model, predictor, dataset_info, task_data_loaders, optimizer, task_idx,
                    pooling_method=args.graph_pooling, device=device,
                    clip_grad=args.clip_grad, orthogonal_push=args.orthogonal_push,
                    normalize_class_h=args.normalize_class_h, identity_projection=identity_projection
                )
                
                dataset_loss += task_loss
                dataset_tasks += 1
            
            # Average loss across tasks for this dataset
            if dataset_tasks > 0:
                avg_dataset_loss = dataset_loss / dataset_tasks
                gc_loss_sum += avg_dataset_loss
                gc_dataset_count += 1
        
        if gc_dataset_count > 0:
            total_gc_loss = args.lambda_gc * (gc_loss_sum / gc_dataset_count)
            gc_count = gc_dataset_count
    
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


def evaluate_node_classification(model, predictor, nc_data, args, split='valid', identity_projection=None):
    """
    Evaluate node classification task only.
    
    Returns:
        Dictionary with node classification metrics
    """
    model.eval()
    predictor.eval()
    
    results = {}
    
    with torch.no_grad():
        nc_data_list, nc_split_idx_list = nc_data
        if len(nc_data_list) > 0:
            if split == 'test':
                # Use inductive evaluation for unseen datasets
                train_metrics, valid_metrics, test_metrics = test_all_induct(
                    model, predictor, nc_data_list, nc_split_idx_list, args.test_batch_size,
                    False, None, None, True, None, 0, identity_projection
                )
                results = {
                    'train': sum(train_metrics) / len(train_metrics) if train_metrics else 0.0,
                    'valid': sum(valid_metrics) / len(valid_metrics) if valid_metrics else 0.0,
                    'test': sum(test_metrics) / len(test_metrics) if test_metrics else 0.0,
                    'individual_test_metrics': test_metrics if test_metrics else []
                }
            else:
                # Use transductive evaluation for seen datasets
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
                            use_full_adj_for_test=(split == 'test')
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
                        
                        # Force aggressive GPU cleanup
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        
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


def evaluate_graph_classification_task(model, predictor, gc_data, args, split='valid', identity_projection=None):
    """
    Evaluate graph classification task only.
    
    Returns:
        Dictionary with graph classification metrics
    """
    model.eval()
    predictor.eval()
    
    results = {}
    
    with torch.no_grad():
        gc_data_list, gc_processed_data_list = gc_data
        
        if len(gc_data_list) > 0:
            all_dataset_results = []
            individual_results = []
            
            for dataset_idx, dataset_info in enumerate(gc_processed_data_list):
                dataset_name = dataset_info['dataset'].name if hasattr(dataset_info['dataset'], 'name') else f'gc_dataset_{dataset_idx}'
                
                # Use precomputed task-filtered datasets
                if split == 'test':
                    task_filtered_splits = dataset_info['task_filtered_splits_test_only']
                else:
                    task_filtered_splits = dataset_info['task_filtered_splits']
                
                # Evaluate each task separately and aggregate results
                task_results = []
                
                for task_idx, task_splits in task_filtered_splits.items():
                    # Check if FUG mapping is present to use index tracking
                    use_fug_tracking = 'fug_mapping' in dataset_info
                    
                    # Create task-specific data loaders for evaluation
                    if split == 'test':
                        # Only use test split for unseen datasets
                        test_only_splits = {'test': task_splits['test']}
                        task_eval_loaders = create_data_loaders(
                            dataset_info['dataset'], 
                            test_only_splits,
                            batch_size=args.gc_test_batch_size,
                            shuffle=False,
                            task_idx=task_idx,
                            use_index_tracking=use_fug_tracking
                        )
                    else:
                        # Use all splits for seen datasets
                        task_eval_loaders = create_data_loaders(
                            dataset_info['dataset'], 
                            task_splits,
                            batch_size=args.gc_test_batch_size,
                            shuffle=False,
                            task_idx=task_idx,
                            use_index_tracking=use_fug_tracking
                        )
                    
                    # Evaluate this specific task
                    task_eval_results = evaluate_graph_classification_single_task(
                        model, predictor, dataset_info, task_eval_loaders, task_idx,
                        pooling_method=args.graph_pooling, device=model.parameters().__next__().device,
                        normalize_class_h=args.normalize_class_h, dataset_name=dataset_name, identity_projection=identity_projection
                    )
                    
                    # Extract the appropriate split result
                    split_result = task_eval_results.get(split, 0.0)
                    task_results.append(split_result)
                
                # Aggregate results across tasks for this dataset
                if task_results:
                    dataset_avg = aggregate_task_metrics(task_results)
                    # For averaging across datasets, extract primary metric if multiple metrics
                    if isinstance(dataset_avg, dict):
                        primary_metric = dataset_avg.get('ap', dataset_avg.get('auc', 0.0))
                        all_dataset_results.append(primary_metric)
                        individual_results.append(dataset_avg)  # Keep full dict for individual results
                    else:
                        all_dataset_results.append(dataset_avg)
                        individual_results.append(dataset_avg)
                else:
                    all_dataset_results.append(0.0)
                    individual_results.append(0.0)
            
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
    
    return results


def joint_evaluation(model, predictor, nc_data, lp_data, gc_data, args, split='valid',
                    identity_projection=None):
    """
    Evaluate enabled tasks and return metrics.
    
    Returns:
        Dictionary with metrics for enabled tasks
    """    

    results = {'nc_metrics': {}, 'lp_metrics': {}, 'gc_metrics': {}}
    
    # Evaluate node classification
    if hasattr(args, 'enable_nc') and args.enable_nc and nc_data is not None and nc_data[0] is not None:
            nc_results = evaluate_node_classification(model, predictor, nc_data, args, split, identity_projection)
            results['nc_metrics'] = nc_results
    
    # Evaluate link prediction  
    if hasattr(args, 'enable_lp') and args.enable_lp and lp_data is not None and lp_data[0] is not None:
            lp_results = evaluate_link_prediction_task(model, predictor, lp_data, args, split, identity_projection)
            results['lp_metrics'] = lp_results
    
    # Evaluate graph classification
    if hasattr(args, 'enable_gc') and args.enable_gc and gc_data is not None and len(gc_data[0]) > 0:
            gc_results = evaluate_graph_classification_task(model, predictor, gc_data, args, split, identity_projection)
            results['gc_metrics'] = gc_results
    
    return results


def run_joint_training(args, device='cuda:0'):
    """
    Main joint training function.
    """
    # Declare global lp_tracker at the very beginning
    global lp_tracker
    
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
    
    # Load and preprocess all data
    with lp_tracker.time_operation('data_preparation') if lp_tracker else nullcontext():
        data_dict = load_and_preprocess_data(args, device)
    
    if lp_tracker:
        lp_tracker.record_memory()
        after_data_stats = lp_tracker.get_memory_stats()
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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 5, gamma=0.5)
    elif args.schedule == 'warmup':
        scheduler = get_cosine_schedule_with_warmup(optimizer, 
                                                  num_warmup_steps=args.epochs // 10, 
                                                  num_training_steps=args.epochs)
    else:
        scheduler = None
    
    print(f"Total parameters: {sum(p.numel() for p in parameters):,}")
    
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
        
        # Joint training step
        train_results = joint_training_step(
            model, predictor, data_dict['nc_train'], data_dict['lp_train'], data_dict['gc_train'],
            optimizer, args, epoch, identity_projection
        )
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Every epoch: Validation on seen datasets (training data) for early stopping
        seen_valid_results = joint_evaluation(
            model, predictor, data_dict['nc_train'], data_dict['lp_train'], data_dict['gc_train'],
            args, 'valid', identity_projection
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
                    model, predictor, data_dict['gc_test'], args, 'test', identity_projection
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
                lp_tracker.log_to_wandb()
            
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
            model, predictor, data_dict['gc_test'], args, 'test', identity_projection
        )
    
    # Print and log final results
    nc_test_metric = nc_test_results.get('test', 0.0)
    lp_test_metric = lp_test_results.get('test', 0.0)
    gc_test_metric = gc_test_results.get('test', 0.0)
    
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
        lp_tracker.log_to_wandb(prefix="final_lp_tracker")
        
        # Add final summary stats to wandb
        summary_stats = lp_tracker.get_summary_stats()
        for key, value in summary_stats.items():
            final_wandb_log[f'final_lp_summary/{key}'] = value
    
    wandb.log(final_wandb_log)
    
    # Return both average metrics and individual dataset metrics
    nc_individual = nc_test_results.get('individual_test_metrics', [])
    lp_individual = lp_test_results.get('individual_test_metrics', [])
    gc_individual = gc_test_results.get('individual_test_metrics', [])
    
    return nc_test_metric, lp_test_metric, gc_test_metric, nc_individual, lp_individual, gc_individual


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
    
    # GPU setup
    try:
        gpu_ids = parse_gpu_spec(args.gpu)
        validate_gpu_availability(gpu_ids)
        print_gpu_info(gpu_ids)
        setup_cuda_visible_devices(gpu_ids)
        
        device = torch.device(f'cuda:{gpu_ids[0]}' if torch.cuda.is_available() else 'cpu')
        
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
        
        # Set different seed for each run
        torch.manual_seed(args.seed + run)
        
        nc_result, lp_result, gc_result, nc_individual, lp_individual, gc_individual = run_joint_training(args, device)
        all_nc_results.append(nc_result)
        all_lp_results.append(lp_result)
        all_gc_results.append(gc_result)
        all_nc_individual_results.append(nc_individual)
        all_lp_individual_results.append(lp_individual)
        all_gc_individual_results.append(gc_individual)
    
    # Aggregate results
    avg_nc = sum(all_nc_results) / len(all_nc_results)
    avg_lp = sum(all_lp_results) / len(all_lp_results)
    avg_gc = sum(all_gc_results) / len(all_gc_results)
    
    # Aggregate individual dataset metrics across runs
    avg_nc_individual = []
    avg_lp_individual = []
    avg_gc_individual = []
    
    if all_nc_individual_results and all_nc_individual_results[0]:
        num_nc_datasets = len(all_nc_individual_results[0])
        for dataset_idx in range(num_nc_datasets):
            dataset_metrics = [run_results[dataset_idx] for run_results in all_nc_individual_results if len(run_results) > dataset_idx]
            avg_nc_individual.append(sum(dataset_metrics) / len(dataset_metrics))
    
    if all_lp_individual_results and all_lp_individual_results[0]:
        num_lp_datasets = len(all_lp_individual_results[0])
        for dataset_idx in range(num_lp_datasets):
            dataset_metrics = [run_results[dataset_idx] for run_results in all_lp_individual_results if len(run_results) > dataset_idx]
            avg_lp_individual.append(sum(dataset_metrics) / len(dataset_metrics))
    
    if all_gc_individual_results and all_gc_individual_results[0]:
        num_gc_datasets = len(all_gc_individual_results[0])
        for dataset_idx in range(num_gc_datasets):
            dataset_metrics = [run_results[dataset_idx] for run_results in all_gc_individual_results if len(run_results) > dataset_idx]
            avg_gc_individual.append(sum(dataset_metrics) / len(dataset_metrics))
    
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
