#!/usr/bin/env python3
"""
Graph Classification using PFN Predictor
Reuses the existing PFNPredictorNodeCls by treating pooled graph embeddings as node embeddings.
"""

import os
import sys
import time
import torch
import wandb
import numpy as np
from transformers import get_cosine_schedule_with_warmup

# Add the project's root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.config import parse_graph_classification_args
from src.data_graph import load_all_graph_datasets, process_graph_features
from src.model import PureGCN_v1, PureGCN, GCN, PFNPredictorNodeCls, IdentityProjection
from src.engine_graph import train_and_evaluate_graph_classification

def create_model_and_predictor(args, device):
    """
    Create GNN model and PFN predictor for graph classification.
    
    Args:
        args: Command line arguments
        device (str): Device for computation
        
    Returns:
        tuple: (model, predictor)
    """
    # Create GNN model
    hidden = args.projection_large_dim if args.use_identity_projection else args.hidden
    if args.model == 'PureGCN':
        model = PureGCN(args.num_layers)
    elif args.model == 'PureGCN_v1':
        model = PureGCN_v1(
            hidden, args.num_layers, hidden, args.dp, args.norm, 
            args.res, args.relu, args.gnn_norm_affine
        )
    elif args.model == 'GCN':
        model = GCN(
            hidden, hidden, args.norm, args.relu, args.num_layers, 
            args.dp, args.multilayer, args.use_gin, args.res, args.gnn_norm_affine
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # Create PFN predictor
    if args.predictor == 'PFN':
        predictor = PFNPredictorNodeCls(
            hidden, args.nhead, args.transformer_layers, args.mlp_layers, 
            args.dp, args.norm, args.seperate, args.degree, None, None, 
            args.sim, args.padding, args.mlp_norm_affine, args.normalize_class_h
        )
    else:
        raise ValueError(f"Unsupported predictor: {args.predictor}")
    
    # Create identity projection if needed
    identity_projection = None
    if args.use_identity_projection:
        identity_projection = IdentityProjection(args.projection_small_dim, args.projection_large_dim)
        identity_projection = identity_projection.to(device)
    
    # Move models to device
    model = model.to(device)
    predictor = predictor.to(device)
    
    return model, predictor, identity_projection


def setup_optimizer_and_scheduler(model, predictor, args, identity_projection=None):
    """
    Setup optimizer and learning rate scheduler.
    
    Args:
        model: GNN model
        predictor: PFN predictor
        args: Command line arguments
        
    Returns:
        tuple: (optimizer, scheduler)
    """
    # Collect all parameters
    parameters = list(model.parameters()) + list(predictor.parameters())
    if identity_projection is not None:
        parameters += list(identity_projection.parameters())
    
    # Create optimizer
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
    
    # Create scheduler
    if args.schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.schedule == 'step':
        step_size = args.epochs // 5
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    elif args.schedule == 'warmup':
        warmup_steps = args.epochs // 10
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=args.epochs
        )
    else:
        scheduler = None
    
    return optimizer, scheduler


def process_datasets_for_models(datasets, processed_data_list, args, device, test_datasets=False):
    """
    Process datasets to handle feature dimensions and create any necessary dummy features.
    
    Args:
        datasets: List of PyTorch Geometric datasets
        processed_data_list: List of processed data information
        args: Command line arguments
        device: Device for computation
        
    Returns:
        tuple: (processed_datasets, processed_data_list, final_num_features)
    """
    processed_datasets = []
    final_num_features = args.hidden
    
    for dataset, dataset_info in zip(datasets, processed_data_list):
        # Extract dataset name for GPSE lookup
        dataset_name = dataset_info.get('name', None)
        
        # Process features using PCA and padding
        pcba_context_only_pca = getattr(args, 'pcba_context_only_pca', False)
        if test_datasets:
            processing_info = process_graph_features(
                dataset, args.hidden, device, 
                args.use_identity_projection, args.projection_small_dim, args.projection_large_dim,
                args.use_full_pca, args.sign_normalize, args.normalize_data,
                args.padding_strategy, args.use_batchnorm,
                args.use_gpse, args.gpse_path, dataset_name,
                args.pca_device, args.incremental_pca_batch_size, args.pca_sample_threshold,
                dataset_info, pcba_context_only_pca, use_pca_cache=args.use_pca_cache, pca_cache_dir=args.pca_cache_dir
            )
        else:
            processing_info = process_graph_features(
                dataset, args.hidden, device, 
                args.use_identity_projection, args.projection_small_dim, args.projection_large_dim,
                args.use_full_pca, args.sign_normalize, args.normalize_data,
                args.padding_strategy, args.use_batchnorm,
                args.use_gpse, args.gpse_path, dataset_name,
                args.pca_device, args.incremental_pca_batch_size, args.pca_sample_threshold,
                dataset_info, pcba_context_only_pca, use_pca_cache=args.use_pca_cache, pca_cache_dir=args.pca_cache_dir
            )
        
        processed_datasets.append(dataset)
        
        # Update dataset info with processing information
        dataset_info.update(processing_info)
    
    return processed_datasets, processed_data_list, final_num_features


def run_single_experiment(args, device='cuda:0'):
    """
    Run a single graph classification experiment.
    
    Args:
        args: Command line arguments
        device (str): Device for computation
        
    Returns:
        dict: Experiment results
    """
    print(f"Running graph classification on device: {device}")
    print(f"Training datasets: {args.train_dataset}")
    print(f"Test datasets: {args.test_dataset}")
    print(f"Graph pooling method: {args.graph_pooling}")
    
    # If user wants to source datasets from copied TAGDataset (e.g., molecular datasets preprocessed by TAGLAS),
    # ensure the path is visible (already copied to inductnode/TAGDataset)
    if args.use_tag_dataset:
        tag_path = os.path.join(project_root, 'TAGDataset')
        if os.path.isdir(tag_path):
            print(f"Using TAGDataset at {tag_path} for dataset roots (TSGFM format)")
            # Prepend to sys.path so that any relative imports inside loaders can find resources if needed
            if tag_path not in sys.path:
                sys.path.insert(0, tag_path)
            # Override default dataset root environment variable consumed by data loaders if they check it
            os.environ['TAG_DATASET_ROOT'] = tag_path
        else:
            print(f"Warning: --use_tag_dataset specified but {tag_path} not found.")
    
    # Configure OGB FUG embeddings if requested
    if args.use_ogb_fug:
        print(f"Enabling OGB FUG embeddings - FUG root: {args.fug_root}, OGB root: {args.ogb_root}")
        os.environ['USE_FUG_EMB'] = '1'
        os.environ['FUG_EMB_ROOT'] = args.fug_root
        os.environ['OGB_ROOT'] = args.ogb_root
    
    # Load training datasets
    train_dataset_names = args.train_dataset.split(',')
    # Use pretraining mode if we have separate test datasets (maximize training data utilization)
    pretraining_mode = args.test_dataset and args.test_dataset != args.train_dataset
    if pretraining_mode:
        print("Pretraining mode enabled: optimizing train/val splits for maximum data utilization")
    train_datasets, train_processed_data_list = load_all_graph_datasets(
        train_dataset_names, device, pretraining_mode, args.context_k, embedding_family=args.embedding_family
    )
    
    if len(train_datasets) == 0:
        raise ValueError("No training datasets loaded successfully")
    
    # Process datasets for model compatibility
    train_datasets, train_processed_data_list, num_features = process_datasets_for_models(
        train_datasets, train_processed_data_list, args, device
    )

    # Load and process test datasets if specified
    test_datasets = None
    test_processed_data_list = None
    if args.test_dataset:
        print(f"Loading test datasets: {args.test_dataset}")
        test_dataset_names = args.test_dataset.split(',')
        # Use context sampling for memory efficiency (sample minimal context from train split)
        test_datasets, test_processed_data_list = load_all_graph_datasets(
            test_dataset_names, device, context_k=args.context_k, embedding_family=args.embedding_family
        )
        
        if len(test_datasets) > 0:
            # Process test datasets
            test_datasets, test_processed_data_list, _ = process_datasets_for_models(
                test_datasets, test_processed_data_list, args, device, True
            )
            print(f"Loaded {len(test_datasets)} test datasets")
    
    # Create model and predictor
    model, predictor, identity_projection = create_model_and_predictor(args, device)

    # Print model information
    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in predictor.parameters())
    if identity_projection is not None:
        total_params += sum(p.numel() for p in identity_projection.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(model, predictor, args, identity_projection)
    
    # Train and evaluate
    start_time = time.time()
    results = train_and_evaluate_graph_classification(
        model, predictor, train_datasets, train_processed_data_list, args,
        optimizer, scheduler, device, test_datasets, test_processed_data_list, identity_projection
    )
    total_time = time.time() - start_time
    
    print(f"\nTraining completed in {total_time:.2f} seconds")
    
    return results

def main():
    """Main function for graph classification."""
    args = parse_graph_classification_args()
    
    # Setup device
    if torch.cuda.is_available():
        if args.gpu == 'auto':
            device = 'cuda:0'
        else:
            device = f'cuda:{args.gpu}'
    else:
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    print(f"Using device: {device}")
    
    # Initialize wandb
    if args.sweep:
        run = wandb.init(project='inductnode-graph')
        # Update args with sweep configuration if needed
        config = wandb.config
        for key in config.keys():
            if hasattr(args, key):
                setattr(args, key, config[key])
    else:
        run = wandb.init(project='inductnode-graph', config=args)

    # Run multiple experiments
    all_results = []
    for run in range(args.runs):
        print(f"\n{'='*60}")
        print(f"Starting run {run + 1}/{args.runs}")
        print(f"{'='*60}")
        
        # Set random seed for reproducibility
        torch.manual_seed(42 + run)
        np.random.seed(42 + run)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42 + run)
        
        try:
            results = run_single_experiment(args, device)
            all_results.append(results)
            
            print(f"\nRun {run + 1} Results:")
            for dataset_name, dataset_results in results.items():
                result_parts = []
                for split_name in ['train', 'val', 'test']:
                    if split_name in dataset_results:
                        result_parts.append(f"{split_name.capitalize()}={dataset_results[split_name]:.4f}")
                print(f"  {dataset_name}: {', '.join(result_parts)}")
        
        except Exception as e:
            print(f"Run {run + 1} failed with error: {e}")
            print(f"Error type: {type(e).__name__}")
            import traceback
            print("Full traceback:")
            traceback.print_exc()
            print("-" * 60)
            continue
    
    # Aggregate results across runs
    if len(all_results) > 0:
        print(f"\n{'='*60}")
        print("FINAL RESULTS ACROSS ALL RUNS")
        print(f"{'='*60}")
        
        # Aggregate results by dataset
        dataset_names = set()
        for result in all_results:
            dataset_names.update(result.keys())
        
        final_metrics = {}
        for dataset_name in dataset_names:
            dataset_results = [r[dataset_name] for r in all_results if dataset_name in r]
            if dataset_results:
                # Check which splits exist in the results
                sample_result = dataset_results[0]
                available_splits = list(sample_result.keys())
                
                print(f"{dataset_name}:")
                
                # Process each available split
                for split_name in ['train', 'val', 'test']:
                    if split_name in available_splits:
                        split_accs = [r[split_name] for r in dataset_results]
                        
                        final_metrics[f"{dataset_name}_{split_name}_mean"] = np.mean(split_accs)
                        final_metrics[f"{dataset_name}_{split_name}_std"] = np.std(split_accs)
                        
                        split_label = split_name.capitalize()
                        print(f"  {split_label}: {np.mean(split_accs):.4f} ± {np.std(split_accs):.4f}")
        
        # Calculate overall average test accuracy for sweep optimization
        all_test_accs = []
        for result in all_results:
            for dataset_name, dataset_results in result.items():
                if not dataset_name.startswith('test_'):  # Only include main datasets, not test datasets
                    if 'test' in dataset_results:
                        all_test_accs.append(dataset_results['test'])
        
        if all_test_accs:
            avg_test_acc = np.mean(all_test_accs)
            final_metrics['avg_test_metric'] = avg_test_acc
            print(f"\nOverall average test metric: {avg_test_acc:.4f}")
        
        # Log final metrics to wandb (don't specify step to avoid interfering with training steps)
        wandb.log(final_metrics)

    else:
        print("No successful runs completed.")
    
    print("\n🎉 Graph classification completed!")


if __name__ == '__main__':
    main()
