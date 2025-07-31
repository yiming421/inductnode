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

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Core imports - reuse from existing scripts
from src.model import PureGCN_v1, PureGCN, PFNPredictorNodeCls, GCN, IdentityProjection
from src.data import load_all_data, load_all_data_train
from src.data_link import load_all_data_link
from src.data_utils import process_data, prepare_link_data, select_link_context, process_link_data
from src.engine import train_all, test_all, test_all_induct  # Node classification engines
from src.engine_link_pred import train_link_prediction, evaluate_link_prediction  # Link prediction engines
from src.gpu_utils import parse_gpu_spec, setup_cuda_visible_devices, get_effective_world_size, validate_gpu_availability, print_gpu_info
from transformers import get_cosine_schedule_with_warmup

# Logging and monitoring
from src.logger import TrainingLogger, LogLevel

from src.config import parse_joint_training_args

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
    Load and preprocess data for both tasks.
    Returns processed datasets for node classification and link prediction.
    """
    print("\n=== Loading and Preprocessing Data ===")
    
    # === Node Classification Data ===
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
                    args.use_batchnorm, args.use_identity_projection, args.projection_small_dim, args.projection_large_dim)

    # Load test data for node classification
    nc_test_data_list, nc_test_split_idx_list = load_all_data(nc_test_datasets)
    
    # Process node classification test data
    for data, split_idx in zip(nc_test_data_list, nc_test_split_idx_list):
        data.x = data.x.to(device)
        data.adj_t = data.adj_t.to(device)
        data.y = data.y.to(device)
        process_data(data, split_idx, args.hidden, args.context_num, False, args.use_full_pca, 
                    args.normalize_data, False, 32, 0, args.padding_strategy, 
                    args.use_batchnorm, args.use_identity_projection, args.projection_small_dim, args.projection_large_dim)
    
    # === Link Prediction Data ===
    print("Loading link prediction datasets...")
    lp_train_datasets = args.lp_train_dataset.split(',')
    lp_test_datasets = args.lp_test_dataset.split(',')
    
    # Load training data for link prediction
    lp_train_data_list, lp_train_split_idx_list = load_all_data_link(lp_train_datasets, device=device)
    
    # Process link prediction training data
    lp_train_context_data = []
    lp_train_masks = []
    lp_train_link_data_all = []
    
    for i, (data, split_idx) in enumerate(zip(lp_train_data_list, lp_train_split_idx_list)):
        # Process link-specific data (no general process_data needed for link prediction)
        process_link_data(data, args, rank=0)
        
        # Prepare link data and select context
        link_data = prepare_link_data(data, split_idx)
        context_data, train_mask = select_link_context(link_data['train'], args.context_k, args.context_neg_ratio,
                                                       args.remove_context_from_train)
        lp_train_context_data.append(context_data)
        lp_train_masks.append(train_mask)
        lp_train_link_data_all.append(link_data)
    
    # Load test data for link prediction
    lp_test_data_list, lp_test_split_idx_list = load_all_data_link(lp_test_datasets, device=device)
    
    # Process link prediction test data
    lp_test_context_data = []
    lp_test_link_data_all = []
    
    for i, (data, split_idx) in enumerate(zip(lp_test_data_list, lp_test_split_idx_list)):
        # Process link-specific data (no general process_data needed for link prediction)
        process_link_data(data, args, rank=0)
        
        # Prepare link data and select context
        link_data = prepare_link_data(data, split_idx)
        context_data, _ = select_link_context(link_data['train'], args.context_k, args.context_neg_ratio, False)
        
        lp_test_context_data.append(context_data)
        lp_test_link_data_all.append(link_data)
    
    return {
        'nc_train': (nc_train_data_list, nc_train_split_idx_list),
        'nc_test': (nc_test_data_list, nc_test_split_idx_list),
        'lp_train': (lp_train_data_list, lp_train_split_idx_list, lp_train_context_data, lp_train_masks, lp_train_link_data_all),
        'lp_test': (lp_test_data_list, lp_test_split_idx_list, lp_test_context_data, lp_test_link_data_all)
    }


def joint_training_step(model, predictor, nc_data, lp_data, optimizer, args, epoch, 
                       identity_projection=None):
    """
    Perform one joint training step combining both tasks.
    
    This function calculates the loss for each task, combines them with weights,
    and performs a single backward pass and optimizer step.
    """
    model.train()
    predictor.train()
    
    total_nc_loss = torch.tensor(0.0, device=optimizer.param_groups[0]['params'][0].device)
    total_lp_loss = torch.tensor(0.0, device=optimizer.param_groups[0]['params'][0].device)
    nc_count = 0
    lp_count = 0
    
    # Unpack data
    nc_data_list, nc_split_idx_list = nc_data
    (lp_data_list, lp_split_idx_list, lp_context_data, lp_masks, lp_link_data_all) = lp_data
    
    # --- 1. Calculate Losses without Optimization ---
    
    # Node Classification Loss
    if len(nc_data_list) > 0 and args.lambda_nc > 0:
        nc_loss = train_all(model, nc_data_list, nc_split_idx_list, optimizer=optimizer, pred=predictor, 
                          batch_size=args.nc_batch_size, degree=False, 
                          orthogonal_push=args.orthogonal_push, normalize_class_h=args.normalize_class_h, 
                          clip_grad=args.clip_grad, rank=0, epoch=epoch, 
                          identity_projection=identity_projection, lambda_=args.lambda_nc)
        if nc_loss is not None:
            total_nc_loss = nc_loss
            nc_count = len(nc_data_list)
    
    # Link Prediction Loss
    if len(lp_data_list) > 0 and args.lambda_lp > 0:
        lp_loss_sum = 0.0
        for i, (data, split_idx) in enumerate(zip(lp_data_list, lp_split_idx_list)):
            link_data_all = lp_link_data_all[i]
            context_data = lp_context_data[i]
            train_mask = lp_masks[i]
            
            if 'train' in link_data_all and link_data_all['train']['edge_pairs'].size(0) > 0:
                # By passing optimizer=None, we signal the function to only return the loss.
                lp_loss = train_link_prediction(
                    model, predictor, data, link_data_all['train'], context_data, train_mask,
                    optimizer=optimizer, batch_size=args.lp_batch_size, 
                    identity_projection=identity_projection, 
                    clip_grad=args.clip_grad, rank=0, orthogonal_push=args.orthogonal_push, 
                    normalize_class_h=args.normalize_class_h, epoch=epoch, 
                    mask_target_edges=args.mask_target_edges, degree=False, lambda_=args.lambda_lp
                )
                if lp_loss is not None:
                    lp_loss_sum += lp_loss
                    lp_count += 1
        
        if lp_count > 0:
            total_lp_loss = lp_loss_sum / lp_count
    
    return {
        'nc_loss': total_nc_loss,
        'lp_loss': total_lp_loss,
        'combined_loss': total_lp_loss + total_nc_loss,
        'nc_count': nc_count,
        'lp_count': lp_count
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
    model.eval()
    predictor.eval()
    
    results = {}
    
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
                    lp_results = evaluate_link_prediction(
                        model, predictor, data, link_data_all[split_key], context_data,
                        args.test_batch_size, None, None, None, identity_projection,
                        0, True, degree=False, k_values=[20, 50, 100]
                    )
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


def joint_evaluation(model, predictor, nc_data, lp_data, args, split='valid',
                    identity_projection=None):
    """
    Evaluate both tasks and return metrics.
    Both nc_data and lp_data must be provided (not None).
    
    Returns:
        Dictionary with metrics for both tasks
    """
    results = {'nc_metrics': {}, 'lp_metrics': {}}
    
    # Evaluate node classification
    if nc_data is not None:
        nc_results = evaluate_node_classification(model, predictor, nc_data, args, split, identity_projection)
        results['nc_metrics'] = nc_results
    
    # Evaluate link prediction  
    if lp_data is not None:
        lp_results = evaluate_link_prediction_task(model, predictor, lp_data, args, split, identity_projection)
        results['lp_metrics'] = lp_results
    
    return results


def run_joint_training(args, device='cuda:0'):
    """
    Main joint training function.
    """
    print(f"\n=== Starting Joint Training ===")
    print(f"Device: {device}")
    print(f"Node Classification Weight (lambda_nc): {args.lambda_nc}")
    print(f"Link Prediction Weight (lambda_lp): {args.lambda_lp}")
    
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
    data_dict = load_and_preprocess_data(args, device)
    
    # Create unified model
    model, predictor, identity_projection = create_unified_model(
        args, args.hidden, device)
    
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
    
    # Training loop
    best_valid_score = 0.0
    best_epoch = 0
    best_model_state = None
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Joint training step
        train_results = joint_training_step(
            model, predictor, data_dict['nc_train'], data_dict['lp_train'],
            optimizer, args, epoch, identity_projection
        )
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Every epoch: Validation on seen datasets (training data) for early stopping
        seen_valid_results = joint_evaluation(
            model, predictor, data_dict['nc_train'], data_dict['lp_train'],
            args, 'valid', identity_projection
        )
        
        # Compute combined validation score on seen datasets
        nc_valid_seen = seen_valid_results['nc_metrics'].get('valid', 0.0)
        lp_valid_seen = seen_valid_results['lp_metrics'].get('valid', 0.0)
        combined_valid_seen = nc_valid_seen + lp_valid_seen
        
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
            nc_unseen_results = evaluate_node_classification(
                model, predictor, data_dict['nc_test'], args, 'test', identity_projection
            )
            lp_unseen_results = evaluate_link_prediction_task(
                model, predictor, data_dict['lp_test'], args, 'test', identity_projection
            )
            
            nc_test_unseen = nc_unseen_results.get('test', 0.0)
            lp_test_unseen = lp_unseen_results.get('test', 0.0)
            combined_test_unseen = nc_test_unseen + lp_test_unseen
            
            # Get individual dataset metrics
            nc_individual = nc_unseen_results.get('individual_test_metrics', [])
            lp_individual = lp_unseen_results.get('individual_test_metrics', [])
            
            # Get dataset names for logging
            nc_test_datasets = args.nc_test_dataset.split(',')
            lp_test_datasets = args.lp_test_dataset.split(',')
            
            unseen_metrics = {
                'test_unseen/nc_metric': nc_test_unseen,
                'test_unseen/lp_metric': lp_test_unseen,
                'test_unseen/combined_score': combined_test_unseen
            }
            
            # Add individual dataset metrics to wandb
            for i, (dataset_name, metric) in enumerate(zip(nc_test_datasets, nc_individual)):
                unseen_metrics[f'test_unseen/nc_{dataset_name.strip()}'] = metric
            
            for i, (dataset_name, metric) in enumerate(zip(lp_test_datasets, lp_individual)):
                unseen_metrics[f'test_unseen/lp_{dataset_name.strip()}'] = metric
            
            # Print individual dataset performance
            print(f"\n--- Unseen Dataset Performance (Epoch {epoch}) ---")
            print(f"Node Classification (Average): {nc_test_unseen:.4f}")
            for i, (dataset_name, metric) in enumerate(zip(nc_test_datasets, nc_individual)):
                print(f"  {dataset_name.strip()}: {metric:.4f}")
            
            print(f"Link Prediction (Average): {lp_test_unseen:.4f}")
            for i, (dataset_name, metric) in enumerate(zip(lp_test_datasets, lp_individual)):
                print(f"  {dataset_name.strip()}: {metric:.4f}")
            print("-----------------------------------------------")
        
        # Logging
        if epoch % args.log_interval == 0:
            epoch_time = time.time() - start_time
            log_message = (f"Epoch {epoch:3d} | Time: {epoch_time:.2f}s | "
                         f"NC Loss: {train_results['nc_loss']:.4f} | "
                         f"LP Loss: {train_results['lp_loss']:.4f} | "
                         f"Combined: {train_results['combined_loss']:.4f} | "
                         f"NC Valid (Seen): {nc_valid_seen:.4f} | LP Valid (Seen): {lp_valid_seen:.4f}")
            
            if unseen_metrics:
                log_message += f" | NC Test (Unseen): {unseen_metrics['test_unseen/nc_metric']:.4f} | LP Test (Unseen): {unseen_metrics['test_unseen/lp_metric']:.4f}"
            
            print(log_message)
            logger.info(log_message, LogLevel.INFO)
            
            # Log to wandb
            wandb_log = {
                'epoch': epoch,
                'train/nc_loss': train_results['nc_loss'],
                'train/lp_loss': train_results['lp_loss'],
                'train/combined_loss': train_results['combined_loss'],
                'valid_seen/nc_metric': nc_valid_seen,
                'valid_seen/lp_metric': lp_valid_seen,
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
    
    # Node Classification Test on unseen datasets
    nc_test_results = evaluate_node_classification(
        model, predictor, data_dict['nc_test'], args, 'test', identity_projection
    )
    
    # Link Prediction Test on unseen datasets
    lp_test_results = evaluate_link_prediction_task(
        model, predictor, data_dict['lp_test'], args, 'test', identity_projection
    )
    
    # Print and log final results
    nc_test_metric = nc_test_results.get('test', 0.0)
    lp_test_metric = lp_test_results.get('test', 0.0)
    
    final_results_msg = (f"Node Classification Test: {nc_test_metric:.4f}\n"
                        f"Link Prediction Test: {lp_test_metric:.4f}")
    print(final_results_msg)
    logger.info(final_results_msg, LogLevel.INFO)
    
    # Final wandb log
    wandb.log({
        'test/nc_metric': nc_test_metric,
        'test/lp_metric': lp_test_metric,
        'test/combined_score': nc_test_metric + lp_test_metric,
        'best_epoch': best_epoch,
        'best_valid_score': best_valid_score
    })
    
    # Return both average metrics and individual dataset metrics
    nc_individual = nc_test_results.get('individual_test_metrics', [])
    lp_individual = lp_test_results.get('individual_test_metrics', [])
    
    return nc_test_metric, lp_test_metric, nc_individual, lp_individual


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
    all_nc_individual_results = []
    all_lp_individual_results = []
    
    for run in range(args.runs):
        print(f"\n{'='*50}")
        print(f"Run {run + 1}/{args.runs}")
        print(f"{'='*50}")
        
        # Set different seed for each run
        torch.manual_seed(args.seed + run)
        
        nc_result, lp_result, nc_individual, lp_individual = run_joint_training(args, device)
        all_nc_results.append(nc_result)
        all_lp_results.append(lp_result)
        all_nc_individual_results.append(nc_individual)
        all_lp_individual_results.append(lp_individual)
    
    # Aggregate results
    avg_nc = sum(all_nc_results) / len(all_nc_results)
    avg_lp = sum(all_lp_results) / len(all_lp_results)
    
    # Aggregate individual dataset metrics across runs
    avg_nc_individual = []
    avg_lp_individual = []
    
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
    
    print(f"\n{'='*50}")
    print(f"Final Results (Average over {args.runs} runs)")
    print(f"{'='*50}")
    print(f"Node Classification: {avg_nc:.4f} ± {torch.std(torch.tensor(all_nc_results)):.4f}")
    print(f"Link Prediction: {avg_lp:.4f} ± {torch.std(torch.tensor(all_lp_results)):.4f}")
    
    # Final sweep metric (for hyperparameter optimization)
    sweep_metric = args.lambda_nc * avg_nc + args.lambda_lp * avg_lp
    print(f"Combined Score: {sweep_metric:.4f}")
    
    # Get dataset names
    nc_test_datasets = args.nc_test_dataset.split(',')
    lp_test_datasets = args.lp_test_dataset.split(',')
    
    # Log final aggregated results
    wandb.init(project='inductnode-joint-summary')
    final_log = {
        'final/avg_nc_metric': avg_nc,
        'final/avg_lp_metric': avg_lp,
        'final/combined_score': sweep_metric,
        'runs': args.runs
    }
    
    # Add individual dataset metrics
    for dataset_name, metric in zip(nc_test_datasets, avg_nc_individual):
        final_log[f'final/nc_{dataset_name.strip()}'] = metric
    
    for dataset_name, metric in zip(lp_test_datasets, avg_lp_individual):
        final_log[f'final/lp_{dataset_name.strip()}'] = metric
    
    wandb.log(final_log)
    
    print("\n🎉 Joint training completed successfully!")

if __name__ == '__main__':
    main()
