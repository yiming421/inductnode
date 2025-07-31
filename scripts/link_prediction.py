import sys
import os
import torch
import numpy as np
import wandb
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import copy
import time

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data_link import load_all_data_link
from src.data_utils import prepare_link_data, select_link_context, process_link_data
from src.checkpoint_utils import (override_args_from_checkpoint, 
                                create_model_from_args, recreate_model_from_checkpoint)
from src.ddp_utils import setup_ddp, cleanup_ddp
from src.config import parse_link_prediction_args
from src.engine_link_pred import train_link_prediction, evaluate_link_prediction
from src.gpu_utils import (
    parse_gpu_spec, 
    setup_cuda_visible_devices, 
    get_effective_world_size,
    validate_gpu_availability,
    print_gpu_info
)

from src.ddp_gpu_monitor import create_ddp_monitor
from src.ddp_monitor import create_ddp_process_monitor
from transformers import get_cosine_schedule_with_warmup

# Import logging system and waste analyzer
from src.logger import TrainingLogger, LogLevel
sys.path.append('/home/maweishuo/inductnode')
from waste_analyzer import DistributedWasteAnalyzer


def run_ddp_lp(rank, world_size, args, results_dict):
    """Main execution function for link prediction, supports DDP."""
    # Initialize logging system
    logger = TrainingLogger(
        rank=rank, 
        world_size=world_size,
        log_level=args.log_level,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        analysis_interval=args.analysis_interval
    )
    
    # Initialize process monitor for both DDP and single GPU modes
    process_monitor = None
    
    if world_size > 1:
        setup_ddp(rank, world_size, args.port)
        if rank == 0:
            wandb.init(project='inductlink', config=args)

        # Create and start process monitor for DDP
        process_monitor = create_ddp_process_monitor(rank, world_size, timeout_minutes=60)
        process_monitor.start_monitoring()
    else:
        # Single GPU mode - initialize wandb if not already initialized
        if not wandb.run and not getattr(args, 'sweep', False):
            wandb.init(project='inductlink', config=args)

    # Handle GPU device selection properly
    if world_size == 1:
        # Single GPU mode - use specified GPU or fallback to rank
        if hasattr(args, 'gpu') and args.gpu != 'auto' and args.gpu.isdigit():
            device = f'cuda:{args.gpu}'
        else:
            device = f'cuda:{rank}'
    else:
        # DDP mode - always use rank-based device assignment
        device = f'cuda:{rank}'
    
    # Initialize systems
    waste_analyzer = DistributedWasteAnalyzer(rank, world_size)
    waste_analyzer.measure_memory_usage("baseline_before_data_loading")
    
    # Setup logging
    logger.setup_start()
    logger.setup_device(device, {'train': args.batch_size, 'test': args.test_batch_size})
        
    # Correct heterogeneous GPU memory management - use the dedicated module
    try:
        # Initialize enhanced GPU monitoring
        torch_device = torch.device(device)
        gpu_monitor = create_ddp_monitor(rank, world_size, torch_device)
        gpu_monitor.start_monitoring(interval=10.0)  # Monitor every 10 seconds
            
    except Exception as e:
        logger.warning(f"GPU memory check failed, using original batch sizes: {e}")
    
    # --- 1. Load and process training datasets ---
    train_dataset_names = args.train_dataset.split(',')
    test_dataset_names = args.test_dataset.split(',')
    logger.setup_datasets(train_dataset_names, test_dataset_names)
    
    # Load training datasets
    train_data_list, train_split_idx_list = load_all_data_link(train_dataset_names, device=device, is_pretraining=args.use_test_split_for_pretraining)

    # Process training datasets
    logger.progress("Processing training datasets")
    
    for i, (data, split_idx) in enumerate(zip(train_data_list, train_split_idx_list)):
        try:
            process_link_data(data, args, rank=rank)
            logger.dataset_processing(data.name, i, len(train_data_list), success=True)
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.oom_error(data.name, args.batch_size)
                raise e
            else:
                logger.dataset_error(data.name, str(e))
                raise e
    
    # Determine the correct input dimension after data processing
    processed_input_dim = args.hidden  # Default
    for data, _ in zip(train_data_list, train_split_idx_list):
        if hasattr(data, 'needs_identity_projection') and data.needs_identity_projection:
            processed_input_dim = args.projection_large_dim
            break
        elif hasattr(data, 'needs_projection') and data.needs_projection:
            processed_input_dim = args.hidden  # projector handles this
            break
        else:
            # Use the actual feature dimension after processing
            processed_input_dim = data.x.size(1)

    # --- 2. Create or load model ---
    
    if args.use_pretrained_model:
        logger.info("Loading pretrained model...")
        model, predictor, att, mlp, projector, identity_projection, model_args = recreate_model_from_checkpoint(
            args.load_checkpoint, processed_input_dim, device
        )
        args = override_args_from_checkpoint(args, model_args, rank)
    else:
        logger.info("Creating new model from scratch...")
        model, predictor, att, mlp, projector, identity_projection = create_model_from_args(
            args, processed_input_dim, device
        )
    
    logger.success("Model and Predictor created successfully!")

    # Wrap models with DDP - only wrap modules that will actually be used
    # Analyze configuration to determine which optional modules will be needed
    needs_att = getattr(args, 'att_pool', False)
    needs_mlp = getattr(args, 'mlp_pool', False)
    needs_projector = getattr(args, 'use_projector', False)
    needs_identity_projection = getattr(args, 'use_identity_projection', False)
    
    # Also check data characteristics for projection modules
    for data, _ in zip(train_data_list, train_split_idx_list):
        if hasattr(data, 'needs_projection') and data.needs_projection:
            needs_projector = True
        if hasattr(data, 'needs_identity_projection') and data.needs_identity_projection:
            needs_identity_projection = True
    
    if rank == 0:
        logger.info(f"DDP wrapping analysis: needs_att={needs_att}, needs_mlp={needs_mlp}, needs_projector={needs_projector}, needs_identity_projection={needs_identity_projection}")
    
    all_modules_list = [model, predictor, att, mlp, projector, identity_projection]
    module_names = ['model', 'predictor', 'att', 'mlp', 'projector', 'identity_projection']
    module_needs = [True, True, needs_att, needs_mlp, needs_projector, needs_identity_projection]
    
    if world_size > 1:
        for i, (module, module_name, is_needed) in enumerate(zip(all_modules_list, module_names, module_needs)):
            if module is not None and any(p.requires_grad for p in module.parameters()):
                if is_needed:
                    all_modules_list[i] = DDP(module, device_ids=[rank], find_unused_parameters=True)
                    if rank == 0:
                        logger.info(f"Wrapped {module_name} with DDP")
                else:
                    if rank == 0:
                        logger.info(f"Skipping DDP wrapping for {module_name} (not needed)")
    model, predictor, att, mlp, projector, identity_projection = all_modules_list

    # Analyze model memory usage after DDP wrapping
    if rank == 0:
        model_names = ['model', 'predictor', 'att', 'mlp', 'projector', 'identity_projection']
        model_breakdown = gpu_monitor.get_model_memory_breakdown(all_modules_list, model_names)
        
        total_model_params = sum(info['parameter_count'] for info in model_breakdown.values())
        total_model_memory = sum(info['total_gb'] for info in model_breakdown.values())
        
        # Log model setup information
        logger.setup_model({
            'type': args.model,
            'hidden_dim': processed_input_dim,
            'total_params': total_model_params,
            'memory_gb': total_model_memory
        })

    # --- 3. Setup Optimizer and Scheduler ---
    parameters = [p for module in all_modules_list if module is not None for p in module.parameters()]
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {args.optimizer}")
        
    if args.schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.schedule == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epochs // 5, gamma=0.5)
    elif args.schedule == 'warmup':
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.epochs // 10, num_training_steps=args.epochs)
    else:
        scheduler = None

    # --- 4. Pre-select context data for consistent training ---
    if rank == 0: print("\n--- Pre-selecting context data for consistent training ---")
    
    # Pre-select context data once for each training dataset to ensure consistency
    train_context_data = []
    train_masks = []
    train_link_data_all = []
    
    for i, (data, split_idx) in enumerate(zip(train_data_list, train_split_idx_list)):
        try:
            link_data = prepare_link_data(data, split_idx, args.train_neg_ratio)
            
            # Validate that we have training data
            if 'train' not in link_data or link_data['train']['edge_pairs'].size(0) == 0:
                if rank == 0:
                    print(f"Warning: No training edges found for dataset {data.name}, skipping...")
                continue
                
            # --- CONTEXT SELECTION SYNCHRONIZATION ---
            # Select context on rank 0 and broadcast to all other ranks to ensure consistency.
            if world_size > 1:
                if rank == 0:
                    # Rank 0 performs the random selection
                    context_data, train_mask = select_link_context(
                        link_data['train'], args.context_k, 
                        args.context_neg_ratio, args.remove_context_from_train
                    )
                    # Prepare data for broadcast
                    objects_to_broadcast = [context_data, train_mask]
                else:
                    # Other ranks prepare placeholders
                    objects_to_broadcast = [None, None]
                
                # Broadcast the selected context from rank 0 to all other ranks
                dist.broadcast_object_list(objects_to_broadcast, src=0)
                
                if rank != 0:
                    # Unpack the received data
                    context_data, train_mask = objects_to_broadcast
            else:
                # Single GPU mode, no broadcasting needed
                context_data, train_mask = select_link_context(
                    link_data['train'], args.context_k, 
                    args.context_neg_ratio, args.remove_context_from_train
                )
            # --- END OF SYNCHRONIZATION ---
            
            train_context_data.append(context_data)
            train_masks.append(train_mask)
            train_link_data_all.append(link_data)
            
            if rank == 0:
                print(f"✅ Context prepared for {data.name}: {context_data['edge_pairs'].size(0)} context edges")
                
        except Exception as e:
            if rank == 0:
                print(f"Error preparing context for dataset {data.name}: {e}")
                import traceback
                print(f"Full traceback: {traceback.format_exc()}")
            continue

    # --- 4.5. Pre-load and process test datasets for periodic evaluation ---
    # Load test datasets once before training to avoid repeated loading in the training loop
    test_dataset_names = args.test_dataset.split(',')
    if rank == 0:
        print(f"\n--- Pre-loading test datasets for periodic evaluation ---")
    
    test_data_list_periodic = []
    test_split_idx_list_periodic = []
    test_link_data_all_periodic = []
    test_context_data_periodic = []
    
    try:
        test_data_raw, test_split_idx_raw = load_all_data_link(test_dataset_names, device=device)
        
        for i, (data, split_idx) in enumerate(zip(test_data_raw, test_split_idx_raw)):
            try:
                process_link_data(data, args, rank=rank)
                
                link_data_all = prepare_link_data(data, split_idx, args.train_neg_ratio)
                
                # --- TEST CONTEXT SELECTION SYNCHRONIZATION ---
                # Also synchronize context selection for test datasets to ensure consistency
                if world_size > 1:
                    if rank == 0:
                        context_data, _ = select_link_context(link_data_all['train'], args.context_k, args.context_neg_ratio, remove_from_train=False)
                        objects_to_broadcast = [context_data]
                    else:
                        objects_to_broadcast = [None]
                    
                    dist.broadcast_object_list(objects_to_broadcast, src=0)
                    
                    if rank != 0:
                        context_data = objects_to_broadcast[0]
                else:
                    context_data, _ = select_link_context(link_data_all['train'], args.context_k, args.context_neg_ratio, remove_from_train=False)
                # --- END OF SYNCHRONIZATION ---
                                
                test_data_list_periodic.append(data)
                test_split_idx_list_periodic.append(split_idx)
                test_link_data_all_periodic.append(link_data_all)
                test_context_data_periodic.append(context_data)
                
                if rank == 0:
                    print(f"✅ Test dataset {data.name} prepared for periodic evaluation")
                    
            except Exception as e:
                if rank == 0:
                    print(f"Error preparing test dataset {data.name} for periodic evaluation: {e}")
                    import traceback
                    print(f"Full traceback: {traceback.format_exc()}")
                continue
        
        if rank == 0:
            print(f"Successfully prepared {len(test_data_list_periodic)} test datasets for periodic evaluation")
            
    except Exception as e:
        if rank == 0:
            print(f"Error loading test datasets for periodic evaluation: {e}")
        test_data_list_periodic = []
        test_split_idx_list_periodic = []
        test_link_data_all_periodic = []
        test_context_data_periodic = []

    # --- 5. Main Training Loop ---
    best_valid_metric = 0
    best_epoch = 0
    # Create a dictionary to hold the state of all best models
    best_states = {
        'model': None,
        'predictor': None,
        'att': None,
        'mlp': None,
        'projector': None,
        'identity_projection': None
    }
    
    # OOM recovery tracking (simplified with decorators)
    oom_count = 0

    # Start training phase
    logger.training_start(args.epochs)
    
    for epoch in range(args.epochs):
        st = time.time()
        
        # Add barrier to ensure all processes start epoch together
        if world_size > 1:
            try:
                dist.barrier()
            except Exception as e:
                print(f"[ERROR] Rank {rank}: Exception at epoch {epoch} start barrier: {e}")
                raise
        
        # --- Train on all training datasets ---
        total_train_loss = 0
        train_dataset_count = 0
        
        for i, (data, split_idx) in enumerate(zip(train_data_list, train_split_idx_list)):
            link_data_all = train_link_data_all[i]
            context_data = train_context_data[i]
            train_mask = train_masks[i]
            
            if 'train' in link_data_all and link_data_all['train']['edge_pairs'].size(0) > 0:
                try:
                    # Standard training step
                    train_loss = train_link_prediction(
                        model, predictor, data, link_data_all['train'], context_data, train_mask,
                        optimizer, args.batch_size, att, mlp, projector, identity_projection, 
                        args.clip_grad, rank, args.orthogonal_push, args.normalize_class_h, 
                        epoch=epoch, mask_target_edges=args.mask_target_edges, degree=args.degree
                    )
                    
                    total_train_loss += train_loss
                    train_dataset_count += 1
                    
                    # Update process monitor activity
                    if process_monitor is not None:
                        process_monitor.update_activity()
                        
                except RuntimeError as e:
                    print(f"[ERROR] Rank {rank}: RuntimeError in training dataset {data.name}: {e}")
                    if "out of memory" in str(e):
                        oom_count += 1
                        if rank == 0:
                            logger.error_oom(data.name, args.batch_size, oom_count)
                    else:
                        if rank == 0:
                            logger.error_training(data.name, str(e))
                    raise  # Re-raise to stop execution and prevent hanging
                except Exception as e:
                    print(f"[ERROR] Rank {rank}: Unexpected exception in training dataset {data.name}: {e}")
                    import traceback
                    print(f"[ERROR] Rank {rank}: Traceback: {traceback.format_exc()}")
                    raise  # Re-raise to stop execution and prevent hanging
            else:
                if rank == 0:
                    logger.warning(f"Skipping dataset {data.name} - no training data")

        # Add barrier after training to ensure all processes finish training together
        if world_size > 1:
            try:
                dist.barrier()
            except Exception as e:
                print(f"[ERROR] Rank {rank}: Exception at epoch {epoch} post-training barrier: {e}")
                raise

        if epoch == 0:
            waste_analyzer.end_timing('training_computation')

        # Step scheduler once per epoch (not per dataset)
        if scheduler is not None:
            scheduler.step()
        
        # Measure gradient synchronization time (communication overhead)
        if epoch == 0 and world_size > 1:
            waste_analyzer.start_timing('gradient_synchronization')
            # DDP automatically synchronizes gradients here
            torch.cuda.synchronize()  # Ensure timing accuracy
            waste_analyzer.end_timing('gradient_synchronization')

        # --- Validate on all training datasets ---
        total_valid_metric = 0
        valid_dataset_count = 0
        for i, (data, split_idx) in enumerate(zip(train_data_list, train_split_idx_list)):
            link_data_all = train_link_data_all[i]
            context_data = train_context_data[i]

            if 'valid' in link_data_all and link_data_all['valid']['edge_pairs'].size(0) > 0:
                try:
                    # Standard validation step
                    valid_results = evaluate_link_prediction(
                        model, predictor, data, link_data_all['valid'], context_data, args.test_batch_size,
                        att, mlp, projector, identity_projection, rank, args.normalize_class_h,
                        degree=args.degree, k_values=[20, 50, 100]
                    )
                    
                    # Use the default metric for this dataset
                    if 'default_metric' in valid_results:
                        metric_value = valid_results['default_metric']
                        metric_name = valid_results.get('default_metric_name', 'unknown')
                    else:
                        # Fallback to hits@100
                        metric_value = valid_results.get('hits@100', 0.0)
                        metric_name = 'hits@100'
                    
                    total_valid_metric += metric_value
                    valid_dataset_count += 1
                    
                    if rank == 0 and epoch % args.log_interval == 0 and args.log_level in [LogLevel.DEBUG, LogLevel.VERBOSE]:
                        logger.debug(f"Dataset {data.name} validation {metric_name}: {metric_value:.4f}")
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if rank == 0:
                            logger.error_oom(data.name, args.test_batch_size, None, is_validation=True)
                    else:
                        if rank == 0:
                            logger.error_validation(data.name, str(e))
        
        # Compute average validation metric with proper zero handling
        if valid_dataset_count > 0:
            avg_valid_metric = total_valid_metric / valid_dataset_count
        else:
            # If no validation datasets succeeded, skip this epoch's validation
            if rank == 0:
                print(f"Warning: No validation datasets succeeded in epoch {epoch}")
            avg_valid_metric = 0.0
        
        # --- Periodic Test Evaluation for Monitoring (every eval_interval epochs) ---
        avg_test_metric = 0.0
        if rank == 0 and epoch % args.eval_interval == 0 and len(test_data_list_periodic) > 0:
            try:
                total_test_metric = 0
                test_dataset_count = 0
                
                logger.training_test_start(epoch)
                
                for i, (data, link_data_all, context_data) in enumerate(zip(test_data_list_periodic, test_link_data_all_periodic, test_context_data_periodic)):
                    try:
                        test_results_dict = evaluate_link_prediction(
                            model, predictor, data, link_data_all['test'], context_data, args.test_batch_size,
                            att, mlp, projector, identity_projection, rank, args.normalize_class_h,
                            degree=args.degree, k_values=[20, 50, 100]
                        )
                        
                        # Use the default metric for this dataset
                        if 'default_metric' in test_results_dict:
                            metric_value = test_results_dict['default_metric']
                            metric_name = test_results_dict.get('default_metric_name', 'unknown')
                        else:
                            metric_value = test_results_dict.get('hits@100', 0.0)
                            metric_name = 'hits@100'
                        
                        total_test_metric += metric_value
                        test_dataset_count += 1

                        # Show individual dataset results - IMPORTANT for monitoring individual performance!
                        if rank == 0:  # Only rank 0 should print to avoid duplicate output
                            print(f"Epoch {epoch} - Test {data.name} {metric_name}: {metric_value:.4f}")
                            
                    except Exception as e:
                        print(f"[ERROR] Exception during test evaluation of {data.name}: {e}")
                        import traceback
                        print(f"[ERROR] Traceback: {traceback.format_exc()}")
                        logger.error_test(data.name, str(e))
                        continue
                
                if test_dataset_count > 0:
                    avg_test_metric = total_test_metric / test_dataset_count
                    logger.training_test_result(epoch, avg_test_metric)
                    
            except Exception as e:
                print(f"[ERROR] Fatal exception during test evaluation: {e}")
                import traceback
                print(f"[ERROR] Traceback: {traceback.format_exc()}")
                avg_test_metric = 0.0
            
        # Add barrier after test evaluation to ensure rank 0 finishes before continuing
        if world_size > 1:
            try:
                dist.barrier()
            except Exception as e:
                print(f"[ERROR] Rank {rank}: Exception at epoch {epoch} post-test-evaluation barrier: {e}")
                raise
            
        if rank == 0 and epoch % args.log_interval == 0:
            en = time.time()
            avg_train_loss = total_train_loss / max(train_dataset_count, 1)
            
            # Enhanced memory monitoring and efficiency analysis
            memory_info = ""
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(device) / 1e9
                memory_reserved = torch.cuda.memory_reserved(device) / 1e9
                memory_info = f", GPU Mem: {memory_allocated:.2f}/{memory_reserved:.2f} GB"
                
                # Print detailed monitoring every analysis_interval epochs  
                if epoch % args.analysis_interval == 0:
                    
                    if args.log_level in [LogLevel.DEBUG, LogLevel.VERBOSE]:
                        gpu_monitor.print_memory_summary()
                        efficiency_report = gpu_monitor.get_memory_efficiency_report()
                        logger.debug(efficiency_report)
                    
                    # Print waste analysis summary - ONLY ON RANK 0 to prevent DDP hanging!
                    if rank == 0 and epoch > 0:  # Only rank 0 and skip epoch 0
                        try:
                            logger.waste_training_summary(epoch, waste_analyzer)
                        except Exception as e:
                            print(f"[ERROR] Rank 0: Exception in waste_training_summary: {e}")
                            import traceback
                            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            
            # Main epoch logging with configurable verbosity
            logger.training_epoch_summary({
                'epoch': epoch,
                'avg_loss': avg_train_loss,
                'avg_valid_metric': avg_valid_metric,
                'time': en-st,
                'memory_info': memory_info,
                'train_dataset_count': train_dataset_count,
                'oom_count': oom_count if oom_count > 0 else None
            })
            
            # Log both validation and test metrics
            log_dict = {'epoch': epoch, 'avg_train_loss': avg_train_loss, 'avg_valid_metric': avg_valid_metric}
            if oom_count > 0:
                log_dict['oom_count'] = oom_count
                log_dict['current_batch_size'] = args.batch_size
                log_dict['current_test_batch_size'] = args.test_batch_size
            if avg_test_metric > 0:
                log_dict['avg_test_metric'] = avg_test_metric
            
            # Add efficiency metrics to wandb logs
            if epoch > 0:  # Only after we have timing data
                efficiency_scores = waste_analyzer.metrics.get('efficiency_scores', {})
                if efficiency_scores:
                    log_dict['memory_efficiency_percent'] = efficiency_scores.get('memory_efficiency_percent', 0)
                    log_dict['compute_efficiency_percent'] = efficiency_scores.get('compute_efficiency_percent', 0)
                    log_dict['overall_efficiency_percent'] = efficiency_scores.get('overall_efficiency_percent', 0)
                    log_dict['resource_waste_factor'] = efficiency_scores.get('waste_factor', 1)
            
            if wandb.run is not None:
                wandb.log(log_dict)

        if avg_valid_metric > best_valid_metric:
            best_valid_metric = avg_valid_metric
            best_epoch = epoch
            
            if rank == 0:
                logger.training_new_best(epoch, avg_valid_metric)
            
            # Save best model states (only rank 0 saves, then broadcast to all ranks)
            if rank == 0:
                best_states['model'] = copy.deepcopy(model.module.state_dict() if hasattr(model, 'module') else model.state_dict())
                best_states['predictor'] = copy.deepcopy(predictor.module.state_dict() if hasattr(predictor, 'module') else predictor.state_dict())
                if att is not None:
                    best_states['att'] = copy.deepcopy(att.module.state_dict() if hasattr(att, 'module') else att.state_dict())
                if mlp is not None:
                    best_states['mlp'] = copy.deepcopy(mlp.module.state_dict() if hasattr(mlp, 'module') else mlp.state_dict())
                if projector is not None:
                    best_states['projector'] = copy.deepcopy(projector.module.state_dict() if hasattr(projector, 'module') else projector.state_dict())
                if identity_projection is not None:
                    best_states['identity_projection'] = copy.deepcopy(identity_projection.module.state_dict() if hasattr(identity_projection, 'module') else identity_projection.state_dict())
            
            # Sync best model states across all ranks in DDP mode
            if world_size > 1:
                # Prepare the list to be broadcasted from rank 0
                objects_to_broadcast = [best_states] if rank == 0 else [None]
                
                # Broadcast the entire 'best_states' dictionary from rank 0 to all others
                dist.broadcast_object_list(objects_to_broadcast, src=0)
                
                # Non-rank-0 processes receive the data
                if rank != 0:
                    best_states = objects_to_broadcast[0]
                    
                # Also sync the metrics so everyone agrees on the best epoch number
                best_metrics_tensor = torch.tensor([best_valid_metric, best_epoch], device=device)
                dist.broadcast(best_metrics_tensor, src=0)
                if rank != 0:
                    best_valid_metric, best_epoch = best_metrics_tensor.tolist()
                    best_epoch = int(best_epoch)

    if rank == 0:
        logger.training_complete(best_valid_metric, best_epoch)

    # --- 6. Inductive Testing Phase ---
    logger.testing_start()
    
    # Load best model weights for all components (ensure all ranks load the same state)
    if best_states['model'] is not None:
        if rank == 0: 
            logger.info("Loading best model states for all components...")
        
        # Load best states into models
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(best_states['model'])
        
        predictor_to_load = predictor.module if hasattr(predictor, 'module') else predictor
        predictor_to_load.load_state_dict(best_states['predictor'])
        
        if att is not None and best_states['att'] is not None:
            att_to_load = att.module if hasattr(att, 'module') else att
            att_to_load.load_state_dict(best_states['att'])
            
        if mlp is not None and best_states['mlp'] is not None:
            mlp_to_load = mlp.module if hasattr(mlp, 'module') else mlp
            mlp_to_load.load_state_dict(best_states['mlp'])
            
        if projector is not None and best_states['projector'] is not None:
            projector_to_load = projector.module if hasattr(projector, 'module') else projector
            projector_to_load.load_state_dict(best_states['projector'])
            
        if identity_projection is not None and best_states['identity_projection'] is not None:
            identity_projection_to_load = identity_projection.module if hasattr(identity_projection, 'module') else identity_projection
            identity_projection_to_load.load_state_dict(best_states['identity_projection'])
    else:
        if rank == 0: 
            logger.warning("No best model found, using the final model state.")

    # Ensure all ranks are synchronized before proceeding to testing
    if world_size > 1:
        dist.barrier()

    # Load and process test datasets (final evaluation)
    test_dataset_names = args.test_dataset.split(',')
    if rank == 0: 
        logger.testing_datasets(test_dataset_names)
    
    # Always reuse the pre-processed test data from periodic evaluation
    final_test_data_list = test_data_list_periodic
    final_test_link_data_all = test_link_data_all_periodic
    final_test_context_data = test_context_data_periodic
    if rank == 0:
        print("Using pre-processed test datasets for final evaluation")

    test_results = []
    for i, (data, link_data_all, context_data) in enumerate(zip(final_test_data_list, final_test_link_data_all, final_test_context_data)):
        try:
            # Validate that test data exists
            if 'test' not in link_data_all or link_data_all['test']['edge_pairs'].size(0) == 0:
                if rank == 0:
                    logger.warning(f"No test edges found for dataset {data.name}, skipping...")
                continue
            
            # Validate context data for test
            if context_data['edge_pairs'].size(0) == 0:
                if rank == 0:
                    logger.warning(f"No context available for test dataset {data.name}, skipping...")
                continue

            test_results_dict = evaluate_link_prediction(
                model, predictor, data, link_data_all['test'], context_data, args.test_batch_size,
                att, mlp, projector, identity_projection, rank, args.normalize_class_h,
                degree=args.degree, k_values=[20, 50, 100]  # Standard Hits@K values
            )
            
            # Get the dataset-specific metric
            default_metric_name = test_results_dict.get('default_metric_name', 'hits@100')
            default_metric_value = test_results_dict.get('default_metric', test_results_dict.get('hits@100', 0.0))
            
            test_results.append({
                'dataset': data.name, 
                'test_results': test_results_dict, 
                'metric': default_metric_name
            })
            
            if rank == 0:
                logger.testing_dataset_result(data.name, default_metric_name, default_metric_value)
                
        except Exception as e:
            if rank == 0:
                logger.error_test(data.name, str(e))
            continue
    
    # --- 7. Final Results and Summary ---
    if rank == 0 and results_dict is not None:
        results_dict[rank] = test_results

    # Stop monitoring and print final summary
    if process_monitor is not None:
        process_monitor.stop_monitoring()
    gpu_monitor.stop_monitoring()
    
    if rank == 0:
        # Generate final summaries using logger
        logger.results_final_summary(gpu_monitor, waste_analyzer)
        
        # Save detailed metrics to file
        waste_analyzer.save_detailed_metrics("distributed_training_waste_analysis.json")
        logger.info("Detailed waste analysis saved to distributed_training_waste_analysis.json")

    # Cleanup DDP
    if world_size > 1:
        cleanup_ddp()

def main():
    """Main function for link prediction."""
    import signal
    import sys
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"\nReceived signal {signum}, shutting down gracefully...")
        # Kill any remaining child processes
        try:
            import psutil
            current_process = psutil.Process()
            children = current_process.children(recursive=True)
            for child in children:
                print(f"Terminating child process {child.pid}")
                child.terminate()
            # Wait for children to terminate
            psutil.wait_procs(children, timeout=10)
            # Force kill any remaining children
            for child in children:
                if child.is_running():
                    print(f"Force killing child process {child.pid}")
                    child.kill()
        except ImportError:
            print("psutil not available, cannot cleanup child processes automatically")
        except Exception as e:
            print(f"Error during signal cleanup: {e}")
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    args = parse_link_prediction_args()

    if args.use_pretrained_model and args.load_checkpoint is None:
        raise ValueError("Must provide --load_checkpoint when --use_pretrained_model is True.")

    # Note: wandb.init() is now handled inside run_ddp_lp() for both single GPU and DDP modes
    # This avoids conflicts and ensures proper initialization in all scenarios
    
    # --- GPU Setup and Validation ---
    try:
        # Parse GPU specification
        gpu_ids = parse_gpu_spec(args.gpu)
        
        # Validate GPU availability
        validate_gpu_availability(gpu_ids)
        
        # Print GPU information
        print_gpu_info(gpu_ids)
        
        # Set up CUDA visible devices for the specified GPUs
        setup_cuda_visible_devices(gpu_ids)
        
        # After setting CUDA_VISIBLE_DEVICES, get the effective world size
        world_size = get_effective_world_size(gpu_ids)
        
    except (ValueError, RuntimeError) as e:
        print(f"GPU setup error: {e}")
        print("Available GPUs:")
        total_gpus = torch.cuda.device_count()
        if total_gpus == 0:
            print("  None")
        else:
            for i in range(total_gpus):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} - {props.total_memory / 1e9:.1f} GB")
        return
    
    # Check if single GPU mode is requested via command line argument or only one GPU available
    use_single_gpu = not args.use_ddp or world_size == 1
    
    # --- DDP or Single-GPU Execution ---
    all_runs_results = []
    if use_single_gpu:
        print(f"Using Single GPU mode on device cuda:0 (Physical GPU {gpu_ids[0]})")
        for run in range(args.runs):
            print(f"\n🏃 Run {run + 1}/{args.runs}")
            results_dict = {0: None} # Simple dict for single-process
            
            try:
                run_ddp_lp(0, 1, args, results_dict)
                if results_dict[0] is not None:
                    all_runs_results.append(results_dict[0])
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"🚨 OOM Error in Run {run + 1}: {e}")
                    print("📊 Reporting zero metrics to W&B to continue sweep...")
                    
                    # Initialize wandb if not already done (for sweep mode)
                    if not wandb.run:
                        wandb.init(project='inductlink', config=args)
                    
                    # Report zero metrics for sweep optimization
                    oom_metrics = {
                        'avg_test_metric': 0.0,
                        'avg_valid_metric': 0.0,
                        'avg_train_loss': float('inf'),
                        'oom_error': True,
                        'oom_run': run + 1,
                        'batch_size': args.batch_size,
                        'test_batch_size': args.test_batch_size,
                        'hidden': args.hidden,
                        'final_test_metric_average': 0.0
                    }
                    
                    # Add individual dataset zero metrics
                    test_dataset_names = args.test_dataset.split(',')
                    for dataset_name in test_dataset_names:
                        oom_metrics[f'{dataset_name}_test_hits_100_mean'] = 0.0
                        oom_metrics[f'{dataset_name}_test_hits_100_std'] = 0.0
                        oom_metrics[f'{dataset_name}_test_hits_100_runs'] = 0
                    
                    wandb.log(oom_metrics)
                    print(f"✅ Zero metrics logged to W&B for OOM in run {run + 1}")
                    
                    # Exit immediately instead of continuing to next run
                    print(f"🛑 Exiting due to OOM error in run {run + 1}")
                    return
                else:
                    # Re-raise non-OOM errors
                    raise e
            except Exception as e:
                print(f"🚨 Unexpected error in Run {run + 1}: {e}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
                
                # For sweeps, still report zero metrics for any fatal error
                if wandb.run or hasattr(args, 'sweep'):
                    if not wandb.run:
                        wandb.init(project='inductlink', config=args)
                    
                    error_metrics = {
                        'avg_test_metric': 0.0,
                        'avg_valid_metric': 0.0,
                        'avg_train_loss': float('inf'),
                        'fatal_error': True,
                        'error_run': run + 1,
                        'final_test_metric_average': 0.0
                    }
                    wandb.log(error_metrics)
                    print(f"✅ Zero metrics logged to W&B for fatal error in run {run + 1}")
                
                # Exit immediately instead of continuing to next run
                print(f"🛑 Exiting due to fatal error in run {run + 1}")
                return
    else:
        print(f"Using DistributedDataParallel with {world_size} GPUs (Physical GPUs {gpu_ids})")
        for run in range(args.runs):
            if run > 0: args.port += 1
            print(f"\n🏃 Run {run + 1}/{args.runs}")
            manager = mp.Manager()
            results_dict = manager.dict()
            mp.spawn(run_ddp_lp, args=(world_size, args, results_dict), nprocs=world_size, join=True)
            if 0 in results_dict:
                all_runs_results.append(results_dict[0])
    
    # --- Aggregate and Log Final Results ---
    if len(all_runs_results) > 0:
        print("\n📊 Final Inductive Test Results")
        print("=" * 60)
        
        aggregated_results = {}
        for run_idx, run_res in enumerate(all_runs_results):
            if run_res is None or len(run_res) == 0:
                print(f"Warning: Run {run_idx + 1} returned no results")
                continue
                
            for res_item in run_res:
                if 'dataset' not in res_item or 'test_results' not in res_item:
                    print(f"Warning: Invalid result format in run {run_idx + 1}")
                    continue
                    
                name = res_item['dataset']
                metric_name = res_item.get('metric', 'hits@100')  # Get dataset-specific metric
                
                if metric_name not in res_item['test_results']:
                    # Fallback to hits@100 if specific metric not available
                    if 'hits@100' in res_item['test_results']:
                        metric_name = 'hits@100'
                    else:
                        print(f"Warning: Missing {metric_name} for dataset {name} in run {run_idx + 1}")
                        continue
                    
                if name not in aggregated_results:
                    aggregated_results[name] = {'metric_name': metric_name, 'test_metric': []}
                aggregated_results[name]['test_metric'].append(res_item['test_results'][metric_name])

        if len(aggregated_results) == 0:
            print("No valid test results found across all runs.")
            return

        final_log = {}
        all_dataset_averages = []  # Collect averages for overall sweep metric
        
        for name, data in aggregated_results.items():
            if len(data['test_metric']) == 0:
                print(f"Warning: No valid results for dataset {name}")
                continue
                
            metric_name = data['metric_name']
            avg_test = np.mean(data['test_metric'])
            std_test = np.std(data['test_metric']) if len(data['test_metric']) > 1 else 0.0
            
            print(f"{name}: Test {metric_name} {avg_test:.4f} ± {std_test:.4f} (n={len(data['test_metric'])})")
            
            # Use clean metric name for logging (remove @ symbol for wandb compatibility)
            clean_metric_name = metric_name.replace('@', '_')
            final_log[f'{name}_test_{clean_metric_name}_mean'] = avg_test
            final_log[f'{name}_test_{clean_metric_name}_std'] = std_test
            final_log[f'{name}_test_{clean_metric_name}_runs'] = len(data['test_metric'])
            
            # Collect for overall average
            all_dataset_averages.append(avg_test)
        
        # Add overall sweep optimization metric (average across all test datasets and runs)
        if len(all_dataset_averages) > 0:
            final_sweep_metric = np.mean(all_dataset_averages)
            final_log['final_test_metric_average'] = final_sweep_metric
            print(f"\n🎯 Final Sweep Metric (avg across all datasets): {final_sweep_metric:.4f}")
        
        if wandb.run is not None and len(final_log) > 0:
            wandb.log(final_log)
    else:
        print("No test results to aggregate.")

    print("\n🎉 Link prediction completed!")

if __name__ == '__main__':
    main()