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
from src.correct_gpu_memory import (
    safe_training_step,
    get_gpu_memory_info,
    calculate_safe_batch_sizes_per_gpu
)
from src.ddp_gpu_monitor import create_ddp_monitor
from transformers import get_cosine_schedule_with_warmup


def run_ddp_lp(rank, world_size, args, results_dict):
    """Main execution function for link prediction, supports DDP."""
    if world_size > 1:
        setup_ddp(rank, world_size, args.port)
        if rank == 0:
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
        
    # Correct heterogeneous GPU memory management - use the dedicated module
    try:
        if rank == 0:
            print("🔗 Inductive Link Prediction Task")
            print("=" * 60)
            print(f"Using device: {device}")
        
        # Use the dedicated memory management module instead of inline logic
        if world_size > 1:
            # Multi-GPU: use heterogeneous memory allocation from the module
            local_batch_size, batch_distribution = calculate_safe_batch_sizes_per_gpu(
                world_size, rank, args.batch_size
            )
            
            args.batch_size = local_batch_size
            args.test_batch_size = local_batch_size  # Keep consistent
            
            if rank == 0:
                print(f"✅ Rank {rank}: Local batch size = {local_batch_size}")
                print(f"✅ Batch distribution across GPUs: {batch_distribution}")
                total_effective = sum(batch_distribution.values()) if batch_distribution else local_batch_size * world_size
                print(f"✅ Total effective batch size: {total_effective}")
            else:
                print(f"✅ Rank {rank}: Local batch size = {local_batch_size} (distribution handled by rank 0)")
        else:
            # Single GPU: simple memory check
            torch_device = torch.device(device)
            memory_info = get_gpu_memory_info(torch_device)
            
            if rank == 0:
                print(f"GPU {rank} Memory: {memory_info['total']:.1f}GB total, {memory_info['free']:.1f}GB free")
            
            estimated_memory_needed = args.batch_size * 0.001  # Rough estimate
            if memory_info['free'] * 0.8 < estimated_memory_needed:
                print(f"⚠️  GPU memory may be insufficient, suggest reducing batch size")
        
        if rank == 0:
            print(f"Final batch sizes - Train: {args.batch_size}, Test: {args.test_batch_size}")
        
        # Initialize enhanced GPU monitoring
        torch_device = torch.device(device)
        gpu_monitor = create_ddp_monitor(rank, world_size, torch_device)
        gpu_monitor.start_monitoring(interval=10.0)  # Monitor every 10 seconds
            
    except Exception as e:
        if rank == 0:
            print("🔗 Inductive Link Prediction Task") 
            print("=" * 60)
            print(f"Using device: {device}")
            print(f"Warning: GPU memory check failed, using original batch sizes: {e}")
    
    # --- 1. Load and process training datasets ---
    train_dataset_names = args.train_dataset.split(',')
    if rank == 0: print(f"Loading training datasets: {train_dataset_names}")
    train_data_list, train_split_idx_list = load_all_data_link(train_dataset_names, device=device)

    # Analyze dataset memory usage and duplication
    if rank == 0:
        dataset_info, duplication_cost = gpu_monitor.analyze_dataset_duplication(
            train_data_list, train_dataset_names
        )
        print(f"⚠️  WARNING: Dataset duplication across {world_size} GPUs wastes {duplication_cost:.2f}GB!")
        if world_size > 1:
            print("💡 Consider: Data parallelism with distributed data loading to reduce memory waste")

    if rank == 0: print("Processing training datasets...")
    for i, (data, split_idx) in enumerate(zip(train_data_list, train_split_idx_list)):
        try:
            process_link_data(data, args, rank=rank)
            if rank == 0:
                print(f"✅ Processed training dataset {i+1}/{len(train_data_list)}: {data.name}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                if rank == 0:
                    print(f"❌ GPU memory insufficient for dataset {data.name}. Consider reducing batch size or using CPU.")
                raise e
            else:
                raise e

    # Determine the correct input dimension after data processing
    # Check if any dataset uses identity projection to set the correct dimension
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

    if rank == 0:
        print(f"Using processed input dimension: {processed_input_dim}")

    # --- 2. Create or load model ---
    if args.use_pretrained_model:
        if rank == 0: print("Loading pretrained model...")
        model, predictor, att, mlp, projector, identity_projection, model_args = recreate_model_from_checkpoint(
            args.load_checkpoint, processed_input_dim, device
        )
        args = override_args_from_checkpoint(args, model_args, rank)
    else:
        if rank == 0: print("Creating a new model from scratch...")
        model, predictor, att, mlp, projector, identity_projection = create_model_from_args(
            args, processed_input_dim, device
        )
    
    if rank == 0: print("✅ Model and Predictor created successfully!")

    # Conditionally wrap models with DDP
    all_modules_list = [model, predictor, att, mlp, projector, identity_projection]
    if world_size > 1:
        for i, module in enumerate(all_modules_list):
            if module is not None and any(p.requires_grad for p in module.parameters()):
                # Only wrap modules that will actually be used in the forward pass
                # projector and identity_projection may not be used if data doesn't need projection
                if i in [4, 5]:  # projector and identity_projection indices
                    # Only wrap if we expect to use these modules based on data characteristics
                    should_wrap = False
                    for data, _ in zip(train_data_list, train_split_idx_list):
                        if (hasattr(data, 'needs_projection') and data.needs_projection) or \
                           (hasattr(data, 'needs_identity_projection') and data.needs_identity_projection):
                            should_wrap = True
                            break
                    if should_wrap:
                        all_modules_list[i] = DDP(module, device_ids=[rank], find_unused_parameters=True)
                else:
                    # Always wrap core modules (model, predictor, att, mlp)
                    all_modules_list[i] = DDP(module, device_ids=[rank], find_unused_parameters=True)
    model, predictor, att, mlp, projector, identity_projection = all_modules_list

    # Analyze model memory usage after DDP wrapping
    if rank == 0:
        model_names = ['model', 'predictor', 'att', 'mlp', 'projector', 'identity_projection']
        model_breakdown = gpu_monitor.get_model_memory_breakdown(all_modules_list, model_names)
        
        total_model_params = sum(info['parameter_count'] for info in model_breakdown.values())
        total_model_memory = sum(info['total_gb'] for info in model_breakdown.values())
        
        print(f"\n📊 Model Memory Analysis:")
        print(f"  Total Parameters: {total_model_params:,}")
        print(f"  Total Model Memory: {total_model_memory:.3f}GB per GPU")
        print(f"  DDP Memory Overhead: {total_model_memory * world_size:.3f}GB across all GPUs")
        
        if world_size > 1:
            communication_overhead = gpu_monitor.measure_ddp_communication_overhead()

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
                
            context_data, train_mask = select_link_context(
                link_data['train'], args.context_k, 
                args.context_neg_ratio, args.remove_context_from_train
            )
            
            # Validate context data
            if context_data['edge_pairs'].size(0) == 0:
                if rank == 0:
                    print(f"Warning: No context edges selected for dataset {data.name}, using minimum context...")
                # Use a minimal context with at least one positive and one negative edge
                pos_indices = torch.where(link_data['train']['labels'] == 1)[0]
                neg_indices = torch.where(link_data['train']['labels'] == 0)[0]
                
                if len(pos_indices) > 0 and len(neg_indices) > 0:
                    min_context_indices = torch.cat([pos_indices[:1], neg_indices[:1]])
                    context_data = {
                        'edge_pairs': link_data['train']['edge_pairs'][min_context_indices],
                        'labels': link_data['train']['labels'][min_context_indices]
                    }
                    train_mask = torch.ones_like(link_data['train']['labels'], dtype=torch.bool)
                    train_mask[min_context_indices] = False
                else:
                    if rank == 0:
                        print(f"Error: Insufficient training data for dataset {data.name}")
                    continue
            
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
    
    # Validate that we have at least one valid training dataset
    if len(train_context_data) == 0:
        raise ValueError("No valid training datasets found. Please check your data and configuration.")
    
    # Update the data lists to only include valid datasets
    valid_indices = list(range(len(train_context_data)))
    train_data_list = [train_data_list[i] for i in valid_indices]
    train_split_idx_list = [train_split_idx_list[i] for i in valid_indices]
    
    # Use same context data for validation as training (for consistency)
    # This ensures prototypes are identical between training and validation
    valid_context_data = train_context_data.copy()  # Use same context as training

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

    if rank == 0: print("\n--- Starting Training Phase ---")
    for epoch in range(args.epochs):
        st = time.time()
        
        # --- Train on all training datasets ---
        total_train_loss = 0
        train_dataset_count = 0
        for i, (data, split_idx) in enumerate(zip(train_data_list, train_split_idx_list)):
            link_data_all = train_link_data_all[i]
            context_data = train_context_data[i]
            train_mask = train_masks[i]
            
            if 'train' in link_data_all and link_data_all['train']['edge_pairs'].size(0) > 0:
                try:
                    # Use safe training step - never dynamically adjust batch size
                    train_loss = safe_training_step(
                        train_link_prediction,
                        model, predictor, data, link_data_all['train'], context_data, train_mask,
                        optimizer, args.batch_size, att, mlp, projector, identity_projection, 
                        args.clip_grad, rank, args.orthogonal_push, args.normalize_class_h, 
                        epoch=epoch, mask_target_edges=args.mask_target_edges, degree=args.degree
                    )
                    total_train_loss += train_loss
                    train_dataset_count += 1
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        oom_count += 1
                        if rank == 0:
                            print(f"❌ GPU out of memory, training failed: {data.name}")
                            print(f"💡 Suggestion: reduce batch size from {args.batch_size} and restart training")
                    else:
                        if rank == 0:
                            print(f"Warning: Training failed for dataset {data.name}: {e}")

        # Step scheduler once per epoch (not per dataset)
        if scheduler is not None:
            scheduler.step()

        # --- Validate on all training datasets ---
        total_valid_metric = 0
        valid_dataset_count = 0
        for i, (data, split_idx) in enumerate(zip(train_data_list, train_split_idx_list)):
            link_data_all = train_link_data_all[i]
            context_data = valid_context_data[i]

            if 'valid' in link_data_all and link_data_all['valid']['edge_pairs'].size(0) > 0:
                try:
                    # Use safe validation step - never dynamically adjust batch size
                    valid_results = safe_training_step(
                        evaluate_link_prediction,
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
                    
                    if rank == 0 and epoch % 10 == 0:
                        print(f"Dataset {data.name} validation {metric_name}: {metric_value:.4f}")
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        if rank == 0:
                            print(f"❌ GPU out of memory, validation failed: {data.name}")
                    else:
                        if rank == 0:
                            print(f"Warning: Validation failed for dataset {data.name}: {e}")
        
        # Compute average validation metric with proper zero handling
        if valid_dataset_count > 0:
            avg_valid_metric = total_valid_metric / valid_dataset_count
        else:
            # If no validation datasets succeeded, skip this epoch's validation
            if rank == 0:
                print(f"Warning: No validation datasets succeeded in epoch {epoch}")
            avg_valid_metric = 0.0
        
        # --- Periodic Test Evaluation for Monitoring (every 20 epochs) ---
        avg_test_metric = 0.0
        if rank == 0 and epoch % 20 == 0:
            # Load test datasets for periodic evaluation
            test_dataset_names = args.test_dataset.split(',')
            test_data_list, test_split_idx_list = load_all_data_link(test_dataset_names, device=device)
            
            total_test_metric = 0
            test_dataset_count = 0
            
            for data, split_idx in zip(test_data_list, test_split_idx_list):
                try:
                    process_link_data(data, args, rank=rank)
                    
                    link_data_all = prepare_link_data(data, split_idx, args.train_neg_ratio)
                    
                    if 'test' not in link_data_all or link_data_all['test']['edge_pairs'].size(0) == 0:
                        continue
                    
                    context_data, _ = select_link_context(link_data_all['train'], args.context_k, args.context_neg_ratio, remove_from_train=False)
                    
                    if context_data['edge_pairs'].size(0) == 0:
                        continue

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
                    
                    print(f"Epoch {epoch} - Test {data.name} {metric_name}: {metric_value:.4f}")
                    
                except Exception as e:
                    print(f"Warning: Test evaluation failed for dataset {data.name}: {e}")
                    continue
            
            if test_dataset_count > 0:
                avg_test_metric = total_test_metric / test_dataset_count
                print(f"Epoch {epoch} - Average Test Metric: {avg_test_metric:.4f}")
            
        if rank == 0 and epoch % 10 == 0:
            en = time.time()
            avg_train_loss = total_train_loss / max(train_dataset_count, 1)
            
            # Enhanced memory monitoring and efficiency analysis
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated(device) / 1e9
                memory_reserved = torch.cuda.memory_reserved(device) / 1e9
                memory_info = f", GPU Mem: {memory_allocated:.2f}/{memory_reserved:.2f} GB"
                
                # Print detailed monitoring every 50 epochs
                if epoch % 50 == 0:
                    gpu_monitor.print_memory_summary()
                    efficiency_report = gpu_monitor.get_memory_efficiency_report()
                    print(efficiency_report)
            else:
                memory_info = ""
            
            print(f"Epoch {epoch}: Avg Loss {avg_train_loss:.4f}, Avg Valid Metric {avg_valid_metric:.4f}, Time: {en-st:.2f}s{memory_info}")
            if train_dataset_count == 0:
                print(f"Warning: No training datasets succeeded in epoch {epoch}")
            if oom_count > 0:
                print(f"Total OOM events so far: {oom_count}")
            
            # Log both validation and test metrics
            log_dict = {'epoch': epoch, 'avg_train_loss': avg_train_loss, 'avg_valid_metric': avg_valid_metric}
            if oom_count > 0:
                log_dict['oom_count'] = oom_count
                log_dict['current_batch_size'] = args.batch_size
                log_dict['current_test_batch_size'] = args.test_batch_size
            if avg_test_metric > 0:
                log_dict['avg_test_metric'] = avg_test_metric
            
            if wandb.run is not None:
                wandb.log(log_dict)

        if avg_valid_metric > best_valid_metric:
            best_valid_metric = avg_valid_metric
            best_epoch = epoch
            
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
        print(f"\n--- Training Complete ---")
        print(f"Best validation metric: {best_valid_metric:.4f} at epoch {best_epoch}")

    # --- 6. Inductive Testing Phase ---
    if rank == 0: print("\n--- Starting Inductive Testing Phase ---")
    
    # Load best model weights for all components (ensure all ranks load the same state)
    if best_states['model'] is not None:
        if rank == 0: print("Loading best model states for all components...")
        
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
        if rank == 0: print("Warning: No best model found, using the final model state.")

    # Ensure all ranks are synchronized before proceeding to testing
    if world_size > 1:
        dist.barrier()

    # Load and process test datasets
    test_dataset_names = args.test_dataset.split(',')
    if rank == 0: print(f"Loading test datasets: {test_dataset_names}")
    test_data_list, test_split_idx_list = load_all_data_link(test_dataset_names, device=device)

    test_results = []
    for data, split_idx in zip(test_data_list, test_split_idx_list):
        try:
            process_link_data(data, args, rank=rank)
            
            link_data_all = prepare_link_data(data, split_idx, args.train_neg_ratio)
            
            # Validate that test data exists
            if 'test' not in link_data_all or link_data_all['test']['edge_pairs'].size(0) == 0:
                if rank == 0:
                    print(f"Warning: No test edges found for dataset {data.name}, skipping...")
                continue
            
            # Use training context for test evaluation (inductive setting)
            context_data, _ = select_link_context(link_data_all['train'], args.context_k, args.context_neg_ratio, remove_from_train=False)
            
            # Validate context data for test
            if context_data['edge_pairs'].size(0) == 0:
                if rank == 0:
                    print(f"Warning: No context available for test dataset {data.name}, skipping...")
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
                print(f"✅ Test completed for {data.name}: {default_metric_name} = {default_metric_value:.4f}")
                
        except Exception as e:
            if rank == 0:
                print(f"Error testing dataset {data.name}: {e}")
            continue
    
    # --- 6. Aggregate and Log Final Results ---
    if rank == 0 and results_dict is not None:
        results_dict[rank] = test_results

    # Stop monitoring and print final summary
    gpu_monitor.stop_monitoring()
    if rank == 0:
        gpu_monitor.print_memory_summary()
        print("\n" + "="*60)
        print("🎯 Final GPU Usage Summary:")
        print("="*60)
        efficiency_report = gpu_monitor.get_memory_efficiency_report()
        print(efficiency_report)

    if world_size > 1:
        cleanup_ddp()

def main():
    """Main function for link prediction."""
    args = parse_link_prediction_args()

    if args.use_pretrained_model and args.load_checkpoint is None:
        raise ValueError("Must provide --load_checkpoint when --use_pretrained_model is True.")

    if not args.use_ddp and not args.sweep:
         wandb.init(project='inductlink', config=args)
    
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
            run_ddp_lp(0, 1, args, results_dict)
            if results_dict[0] is not None:
                all_runs_results.append(results_dict[0])
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
        
        if wandb.run is not None and len(final_log) > 0:
            wandb.log(final_log)
    else:
        print("No test results to aggregate.")

    print("\n🎉 Link prediction completed!")

if __name__ == '__main__':
    main()