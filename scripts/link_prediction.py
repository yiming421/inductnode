import sys
import os
import torch
import numpy as np
import argparse
import wandb
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
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
from src.checkpoint_utils import (override_args_from_checkpoint, save_checkpoint, 
                                create_model_from_args, recreate_model_from_checkpoint)
from src.ddp_utils import setup_ddp, cleanup_ddp
from src.config import parse_link_prediction_args
from src.engine_link_pred import train_link_prediction, evaluate_link_prediction
from transformers import get_cosine_schedule_with_warmup


def run_ddp_lp(rank, world_size, args, results_dict):
    """Main execution function for link prediction, supports DDP."""
    if world_size > 1:
        setup_ddp(rank, world_size, args.port)
        if rank == 0:
            wandb.init(project='inductlink', config=args)

    device = f'cuda:{rank}'
    if rank == 0:
        print("🔗 Inductive Link Prediction Task")
        print("=" * 60)
        print(f"Using device: {device}")
    
    # --- 1. Load and process training datasets ---
    train_dataset_names = args.train_dataset.split(',')
    if rank == 0: print(f"Loading training datasets: {train_dataset_names}")
    train_data_list, train_split_idx_list = load_all_data_link(train_dataset_names)

    # Process all training data once before training loop
    for data, split_idx in zip(train_data_list, train_split_idx_list):
        data.x = data.x.to(device)
        data.adj_t = data.adj_t.to(device)
        process_link_data(data, args, rank=rank)

    processed_input_dim = args.hidden

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
                all_modules_list[i] = DDP(module, device_ids=[rank], find_unused_parameters=True)
    model, predictor, att, mlp, projector, identity_projection = all_modules_list

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

    # --- 4. Main Training Loop ---
    best_valid_hits = 0
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

    if rank == 0: print("\n--- Starting Training Phase ---")
    for epoch in range(args.epochs):
        st = time.time()
        
        # --- Train on all training datasets ---
        total_train_loss = 0
        for data, split_idx in zip(train_data_list, train_split_idx_list):
            link_data_all = prepare_link_data(data, split_idx, args.train_neg_ratio)
            context_data, train_mask = select_link_context(link_data_all['train'], args.context_k, 
                                                           args.context_neg_ratio, args.remove_context_from_train)
            
            if 'train' in link_data_all:
                train_loss = train_link_prediction(
                    model, predictor, data, link_data_all['train'], context_data, train_mask,
                    optimizer, args.batch_size, att, mlp, projector, identity_projection, 
                    args.clip_grad, rank, args.orthogonal_push, args.normalize_class_h, 
                    epoch=epoch, mask_target_edges=args.mask_target_edges, degree=args.degree
                )
                total_train_loss += train_loss
            
            if scheduler is not None:
                scheduler.step()

        # --- Validate on all training datasets ---
        total_valid_hits = 0
        for data, split_idx in zip(train_data_list, train_split_idx_list):
            link_data_all = prepare_link_data(data, split_idx, args.train_neg_ratio)
            # For validation, we can use a dummy mask or re-select context
            context_data, _ = select_link_context(link_data_all['train'], args.context_k, args.context_neg_ratio, remove_from_train=False)

            if 'valid' in link_data_all:
                valid_results = evaluate_link_prediction(
                    model, predictor, data, link_data_all['valid'], context_data, args.test_batch_size,
                    att, mlp, projector, identity_projection, rank, args.normalize_class_h,
                    degree=args.degree, k_values=[100]  # Use Hits@100 as primary metric
                )
                total_valid_hits += valid_results['hits@100']
        
        avg_valid_hits = total_valid_hits / len(train_data_list)
            
        if rank == 0 and epoch % 10 == 0:
            en = time.time()
            print(f"Epoch {epoch}: Avg Loss {(total_train_loss / len(train_data_list)):.4f}, Avg Valid Hits@100 {avg_valid_hits:.4f}, Time: {en-st:.2f}s")
            wandb.log({'epoch': epoch, 'avg_train_loss': total_train_loss / len(train_data_list), 'avg_valid_hits100': avg_valid_hits})

        if avg_valid_hits > best_valid_hits:
            best_valid_hits = avg_valid_hits
            best_epoch = epoch
            if rank == 0:
                # Save all relevant model states
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
    
    if rank == 0:
        print(f"\n--- Training Complete ---")
        print(f"Best validation Hits@100: {best_valid_hits:.4f} at epoch {best_epoch}")

    # --- 5. Inductive Testing Phase ---
    if rank == 0: print("\n--- Starting Inductive Testing Phase ---")
    
    # Load best model weights for all components
    if best_states['model'] is not None:
        if rank == 0: print("Loading best model states for all components...")
        for name, state in best_states.items():
            if state is not None:
                module = locals().get(name)
                if module is not None:
                    module_to_load = module.module if hasattr(module, 'module') else module
                    module_to_load.load_state_dict(state)
    else:
        if rank == 0: print("Warning: No best model found, using the final model state.")

    # Load and process test datasets
    test_dataset_names = args.test_dataset.split(',')
    if rank == 0: print(f"Loading test datasets: {test_dataset_names}")
    test_data_list, test_split_idx_list = load_all_data_link(test_dataset_names)

    test_results = []
    for data, split_idx in zip(test_data_list, test_split_idx_list):
        data.x = data.x.to(device)
        data.adj_t = data.adj_t.to(device)
        process_link_data(data, args, rank=rank)
        
        link_data_all = prepare_link_data(data, split_idx, args.train_neg_ratio)
        context_data, _ = select_link_context(link_data_all['train'], args.context_k, args.context_neg_ratio, remove_from_train=False)

        if 'test' in link_data_all:
            test_results_dict = evaluate_link_prediction(
                model, predictor, data, link_data_all['test'], context_data, args.test_batch_size,
                att, mlp, projector, identity_projection, rank, args.normalize_class_h,
                degree=args.degree, k_values=[20, 50, 100]  # Standard Hits@K values
            )
            test_results.append({'dataset': data.name, 'test_results': test_results_dict})
    
    # --- 6. Aggregate and Log Final Results ---
    if rank == 0 and results_dict is not None:
        results_dict[rank] = test_results

    if world_size > 1:
        cleanup_ddp()

def main():
    """Main function for link prediction."""
    args = parse_link_prediction_args()

    if args.use_pretrained_model and args.load_checkpoint is None:
        raise ValueError("Must provide --load_checkpoint when --use_pretrained_model is True.")

    if not args.use_ddp and not args.sweep:
         wandb.init(project='inductlink', config=args)
    
    # --- DDP or Single-GPU Execution ---
    all_runs_results = []
    if args.use_ddp:
        world_size = torch.cuda.device_count()
        if world_size == 0: return
        print(f"Using DistributedDataParallel with {world_size} GPUs")
        for run in range(args.runs):
            if run > 0: args.port += 1
            print(f"\n🏃 Run {run + 1}/{args.runs}")
            manager = mp.Manager()
            results_dict = manager.dict()
            mp.spawn(run_ddp_lp, args=(world_size, args, results_dict), nprocs=world_size, join=True)
            if 0 in results_dict:
                all_runs_results.append(results_dict[0])
    else: # Single GPU
        for run in range(args.runs):
            print(f"\n🏃 Run {run + 1}/{args.runs}")
            results_dict = {0: None} # Simple dict for single-process
            run_ddp_lp(0, 1, args, results_dict)
            if results_dict[0] is not None:
                all_runs_results.append(results_dict[0])
    
    # --- Aggregate and Log Final Results ---
    if len(all_runs_results) > 0:
        print("\n📊 Final Inductive Test Results")
        print("=" * 60)
        
        # { 'dataset_name': {'test_hits100': [run1_hits100, run2_hits100, ...]} }
        aggregated_results = {}
        for run_res in all_runs_results:
            for res_item in run_res:
                name = res_item['dataset']
                if name not in aggregated_results:
                    aggregated_results[name] = {'test_hits100': []}
                aggregated_results[name]['test_hits100'].append(res_item['test_results']['hits@100'])

        final_log = {}
        for name, data in aggregated_results.items():
            avg_test = np.mean(data['test_hits100'])
            std_test = np.std(data['test_hits100'])
            
            print(f"{name}: Test Hits@100 {avg_test:.4f} ± {std_test:.4f}")
            
            final_log[f'{name}_test_hits100_mean'] = avg_test
            final_log[f'{name}_test_hits100_std'] = std_test
        
        if wandb.run is not None:
            wandb.log(final_log)

    print("\n🎉 Link prediction completed!")

if __name__ == '__main__':
    main() 