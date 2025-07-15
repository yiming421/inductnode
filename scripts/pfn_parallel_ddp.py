import os
import sys
# Add the project's root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import wandb
import copy
from transformers import get_cosine_schedule_with_warmup

from src.model import PureGCN_v1, PureGCN, PFNPredictorNodeCls, GCN, AttentionPool, MLP, IdentityProjection
from src.data import load_all_data, load_all_data_train
from src.checkpoint_utils import (
    save_checkpoint,
    load_checkpoint_config, 
    override_args_from_checkpoint, 
    load_checkpoint_states
)
from src.config import parse
from src.data_utils import process_data
from src.ddp_utils import setup_ddp, cleanup_ddp
from src.engine import train_all, test_all, test_all_induct


def run_ddp(rank, world_size, args, results_dict=None):
    # Setup distributed training
    setup_ddp(rank, world_size, args.port)
    # Only initialize wandb on rank 0
    if rank == 0:
        if args.sweep:
            wandb.init(project='inductnode')
            config = wandb.config
            for key in config.keys():
                setattr(args, key, config[key])
        else:
            wandb.init(project='inductnode', config=args)

    # Check if we should load a checkpoint and extract configuration
    checkpoint = None
    if args.load_checkpoint is not None:
        if rank == 0:
            print(f"Loading checkpoint from: {args.load_checkpoint}")
            print("Extracting model configuration from checkpoint...")
        
        # Load checkpoint configuration using the new utility function
        checkpoint_info, checkpoint = load_checkpoint_config(args.load_checkpoint)
        
        # Override current args with checkpoint's configuration
        if 'args' in checkpoint:
            checkpoint_args = checkpoint['args']
            args = override_args_from_checkpoint(args, checkpoint_args, rank)
        else:
            if rank == 0:
                print("Warning: No argument configuration found in checkpoint, using current arguments")

    # Adjust hidden dimension for identity projection (after potential override)
    if args.use_identity_projection:
        original_hidden = args.hidden
        args.hidden = args.projection_large_dim
        if rank == 0:
            print(f"Identity projection: Adjusted hidden dimension from {original_hidden} to {args.hidden}")

    # Validate nhead and hidden dimension compatibility (after potential override)
    if args.hidden % args.nhead != 0:
        if rank == 0:
            print(f"ERROR: Hidden dimension ({args.hidden}) must be divisible by number of heads ({args.nhead})")
            print("This is likely due to a checkpoint with incompatible configuration.")
        raise ValueError(f"Hidden dimension ({args.hidden}) must be divisible by number of heads ({args.nhead})")

    # CONDITIONAL LR ADJUSTMENT - Prevent deep transformer + high LR disasters (after potential override)
    if args.transformer_layers >= args.safe_transformer_layers:     
        original_lr = args.lr
        max_safe_lr = args.safe_lr  # Much lower for deep transformers
        if args.lr > max_safe_lr:
            args.lr = max_safe_lr
            if rank == 0:
                print(f"WARNING: Deep transformer ({args.transformer_layers} layers) + high LR detected!")
                print(f"Automatically reducing LR from {original_lr} to {args.lr} for stability")
                wandb.log({'lr_auto_adjusted': True, 'original_lr': original_lr, 'adjusted_lr': args.lr})
        else:
            if rank == 0:
                wandb.log({'lr_auto_adjusted': False})
    else:
        if rank == 0:
            wandb.log({'lr_auto_adjusted': False})

    device = f'cuda:{rank}'
    print(f"[RANK {rank}] Device assigned: {device}", flush=True)
    
    # Handle distributed training parameters
    if world_size > 1:
        original_batch_size = args.batch_size
        args.batch_size = args.batch_size // world_size  # Divide by world_size for work distribution
        
        if rank == 0:
            print(f"Distributed training with {world_size} GPUs")
            print(f"Per-GPU batch size: {args.batch_size} (total effective: {original_batch_size})")
            print(f"Learning rate: {args.lr} (unchanged for identical single-GPU behavior)")
        
        print(f"[RANK {rank}] My batch size: {args.batch_size}, My LR: {args.lr}", flush=True)

    # If loading from checkpoint, we only need test datasets
    # otherwise, we need training datasets
    if args.load_checkpoint is None:
        train_dataset = args.train_dataset.split(',')
        data_list, split_idx_list = load_all_data_train(train_dataset)
        if args.skip_datasets:
            skip_idx = []

        for i, (data, split_idx) in enumerate(zip(data_list, split_idx_list)):
            data.x = data.x.to(device)
            data.adj_t = data.adj_t.to(device)
            data.y = data.y.to(device)
            if args.skip_datasets and data.x.size(1) < args.hidden:
                if rank == 0:
                    print(f"Skipping dataset {data.name} because it has less than {args.hidden} features")
                skip_idx.append(i)
                continue
            process_data(data, split_idx, args.hidden, args.context_num, args.sign_normalize, args.use_full_pca, 
                         args.normalize_data, args.use_projector, args.min_pca_dim, rank, 
                         args.padding_strategy, args.use_batchnorm, args.use_identity_projection,
                         args.projection_small_dim, args.projection_large_dim)

        if args.skip_datasets:
            data_list = [data for i, data in enumerate(data_list) if i not in skip_idx]
            split_idx_list = [split_idx for i, split_idx in enumerate(split_idx_list) if i not in skip_idx]
    else:
        # For checkpoint loading, we don't need to load training data
        data_list, split_idx_list = [], []

    att, mlp = None, None

    if args.att_pool:
        att = AttentionPool(args.hidden, args.hidden // args.nhead, args.nhead, args.dp)
        att = att.to(device)
    if args.mlp_pool:
        mlp = MLP(args.hidden, args.hidden, args.hidden, 2, args.dp, args.norm, False, args.mlp_norm_affine)
        mlp = mlp.to(device)
    
    # Initialize projector if needed
    projector = None
    if args.use_projector:
        projector = MLP(args.min_pca_dim, args.hidden, args.hidden, 2, args.dp, args.norm, False, args.mlp_norm_affine)
        projector = projector.to(device)
        if rank == 0:
            print(f"Created projector: {args.min_pca_dim} -> {args.hidden}")

    # Initialize identity projection if needed
    identity_projection = None
    if args.use_identity_projection:
        identity_projection = IdentityProjection(args.projection_small_dim, args.projection_large_dim)
        identity_projection = identity_projection.to(device)
        if rank == 0:
            print(f"Created identity projection: {args.projection_small_dim} -> {args.projection_large_dim}")

    if args.model == 'PureGCN':
        model = PureGCN(args.num_layers)
    elif args.model == 'PureGCN_v1':
        model = PureGCN_v1(args.hidden, args.num_layers, args.hidden, args.dp, args.norm, args.res,
                            args.relu, args.gnn_norm_affine)
    elif args.model == 'GCN':
        model = GCN(args.hidden, args.hidden, args.norm, args.relu, args.num_layers, args.dp,
                    args.multilayer, args.use_gin, args.res, args.gnn_norm_affine)
    else:
        raise NotImplementedError

    if args.predictor == 'PFN':
        predictor = PFNPredictorNodeCls(args.hidden, args.nhead, args.transformer_layers, 
                                        args.mlp_layers, args.dp, args.norm, args.seperate, 
                                        args.degree, att, mlp, args.sim, args.padding, 
                                        args.mlp_norm_affine, args.normalize_class_h)
    else:
        raise NotImplementedError

    model = model.to(device)
    predictor = predictor.to(device)

    # Wrap with DDP only if the module has parameters
    if any(p.requires_grad for p in model.parameters()):
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    if any(p.requires_grad for p in predictor.parameters()):
        predictor = DDP(predictor, device_ids=[rank], find_unused_parameters=True)
    
    if att is not None and any(p.requires_grad for p in att.parameters()):
        att = DDP(att, device_ids=[rank], find_unused_parameters=True)
    if mlp is not None and any(p.requires_grad for p in mlp.parameters()):
        mlp = DDP(mlp, device_ids=[rank], find_unused_parameters=True)
    if projector is not None and any(p.requires_grad for p in projector.parameters()):
        projector = DDP(projector, device_ids=[rank], find_unused_parameters=True)
    if identity_projection is not None and any(p.requires_grad for p in identity_projection.parameters()):
        identity_projection = DDP(identity_projection, device_ids=[rank], find_unused_parameters=True)

    # Collect parameters
    parameters = []
    # Use a helper to add parameters from a module if it's not None
    def add_params(module):
        if module is not None:
            parameters.extend(list(module.parameters()))

    add_params(model)
    add_params(predictor)
    
    # Add projector parameters if using projector
    if args.use_projector and projector is not None:
        add_params(projector)
        if rank == 0:
            print(f"Added {sum(p.numel() for p in projector.parameters())} projector parameters")
    
    # Add identity projection parameters if using identity projection
    if args.use_identity_projection and identity_projection is not None:
        add_params(identity_projection)
        if rank == 0:
            print(f"Added {sum(p.numel() for p in identity_projection.parameters())} identity projection parameters")

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError

    if args.schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.schedule == 'step':
        step_size = args.epochs // 5
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    elif args.schedule == 'warmup':
        warmup_steps = args.epochs // 10
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=args.epochs)
    else:
        scheduler = None

    if rank == 0:
        print(f'number of parameters: {sum(p.numel() for p in parameters)}', flush=True)

    # Check if we should load a checkpoint and skip training
    if args.load_checkpoint is not None:
        if rank == 0:
            print("Skipping training and going directly to testing...")
        
        # Load model states using the new utility function
        best_epoch, best_valid, final_test = load_checkpoint_states(
            checkpoint, model, predictor, optimizer, att, mlp, projector, 
            identity_projection, scheduler, rank
        )
        
        # Skip training loop entirely
        st_all = time.time()  # Start timing for testing only
        
    else:
        # Regular training path
        st_all = time.time()
        best_valid = 0
        final_test = 0
        best_epoch = 0
        # Create a dictionary to hold the state of all best models
        best_states = {
            'model': None, 'predictor': None, 'att': None, 'mlp': None,
            'projector': None, 'identity_projection': None
        }

        for epoch in range(args.epochs):
            st = time.time()
            if rank == 0:
                print(f"Epoch {epoch}", flush=True)
            
            # Set epoch for distributed samplers in train function
            # This will be handled inside the train function for each dataset
            
            loss = train_all(model, data_list, split_idx_list, optimizer, predictor, args.batch_size,
                             args.degree, att, mlp, args.orthogonal_push, args.normalize_class_h, 
                             args.clip_grad, projector, rank, epoch, identity_projection)
            if scheduler is not None:
                scheduler.step()
            
            train_metric, valid_metric, test_metric = \
            test_all(model, predictor, data_list, split_idx_list, args.test_batch_size, args.degree, 
                     att, mlp, args.normalize_class_h, projector, rank, identity_projection)
            
            if rank == 0:
                wandb.log({'train_loss': loss, 'train_metric': train_metric, 'valid_metric': valid_metric, 
                           'test_metric': test_metric})
                en = time.time()
                print(f"Epoch time: {en-st}", flush=True)
            
            if valid_metric >= best_valid:
                best_valid = valid_metric
                best_epoch = epoch
                final_test = test_metric
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

            if epoch - best_epoch >= 200:
                break

        if rank == 0:
            print(f"Memory: {torch.cuda.max_memory_allocated() / 1e9} GB", flush=True)
            print(f"Total time: {time.time()-st_all}", flush=True)
            wandb.log({'final_test': final_test})
            print(f"Best epoch: {best_epoch}", flush=True)

        # Load best model weights for all components
        if best_states['model'] is not None:
            if rank == 0: print("Loading best model states for all components...")
            all_modules = {'model': model, 'predictor': predictor, 'att': att, 'mlp': mlp, 'projector': projector, 'identity_projection': identity_projection}
            for name, state in best_states.items():
                if state is not None and all_modules.get(name) is not None:
                    module_to_load = all_modules[name].module if hasattr(all_modules[name], 'module') else all_modules[name]
                    module_to_load.load_state_dict(state)
        else:
            if rank == 0: print("Warning: No best model found, using the final model state.")
    
    test_dataset = args.test_dataset.split(',')
    data_list, split_idx_list = load_all_data(test_dataset)

    for data, split_idx in zip(data_list, split_idx_list):
        data.x = data.x.to(device)
        data.y = data.y.to(device)
        data.adj_t = data.adj_t.to(device)
        process_data(data, split_idx, args.hidden, args.context_num, args.sign_normalize, args.use_full_pca, 
                     args.normalize_data, args.use_projector, args.min_pca_dim, rank, 
                     args.padding_strategy, args.use_batchnorm, args.use_identity_projection,
                     args.projection_small_dim, args.projection_large_dim)

    st_total = time.time()
    st = time.time()
    train_metric_list, valid_metric_list, test_metric_list = \
    test_all_induct(model, predictor, data_list, split_idx_list, args.test_batch_size, 
                    args.degree, att, mlp, args.normalize_class_h, projector, rank, identity_projection)
    
    train_metric = sum(train_metric_list) / len(train_metric_list)
    valid_metric = sum(valid_metric_list) / len(valid_metric_list)
    test_metric = sum(test_metric_list) / len(test_metric_list)
    
    if rank == 0:
        print(f"Test time: {time.time()-st}", flush=True)
        for i, data in enumerate(data_list):
            print(f"Dataset {data.name}")
            print(f"Train {train_metric_list[i]}")
            print(f"Valid {valid_metric_list[i]}")
            print(f"Test {test_metric_list[i]}")
            wandb.log({f'{data.name}_train_metric': train_metric_list, 
                       f'{data.name}_valid_metric': valid_metric_list[i], 
                       f'{data.name}_test_metric': test_metric_list[i]})
        
        print(f'induct_train_metric: {train_metric}, induct_valid_metric: {valid_metric}, induct_test_metric: {test_metric}', flush=True)
        wandb.log({'induct_train_metric': train_metric, 'induct_valid_metric': valid_metric, 
                   'induct_test_metric': test_metric})
        print(f"Total time: {time.time()-st_total}", flush=True)

    # Store results for collection (only rank 0 stores results)
    if rank == 0 and results_dict is not None:
        results_dict['train_metric'] = train_metric
        results_dict['valid_metric'] = valid_metric
        results_dict['test_metric'] = test_metric

    # Save checkpoint after training completion
    if args.save_checkpoint and args.load_checkpoint is None:
        # Only save checkpoint if we actually trained (not loading from checkpoint)
        best_metrics = {
            'train_metric': train_metric,
            'valid_metric': valid_metric,
            'test_metric': test_metric,
            'best_epoch': best_epoch,
            'final_test': final_test,
            'best_valid': best_valid
        }
        
        checkpoint_path = save_checkpoint(
            model, predictor, optimizer, args, best_metrics, best_epoch,
            att=att, mlp=mlp, projector=projector, identity_projection=identity_projection,
            scheduler=scheduler, rank=rank
        )
        
        # Log checkpoint path to wandb
        if rank == 0:
            wandb.log({'checkpoint_path': checkpoint_path})
    elif args.save_checkpoint and args.load_checkpoint is not None:
        if rank == 0:
            print("Skipping checkpoint save since we loaded from checkpoint (testing only)")
            print(f"Original checkpoint: {args.load_checkpoint}")

    cleanup_ddp()
    return train_metric, valid_metric, test_metric

def main():
    args = parse()
    
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No CUDA devices are available. This script requires GPUs. Exiting.", flush=True)
        return

    print(f"Using DistributedDataParallel with {world_size} GPUs")
    avg_train_metric = 0
    avg_valid_metric = 0
    avg_test_metric = 0
    
    # Create a manager for shared results
    manager = mp.Manager()
    
    for run_idx in range(args.runs):
        print(f"Run {run_idx + 1}/{args.runs}")
        # Create shared dictionary for results
        results_dict = manager.dict()
        
        # Spawn processes for DDP
        mp.spawn(run_ddp, args=(world_size, args, results_dict), nprocs=world_size, join=True)
        
        # Collect results from shared dictionary
        train_metric = results_dict.get('train_metric', 0.0)
        valid_metric = results_dict.get('valid_metric', 0.0)
        test_metric = results_dict.get('test_metric', 0.0)
        
        print(f"Run {run_idx + 1} Results:")
        print(f"  Train: {train_metric:.4f}")
        print(f"  Valid: {valid_metric:.4f}")
        print(f"  Test: {test_metric:.4f}")
        
        avg_train_metric += train_metric
        avg_valid_metric += valid_metric
        avg_test_metric += test_metric
    
    avg_train_metric /= args.runs
    avg_valid_metric /= args.runs
    avg_test_metric /= args.runs
    
    print('Average Train Metric')
    print(avg_train_metric)
    print('Average Valid Metric')
    print(avg_valid_metric)
    print('Average Test Metric')
    print(avg_test_metric)
    
    wandb.init(project='inductnode')
    wandb.log({
        'avg_train_metric': avg_train_metric, 
        'avg_valid_metric': avg_valid_metric, 
        'avg_test_metric': avg_test_metric,
    })
    print(f"Logged average metrics to wandb for sweep optimization: avg_test_metric={avg_test_metric:.4f}")

if __name__ == '__main__':
    main() 