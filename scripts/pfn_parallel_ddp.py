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
from src.gpu_utils import (
    parse_gpu_spec, 
    setup_cuda_visible_devices, 
    get_effective_world_size,
    validate_gpu_availability,
    print_gpu_info
)


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

    test_dataset = args.test_dataset.split(',')
    test_data_list, test_split_idx_list = load_all_data(test_dataset)

    for data, split_idx in zip(test_data_list, test_split_idx_list):
        data.x = data.x.to(device)
        data.adj_t = data.adj_t.to(device)
        data.y = data.y.to(device)
        process_data(data, split_idx, args.hidden, args.context_num, args.sign_normalize, args.use_full_pca, 
                     args.normalize_data, args.use_projector, args.min_pca_dim, 0, 
                     args.padding_strategy, args.use_batchnorm, args.use_identity_projection,
                     args.projection_small_dim, args.projection_large_dim)

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
            
            # Get individual dataset metrics during training for better monitoring
            train_metric_list, valid_metric_list, test_metric_list = \
            test_all_induct(model, predictor, data_list, split_idx_list, args.test_batch_size, args.degree, 
                           att, mlp, args.normalize_class_h, projector, rank, identity_projection)
            
            # Calculate aggregated metrics (geometric mean)
            train_metric = (sum(train_metric_list) / len(train_metric_list)) if train_metric_list else 0
            valid_metric = (sum(valid_metric_list) / len(valid_metric_list)) if valid_metric_list else 0
            test_metric = (sum(test_metric_list) / len(test_metric_list)) if test_metric_list else 0
            
            if rank == 0:
                # Log aggregated metrics
                log_dict = {
                    'epoch': epoch,
                    'train_loss': loss, 
                    'train_metric_avg': train_metric, 
                    'valid_metric_avg': valid_metric, 
                    'test_metric_avg': test_metric
                }
                
                # Log individual dataset metrics during training for progress tracking
                for i, data in enumerate(data_list):
                    dataset_name = data.name
                    log_dict.update({
                        f'train_{dataset_name}_test_metric': test_metric_list[i]
                    })
                    print(f"  {dataset_name}: Train={train_metric_list[i]:.4f}, Valid={valid_metric_list[i]:.4f}, Test={test_metric_list[i]:.4f}")
                
                wandb.log(log_dict)
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

    st = time.time()
    train_metric_list, valid_metric_list, test_metric_list = \
    test_all_induct(model, predictor, test_data_list, test_split_idx_list, args.test_batch_size, 
                    args.degree, att, mlp, args.normalize_class_h, projector, rank, identity_projection)
    
    train_metric = sum(train_metric_list) / len(train_metric_list)
    valid_metric = sum(valid_metric_list) / len(valid_metric_list)
    test_metric = sum(test_metric_list) / len(test_metric_list)
    
    if rank == 0:
        print(f"\n=== FINAL TEST RESULTS ===")
        print(f"Test time: {time.time()-st:.2f} seconds", flush=True)
        
        # Create comprehensive logging dictionary
        final_log_dict = {
            'induct_test_metric_avg': test_metric,
            'total_time': time.time()-st
        }
        
        # Log individual dataset results with better formatting
        print(f"\nIndividual Dataset Results:")
        for i, data in enumerate(test_data_list):
            dataset_name = data.name
            print(f"  {dataset_name}:")
            print(f"    Train: {train_metric_list[i]:.4f}")
            print(f"    Valid: {valid_metric_list[i]:.4f}")
            print(f"    Test:  {test_metric_list[i]:.4f}")
            
            # Add individual dataset metrics to logging
            final_log_dict.update({
                f'final_{dataset_name}_train_metric': train_metric_list[i],
                f'final_{dataset_name}_valid_metric': valid_metric_list[i],
                f'final_{dataset_name}_test_metric': test_metric_list[i]
            })
        
        print(f"\nAverage Metrics Across All Datasets:")
        print(f"  Train: {train_metric:.4f}")
        print(f"  Valid: {valid_metric:.4f}")
        print(f"  Test:  {test_metric:.4f}")
        print(f"Total time: {time.time()-st_total:.2f} seconds", flush=True)
        
        # Log everything to wandb
        wandb.log(final_log_dict)

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

def run_single_gpu(args, device='cuda:0'):
    """
    Single GPU execution function that reuses most of the logic from run_ddp
    but without distributed training components.
    """
    # Initialize wandb
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
        print(f"Loading checkpoint from: {args.load_checkpoint}")
        print("Extracting model configuration from checkpoint...")
        
        # Load checkpoint configuration using the new utility function
        checkpoint_info, checkpoint = load_checkpoint_config(args.load_checkpoint)
        
        # Override current args with checkpoint's configuration
        if 'args' in checkpoint:
            checkpoint_args = checkpoint['args']
            args = override_args_from_checkpoint(args, checkpoint_args, rank=0)
        else:
            print("Warning: No argument configuration found in checkpoint, using current arguments")

    # Adjust hidden dimension for identity projection (after potential override)
    if args.use_identity_projection:
        original_hidden = args.hidden
        args.hidden = args.projection_large_dim
        print(f"Identity projection: Adjusted hidden dimension from {original_hidden} to {args.hidden}")

    # Validate nhead and hidden dimension compatibility (after potential override)
    if args.hidden % args.nhead != 0:
        print(f"ERROR: Hidden dimension ({args.hidden}) must be divisible by number of heads ({args.nhead})")
        print("This is likely due to a checkpoint with incompatible configuration.")
        raise ValueError(f"Hidden dimension ({args.hidden}) must be divisible by number of heads ({args.nhead})")

    # CONDITIONAL LR ADJUSTMENT - Prevent deep transformer + high LR disasters (after potential override)
    if args.transformer_layers >= args.safe_transformer_layers:     
        original_lr = args.lr
        max_safe_lr = args.safe_lr  # Much lower for deep transformers
        if args.lr > max_safe_lr:
            args.lr = max_safe_lr
            print(f"WARNING: Deep transformer ({args.transformer_layers} layers) + high LR detected!")
            print(f"Automatically reducing LR from {original_lr} to {args.lr} for stability")
            wandb.log({'lr_auto_adjusted': True, 'original_lr': original_lr, 'adjusted_lr': args.lr})
        else:
            wandb.log({'lr_auto_adjusted': False})
    else:
        wandb.log({'lr_auto_adjusted': False})

    print(f"Device assigned: {device}", flush=True)

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
                print(f"Skipping dataset {data.name} because it has less than {args.hidden} features")
                skip_idx.append(i)
                continue
            process_data(data, split_idx, args.hidden, args.context_num, args.sign_normalize, args.use_full_pca, 
                         args.normalize_data, args.use_projector, args.min_pca_dim, 0, 
                         args.padding_strategy, args.use_batchnorm, args.use_identity_projection,
                         args.projection_small_dim, args.projection_large_dim)

        if args.skip_datasets:
            data_list = [data for i, data in enumerate(data_list) if i not in skip_idx]
            split_idx_list = [split_idx for i, split_idx in enumerate(split_idx_list) if i not in skip_idx]
    else:
        # For checkpoint loading, we don't need to load training data
        data_list, split_idx_list = [], []

    test_dataset = args.test_dataset.split(',')
    test_data_list, test_split_idx_list = load_all_data(test_dataset)

    for data, split_idx in zip(test_data_list, test_split_idx_list):
        data.x = data.x.to(device)
        data.y = data.y.to(device)
        data.adj_t = data.adj_t.to(device)
        process_data(data, split_idx, args.hidden, args.context_num, args.sign_normalize, args.use_full_pca, 
                     args.normalize_data, args.use_projector, args.min_pca_dim, 0, 
                     args.padding_strategy, args.use_batchnorm, args.use_identity_projection,
                     args.projection_small_dim, args.projection_large_dim)

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
        print(f"Created projector: {args.min_pca_dim} -> {args.hidden}")

    # Initialize identity projection if needed
    identity_projection = None
    if args.use_identity_projection:
        identity_projection = IdentityProjection(args.projection_small_dim, args.projection_large_dim)
        identity_projection = identity_projection.to(device)
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

    # Note: No DDP wrapping for single GPU execution

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
        print(f"Added {sum(p.numel() for p in projector.parameters())} projector parameters")
    
    # Add identity projection parameters if using identity projection
    if args.use_identity_projection and identity_projection is not None:
        add_params(identity_projection)
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

    print(f'number of parameters: {sum(p.numel() for p in parameters)}', flush=True)

    # Check if we should load a checkpoint and skip training
    if args.load_checkpoint is not None:
        print("Skipping training and going directly to testing...")
        
        # Load model states using the new utility function
        best_epoch, best_valid, final_test = load_checkpoint_states(
            checkpoint, model, predictor, optimizer, att, mlp, projector, 
            identity_projection, scheduler, rank=0
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
            print(f"Epoch {epoch}", flush=True)
            
            loss = train_all(model, data_list, split_idx_list, optimizer, predictor, args.batch_size,
                             args.degree, att, mlp, args.orthogonal_push, args.normalize_class_h, 
                             args.clip_grad, projector, rank=0, epoch=epoch, identity_projection=identity_projection)
            if scheduler is not None:
                scheduler.step()
            
            # Get individual dataset metrics during training for better monitoring
            train_metric_list, valid_metric_list, test_metric_list = \
            test_all_induct(model, predictor, data_list, split_idx_list, args.test_batch_size, args.degree, 
                           att, mlp, args.normalize_class_h, projector, rank=0, identity_projection=identity_projection)
            
            # Calculate aggregated metrics (arithmetic mean for single GPU)
            train_metric = sum(train_metric_list) / len(train_metric_list) if train_metric_list else 0
            valid_metric = sum(valid_metric_list) / len(valid_metric_list) if valid_metric_list else 0
            test_metric = sum(test_metric_list) / len(test_metric_list) if test_metric_list else 0

            induct_train_metric_list, induct_valid_metric_list, induct_test_metric_list = \
            test_all_induct(model, predictor, test_data_list, test_split_idx_list, args.test_batch_size, args.degree, 
                            att, mlp, args.normalize_class_h, projector, rank=0, identity_projection=identity_projection)

            induct_train_metric = sum(induct_train_metric_list) / len(induct_train_metric_list)
            induct_valid_metric = sum(induct_valid_metric_list) / len(induct_valid_metric_list)
            induct_test_metric = sum(induct_test_metric_list) / len(induct_test_metric_list)
            
            # Log aggregated metrics
            log_dict = {
                'epoch': epoch,
                'train_loss': loss, 
                'train_metric_avg': train_metric, 
                'valid_metric_avg': valid_metric, 
                'test_metric_avg': test_metric,
                'induct_test_metric_avg': induct_test_metric
            }
            
            # Log individual dataset metrics during training for progress tracking
            for i, data in enumerate(data_list):
                dataset_name = data.name
                log_dict.update({
                    f'train_{dataset_name}_test_metric': test_metric_list[i],
                })
                print(f"  {dataset_name}: Train={train_metric_list[i]:.4f}, Valid={valid_metric_list[i]:.4f}, Test={test_metric_list[i]:.4f}")
            
            for i, data in enumerate(test_data_list):
                dataset_name = data.name
                log_dict.update({
                    f'induct_{dataset_name}_test_metric': induct_test_metric_list[i],
                })
                print(f"  {dataset_name}: Induct Train={induct_train_metric_list[i]:.4f}, "
                      f"Induct Valid={induct_valid_metric_list[i]:.4f}, Induct Test={induct_test_metric_list[i]:.4f}")
            
            wandb.log(log_dict)
            en = time.time()
            print(f"Epoch time: {en-st}", flush=True)
            
            if valid_metric >= best_valid:
                best_valid = valid_metric
                best_epoch = epoch
                final_test = test_metric
                # Save all relevant model states
                best_states['model'] = copy.deepcopy(model.state_dict())
                best_states['predictor'] = copy.deepcopy(predictor.state_dict())
                if att is not None:
                    best_states['att'] = copy.deepcopy(att.state_dict())
                if mlp is not None:
                    best_states['mlp'] = copy.deepcopy(mlp.state_dict())
                if projector is not None:
                    best_states['projector'] = copy.deepcopy(projector.state_dict())
                if identity_projection is not None:
                    best_states['identity_projection'] = copy.deepcopy(identity_projection.state_dict())

        print(f"Memory: {torch.cuda.max_memory_allocated() / 1e9} GB", flush=True)
        print(f"Training completed in: {time.time()-st_all:.2f} seconds", flush=True)
        wandb.log({'final_test_training': final_test, 'best_epoch': best_epoch, 'best_valid': best_valid})
        print(f"Best epoch: {best_epoch}, Best valid: {best_valid:.4f}, Final test: {final_test:.4f}", flush=True)

        # Load best model weights for all components
        if best_states['model'] is not None:
            print("Loading best model states for all components...")
            model.load_state_dict(best_states['model'])
            predictor.load_state_dict(best_states['predictor'])
            if att is not None and best_states['att'] is not None:
                att.load_state_dict(best_states['att'])
            if mlp is not None and best_states['mlp'] is not None:
                mlp.load_state_dict(best_states['mlp'])
            if projector is not None and best_states['projector'] is not None:
                projector.load_state_dict(best_states['projector'])
            if identity_projection is not None and best_states['identity_projection'] is not None:
                identity_projection.load_state_dict(best_states['identity_projection'])
        else:
            print("Warning: No best model found, using the final model state.")

    train_metric_list, valid_metric_list, test_metric_list = \
    test_all_induct(model, predictor, test_data_list, test_split_idx_list, args.test_batch_size, args.degree, 
                    att, mlp, args.normalize_class_h, projector, rank=0, identity_projection=identity_projection)

    train_metric = sum(train_metric_list) / len(train_metric_list)
    valid_metric = sum(valid_metric_list) / len(valid_metric_list)
    test_metric = sum(test_metric_list) / len(test_metric_list)

    print(f"\n=== FINAL TEST RESULTS ===")
    print(f"Test time: {time.time()-st:.2f} seconds", flush=True)
    
    en_all = time.time()
    total_time = en_all - st_all
    # Create comprehensive logging dictionary
    final_log_dict = {
        'induct_test_metric_avg': test_metric,
        'final_test_time': time.time()-st,
        'best_epoch': best_epoch,
        'total_time': total_time
    }
    
    # Log individual dataset results with better formatting
    print(f"\nIndividual Dataset Results:")
    for i, data in enumerate(test_data_list):
        dataset_name = data.name
        print(f"  {dataset_name}:")
        print(f"    Train: {train_metric_list[i]:.4f}")
        print(f"    Valid: {valid_metric_list[i]:.4f}")
        print(f"    Test:  {test_metric_list[i]:.4f}")
        
        # Add individual dataset metrics to logging
        final_log_dict.update({
            f'final_{dataset_name}_test_metric': test_metric_list[i]
        })
    
    print(f"\nAverage Metrics Across All Datasets:")
    print(f"  Train: {train_metric:.4f}")
    print(f"  Valid: {valid_metric:.4f}")
    print(f"  Test:  {test_metric:.4f}")
    
    # Log everything to wandb
    wandb.log(final_log_dict)

    en_all = time.time()
    total_time = en_all - st_all
    print(f"\\nTotal time: {total_time:.2f} seconds", flush=True)

    # Save checkpoint if requested
    if args.save_checkpoint and args.load_checkpoint is None:
        best_metrics = {
            'best_epoch': best_epoch,
            'best_valid': best_valid,
            'final_train': train_metric,
            'final_valid': valid_metric,
            'final_test': test_metric,
            'total_time': total_time
        }
        
        checkpoint_path = save_checkpoint(
            model, predictor, optimizer, args, best_metrics, best_epoch,
            att=att, mlp=mlp, projector=projector, identity_projection=identity_projection,
            scheduler=scheduler, rank=0
        )
        
        # Log checkpoint path to wandb
        wandb.log({'checkpoint_path': checkpoint_path})
    elif args.save_checkpoint and args.load_checkpoint is not None:
        print("Skipping checkpoint save since we loaded from checkpoint (testing only)")
        print(f"Original checkpoint: {args.load_checkpoint}")

    return train_metric, valid_metric, test_metric

def main():
    args = parse()
    
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
    use_single_gpu = getattr(args, 'single_gpu', False) or world_size == 1
    
    if use_single_gpu:
        print(f"Using Single GPU mode on device cuda:0 (Physical GPU {gpu_ids[0]})")
        avg_train_metric = 0
        avg_valid_metric = 0
        avg_test_metric = 0
        
        for run_idx in range(args.runs):
            print(f"Run {run_idx + 1}/{args.runs}")
            
            # Run single GPU execution
            train_metric, valid_metric, test_metric = run_single_gpu(args, device='cuda:0')
            
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
        
    else:
        print(f"Using DistributedDataParallel with {world_size} GPUs (Physical GPUs {gpu_ids})")
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