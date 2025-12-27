import os
import torch
import datetime
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.model import PFNPredictorNodeCls


def save_checkpoint(model, predictor, optimizer, args, best_metrics, epoch,
                   att=None, mlp=None, projector=None, identity_projection=None,
                   scheduler=None, rank=0):
    """Save checkpoint with all training state"""
    if rank != 0:
        return  # Only rank 0 saves checkpoints

    # Check if checkpoint threshold is met (only if threshold is set)
    if hasattr(args, 'checkpoint_threshold') and args.checkpoint_threshold > 0:
        # Calculate combined score from best_metrics
        combined_score = best_metrics.get('final_test', 0)  # final_test contains the combined score
        if combined_score < args.checkpoint_threshold:
            print(f"❌ Checkpoint not saved: combined score {combined_score:.4f} below threshold {args.checkpoint_threshold:.4f}")
            return None
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Generate checkpoint name
    if args.checkpoint_name is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_epoch_{epoch}_{timestamp}.pt"
    else:
        checkpoint_name = args.checkpoint_name
        if not checkpoint_name.endswith('.pt'):
            checkpoint_name += '.pt'
    
    checkpoint_path = os.path.join(args.checkpoint_dir, checkpoint_name)
    
    # Prepare checkpoint data
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'predictor_state_dict': predictor.module.state_dict() if hasattr(predictor, 'module') else predictor.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metrics': best_metrics,
        'args': vars(args),  # Save all arguments
        'model_config': {
            'model_type': args.model,
            'predictor_type': args.predictor,
            'hidden_dim': args.hidden,
            'num_layers': args.num_layers,
            'transformer_layers': args.transformer_layers,
            'nhead': args.nhead,
        }
    }
    
    # Add scheduler state if available
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    # Add optional components
    if att is not None:
        checkpoint['att_state_dict'] = att.module.state_dict() if hasattr(att, 'module') else att.state_dict()
    if mlp is not None:
        checkpoint['mlp_state_dict'] = mlp.module.state_dict() if hasattr(mlp, 'module') else mlp.state_dict()
    if projector is not None:
        checkpoint['projector_state_dict'] = projector.module.state_dict() if hasattr(projector, 'module') else projector.state_dict()
    if identity_projection is not None:
        checkpoint['identity_projection_state_dict'] = identity_projection.module.state_dict() if hasattr(identity_projection, 'module') else identity_projection.state_dict()
    
    # Save checkpoint
    torch.save(checkpoint, checkpoint_path)

    # Show threshold information if applicable
    if hasattr(args, 'checkpoint_threshold') and args.checkpoint_threshold > 0:
        combined_score = best_metrics.get('final_test', 0)
        print(f"✅ Checkpoint saved: combined score {combined_score:.4f} meets threshold {args.checkpoint_threshold:.4f}")

    print(f"💾 Checkpoint saved to: {checkpoint_path}")
    print(f"📊 Best metrics: {best_metrics}")
    
    # Also save a "latest" checkpoint for easy loading
    latest_path = os.path.join(args.checkpoint_dir, "latest_checkpoint.pt")
    torch.save(checkpoint, latest_path)
    print(f"🔄 Latest checkpoint saved to: {latest_path}")
    
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, predictor, optimizer=None, 
                   att=None, mlp=None, projector=None, identity_projection=None, 
                   scheduler=None, device='cpu'):
    """Load checkpoint and restore training state"""
    
    print(f"🔄 Loading checkpoint from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model states
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if hasattr(predictor, 'module'):
        predictor.module.load_state_dict(checkpoint['predictor_state_dict'])
    else:
        predictor.load_state_dict(checkpoint['predictor_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if available
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Load optional components
    if att is not None and 'att_state_dict' in checkpoint:
        if hasattr(att, 'module'):
            att.module.load_state_dict(checkpoint['att_state_dict'])
        else:
            att.load_state_dict(checkpoint['att_state_dict'])
    
    if mlp is not None and 'mlp_state_dict' in checkpoint:
        if hasattr(mlp, 'module'):
            mlp.module.load_state_dict(checkpoint['mlp_state_dict'])
        else:
            mlp.load_state_dict(checkpoint['mlp_state_dict'])
    
    if projector is not None and 'projector_state_dict' in checkpoint:
        if hasattr(projector, 'module'):
            projector.module.load_state_dict(checkpoint['projector_state_dict'])
        else:
            projector.load_state_dict(checkpoint['projector_state_dict'])
    
    if identity_projection is not None and 'identity_projection_state_dict' in checkpoint:
        if hasattr(identity_projection, 'module'):
            identity_projection.module.load_state_dict(checkpoint['identity_projection_state_dict'])
        else:
            identity_projection.load_state_dict(checkpoint['identity_projection_state_dict'])
    
    # Return checkpoint info
    info = {
        'epoch': checkpoint['epoch'],
        'best_metrics': checkpoint['best_metrics'],
        'args': checkpoint['args'],
        'model_config': checkpoint['model_config']
    }
    
    print(f"✅ Checkpoint loaded successfully!")
    print(f"📊 Epoch: {info['epoch']}")
    print(f"🎯 Best metrics: {info['best_metrics']}")
    
    return info


def load_checkpoint_config(checkpoint_path):
    """Load only the configuration from checkpoint without loading model states"""
    
    print(f"🔄 Loading checkpoint configuration from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Return checkpoint info
    info = {
        'epoch': checkpoint['epoch'],
        'best_metrics': checkpoint['best_metrics'],
        'args': checkpoint['args'],
        'model_config': checkpoint.get('model_config', {})
    }
    
    print(f"✅ Checkpoint configuration loaded successfully!")
    print(f"📊 Epoch: {info['epoch']}")
    print(f"🎯 Best metrics: {info['best_metrics']}")
    
    return info, checkpoint


def override_args_from_checkpoint(args, checkpoint_args, rank=0):
    """Override current arguments with checkpoint configuration"""

    if rank == 0:
        print("Overriding current arguments with checkpoint configuration:")
        print(f"  Original hidden: {args.hidden} -> Checkpoint hidden: {checkpoint_args['hidden']}")
        print(f"  Original num_layers: {args.num_layers} -> Checkpoint num_layers: {checkpoint_args['num_layers']}")
        print(f"  Original transformer_layers: {args.transformer_layers} -> Checkpoint transformer_layers: {checkpoint_args['transformer_layers']}")
        print(f"  Original nhead: {args.nhead} -> Checkpoint nhead: {checkpoint_args['nhead']}")
        print(f"  Original model: {args.model} -> Checkpoint model: {checkpoint_args['model']}")
        print(f"  Original predictor: {args.predictor} -> Checkpoint predictor: {checkpoint_args['predictor']}")
        if 'head_num_layers' in checkpoint_args:
            print(f"  Original head_num_layers: {getattr(args, 'head_num_layers', 2)} -> Checkpoint head_num_layers: {checkpoint_args['head_num_layers']}")

    # Override key model architecture parameters (with fallback for missing keys)
    args.hidden = checkpoint_args['hidden']
    args.num_layers = checkpoint_args['num_layers']
    args.transformer_layers = checkpoint_args['transformer_layers']
    args.nhead = checkpoint_args['nhead']
    args.model = checkpoint_args['model']
    args.predictor = checkpoint_args['predictor']
    args.mlp_layers = checkpoint_args['mlp_layers']
    args.dp = checkpoint_args['dp']
    args.norm = checkpoint_args['norm']
    args.seperate = checkpoint_args['seperate']
    args.padding = checkpoint_args['padding']
    args.sim = checkpoint_args['sim']
    args.normalize_class_h = checkpoint_args['normalize_class_h']
    args.use_identity_projection = checkpoint_args['use_identity_projection']
    args.projection_small_dim = checkpoint_args['projection_small_dim']
    args.projection_large_dim = checkpoint_args['projection_large_dim']
    args.gnn_norm_affine = checkpoint_args['gnn_norm_affine']
    args.mlp_norm_affine = checkpoint_args['mlp_norm_affine']
    args.relu = checkpoint_args['relu']
    args.res = checkpoint_args['res']
    args.multilayer = checkpoint_args['multilayer']
    args.use_gin = checkpoint_args['use_gin']

    # Override head_num_layers if present in checkpoint (added for backwards compatibility)
    if 'head_num_layers' in checkpoint_args:
        args.head_num_layers = checkpoint_args['head_num_layers']

    if rank == 0:
        print("Model configuration successfully overridden from checkpoint!")

    return args


def create_gnn_from_config(model_config, args_dict, input_dim):
    """Creates a GNN model from a configuration dictionary."""
    from .model import PureGCN, PureGCN_v1, GCN
    model_type = model_config.get('model_type') or args_dict.get('model')
    
    if model_type == 'PureGCN':
        model = PureGCN(args_dict['num_layers'])
    elif model_type == 'PureGCN_v1':
        model = PureGCN_v1(input_dim, args_dict['num_layers'], args_dict['hidden'], 
                          args_dict['dp'], args_dict['norm'], args_dict['res'], 
                          args_dict['relu'], args_dict['gnn_norm_affine'])
    elif model_type == 'GCN':
        model = GCN(input_dim, args_dict['hidden'], args_dict['norm'], 
                   args_dict['relu'], args_dict['num_layers'], args_dict['dp'], 
                   args_dict['multilayer'], args_dict['use_gin'], args_dict['res'], 
                   args_dict['gnn_norm_affine'])
    else:
        raise ValueError(f"Unknown GNN model type: {model_type}")
    return model

def create_pfn_components_from_config(args_dict, device='cpu'):
    """Creates all PFN-related components from a configuration dictionary."""
    from .model import AttentionPool, MLP, IdentityProjection
    use_att_pool = args_dict.get('att_pool', False)
    use_mlp_pool = args_dict.get('mlp_pool', False)
    use_projector = args_dict.get('use_projector', False)
    use_identity_projection = args_dict.get('use_identity_projection', False)

    att = AttentionPool(
        args_dict['hidden'], 
        args_dict['hidden'] // args_dict['nhead'], 
        args_dict['nhead'], 
        args_dict['dp']
    ) if use_att_pool else None
    
    mlp = MLP(
        args_dict['hidden'], 
        args_dict['hidden'], 
        args_dict['hidden'], 
        args_dict['mlp_layers'], 
        args_dict['dp'], 
        args_dict['norm'], 
        False, 
        args_dict['mlp_norm_affine']
    ) if use_mlp_pool else None
    
    projector = MLP(
        args_dict['min_pca_dim'], 
        args_dict['hidden'], 
        args_dict['hidden'], 
        2, 
        args_dict['dp'], 
        args_dict['norm'], 
        False, 
        args_dict['mlp_norm_affine']
    ) if use_projector else None
    
    identity_projection = IdentityProjection(
        args_dict['projection_small_dim'], 
        args_dict['projection_large_dim']
    ) if use_identity_projection else None

    return att, mlp, projector, identity_projection


def create_model_from_args(args, input_dim, device):
    """Creates a GNN model and all PFN components from command-line arguments."""
    from .model import PureGCN, PureGCN_v1, GCN, AttentionPool, MLP, IdentityProjection
    
    # GNN Model
    if args.model == 'PureGCN':
        model = PureGCN(args.num_layers)
    elif args.model == 'PureGCN_v1':
        model = PureGCN_v1(input_dim, args.num_layers, args.hidden, args.dp, args.norm, args.res, args.relu, args.gnn_norm_affine)
    elif args.model == 'GCN':
        model = GCN(input_dim, args.hidden, args.norm, args.relu, args.num_layers, args.dp, args.multilayer, args.use_gin, args.res, args.gnn_norm_affine)
    else:
        raise ValueError(f"Unknown model type: {args.model}")
    model = model.to(device)

    # PFN Components
    att = AttentionPool(args.hidden, args.hidden // args.nhead, args.nhead, args.dp).to(device) if args.att_pool else None
    mlp = MLP(args.hidden, args.hidden, args.hidden, args.mlp_layers, args.dp, args.norm, False, args.mlp_norm_affine).to(device) if args.mlp_pool else None
    
    # Predictor
    predictor = PFNPredictorNodeCls(
            args.hidden, args.nhead, args.transformer_layers, args.mlp_layers, args.dp, args.norm,
            separate_att=args.seperate, degree=args.degree, att=att, mlp=mlp, sim=args.sim,
            padding=args.padding, norm_affine=args.mlp_norm_affine, normalize=args.normalize_class_h,
            use_first_half_embedding=getattr(args, 'use_first_half_embedding', False),
            use_full_embedding=getattr(args, 'use_full_embedding', False),
            use_matching_network=getattr(args, 'use_matching_network', False),
            matching_network_projection=getattr(args, 'matching_network_projection', 'linear'),
            matching_network_temperature=getattr(args, 'matching_network_temperature', 0.1),
            matching_network_learnable_temp=getattr(args, 'matching_network_learnable_temp', True)
        ).to(device) if args.predictor == 'PFN' else None
    
    projector = MLP(args.min_pca_dim, args.hidden, args.hidden, 2, args.dp, args.norm, False, args.mlp_norm_affine).to(device) if args.use_projector else None
    identity_projection = IdentityProjection(args.projection_small_dim, args.projection_large_dim).to(device) if args.use_identity_projection else None
    
    return model, predictor, att, mlp, projector, identity_projection

def recreate_model_from_checkpoint(checkpoint_path, input_dim, device):
    """
    Recreates a model, including GNN, Predictor, and all PFN components, from a checkpoint file.
    This function handles both architecture recreation and state loading.
    """
    
    print(f"Recreating model from checkpoint: {checkpoint_path}")
    
    # 1. Load configuration and the entire checkpoint object
    checkpoint_info, checkpoint = load_checkpoint_config(checkpoint_path)
    model_config = checkpoint_info['model_config']
    args_dict = checkpoint_info['args']
    
    # Encapsulate args into an object for easier handling
    from argparse import Namespace
    args = Namespace(**args_dict)

    # 2. Create the model and all components with the saved architecture
    model = create_gnn_from_config(model_config, args_dict, input_dim).to(device)
    att, mlp, projector, identity_projection = create_pfn_components_from_config(args_dict, device)
    
    # Move components to device
    if att is not None:
        att = att.to(device)
    if mlp is not None:
        mlp = mlp.to(device)
    if projector is not None:
        projector = projector.to(device)
    if identity_projection is not None:
        identity_projection = identity_projection.to(device)
    
    predictor = PFNPredictorNodeCls(
        args.hidden, args.nhead, args.transformer_layers, args.mlp_layers, args.dp, args.norm,
        separate_att=args.seperate, degree=args.degree, att=att, mlp=mlp, sim=args.sim,
        padding=args.padding, norm_affine=args.mlp_norm_affine, normalize=args.normalize_class_h,
        use_first_half_embedding=getattr(args, 'use_first_half_embedding', False),
        use_full_embedding=getattr(args, 'use_full_embedding', False),
        use_matching_network=getattr(args, 'use_matching_network', False),
        matching_network_projection=getattr(args, 'matching_network_projection', 'linear'),
        matching_network_temperature=getattr(args, 'matching_network_temperature', 0.1),
        matching_network_learnable_temp=getattr(args, 'matching_network_learnable_temp', True)
    ).to(device) if args.predictor == 'PFN' else None
    
    # 3. Load the saved states into the newly created objects
    load_checkpoint_states(
        checkpoint, model, predictor=predictor,
        att=att, mlp=mlp, projector=projector, identity_projection=identity_projection
    )
    
    print("✅ Model recreated and weights loaded successfully!")
    
    return model, predictor, att, mlp, projector, identity_projection, args_dict


def load_checkpoint_states(checkpoint, model, predictor=None, optimizer=None, 
                          att=None, mlp=None, projector=None, identity_projection=None, 
                          scheduler=None, rank=0):
    """Load model states from an already-loaded checkpoint"""
    
    if rank == 0:
        print("Loading model states from checkpoint...")
    
    # Load model states
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if predictor is not None:
        if hasattr(predictor, 'module'):
            predictor.module.load_state_dict(checkpoint['predictor_state_dict'])
        else:
            predictor.load_state_dict(checkpoint['predictor_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if available
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Load optional components
    if att is not None and 'att_state_dict' in checkpoint:
        if hasattr(att, 'module'):
            att.module.load_state_dict(checkpoint['att_state_dict'])
        else:
            att.load_state_dict(checkpoint['att_state_dict'])
    
    if mlp is not None and 'mlp_state_dict' in checkpoint:
        if hasattr(mlp, 'module'):
            mlp.module.load_state_dict(checkpoint['mlp_state_dict'])
        else:
            mlp.load_state_dict(checkpoint['mlp_state_dict'])
    
    if projector is not None and 'projector_state_dict' in checkpoint:
        if hasattr(projector, 'module'):
            projector.module.load_state_dict(checkpoint['projector_state_dict'])
        else:
            projector.load_state_dict(checkpoint['projector_state_dict'])
    
    if identity_projection is not None and 'identity_projection_state_dict' in checkpoint:
        if hasattr(identity_projection, 'module'):
            identity_projection.module.load_state_dict(checkpoint['identity_projection_state_dict'])
        else:
            identity_projection.load_state_dict(checkpoint['identity_projection_state_dict'])
    
    # Extract info from checkpoint
    best_epoch = checkpoint['epoch']
    best_valid = checkpoint['best_metrics'].get('best_valid', 0)
    final_test = checkpoint['best_metrics'].get('final_test', 0)
    
    if rank == 0:
        print(f"✅ Checkpoint loaded successfully!")
        print(f"📊 Epoch: {best_epoch}")
        print(f"🎯 Best validation metric: {best_valid}")
        print(f"🚀 Final test metric: {final_test}")
    
    return best_epoch, best_valid, final_test 