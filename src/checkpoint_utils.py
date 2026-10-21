"""Checkpoint I/O for the shared GILT architecture."""
from pathlib import Path
import torch
ARCHITECTURE_KEYS = ('hidden', 'num_layers', 'transformer_layers', 'nhead', 'mlp_layers', 'dp', 'norm', 'gnn_norm_affine', 'mlp_norm_affine', 'ffn_expansion_ratio', 'normalize_class_h', 'head_num_layers', 'nc_head_num_layers', 'lp_head_num_layers', 'gc_head_num_layers', 'mplp_signature_dim', 'mplp_num_hops', 'mplp_feature_combine', 'mplp_prop_type', 'mplp_signature_sampling', 'mplp_use_subgraph', 'mplp_use_degree')

def load_checkpoint_config(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    for key in ('args', 'model_state_dict', 'predictor_state_dict'):
        if key not in checkpoint:
            raise ValueError(f'Checkpoint is missing {key}')
    return ({'args': checkpoint['args'], 'epoch': checkpoint.get('epoch', -1)}, checkpoint)

def override_args_from_checkpoint(args, checkpoint_args, rank=0, predictor_state_dict=None):
    required = {'model': 'PureGCN_v1', 'predictor': 'PFN', 'relu': False, 'res': False, 'use_identity_projection': False, 'use_graphpfn': False, 'use_matching_network': False, 'use_first_half_embedding': False, 'use_full_embedding': False, 'transformer_ffn_only': False, 'skip_token_formulation': False, 'lp_head_type': 'mplp', 'mplp_context_calibrate': False, 'padding': 'zero', 'seperate': True, 'transformer_norm_type': 'pre', 'nc_proto_pooling': 'mean', 'sim': 'dot', 'nc_sim': 'dot', 'gc_sim': 'dot'}
    for key, expected in required.items():
        if checkpoint_args.get(key, expected) != expected:
            raise ValueError(f'Checkpoint uses an unsupported architecture option: {key}')
    for key in ARCHITECTURE_KEYS:
        if key in checkpoint_args:
            setattr(args, key, checkpoint_args[key])
    if 'graph_pooling' in checkpoint_args:
        args.graph_pooling = checkpoint_args['graph_pooling']
    return args

def load_checkpoint_states(checkpoint, model, predictor, **kwargs):
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    predictor.load_state_dict(checkpoint['predictor_state_dict'], strict=True)
    metrics = checkpoint.get('best_metrics', {})
    return (checkpoint.get('epoch', -1), metrics.get('best_valid', 0.0), metrics.get('final_test', 0.0))

def save_checkpoint(model, predictor, optimizer, args, best_metrics, epoch, **kwargs):
    directory = Path(args.checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (args.checkpoint_name or 'gilt_model.pt')
    if path.suffix != '.pt':
        path = path.with_suffix('.pt')
    torch.save({'model_state_dict': model.state_dict(), 'predictor_state_dict': predictor.state_dict(), 'args': vars(args), 'epoch': epoch, 'best_metrics': best_metrics}, path)
    print(f'Saved checkpoint: {path}')
    return str(path)
