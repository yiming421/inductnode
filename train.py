import os
'\nJoint Training Script for Node Classification and Link Prediction\n'
import os
import sys
import time
import copy
import random
import secrets
import torch
import psutil
import numpy as np
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from src.model import PureGCN_v1, PFNPredictorNodeCls
from src.data_nc import load_all_data, load_all_data_train
from src.data_lp import load_all_data_link
from src.data_gc import load_all_graph_datasets, process_graph_features, create_data_loaders, create_task_filtered_datasets
from src.data_utils import process_data, prepare_link_data, select_link_context, process_link_data
from src.engine_nc import train_all, test_all, test_all_induct
from src.engine_lp import train_link_prediction, evaluate_link_prediction, evaluate_link_prediction_with_tta
from src.engine_gc import train_graph_classification_single_task, evaluate_graph_classification_single_task, evaluate_graph_classification_single_task_with_tta, aggregate_task_metrics
from src.config import parse_joint_training_args
from src.checkpoint_utils import load_checkpoint_config, override_args_from_checkpoint, load_checkpoint_states, save_checkpoint
REPORTED_CONTEXT_SEED_BASES = '378323818,1058536330,665380492,915013419,1634403476'

def _select_primary_metric(metric_dict, override=None, prefer='auc'):
    if not isinstance(metric_dict, dict):
        return metric_dict
    if not metric_dict:
        return 0.0
    if override and override != 'auto':
        return metric_dict.get(override, metric_dict.get(prefer, metric_dict.get('ap', next(iter(metric_dict.values())))))
    if prefer == 'ap':
        return metric_dict.get('ap', metric_dict.get('auc', next(iter(metric_dict.values()))))
    return metric_dict.get('auc', metric_dict.get('ap', next(iter(metric_dict.values()))))

def _parse_unseen_test_context_seed_bases(args):
    raw = getattr(args, 'unseen_test_context_seed_bases', None)
    if raw is None or str(raw).strip() == '':
        return None
    cached = getattr(args, '_unseen_test_context_seed_bases_parsed', None)
    if cached is not None:
        return cached
    try:
        seeds = [int(part.strip()) for part in str(raw).split(',') if part.strip()]
    except ValueError as exc:
        raise ValueError('--unseen_test_context_seed_bases must be a comma-separated list of integers') from exc
    if not seeds:
        return None
    setattr(args, '_unseen_test_context_seed_bases_parsed', seeds)
    return seeds

def _get_unseen_test_context_seed_base(args):
    """
    Return a seed base for unseen-test context resampling.
    - explicit override: use args.unseen_test_context_seed_base when provided
    - explicit list: use the run-indexed seed from args.unseen_test_context_seed_bases
    - fixed: reuse the deterministic seed for the current run
    - per_run: derive once from OS entropy and cache on args (different each run)
    """
    explicit_seed = getattr(args, 'unseen_test_context_seed_base', None)
    if explicit_seed is not None:
        return int(explicit_seed)
    explicit_seed_list = _parse_unseen_test_context_seed_bases(args)
    if explicit_seed_list is not None:
        run_idx = int(getattr(args, '_current_run_idx', 0))
        if run_idx >= len(explicit_seed_list):
            raise ValueError(f'--unseen_test_context_seed_bases provides {len(explicit_seed_list)} seeds but run index {run_idx + 1} was requested')
        return int(explicit_seed_list[run_idx])
    mode = str(getattr(args, 'unseen_test_context_seed_mode', 'per_run')).lower()
    if mode == 'fixed':
        return int(getattr(args, '_current_run_seed', getattr(args, 'seed', 42)))
    if mode != 'per_run':
        print(f"[Unseen Test Context] Unknown seed mode '{mode}', defaulting to 'per_run'")
    cached_seed = getattr(args, '_unseen_test_context_seed_base', None)
    if cached_seed is None:
        cached_seed = int(secrets.randbits(31))
        setattr(args, '_unseen_test_context_seed_base', cached_seed)
        print(f'[Unseen Test Context] Using run seed base {cached_seed} (mode=per_run)')
    return int(cached_seed)

def _create_gnn_backbone(args, gnn_input_dim, hidden, device):
    return PureGCN_v1(gnn_input_dim, num_layers=args.num_layers, hidden=hidden, dp=args.dp, norm=args.norm, norm_affine=args.gnn_norm_affine).to(device)

def _get_module_device(module, fallback_module=None):
    """Best-effort device resolution for modules that may be parameter-free."""
    for candidate in (module, fallback_module):
        if candidate is None:
            continue
        try:
            return next(candidate.parameters()).device
        except StopIteration:
            continue
    return torch.device('cpu')
lp_tracker = None
gc_tracker = None

def parse_context_overrides(override_string):
    if not override_string:
        return {}
    overrides = {}
    for pair in override_string.split(','):
        if ':' in pair:
            dataset, shots = pair.strip().split(':')
            overrides[dataset.strip()] = int(shots.strip())
    return overrides

def parse_context_bounds(bounds_string):
    import re
    if not bounds_string:
        return {'nc': (10, 30), 'lp': (64, 192), 'gc': (8, 24)}
    pattern = '\\((\\d+),(\\d+)\\)'
    matches = re.findall(pattern, bounds_string)
    if len(matches) != 3:
        raise ValueError(f'Expected 3 bound pairs for NC/LP/GC, got {len(matches)}: {bounds_string}')
    bounds = {'nc': (int(matches[0][0]), int(matches[0][1])), 'lp': (int(matches[1][0]), int(matches[1][1])), 'gc': (int(matches[2][0]), int(matches[2][1]))}
    for task_type, (lower, upper) in bounds.items():
        if lower >= upper:
            raise ValueError(f'Invalid bounds for {task_type.upper()}: lower ({lower}) must be < upper ({upper})')
        if lower <= 0:
            raise ValueError(f'Invalid bounds for {task_type.upper()}: lower bound ({lower}) must be > 0')
    return bounds

def resolve_context_shots(dataset_name, task_type, args, epoch=None):
    override_attr = f'{task_type}_context_overrides'
    override_string = getattr(args, override_attr, None)
    override_map = parse_context_overrides(override_string)
    if dataset_name in override_map:
        print(f"[Context Override] {task_type.upper()} dataset '{dataset_name}': using {override_map[dataset_name]} context shots")
        return override_map[dataset_name]
    sampling_plan = getattr(args, 'context_sampling_plan', 'ori')
    if sampling_plan == 'ori':
        defaults = {'nc': args.context_num, 'lp': args.context_k, 'gc': args.context_graph_num}
        return defaults[task_type]
    elif sampling_plan == 'random':
        if epoch is None:
            defaults = {'nc': args.context_num, 'lp': args.context_k, 'gc': args.context_graph_num}
            context_shots = defaults[task_type]
            print(f"[Context Evaluation] {task_type.upper()} dataset '{dataset_name}': using {context_shots} TRAINING context shots (evaluation mode, fixed)")
            return context_shots
        import random
        bounds = parse_context_bounds(getattr(args, 'context_bounds', None))
        lower, upper = bounds[task_type]
        context_shots = random.randint(lower, upper)
        print(f"[Context Random] {task_type.upper()} dataset '{dataset_name}': using {context_shots} TRAINING context shots (range: {lower}-{upper})")
        return context_shots
    elif sampling_plan == 'decay':
        if epoch is None:
            defaults = {'nc': args.context_num, 'lp': args.context_k, 'gc': args.context_graph_num}
            context_shots = defaults[task_type]
            print(f"[Context Evaluation] {task_type.upper()} dataset '{dataset_name}': using {context_shots} context shots (evaluation mode, ignoring decay)")
            return context_shots
        bounds = parse_context_bounds(getattr(args, 'context_bounds', None))
        lower, upper = bounds[task_type]
        total_epochs = getattr(args, 'epochs', 100)
        progress = min(1.0, epoch / max(1, total_epochs - 1))
        context_shots = int(upper - progress * (upper - lower))
        context_shots = max(lower, min(upper, context_shots))
        print(f"[Context Decay] {task_type.upper()} dataset '{dataset_name}': using {context_shots} context shots (epoch {epoch}, progress {progress:.3f})")
        return context_shots
    else:
        raise ValueError(f'Unknown context sampling plan: {sampling_plan}')
    defaults = {'nc': args.context_num, 'lp': args.context_k, 'gc': args.context_graph_num}
    return defaults[task_type]

def refresh_nc_contexts(nc_data, args=None, epoch=None):
    """Refresh node classification context samples"""
    if nc_data[0] is None:
        return
    nc_data_list, nc_split_idx_list, nc_external_embeddings = nc_data
    print('  🔄 Refreshing NC contexts...')
    for data, split_idx in zip(nc_data_list, nc_split_idx_list):
        if hasattr(data, 'context_sample'):
            if args is not None:
                context_shots = resolve_context_shots(data.name, 'nc', args, epoch)
                current_context_size = context_shots
            else:
                current_context_size = len(data.context_sample) // len(data.y.unique())
            from src.data_utils import select_k_shot_context
            new_context_sample = select_k_shot_context(data, current_context_size, split_idx['train'])
            data.context_sample = new_context_sample.to(data.context_sample.device)
            print(f'    ✓ Refreshed {data.name}: {len(new_context_sample)} context samples ({current_context_size} per class, random sampling)')

def refresh_lp_contexts(lp_data, args, epoch=None):
    """Refresh link prediction context samples"""
    if lp_data[0] is None:
        return
    lp_data_list, lp_split_idx_list, lp_context_data, lp_masks, lp_link_data_all = lp_data
    print('  🔄 Refreshing LP contexts...')
    for i, (data, split_idx) in enumerate(zip(lp_data_list, lp_split_idx_list)):
        context_shots = resolve_context_shots(data.name, 'lp', args, epoch)
        link_data = lp_link_data_all[i]
        if 'train' in link_data and link_data['train']['edge_pairs'].size(0) > 0:
            new_context_data, new_train_mask = select_link_context(link_data['train'], context_shots, args.context_neg_ratio, args.remove_context_from_train)
            lp_context_data[i] = new_context_data
            lp_masks[i] = new_train_mask
            print(f'    ✓ Refreshed {data.name}: {context_shots} context shots')

def refresh_gc_contexts(gc_data, args, epoch=None):
    """Refresh graph classification context samples"""
    if len(gc_data[0]) == 0:
        return
    gc_data_list, gc_processed_data_list = gc_data
    print('  🔄 Refreshing GC contexts...')
    for dataset_info in gc_processed_data_list:
        if 'dataset' in dataset_info:
            dataset_name = dataset_info['dataset'].name if hasattr(dataset_info['dataset'], 'name') else 'GC dataset'
            context_shots = resolve_context_shots(dataset_name, 'gc', args, epoch)
            task_filtered_splits = create_task_filtered_datasets(dataset_info['dataset'], dataset_info['split_idx'])
            dataset_info['task_filtered_splits'] = task_filtered_splits
            print(f'    ✓ Refreshed {dataset_name}: context samples regenerated (target: {context_shots} context shots)')

def refresh_contexts_if_needed(epoch, args, data_dict):
    """Refresh contexts for all tasks if needed based on interval"""
    if args.context_refresh_interval <= 0 or epoch % args.context_refresh_interval != 0:
        return
    print(f'\n🔄 Refreshing contexts at epoch {epoch} (interval: {args.context_refresh_interval})')
    print(f"   Sampling plan: {getattr(args, 'context_sampling_plan', 'ori')}")
    refresh_seed = args.seed + epoch // args.context_refresh_interval
    torch.manual_seed(refresh_seed)
    if getattr(args, 'enable_nc', True) and data_dict['nc_train'][0] is not None:
        refresh_nc_contexts(data_dict['nc_train'], args, epoch)
    if getattr(args, 'enable_lp', True) and data_dict['lp_train'][0] is not None:
        refresh_lp_contexts(data_dict['lp_train'], args, epoch)
    if getattr(args, 'enable_gc', True) and len(data_dict['gc_train'][0]) > 0:
        refresh_gc_contexts(data_dict['gc_train'], args, epoch)
    print('  ✅ Context refresh completed\n')

def setup_graph_dataset_environment(args):
    graph_data_needed = bool(getattr(args, 'enable_gc', False) or False)
    if not graph_data_needed:
        os.environ.pop('USE_FUG_EMB', None)
        os.environ.pop('TAG_DATASET_ROOT', None)
        os.environ.pop('USE_ORIGINAL_FEATURES', None)
        return
    print(f'Enabling ORIGINAL FEATURES for graph classification')
    print(f'OGB root: {args.ogb_root}, TU root: ./dataset/TU')
    os.environ['USE_ORIGINAL_FEATURES'] = '1'
    os.environ['OGB_ROOT'] = args.ogb_root
    os.environ['TU_ROOT'] = './dataset/TU'
    os.environ.pop('USE_FUG_EMB', None)
    os.environ.pop('TAG_DATASET_ROOT', None)

def process_datasets_for_models(datasets, processed_data_list, args, device, test_datasets=False):
    """
    Process datasets to handle feature dimensions and create any necessary dummy features.
    Adapted from graph_classification.py for joint training.
    """
    processed_datasets = []
    final_num_features = args.hidden
    for dataset, dataset_info in zip(datasets, processed_data_list):
        is_tsgfm_dataset = False
        use_pca_cache_for_dataset = args.use_pca_cache and (not False)
        if test_datasets:
            processing_info = process_graph_features(dataset, args.hidden, device, args.projection_large_dim, args.use_full_pca, False, args.normalize_data, args.use_batchnorm, pca_device=args.pca_device, incremental_pca_batch_size=args.incremental_pca_batch_size, pca_sample_threshold=500000, processed_data=dataset_info, pcba_context_only_pca=False, use_pca_cache=use_pca_cache_for_dataset, pca_cache_dir=args.pca_cache_dir, dataset_name=getattr(dataset, 'name', None))
        else:
            processing_info = process_graph_features(dataset, args.hidden, device, args.projection_large_dim, args.use_full_pca, False, args.normalize_data, args.use_batchnorm, pca_device=args.pca_device, incremental_pca_batch_size=args.incremental_pca_batch_size, pca_sample_threshold=500000, processed_data=dataset_info, pcba_context_only_pca=False, use_pca_cache=use_pca_cache_for_dataset, pca_cache_dir=args.pca_cache_dir, dataset_name=getattr(dataset, 'name', None))
        processed_datasets.append(dataset)
        dataset_info.update(processing_info)
    return (processed_datasets, processed_data_list, final_num_features)

def create_unified_model(args, input_dim, device):
    """Construct the shared GILT encoder and its three prediction heads."""
    model = _create_gnn_backbone(args, args.hidden, args.hidden, device)
    predictor = PFNPredictorNodeCls(hidden_dim=args.hidden, nhead=args.nhead, num_layers=args.transformer_layers, mlp_layers=args.mlp_layers, dropout=args.dp, norm=args.norm, norm_affine=args.mlp_norm_affine, normalize=args.normalize_class_h, norm_type='pre', ffn_expansion_ratio=args.ffn_expansion_ratio, head_num_layers=args.head_num_layers, nc_head_num_layers=args.nc_head_num_layers, lp_head_num_layers=args.lp_head_num_layers, gc_head_num_layers=args.gc_head_num_layers, mplp_signature_dim=args.mplp_signature_dim, mplp_num_hops=args.mplp_num_hops, mplp_feature_combine=args.mplp_feature_combine, mplp_prop_type=args.mplp_prop_type, mplp_signature_sampling=args.mplp_signature_sampling, mplp_use_subgraph=args.mplp_use_subgraph, mplp_use_degree=args.mplp_use_degree).to(device)
    return (model, predictor, None, None)

def load_and_preprocess_data(args, device, skip_training_data=False, gc_tracker=None):
    """
    Load and preprocess data for enabled tasks.
    Returns processed datasets for node classification, link prediction, and graph classification.
    
    Args:
        skip_training_data: If True, skip loading training datasets (for pretrained model evaluation)
    """
    global lp_tracker
    print('\n=== Loading and Preprocessing Data ===')

    def move_nc_tensors_to_cpu(data_obj, split_idx):
        """Move NC data tensors and split indices to CPU to keep GPU memory free until use."""
        if hasattr(data_obj, 'x'):
            data_obj.x = data_obj.x.cpu()
        if hasattr(data_obj, 'adj_t'):
            data_obj.adj_t = data_obj.adj_t.cpu()
        if hasattr(data_obj, 'y'):
            data_obj.y = data_obj.y.cpu()
        if hasattr(data_obj, 'context_sample'):
            data_obj.context_sample = data_obj.context_sample.cpu()
        if hasattr(data_obj, 'x_pca'):
            data_obj.x_pca = data_obj.x_pca.cpu()
        if hasattr(data_obj, 'x_original'):
            data_obj.x_original = data_obj.x_original.cpu()
        split_idx['train'] = split_idx['train'].cpu()
        split_idx['valid'] = split_idx['valid'].cpu()
        split_idx['test'] = split_idx['test'].cpu()
    nc_train_data_list, nc_train_split_idx_list = (None, None)
    nc_test_data_list, nc_test_split_idx_list = (None, None)
    nc_valid_data_list, nc_valid_split_idx_list = (None, None)
    nc_train_external_embeddings = None
    nc_test_external_embeddings = None
    nc_valid_external_embeddings = None
    if args.enable_nc:
        print('Loading node classification datasets...')
        nc_train_datasets = [d.strip() for d in args.nc_train_dataset.split(',') if d.strip()]
        nc_test_datasets = [d.strip() for d in args.nc_test_dataset.split(',') if d.strip()]
        nc_valid_datasets = [d.strip() for d in str(getattr(args, 'nc_valid_dataset', '')).split(',') if d.strip()]
        nc_make_undirected = getattr(args, 'nc_graph_direction', 'undirected') != 'directed'
        nc_direction_label = 'undirected' if nc_make_undirected else 'directed'
        print(f'NC graph direction mode: {nc_direction_label}')
        webkb_datasets = {'Cornell', 'Texas', 'Wisconsin'}
        expand_webkb_splits = any((ds.strip() in webkb_datasets for ds in nc_test_datasets))
        expand_webkb_splits_valid = any((ds.strip() in webkb_datasets for ds in nc_valid_datasets))
        if expand_webkb_splits:
            print('NC test split mode: using all official WebKB splits')
        if expand_webkb_splits_valid:
            print('NC held-out validation split mode: using all official WebKB splits')
        if args.split_rebalance_strategy == 'legacy':
            large_datasets = ['ogbn-products', 'ogbn-papers100M']
            if any((dataset in nc_train_datasets for dataset in large_datasets)):
                print(f"⚠️  WARNING: Detected large dataset(s) {[d for d in large_datasets if d in nc_train_datasets]} with 'legacy' split strategy.")
                print(f"⚠️  Automatically changing split_rebalance_strategy from 'legacy' to 'small_valid' to prevent infinite training loops.")
                args.split_rebalance_strategy = 'smallest_for_valid'
        if not skip_training_data:
            nc_train_data_list, nc_train_split_idx_list = load_all_data_train(nc_train_datasets, split_strategy=args.split_rebalance_strategy, make_undirected=nc_make_undirected)
        else:
            print('  Skipping NC training data loading (using pretrained model)')
        nc_test_data_list, nc_test_split_idx_list = load_all_data(nc_test_datasets, make_undirected=nc_make_undirected, expand_webkb_splits=expand_webkb_splits)
        if nc_valid_datasets:
            nc_valid_data_list, nc_valid_split_idx_list = load_all_data(nc_valid_datasets, make_undirected=nc_make_undirected, expand_webkb_splits=expand_webkb_splits_valid)
            print(f'Loaded {len(nc_valid_data_list)} NC held-out validation datasets')
        if not skip_training_data:
            for i, (data, split_idx) in enumerate(zip(nc_train_data_list, nc_train_split_idx_list)):
                data.x = data.x.to(device)
                data.adj_t = data.adj_t.to(device)
                data.y = data.y.to(device)
                external_emb = nc_train_external_embeddings[i] if nc_train_external_embeddings else None
                process_data(data, split_idx, args.hidden, args.context_num, False, args.use_full_pca, args.normalize_data, 0, args.use_batchnorm, args.pca_device, args.incremental_pca_batch_size, process_test_only=False)
                move_nc_tensors_to_cpu(data, split_idx)
        for i, (data, split_idx) in enumerate(zip(nc_test_data_list, nc_test_split_idx_list)):
            data.x = data.x.to(device)
            data.adj_t = data.adj_t.to(device)
            data.y = data.y.to(device)
            external_emb = nc_test_external_embeddings[i] if nc_test_external_embeddings else None
            context_shots = resolve_context_shots(data.name, 'nc', args, epoch=None)
            process_data(data, split_idx, args.hidden, context_shots, False, args.use_full_pca, args.normalize_data, 0, args.use_batchnorm, args.pca_device, args.incremental_pca_batch_size, process_test_only=False)
            move_nc_tensors_to_cpu(data, split_idx)
        if nc_valid_data_list is not None:
            for i, (data, split_idx) in enumerate(zip(nc_valid_data_list, nc_valid_split_idx_list)):
                data.x = data.x.to(device)
                data.adj_t = data.adj_t.to(device)
                data.y = data.y.to(device)
                external_emb = nc_valid_external_embeddings[i] if nc_valid_external_embeddings else None
                context_shots = resolve_context_shots(data.name, 'nc', args, epoch=None)
                process_data(data, split_idx, args.hidden, context_shots, False, args.use_full_pca, args.normalize_data, 0, args.use_batchnorm, args.pca_device, args.incremental_pca_batch_size, process_test_only=False)
                move_nc_tensors_to_cpu(data, split_idx)
    else:
        print('Node classification task disabled, skipping dataset loading...')
    lp_train_data_list, lp_train_split_idx_list = (None, None)
    lp_train_context_data, lp_train_masks, lp_train_link_data_all = ([], [], [])
    lp_test_data_list, lp_test_split_idx_list = (None, None)
    lp_test_context_data, lp_test_link_data_all = ([], [])
    lp_valid_data_list, lp_valid_split_idx_list = (None, None)
    lp_valid_context_data, lp_valid_link_data_all = ([], [])
    lp_valid_masks = []
    if args.enable_lp:
        print('Loading link prediction datasets...')
        lp_train_datasets = [d.strip() for d in args.lp_train_dataset.split(',') if d.strip()]
        lp_test_datasets = [d.strip() for d in args.lp_test_dataset.split(',') if d.strip()]
        lp_valid_datasets = [d.strip() for d in str(getattr(args, 'lp_valid_dataset', '')).split(',') if d.strip()]
        if not skip_training_data:
            lp_train_data_list, lp_train_split_idx_list = load_all_data_link(lp_train_datasets, device='cpu')
            print(f'[MEMORY_FIX] Loaded {len(lp_train_data_list)} training datasets on CPU (was loading to GPU before!)')
        else:
            lp_train_data_list, lp_train_split_idx_list = ([], [])
            print('  Skipping LP training data loading (using pretrained model)')
        lp_train_context_data = []
        lp_train_masks = []
        lp_train_link_data_all = []
        if not skip_training_data:
            for i, (data, split_idx) in enumerate(zip(lp_train_data_list, lp_train_split_idx_list)):
                original_x_device = data.x.device
                if data.x.device.type == 'cpu':
                    print(f'[MEMORY_FIX] Moving {data.name} features to GPU for fast PCA computation')
                    data.x = data.x.to(device)
                process_link_data(data, args, rank=0)
                data.x = data.x.to(original_x_device)
                if hasattr(data, 'x_pca') and data.x_pca is not None:
                    data.x_pca = data.x_pca.to(original_x_device)
                if hasattr(data, 'context_sample') and data.context_sample is not None:
                    data.context_sample = data.context_sample.to(original_x_device)
                    print(f'[MEMORY_FIX] Moved {data.name} context_sample back to CPU')
                print(f'[MEMORY_FIX] Moved {data.name} processed features back to CPU')
                link_data = prepare_link_data(data, split_idx)
                context_dataset_name = getattr(data, 'lp_base_dataset_name', data.name)
                context_shots = resolve_context_shots(context_dataset_name, 'lp', args, epoch=None)
                context_data, train_mask = select_link_context(link_data['train'], context_shots, args.context_neg_ratio, args.remove_context_from_train)
                lp_train_context_data.append(context_data)
                lp_train_masks.append(train_mask)
                lp_train_link_data_all.append(link_data)
        else:
            print('  Skipping LP training data processing (using pretrained model)')
        lp_test_data_list, lp_test_split_idx_list = load_all_data_link(lp_test_datasets, device='cpu')
        print(f'[MEMORY_FIX] Loaded {len(lp_test_data_list)} test datasets on CPU (was loading to GPU before!)')
        lp_test_context_data = []
        lp_test_link_data_all = []
        for i, (data, split_idx) in enumerate(zip(lp_test_data_list, lp_test_split_idx_list)):
            original_x_device = data.x.device
            if data.x.device.type == 'cpu':
                print(f'[MEMORY_FIX] Moving {data.name} features to GPU for fast PCA computation')
                data.x = data.x.to(device)
            process_link_data(data, args, rank=0)
            data.x = data.x.to(original_x_device)
            if hasattr(data, 'x_pca') and data.x_pca is not None:
                data.x_pca = data.x_pca.to(original_x_device)
                print(f'[MEMORY_FIX] Moved {data.name} x_pca back to CPU')
            if hasattr(data, 'context_sample') and data.context_sample is not None:
                data.context_sample = data.context_sample.to(original_x_device)
                print(f'[MEMORY_FIX] Moved {data.name} context_sample back to CPU')
            print(f'[MEMORY_FIX] Moved {data.name} processed features back to CPU')
            link_data = prepare_link_data(data, split_idx)
            context_dataset_name = getattr(data, 'lp_base_dataset_name', data.name)
            context_shots = resolve_context_shots(context_dataset_name, 'lp', args, epoch=None)
            context_data, _ = select_link_context(link_data['train'], context_shots, args.context_neg_ratio, False)
            lp_test_context_data.append(context_data)
            lp_test_link_data_all.append(link_data)
        if lp_valid_datasets:
            lp_valid_data_list, lp_valid_split_idx_list = load_all_data_link(lp_valid_datasets, device='cpu')
            print(f'[MEMORY_FIX] Loaded {len(lp_valid_data_list)} held-out LP validation datasets on CPU')
            for i, (data, split_idx) in enumerate(zip(lp_valid_data_list, lp_valid_split_idx_list)):
                original_x_device = data.x.device
                if data.x.device.type == 'cpu':
                    print(f'[MEMORY_FIX] Moving {data.name} features to GPU for fast PCA computation')
                    data.x = data.x.to(device)
                process_link_data(data, args, rank=0)
                data.x = data.x.to(original_x_device)
                if hasattr(data, 'x_pca') and data.x_pca is not None:
                    data.x_pca = data.x_pca.to(original_x_device)
                    print(f'[MEMORY_FIX] Moved {data.name} x_pca back to CPU')
                if hasattr(data, 'context_sample') and data.context_sample is not None:
                    data.context_sample = data.context_sample.to(original_x_device)
                    print(f'[MEMORY_FIX] Moved {data.name} context_sample back to CPU')
                print(f'[MEMORY_FIX] Moved {data.name} processed features back to CPU')
                link_data = prepare_link_data(data, split_idx)
                context_dataset_name = getattr(data, 'lp_base_dataset_name', data.name)
                context_shots = resolve_context_shots(context_dataset_name, 'lp', args, epoch=None)
                context_data, _ = select_link_context(link_data['train'], context_shots, args.context_neg_ratio, False)
                lp_valid_context_data.append(context_data)
                lp_valid_link_data_all.append(link_data)
                lp_valid_masks.append(None)
    else:
        print('Link prediction task disabled, skipping dataset loading...')
    gc_train_data_list, gc_train_processed_data_list = ([], [])
    gc_test_data_list, gc_test_processed_data_list = ([], [])
    gc_valid_data_list, gc_valid_processed_data_list = ([], [])
    gc_training_enabled = bool(getattr(args, 'enable_gc', False))
    gc_downstream_eval_enabled = bool(False)
    gc_data_loading_enabled = gc_training_enabled or gc_downstream_eval_enabled
    if gc_data_loading_enabled:
        print('Loading graph classification datasets...')
        setup_graph_dataset_environment(args)
        gc_train_datasets = args.gc_train_dataset.split(',') if gc_training_enabled else []
        gc_test_datasets = args.gc_test_dataset.split(',')
        gc_valid_datasets = [d.strip() for d in str(getattr(args, 'gc_valid_dataset', '')).split(',') if d.strip()]
        if gc_training_enabled and (not skip_training_data):
            use_cache_aware = False
            print(f'Using standard loading (original features, no PCA cache)')
            gc_train_data_list, gc_train_processed_data_list = load_all_graph_datasets(gc_train_datasets, device, pretraining_mode=True, context_k=args.context_graph_num)
            if len(gc_train_data_list) > 0:
                gc_train_data_list, gc_train_processed_data_list, _ = process_datasets_for_models(gc_train_data_list, gc_train_processed_data_list, args, device)
                print('Precomputing task-filtered splits for training datasets...')
                for dataset_info in gc_train_processed_data_list:
                    task_filtered_splits = create_task_filtered_datasets(dataset_info['dataset'], dataset_info['split_idx'])
                    dataset_info['task_filtered_splits'] = task_filtered_splits
        else:
            gc_train_data_list, gc_train_processed_data_list = ([], [])
            if gc_training_enabled and skip_training_data:
                print('  Skipping GC training data loading (using pretrained model)')
            elif not gc_training_enabled:
                print('  Skipping GC training data loading (GC training disabled)')
        gc_test_data_list = []
        gc_test_processed_data_list = []
        use_cache_aware = False
        print(f'Using standard loading for test datasets (original features, no PCA cache)')
        gc_test_context_seed_base = _get_unseen_test_context_seed_base(args)
        print(f"[GC Test Context] Seed mode: {getattr(args, 'unseen_test_context_seed_mode', 'per_run')}, base={gc_test_context_seed_base}")
        py_rng_state = random.getstate()
        np_rng_state = np.random.get_state()
        torch_rng_state = torch.random.get_rng_state()
        cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            for dataset_idx, dataset_name in enumerate(gc_test_datasets):
                dataset_name = dataset_name.strip()
                dataset_seed = int(gc_test_context_seed_base + dataset_idx * 1000)
                random.seed(dataset_seed)
                np.random.seed(dataset_seed)
                torch.manual_seed(dataset_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(dataset_seed)
                context_shots = resolve_context_shots(dataset_name, 'gc', args, epoch=None)
                single_data_list, single_processed_list = load_all_graph_datasets([dataset_name], device, context_k=context_shots)
                print(f'  {dataset_name}: Loaded (original features, no cache)')
                gc_test_data_list.extend(single_data_list)
                gc_test_processed_data_list.extend(single_processed_list)
        finally:
            random.setstate(py_rng_state)
            np.random.set_state(np_rng_state)
            torch.random.set_rng_state(torch_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
        if len(gc_test_data_list) > 0:
            gc_test_data_list, gc_test_processed_data_list, _ = process_datasets_for_models(gc_test_data_list, gc_test_processed_data_list, args, device, test_datasets=True)
            print('Precomputing task-filtered splits for test datasets...')
            for dataset_info in gc_test_processed_data_list:
                task_filtered_splits_test = create_task_filtered_datasets(dataset_info['dataset'], dataset_info['split_idx'], 'test')
                dataset_info['task_filtered_splits'] = task_filtered_splits_test
                dataset_info['task_filtered_splits_test_only'] = task_filtered_splits_test
        if gc_valid_datasets:
            print('Loading held-out validation data for graph classification...')
            use_cache_aware = False
            for dataset_name in gc_valid_datasets:
                context_shots = resolve_context_shots(dataset_name, 'gc', args, epoch=None)
                single_data_list, single_processed_list = load_all_graph_datasets([dataset_name], device, context_k=context_shots)
                gc_valid_data_list.extend(single_data_list)
                gc_valid_processed_data_list.extend(single_processed_list)
            if len(gc_valid_data_list) > 0:
                gc_valid_data_list, gc_valid_processed_data_list, _ = process_datasets_for_models(gc_valid_data_list, gc_valid_processed_data_list, args, device, test_datasets=True)
                print('Precomputing task-filtered splits for held-out validation datasets...')
                for dataset_info in gc_valid_processed_data_list:
                    task_filtered_splits_valid = create_task_filtered_datasets(dataset_info['dataset'], dataset_info['split_idx'], 'val')
                    dataset_info['task_filtered_splits'] = task_filtered_splits_valid
                    dataset_info['task_filtered_splits_test_only'] = task_filtered_splits_valid
                print(f'Loaded {len(gc_valid_data_list)} GC held-out validation datasets')
    else:
        print('Graph classification data loading disabled, skipping dataset loading...')
    nc_train_loaders = []
    data_dict = {'nc_train': (nc_train_data_list, nc_train_split_idx_list, nc_train_external_embeddings), 'nc_test': (nc_test_data_list, nc_test_split_idx_list, nc_test_external_embeddings), 'nc_valid': (nc_valid_data_list, nc_valid_split_idx_list, nc_valid_external_embeddings), 'nc_train_loaders': nc_train_loaders, 'lp_train': (lp_train_data_list, lp_train_split_idx_list, lp_train_context_data, lp_train_masks, lp_train_link_data_all), 'lp_test': (lp_test_data_list, lp_test_split_idx_list, lp_test_context_data, lp_test_link_data_all), 'lp_valid': (lp_valid_data_list, lp_valid_split_idx_list, lp_valid_context_data, lp_valid_masks, lp_valid_link_data_all), 'gc_train': (gc_train_data_list, gc_train_processed_data_list), 'gc_test': (gc_test_data_list, gc_test_processed_data_list), 'gc_valid': (gc_valid_data_list, gc_valid_processed_data_list)}
    return data_dict

def get_task_schedule(epoch, args):
    return {'nc': args.enable_nc, 'lp': args.enable_lp, 'gc': args.enable_gc}

def joint_training_step(model, predictor, nc_data, lp_data, gc_data, optimizer, args, epoch, optimizer_nc=None, optimizer_lp=None, optimizer_gc=None):
    global lp_tracker
    model.train()
    predictor.train()
    gc_encoder = model
    gc_encoder.train()
    opt_nc = optimizer_nc if optimizer_nc is not None else optimizer
    opt_lp = optimizer_lp if optimizer_lp is not None else optimizer
    opt_gc = optimizer_gc if optimizer_gc is not None else optimizer
    if args.use_separate_optimizers:
        original_lambda_nc = args.lambda_nc
        original_lambda_lp = args.lambda_lp
        original_lambda_gc = args.lambda_gc
        args.lambda_nc = 1.0
        args.lambda_lp = 1.0
        args.lambda_gc = 1.0
    device = optimizer.param_groups[0]['params'][0].device
    total_nc_loss = torch.tensor(0.0, device=device)
    total_lp_loss = torch.tensor(0.0, device=device)
    total_gc_loss = torch.tensor(0.0, device=device)
    nc_count = 0
    lp_count = 0
    gc_count = 0
    lp_gate_sum = 0.0
    lp_gate_count = 0
    lp_gate_calib_sum = 0.0
    lp_gate_calib_count = 0
    lp_struct_loss_sum = 0.0
    lp_struct_loss_count = 0
    nc_nll_loss = 0.0
    nc_de_loss = 0.0
    nc_contrastive_loss = 0.0
    nc_data_list, nc_split_idx_list, nc_external_embeddings = nc_data
    lp_data_list, lp_split_idx_list, lp_context_data, lp_masks, lp_link_data_all = lp_data
    gc_data_list, gc_processed_data_list = gc_data
    active_tasks = get_task_schedule(epoch, args)
    if epoch == 0 or False:
        active_task_names = [name.upper() for name, active in active_tasks.items() if active]
        print(f"\n{'=' * 60}")
        print(f"[JOINT TRAINING] Epoch {epoch}: Active tasks = {', '.join(active_task_names)}")
        print(f"{'=' * 60}\n")
    if active_tasks['nc'] and hasattr(args, 'enable_nc') and args.enable_nc and (nc_data_list is not None) and (len(nc_data_list) > 0) and (args.lambda_nc > 0):
        nc_loss_result = train_all(model, nc_data_list, nc_split_idx_list, optimizer=opt_nc, pred=predictor, batch_size=args.nc_batch_size, degree=False, orthogonal_push=args.orthogonal_push, normalize_class_h=args.normalize_class_h, clip_grad=args.clip_grad, projector=None, rank=0, epoch=epoch, identity_projection=None, lambda_=args.lambda_nc, args=args)
        if nc_loss_result is not None:
            if isinstance(nc_loss_result, dict):
                total_nc_loss = nc_loss_result['total']
                nc_nll_loss = nc_loss_result['nll']
                nc_de_loss = nc_loss_result['de']
                nc_contrastive_loss = nc_loss_result.get('contrastive', 0)
                nc_original_loss = nc_loss_result.get('original_total', 0)
                nc_augmented_loss = nc_loss_result.get('augmented_total', 0)
                nc_num_original = nc_loss_result.get('num_original', 0)
                nc_num_augmented = nc_loss_result.get('num_augmented', 0)
            else:
                total_nc_loss = nc_loss_result
                nc_nll_loss = nc_loss_result
                nc_de_loss = 0.0
                nc_contrastive_loss = 0.0
                nc_original_loss = 0
                nc_augmented_loss = 0
                nc_num_original = 0
                nc_num_augmented = 0
            nc_count = len(nc_data_list)
    if active_tasks['lp'] and hasattr(args, 'enable_lp') and args.enable_lp and (lp_data_list is not None) and (len(lp_data_list) > 0) and (args.lambda_lp > 0):
        lp_tracker = None
        lp_loss_sum = 0.0
        for i, (data, split_idx) in enumerate(zip(lp_data_list, lp_split_idx_list)):
            link_data_all = lp_link_data_all[i]
            context_data = lp_context_data[i]
            train_mask = lp_masks[i]
            if 'train' in link_data_all and link_data_all['train']['edge_pairs'].size(0) > 0:
                data.x = data.x.to(device)
                data.adj_t = data.adj_t.to(device)
                if hasattr(data, 'full_adj_t') and data.full_adj_t is not None:
                    data.full_adj_t = data.full_adj_t.to(device)
                lp_loss = train_link_prediction(model, predictor, data, link_data_all['train'], context_data, train_mask, optimizer=opt_lp, batch_size=args.lp_batch_size, identity_projection=None, clip_grad=args.clip_grad, rank=0, orthogonal_push=args.orthogonal_push, normalize_class_h=args.normalize_class_h, epoch=epoch, mask_target_edges=args.mask_target_edges, degree=False, lambda_=args.lambda_lp, args=args)
                data.x = data.x.cpu()
                data.adj_t = data.adj_t.cpu()
                if hasattr(data, 'full_adj_t') and data.full_adj_t is not None:
                    data.full_adj_t = data.full_adj_t.cpu()
                if hasattr(data, 'x_pca') and data.x_pca is not None:
                    data.x_pca = data.x_pca.cpu()
                if hasattr(data, 'context_sample') and data.context_sample is not None:
                    data.context_sample = data.context_sample.cpu()
                if lp_loss is not None:
                    lp_loss_sum += lp_loss
                    lp_count += 1
                    gate_val = getattr(getattr(predictor, 'lp_head', None), 'last_gate_mean_train', None)
                    if gate_val is not None:
                        lp_gate_sum += float(gate_val)
                        lp_gate_count += 1
                    calib_val = getattr(getattr(predictor, 'lp_head', None), 'last_gate_calib_ms_train', None)
                    if calib_val is not None:
                        lp_gate_calib_sum += float(calib_val)
                        lp_gate_calib_count += 1
                    struct_loss_val = getattr(getattr(predictor, 'lp_head', None), 'last_struct_loss_train', None)
                    if struct_loss_val is not None:
                        lp_struct_loss_sum += float(struct_loss_val)
                        lp_struct_loss_count += 1
        if lp_count > 0:
            total_lp_loss = lp_loss_sum / lp_count
    if active_tasks['gc'] and hasattr(args, 'enable_gc') and args.enable_gc and (len(gc_data_list) > 0) and (args.lambda_gc > 0):
        gc_loss_sum = 0.0
        gc_dataset_count = 0
        use_multi_dataset_sampling = False
        use_gc_vectorized = getattr(args, 'gc_multitask_vectorized', False)
        for dataset_idx, dataset_info in enumerate(gc_processed_data_list):
            task_filtered_splits = dataset_info['task_filtered_splits']
            dataset_loss = 0.0
            dataset_tasks = 0
            sample_graph = dataset_info['dataset'][0]
            is_multitask = sample_graph.y.numel() > 1
            if use_gc_vectorized and is_multitask:
                use_index_tracking = 'feature_mapping' in dataset_info or 'tsgfm_mapping' in dataset_info or 'tag_mapping' in dataset_info
                unfiltered_loaders = create_data_loaders(dataset_info['dataset'], dataset_info['split_idx'], batch_size=args.gc_batch_size, shuffle=True, task_idx=None, use_index_tracking=use_index_tracking)
                from src.engine_gc import train_graph_classification_multitask_vectorized
                dataset_loss = train_graph_classification_multitask_vectorized(gc_encoder, predictor, dataset_info, unfiltered_loaders, opt_gc, pooling_method=args.graph_pooling, device=device, clip_grad=args.clip_grad, orthogonal_push=args.orthogonal_push, normalize_class_h=args.normalize_class_h, identity_projection=None, args=args, lambda_=args.lambda_gc)
                dataset_tasks = 1
            else:
                for task_idx, task_splits in task_filtered_splits.items():
                    use_index_tracking = 'feature_mapping' in dataset_info or 'tsgfm_mapping' in dataset_info or 'tag_mapping' in dataset_info
                    task_data_loaders = create_data_loaders(dataset_info['dataset'], task_splits, batch_size=args.gc_batch_size, shuffle=True, task_idx=task_idx, use_index_tracking=use_index_tracking)
                    task_loss = train_graph_classification_single_task(gc_encoder, predictor, dataset_info, task_data_loaders, opt_gc, task_idx, pooling_method=args.graph_pooling, device=device, clip_grad=args.clip_grad, orthogonal_push=args.orthogonal_push, normalize_class_h=args.normalize_class_h, identity_projection=None, lambda_=args.lambda_gc)
                    dataset_loss += task_loss
                    dataset_tasks += 1
            if dataset_tasks > 0:
                avg_dataset_loss = dataset_loss / dataset_tasks
                gc_loss_sum += avg_dataset_loss
                gc_dataset_count += 1
        if gc_dataset_count > 0:
            total_gc_loss = gc_loss_sum / gc_dataset_count
            gc_count = gc_dataset_count
    combined_loss = total_nc_loss + total_lp_loss + total_gc_loss + 0.0
    if args.use_separate_optimizers:
        args.lambda_nc = original_lambda_nc
        args.lambda_lp = original_lambda_lp
        args.lambda_gc = original_lambda_gc
    result_dict = {'nc_loss': total_nc_loss, 'lp_loss': total_lp_loss, 'gc_loss': total_gc_loss, 'combined_loss': combined_loss, 'nc_count': nc_count, 'lp_count': lp_count, 'gc_count': gc_count}
    return result_dict

def evaluate_node_classification(model, predictor, nc_data, args, split='valid', identity_projection=None, projector=None, nc_loaders=None, epoch=0):
    import time
    eval_start_time = time.time()
    print(f'Starting node classification evaluation ({split} split)...')
    model.eval()
    predictor.eval()
    results = {}
    nc_profile_enabled = split == 'test' and bool(False)
    nc_profile_proc = psutil.Process(os.getpid()) if nc_profile_enabled else None
    try:
        nc_profile_device = next(model.parameters()).device
    except StopIteration:
        try:
            nc_profile_device = next(predictor.parameters()).device
        except StopIteration:
            nc_profile_device = torch.device('cpu')

    def _nc_profile_snapshot():
        if not nc_profile_enabled:
            return None
        if isinstance(nc_profile_device, torch.device) and nc_profile_device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize(nc_profile_device)
        cpu_times = nc_profile_proc.cpu_times()
        return (time.perf_counter(), cpu_times.user + cpu_times.system)

    def _nc_profile_log(stage, start_snapshot=None, extra=''):
        if not nc_profile_enabled:
            return
        now_snapshot = _nc_profile_snapshot()
        wall_dt = 0.0
        cpu_dt = 0.0
        cpu_over_wall = 0.0
        if start_snapshot is not None:
            wall_dt = now_snapshot[0] - start_snapshot[0]
            cpu_dt = now_snapshot[1] - start_snapshot[1]
            if wall_dt > 1e-09:
                cpu_over_wall = 100.0 * cpu_dt / wall_dt
        sys_cpu = psutil.cpu_percent(interval=None)
        rss_gb = nc_profile_proc.memory_info().rss / 1024 ** 3
        gpu_alloc_gb = 0.0
        gpu_reserved_gb = 0.0
        if isinstance(nc_profile_device, torch.device) and nc_profile_device.type == 'cuda' and torch.cuda.is_available():
            dev_idx = nc_profile_device.index if nc_profile_device.index is not None else torch.cuda.current_device()
            gpu_alloc_gb = torch.cuda.memory_allocated(dev_idx) / 1024 ** 3
            gpu_reserved_gb = torch.cuda.memory_reserved(dev_idx) / 1024 ** 3
        extra_text = f' | {extra}' if extra else ''
        print(f'[NC_EVAL_PROFILE] {stage} | wall={wall_dt:.3f}s cpu={cpu_dt:.3f}s cpu/wall={cpu_over_wall:.1f}% sys_cpu={sys_cpu:.1f}% rss={rss_gb:.2f}GB gpu_alloc={gpu_alloc_gb:.2f}GB gpu_reserved={gpu_reserved_gb:.2f}GB{extra_text}')
    if nc_profile_enabled:
        psutil.cpu_percent(interval=None)
        _nc_profile_log('enabled', extra=f"split={split} use_tta={getattr(args, 'use_test_time_augmentation', False)} tta_aug={getattr(args, 'tta_num_augmentations', 0)} context_samples={getattr(args, 'unseen_test_context_samples', 1)} batch_size={getattr(args, 'test_batch_size', -1)}")
    with torch.no_grad():
        nc_data_list, nc_split_idx_list, nc_external_embeddings = nc_data
        if len(nc_data_list) > 0:
            if split == 'test':
                print(f'  Processing {len(nc_data_list)} unseen test datasets...')
                datasets_start_time = time.time()
                test_eval_profile_start = _nc_profile_snapshot()
                use_tta = getattr(args, 'use_test_time_augmentation', False)
                tta_gate_by_valid = getattr(args, 'tta_gate_by_valid', True)
                num_context_samples = max(1, int(getattr(args, 'unseen_test_context_samples', 1)))
                if num_context_samples > 1:
                    print(f'  Averaging test metrics over {num_context_samples} random context resamples')
                if nc_profile_enabled:
                    views_per_dataset = getattr(args, 'tta_num_augmentations', 0) + (1 if getattr(args, 'tta_include_original', True) else 0) if use_tta else 1
                    planned_eval_calls = len(nc_data_list) * num_context_samples * views_per_dataset
                    _nc_profile_log('plan', extra=f'datasets={len(nc_data_list)} context_samples={num_context_samples} views_per_dataset={views_per_dataset} total_view_evals~={planned_eval_calls}')

                def _average_metric_lists(metric_lists):
                    if not metric_lists:
                        return []
                    if not isinstance(metric_lists[0], (list, tuple)):
                        return sum(metric_lists) / len(metric_lists)
                    num_items = len(metric_lists[0])
                    avg = []
                    for i in range(num_items):
                        vals = [ml[i] for ml in metric_lists if len(ml) > i]
                        avg.append(sum(vals) / len(vals) if vals else 0.0)
                    return avg

                def _run_eval_once():
                    if use_tta:
                        from src.engine_nc import test_all_induct_with_tta
                        return test_all_induct_with_tta(model, predictor, nc_data_list, nc_split_idx_list, args.test_batch_size, False, None, None, True, None, 0, None, None, use_cs=args.use_cs, cs_num_iters=args.cs_num_iters, cs_alpha=args.cs_alpha, args=args)
                    return test_all_induct(model, predictor, nc_data_list, nc_split_idx_list, args.test_batch_size, False, None, None, True, None, 0, None, None, use_cs=args.use_cs, cs_num_iters=args.cs_num_iters, cs_alpha=args.cs_alpha, args=args, epoch=epoch)
                if num_context_samples == 1:
                    eval_once_start = _nc_profile_snapshot()
                    train_metrics, valid_metrics, test_metrics = _run_eval_once()
                    _nc_profile_log('eval_once', eval_once_start, extra=f'context_sample_idx=0 datasets={len(nc_data_list)}')
                else:
                    from src.data_utils import select_k_shot_context
                    original_contexts = []
                    for data in nc_data_list:
                        ctx = data.context_sample.clone() if hasattr(data, 'context_sample') and data.context_sample is not None else None
                        original_contexts.append(ctx)
                    all_train_metrics = []
                    all_valid_metrics = []
                    all_test_metrics = []
                    base_seed = _get_unseen_test_context_seed_base(args)
                    for sample_idx in range(num_context_samples):
                        context_resample_start = _nc_profile_snapshot()
                        torch.manual_seed(base_seed + sample_idx)
                        for data, split_idx in zip(nc_data_list, nc_split_idx_list):
                            context_shots = resolve_context_shots(data.name, 'nc', args, epoch=None)
                            if len(split_idx['train']) > 0:
                                context_source_split = split_idx['train']
                            else:
                                context_source_split = split_idx['test']
                            context_source_split = context_source_split.to(data.y.device)
                            new_context = select_k_shot_context(data, context_shots, context_source_split)
                            data.context_sample = new_context.to(data.y.device)
                        _nc_profile_log('context_resample', context_resample_start, extra=f'sample_idx={sample_idx} seed={base_seed + sample_idx}')
                        eval_once_start = _nc_profile_snapshot()
                        train_m, valid_m, test_m = _run_eval_once()
                        _nc_profile_log('eval_once', eval_once_start, extra=f'context_sample_idx={sample_idx} datasets={len(nc_data_list)}')
                        all_train_metrics.append(train_m)
                        all_valid_metrics.append(valid_m)
                        all_test_metrics.append(test_m)
                    for data, ctx in zip(nc_data_list, original_contexts):
                        if ctx is not None:
                            data.context_sample = ctx.to(data.y.device)
                    train_metrics = _average_metric_lists(all_train_metrics)
                    valid_metrics = _average_metric_lists(all_valid_metrics)
                    test_metrics = _average_metric_lists(all_test_metrics)
                datasets_time = time.time() - datasets_start_time
                if use_tta and tta_gate_by_valid:
                    tta_suffix = ' (TTA gated)'
                else:
                    tta_suffix = ' (with TTA)' if use_tta else ''
                print(f'  All {len(nc_data_list)} datasets completed in {datasets_time:.2f}s{tta_suffix}')
                _nc_profile_log('test_eval_total', test_eval_profile_start, extra=f'datasets={len(nc_data_list)} use_tta={use_tta} context_samples={num_context_samples}')
                results = {'train': sum(train_metrics) / len(train_metrics) if train_metrics else 0.0, 'valid': sum(valid_metrics) / len(valid_metrics) if valid_metrics else 0.0, 'test': sum(test_metrics) / len(test_metrics) if test_metrics else 0.0, 'individual_test_metrics': test_metrics if test_metrics else [], 'evaluation_time': datasets_time}
            else:
                train_metrics, valid_metrics, test_metrics = test_all(model, predictor, nc_data_list, nc_split_idx_list, args.test_batch_size, False, None, None, True, None, 0, None, None, use_cs=args.use_cs, cs_num_iters=args.cs_num_iters, cs_alpha=args.cs_alpha)
                results = {'train': train_metrics if isinstance(train_metrics, (int, float)) else sum(train_metrics) / len(train_metrics), 'valid': valid_metrics if isinstance(valid_metrics, (int, float)) else sum(valid_metrics) / len(valid_metrics), 'test': test_metrics if isinstance(test_metrics, (int, float)) else sum(test_metrics) / len(test_metrics), 'individual_test_metrics': test_metrics if isinstance(test_metrics, list) else [test_metrics]}
    eval_total_time = time.time() - eval_start_time
    print(f'Node classification evaluation ({split} split) completed in {eval_total_time:.2f}s')
    return results

@torch.no_grad()
def evaluate_link_prediction_task(model, predictor, lp_data, args, split='valid', identity_projection=None):
    model.eval()
    predictor.eval()
    data_list, split_list, contexts = lp_data[:3]
    links = lp_data[3] if split == 'test' else lp_data[4]
    device = _get_module_device(model, predictor)
    samples = max(1, args.unseen_test_context_samples) if split == 'test' else 1
    metrics = []
    for index, data in enumerate(data_list):
        split_key = split if split in links[index] else 'test'
        if links[index][split_key]['edge_pairs'].size(0) == 0:
            raise ValueError(f'{data.name} has no {split_key} edges')
        data.x = data.x.to(device)
        data.adj_t = data.adj_t.to(device)
        if getattr(data, 'full_adj_t', None) is not None:
            data.full_adj_t = data.full_adj_t.to(device)
        values = []
        for sample in range(samples):
            context = contexts[index]
            if samples > 1:
                torch.manual_seed(_get_unseen_test_context_seed_base(args) + index * 1000 + sample)
                source = links[index].get('train')
                if source is not None and source['edge_pairs'].size(0) > 0:
                    context, _ = select_link_context(source, resolve_context_shots(data.name, 'lp', args), args.context_neg_ratio, False)
            use_tta = split == 'test' and args.lp_use_test_time_augmentation
            evaluator = evaluate_link_prediction_with_tta if use_tta else evaluate_link_prediction
            options = {}
            if use_tta:
                options = dict(num_augmentations=args.lp_tta_num_augmentations if args.lp_tta_num_augmentations is not None else args.tta_num_augmentations, normalize_views=args.lp_tta_normalize_views, use_batchnorm=args.use_batchnorm, linear_projection=args.lp_tta_linear_projection, context_gate=args.lp_tta_context_gate, context_gate_tolerance=args.lp_tta_context_gate_tolerance, train_gate=args.lp_tta_train_gate, train_gate_edges=links[index].get('train'), train_gate_pos_samples=args.lp_tta_train_gate_pos_samples, train_gate_neg_ratio=args.lp_tta_train_gate_neg_ratio, train_gate_hits_k=args.lp_tta_train_gate_hits_k, train_gate_tolerance=args.lp_tta_train_gate_tolerance)
            result = evaluator(model, predictor, data, links[index][split_key], context, args.test_batch_size, None, None, None, None, 0, True, degree=False, k_values=[20, 50, 100], use_full_adj_for_test=split == 'test', lp_metric=args.lp_metric, lp_concat_common_neighbors=False, **options)
            values.append(result['default_metric'])
        metrics.append(sum(values) / len(values))
    mean = sum(metrics) / len(metrics) if metrics else 0.0
    return dict(train=mean, valid=mean, test=mean, individual_test_metrics=metrics)

def evaluate_graph_classification_task(model, predictor, gc_data, args, split='valid', identity_projection=None, gc_tracker=None, gc_model=None):
    """
    Evaluate graph classification task only.
    
    Returns:
        Dictionary with graph classification metrics
    """
    import time
    eval_start_time = time.time()
    print(f'Starting graph classification evaluation ({split} split)...')
    gc_encoder = model
    gc_encoder.eval()
    predictor.eval()
    gc_device = _get_module_device(gc_encoder, predictor)
    results = {}
    with torch.no_grad():
        gc_data_list, gc_processed_data_list = gc_data
        if len(gc_data_list) > 0:
            all_dataset_results = []
            individual_results = []
            datasets_start_time = time.time()
            print(f'  Processing {len(gc_processed_data_list)} datasets...')
            for dataset_idx, dataset_info in enumerate(gc_processed_data_list):
                dataset_start_time = time.time()
                dataset_name = dataset_info['dataset'].name if hasattr(dataset_info['dataset'], 'name') else f'gc_dataset_{dataset_idx}'
                if split == 'test':
                    task_filtered_splits = dataset_info['task_filtered_splits_test_only']
                else:
                    task_filtered_splits = dataset_info['task_filtered_splits']
                task_results = []
                tasks_start_time = time.time()
                sample_graph = dataset_info['dataset'][0]
                is_multitask = sample_graph.y.numel() > 1
                if is_multitask and getattr(args, 'gc_multitask_vectorized', False):
                    print(f'    Dataset {dataset_idx} ({dataset_name}): Vectorized multi-task evaluation')
                    from src.engine_gc import evaluate_graph_classification_multitask_vectorized
                    use_index_tracking = 'feature_mapping' in dataset_info or 'tsgfm_mapping' in dataset_info or 'tag_mapping' in dataset_info
                    if split == 'test':
                        test_only_splits = {'test': dataset_info['split_idx']['test']}
                        shuffle_train_eval = False
                        task_eval_loaders = create_data_loaders(dataset_info['dataset'], test_only_splits, batch_size=args.gc_test_batch_size, shuffle=shuffle_train_eval, task_idx=None, use_index_tracking=use_index_tracking)
                    else:
                        shuffle_train_eval = split == 'train' and getattr(args, 'gc_train_eval_max_batches', 0) > 0 and getattr(args, 'gc_train_eval_shuffle', True)
                        task_eval_loaders = create_data_loaders(dataset_info['dataset'], dataset_info['split_idx'], batch_size=args.gc_test_batch_size, shuffle=shuffle_train_eval, task_idx=None, use_index_tracking=use_index_tracking)
                    gc_vec_eval_fn = evaluate_graph_classification_multitask_vectorized
                    gc_vec_eval_kwargs = {}
                    if split == 'test' and getattr(args, 'gc_use_test_time_augmentation', False):
                        from src.engine_gc import evaluate_graph_classification_multitask_vectorized_with_tta
                        gc_vec_eval_fn = evaluate_graph_classification_multitask_vectorized_with_tta
                        gc_vec_eval_kwargs = {'num_augmentations': getattr(args, 'gc_tta_num_augmentations', None) if getattr(args, 'gc_tta_num_augmentations', None) is not None else getattr(args, 'tta_num_augmentations', 5), 'normalize_views': getattr(args, 'gc_tta_normalize_views', False), 'use_batchnorm': getattr(args, 'use_batchnorm', False), 'linear_projection': getattr(args, 'gc_tta_linear_projection', False), 'feature_projection': getattr(args, 'gc_tta_feature_projection', True), 'edge_dropout_rate': getattr(args, 'gc_tta_edge_dropout_rate', 0.0)}
                    task_eval_results = gc_vec_eval_fn(gc_encoder, predictor, dataset_info, task_eval_loaders, pooling_method=args.graph_pooling, device=gc_device, normalize_class_h=args.normalize_class_h, dataset_name=dataset_name, identity_projection=None, args=args, **gc_vec_eval_kwargs)
                    result_key = 'val' if split == 'valid' else split
                    split_result = task_eval_results.get(result_key, 0.0)
                    task_results.append(split_result)
                else:
                    print(f'    Dataset {dataset_idx} ({dataset_name}): Processing {len(task_filtered_splits)} tasks...')
                    for task_idx, task_splits in task_filtered_splits.items():
                        task_start_time = time.time()
                        setup_start_time = time.time()
                        use_index_tracking = 'feature_mapping' in dataset_info or 'tsgfm_mapping' in dataset_info or 'tag_mapping' in dataset_info
                        dataloader_start_time = time.time()
                        if split == 'test':
                            test_only_splits = {'test': task_splits['test']}
                            shuffle_train_eval = False
                            task_eval_loaders = create_data_loaders(dataset_info['dataset'], test_only_splits, batch_size=args.gc_test_batch_size, shuffle=shuffle_train_eval, task_idx=task_idx, use_index_tracking=use_index_tracking)
                        else:
                            shuffle_train_eval = split == 'train' and getattr(args, 'gc_train_eval_max_batches', 0) > 0 and getattr(args, 'gc_train_eval_shuffle', True)
                            task_eval_loaders = create_data_loaders(dataset_info['dataset'], task_splits, batch_size=args.gc_test_batch_size, shuffle=shuffle_train_eval, task_idx=task_idx, use_index_tracking=use_index_tracking)
                        dataloader_time = time.time() - dataloader_start_time
                        profiling_enabled = False
                        eval_start_time = time.time()
                        gc_eval_fn = evaluate_graph_classification_single_task_with_tta if split == 'test' and getattr(args, 'gc_use_test_time_augmentation', False) else evaluate_graph_classification_single_task
                        gc_eval_kwargs = {}
                        if gc_eval_fn is evaluate_graph_classification_single_task_with_tta:
                            gc_eval_kwargs = {'num_augmentations': getattr(args, 'gc_tta_num_augmentations', None) if getattr(args, 'gc_tta_num_augmentations', None) is not None else getattr(args, 'tta_num_augmentations', 5), 'normalize_views': getattr(args, 'gc_tta_normalize_views', False), 'use_batchnorm': getattr(args, 'use_batchnorm', False), 'linear_projection': getattr(args, 'gc_tta_linear_projection', False), 'feature_projection': getattr(args, 'gc_tta_feature_projection', True), 'edge_dropout_rate': getattr(args, 'gc_tta_edge_dropout_rate', 0.0)}
                        task_eval_results = gc_eval_fn(gc_encoder, predictor, dataset_info, task_eval_loaders, task_idx, pooling_method=args.graph_pooling, device=gc_device, normalize_class_h=args.normalize_class_h, dataset_name=dataset_name, identity_projection=None, args=args, **gc_eval_kwargs)
                        eval_time = time.time() - eval_start_time
                        result_key = 'val' if split == 'valid' else split
                        split_result = task_eval_results.get(result_key, 0.0)
                        task_results.append(split_result)
                        task_time = time.time() - task_start_time
                        overhead_time = task_time - eval_time - dataloader_time
                        if isinstance(split_result, dict):
                            display_metric = _select_primary_metric(split_result, override=getattr(args, 'gc_metric', 'auto'), prefer='ap')
                            print(f'      Task {task_idx}: {display_metric:.4f} (eval: {eval_time:.2f}s)')
                        else:
                            print(f'      Task {task_idx}: {split_result:.4f} (eval: {eval_time:.2f}s)')
                if task_results:
                    dataset_avg = aggregate_task_metrics(task_results)
                    if isinstance(dataset_avg, dict):
                        primary_metric = _select_primary_metric(dataset_avg, override=getattr(args, 'gc_metric', 'auto'), prefer='ap')
                        all_dataset_results.append(primary_metric)
                        individual_results.append(primary_metric)
                    else:
                        all_dataset_results.append(dataset_avg)
                        individual_results.append(dataset_avg)
                else:
                    all_dataset_results.append(0.0)
                    individual_results.append(0.0)
                dataset_time = time.time() - dataset_start_time
                tasks_time = time.time() - tasks_start_time
                print(f'    Dataset {dataset_idx} ({dataset_name}): completed in {dataset_time:.2f}s (tasks: {tasks_time:.2f}s)')
            if all_dataset_results:
                avg_result = sum(all_dataset_results) / len(all_dataset_results)
                results = {'train': avg_result, 'valid': avg_result, 'test': avg_result, 'individual_test_metrics': individual_results}
            else:
                results = {'train': 0.0, 'valid': 0.0, 'test': 0.0, 'individual_test_metrics': []}
            datasets_total_time = time.time() - datasets_start_time
            print(f'  All {len(gc_processed_data_list)} datasets completed in {datasets_total_time:.2f}s')
    eval_total_time = time.time() - eval_start_time
    print(f'Graph classification evaluation ({split} split) completed in {eval_total_time:.2f}s')
    return results

def joint_evaluation(model, predictor, nc_data, lp_data, gc_data, args, split='valid', identity_projection=None, projector=None, gc_tracker=None, nc_loaders=None, epoch=0, gc_model=None):
    results = {'nc_metrics': {}, 'lp_metrics': {}, 'gc_metrics': {}}
    if hasattr(args, 'enable_nc') and args.enable_nc and (nc_data is not None) and (nc_data[0] is not None):
        nc_results = evaluate_node_classification(model, predictor, nc_data, args, split, None, None, None, epoch=epoch)
        results['nc_metrics'] = nc_results
    if hasattr(args, 'enable_lp') and args.enable_lp and (lp_data is not None) and (lp_data[0] is not None):
        lp_results = evaluate_link_prediction_task(model, predictor, lp_data, args, split, None)
        results['lp_metrics'] = lp_results
    if hasattr(args, 'enable_gc') and args.enable_gc and (gc_data is not None) and (len(gc_data[0]) > 0):
        gc_results = evaluate_graph_classification_task(model, predictor, gc_data, args, split, None, None, gc_model=None)
        results['gc_metrics'] = gc_results
    return results

def run_joint_training(args, device='cpu'):
    """Train on source splits or evaluate an immutable checkpoint."""
    checkpoint = None
    if args.use_pretrained_model:
        _, checkpoint = load_checkpoint_config(args.load_checkpoint)
        override_args_from_checkpoint(args, checkpoint['args'], predictor_state_dict=checkpoint['predictor_state_dict'])
    data = load_and_preprocess_data(args, device, skip_training_data=args.use_pretrained_model)
    model, predictor, _, _ = create_unified_model(args, args.hidden, device)
    if checkpoint is not None:
        load_checkpoint_states(checkpoint, model, predictor)
    else:
        params = list(model.parameters()) + list(predictor.parameters())
        optimizer_cls = torch.optim.Adam if args.optimizer == 'adam' else torch.optim.AdamW
        optimizers = {task: optimizer_cls(params, lr=getattr(args, 'lr_' + task) or args.lr, weight_decay=args.weight_decay, eps=args.eps) for task in ('nc', 'lp', 'gc')}
        if not args.use_separate_optimizers:
            optimizers = dict.fromkeys(('nc', 'lp', 'gc'), optimizers['nc'])
        schedulers = []
        for opt in dict.fromkeys(optimizers.values()):
            if args.schedule == 'cosine':
                schedulers.append(torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs))
            elif args.schedule == 'step':
                schedulers.append(torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, args.epochs // 5), gamma=0.5))
        best_score, best_epoch, best_state = (float('-inf'), -1, None)
        for epoch in range(args.epochs):
            refresh_contexts_if_needed(epoch, args, data)
            losses = joint_training_step(model, predictor, data['nc_train'], data['lp_train'], data['gc_train'], optimizers['nc'], args, epoch, optimizer_nc=optimizers['nc'], optimizer_lp=optimizers['lp'], optimizer_gc=optimizers['gc'])
            for scheduler in schedulers:
                scheduler.step()
            valid = joint_evaluation(model, predictor, data['nc_train'], data['lp_train'], data['gc_train'], args, split='valid', epoch=epoch)
            score = sum((valid[task + '_metrics'].get('valid', 0.0) for task in ('nc', 'lp', 'gc')))
            print(f"Epoch {epoch + 1}/{args.epochs}: loss={float(losses['combined_loss']):.6f}, source validation={score:.6f}")
            if score > best_score:
                best_score, best_epoch = (score, epoch)
                best_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(predictor.state_dict()))
        if best_state is None:
            raise RuntimeError('No finite source-validation score was obtained')
        model.load_state_dict(best_state[0])
        predictor.load_state_dict(best_state[1])
        if args.save_checkpoint:
            save_checkpoint(model, predictor, optimizers['nc'], args, {'best_valid': best_score}, best_epoch)
    model.eval()
    predictor.eval()
    results = joint_evaluation(model, predictor, data['nc_test'], data['lp_test'], data['gc_test'], args, split='test')
    return tuple((results[k + '_metrics'].get('test', 0.0) for k in ('nc', 'lp', 'gc'))) + tuple((results[k + '_metrics'].get('individual_test_metrics', []) for k in ('nc', 'lp', 'gc')))

def set_all_random_seeds(seed):
    """Set all random seeds for reproducible results."""
    print(f'Setting all random seeds to: {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print('✓ CUDA deterministic settings enabled')
    print('✓ All random seeds initialized for reproducible inference')

def main():
    args = parse_joint_training_args()
    if args.use_pretrained_model and (not args.load_checkpoint or not os.path.isfile(args.load_checkpoint)):
        raise ValueError('Evaluation requires --load_checkpoint pointing to a local checkpoint')
    if args.load_checkpoint and (not args.use_pretrained_model):
        raise ValueError('Use --use_pretrained_model true to evaluate a checkpoint')
    if args.runs < 1 or args.epochs < 1:
        raise ValueError('runs and epochs must be positive')
    if not any((args.enable_nc, args.enable_lp, args.enable_gc)):
        raise ValueError('Enable at least one task')
    if args.gpu == 'cpu':
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:' + args.gpu if torch.cuda.is_available() else 'cpu')
    rows = []
    for run in range(args.runs):
        cfg = copy.deepcopy(args)
        cfg._current_run_idx = run
        cfg._current_run_seed = args.seed + run
        set_all_random_seeds(cfg._current_run_seed)
        if args.runs > 1 and (not args.use_pretrained_model):
            cfg.checkpoint_name = (args.checkpoint_name or 'gilt_model') + f'_run{run}'
        rows.append(run_joint_training(cfg, device))
    for task_idx, task in enumerate(('nc', 'lp', 'gc')):
        if not getattr(args, 'enable_' + task):
            continue
        names = getattr(args, task + '_test_dataset').split(',')
        for dataset_idx, name in enumerate(names):
            values = [float(_select_primary_metric(row[task_idx + 3][dataset_idx])) for row in rows]
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            print(f'{task.upper()} {name}: {100 * np.mean(values):.4f} +/- {100 * std:.4f} (runs={len(values)})')
if __name__ == '__main__':
    main()
