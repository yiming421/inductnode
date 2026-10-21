"""
Training and evaluation engine for graph classification using PFN predictor.
Reuses the existing PFNPredictorNodeCls by treating pooled graph embeddings as node embeddings.
"""
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from sklearn.metrics import roc_auc_score, average_precision_score
from .data_gc import create_graph_batch
from .utils import process_node_features
from .data_utils import batch_edge_dropout, feature_dropout, apply_random_projection_augmentation

def apply_feature_dropout_if_enabled(x, args, rank=0, training=True):
    if args is not None and hasattr(args, 'feature_dropout_enabled') and args.feature_dropout_enabled and hasattr(args, 'feature_dropout_rate') and (args.feature_dropout_rate > 0):
        dropout_type = getattr(args, 'feature_dropout_type', 'element_wise')
        verbose = getattr(args, 'verbose_feature_dropout', False) and rank == 0
        return feature_dropout(x, args.feature_dropout_rate, training=training, dropout_type=dropout_type, verbose=verbose)
    return x

def _has_trainable_parameters(module):
    """Return True if module has any trainable parameter."""
    if module is None:
        return False
    return any((p.requires_grad for p in module.parameters()))

def _is_puregcn_v1_parameter_free(model_module):
    """
    Conservative check for parameter-free PureGCN_v1 forward path.

    Returns:
        (is_safe, reason)
    """
    if model_module is None:
        return (False, 'model_missing')
    if model_module.__class__.__name__ != 'PureGCN_v1':
        return (False, 'model_not_puregcn_v1')
    if not isinstance(getattr(model_module, 'lin', None), nn.Identity):
        return (False, 'input_projection_not_identity')
    if getattr(model_module, 'use_virtual_node', False):
        return (False, 'virtual_node_enabled')
    if float(getattr(model_module, 'dp', 0.0)) > 0:
        return (False, 'model_dropout_enabled')
    if getattr(model_module, 'norm', False):
        norms = getattr(model_module, 'norms', None)
        if norms is None:
            return (False, 'missing_norm_layers')
        for layer_norm in norms:
            if getattr(layer_norm, 'elementwise_affine', False):
                return (False, 'gnn_norm_affine_enabled')
    if _has_trainable_parameters(model_module):
        return (False, 'model_has_trainable_params')
    return (True, None)

def _resolve_static_embedding_cache_policy(args, model, identity_projection=None, hard_blockers=None):
    """Shared cache policy resolution for GC train loops."""
    cache_mode = getattr(args, 'static_embedding_cache', 'auto') if args is not None else 'auto'
    if cache_mode not in ('off', 'auto', 'force'):
        cache_mode = 'auto'
    model_module = model.module if hasattr(model, 'module') else model
    model_has_trainable_params = _has_trainable_parameters(model_module)
    identity_has_trainable_params = _has_trainable_parameters(None)
    puregcn_cacheable, puregcn_reason = _is_puregcn_v1_parameter_free(model_module)
    blockers = list(hard_blockers or [])
    cache_notes = []
    use_static_embedding_cache = False
    if cache_mode == 'off':
        cache_notes.append('policy_off')
    elif len(blockers) > 0:
        cache_notes.extend(blockers)
    elif cache_mode == 'auto':
        if not puregcn_cacheable:
            cache_notes.append(puregcn_reason or 'model_not_cacheable')
        if model_has_trainable_params:
            cache_notes.append('model_has_trainable_params')
        if identity_has_trainable_params:
            cache_notes.append('identity_projection_has_trainable_params')
        use_static_embedding_cache = len(cache_notes) == 0
    else:
        if model_has_trainable_params:
            cache_notes.append('force_rejected_model_has_trainable_params')
        if identity_has_trainable_params:
            cache_notes.append('force_rejected_identity_projection_has_trainable_params')
        use_static_embedding_cache = len(cache_notes) == 0
        if use_static_embedding_cache and (not puregcn_cacheable):
            cache_notes.append(f'forced_without_strict_check:{puregcn_reason or chr(63)}')
    cache_notes = list(dict.fromkeys(cache_notes))
    return (use_static_embedding_cache, cache_mode, cache_notes)

def refresh_gc_context_if_needed(dataset_info, batch_idx, epoch, args, device='cuda', task_idx=None):
    if getattr(args, 'context_batch_refresh_interval', 0) <= 0:
        return
    if batch_idx > 0 and batch_idx % args.context_batch_refresh_interval == 0:
        refresh_seed = args.seed + epoch * 10000 + batch_idx
        torch.manual_seed(refresh_seed)
        dataset = dataset_info['dataset']
        dataset_name = getattr(dataset, 'name', 'unknown')
        if 'context_graphs' in dataset_info:
            refresh_gc_context_graphs(dataset_info, args, device)
            if task_idx is not None:
                print(f'🔄 GC Dataset {dataset_name} context refreshed for task {task_idx} at batch {batch_idx}')
            else:
                print(f'🔄 GC Dataset {dataset_name} context refreshed (all tasks) at batch {batch_idx}')

def refresh_gc_context_graphs(dataset_info, args, device):
    """
    Refresh context graphs structure by resampling context graphs for each task/class.
    """
    from .data_gc import prepare_graph_data_for_pfn
    if 'context_graphs' not in dataset_info:
        return
    context_k = getattr(args, 'context_k', 5)
    refreshed_data = prepare_graph_data_for_pfn(dataset_info['dataset'], dataset_info['split_idx'], context_k, device)
    dataset_info['context_graphs'] = refreshed_data['context_graphs']

def _get_node_embedding_table(dataset, task_idx, device, dataset_info=None):
    if dataset_info and dataset_info.get('_gc_tta_node_embs_override', None) is not None:
        override = dataset_info['_gc_tta_node_embs_override']
        if 'feature_mapping' in dataset_info:
            return override
        return override.to(device)
    if dataset_info and 'feature_mapping' in dataset_info:
        feature_mapping = dataset_info['feature_mapping']
        if feature_mapping and 'node_embs' in feature_mapping:
            return feature_mapping['node_embs']
    if hasattr(dataset, 'node_embs') and dataset.node_embs is not None:
        return dataset.node_embs.to(device)
    print(f'[ERROR] No node_embs found in dataset - expected under unified setting')
    return None

def _apply_gc_tta_edge_dropout_if_needed(batch_data, dataset_info):
    """Apply deterministic per-view undirected edge-pair dropout for GC TTA only."""
    if not dataset_info:
        return batch_data
    dropout_rate = float(dataset_info.get('_gc_tta_edge_dropout_rate', 0.0) or 0.0)
    if dropout_rate <= 0.0 or not hasattr(batch_data, 'edge_index') or batch_data.edge_index is None:
        return batch_data
    dropout_rate = min(max(dropout_rate, 0.0), 1.0)
    edge_index = batch_data.edge_index
    num_edges = edge_index.size(1)
    if num_edges == 0:
        return batch_data
    row, col = (edge_index[0], edge_index[1])
    lo = torch.minimum(row, col)
    hi = torch.maximum(row, col)
    pair_key = lo * int(batch_data.num_nodes) + hi
    unique_key, inverse = torch.unique(pair_key, sorted=False, return_inverse=True)
    num_pairs = unique_key.numel()
    if num_pairs == 0:
        return batch_data
    base_seed = int(dataset_info.get('_gc_tta_edge_dropout_seed', 0) or 0)
    counter = int(dataset_info.get('_gc_tta_edge_dropout_counter', 0) or 0)
    dataset_info['_gc_tta_edge_dropout_counter'] = counter + 1
    seed = base_seed + counter
    devices = []
    if edge_index.is_cuda:
        devices = [edge_index.device.index if edge_index.device.index is not None else torch.cuda.current_device()]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        pair_keep = torch.rand(num_pairs, device=edge_index.device) > dropout_rate
        if not pair_keep.any():
            pair_keep[torch.randint(num_pairs, (1,), device=edge_index.device)] = True
        keep_mask = pair_keep[inverse]
    stats = dataset_info.setdefault('_gc_tta_edge_dropout_stats', [])
    stats.append((int(num_pairs), int(pair_keep.sum().item()), int(num_edges), int(keep_mask.sum().item())))
    batch_data = batch_data.clone()
    batch_data.edge_index = edge_index[:, keep_mask]
    if hasattr(batch_data, 'edge_attr') and batch_data.edge_attr is not None:
        batch_data.edge_attr = batch_data.edge_attr[keep_mask]
    return batch_data

def _safe_lookup_node_embeddings_micro_optimized(node_emb_table: torch.Tensor, x: torch.Tensor, context: str='', batch_data=None, dataset_info=None) -> torch.Tensor:
    """
    Micro-transfer optimized version - eliminates thousands of GPU-CPU transfers.
    Handles external mapping for both OGB (integer features) and TU (float features).
    """
    if dataset_info and 'feature_mapping' in dataset_info:
        if batch_data is None or not hasattr(batch_data, 'original_graph_indices'):
            raise ValueError(f'[FUG] Batch data with original_graph_indices required for external mapping in {context}')
        feature_mapping = dataset_info['feature_mapping']
        node_mapping = feature_mapping['node_index_mapping']
        original_indices_cpu = batch_data.original_graph_indices.cpu()
        try:
            indices_list = [node_mapping[g_idx.item()] for g_idx in original_indices_cpu]
        except KeyError as e:
            raise ValueError(f'[FUG] Original graph index {e.args[0]} not found.') from e
        all_fug_indices_cpu = torch.cat(indices_list)
        processed_embeddings_cpu = node_emb_table[all_fug_indices_cpu]
        if processed_embeddings_cpu.shape[0] != x.shape[0]:
            raise ValueError(f'[FUG] Size mismatch after mapping: got {processed_embeddings_cpu.shape[0]} embeddings for {x.shape[0]} nodes')
        target_device = x.device if x.numel() > 0 else 'cuda'
        result = processed_embeddings_cpu.to(target_device)
        return result
    raise ValueError(f'[FUG] _safe_lookup_node_embeddings_micro_optimized called without feature mapping in {context}')

def _safe_lookup_node_embeddings(node_emb_table: torch.Tensor, x: torch.Tensor, context: str='', batch_data=None, dataset_info=None) -> torch.Tensor:
    if dataset_info and 'feature_mapping' in dataset_info:
        return _safe_lookup_node_embeddings_micro_optimized(node_emb_table, x, context, batch_data, dataset_info)
    if x.dim() == 2 and x.size(1) > 1 and x.dtype.is_floating_point:
        return x
    if x.dim() == 2 and x.size(1) == 1:
        x = x.squeeze(1)
    if x.dtype != torch.long:
        if not torch.allclose(x, x.round()):
            raise ValueError(f'[SAFE_LOOKUP]{context} x contains non-integer values; cannot treat as indices. dtype={x.dtype}')
        x = x.long()
    if x.dim() != 1:
        raise ValueError(f'[SAFE_LOOKUP]{context} Expected index tensor 1D after processing, got shape {tuple(x.shape)}')
    if x.numel() == 0:
        return x.new_empty((0, node_emb_table.size(1)))
    min_idx = int(x.min().item())
    max_idx = int(x.max().item())
    table_size = node_emb_table.size(0)
    if min_idx < 0 or max_idx >= table_size:
        raise IndexError(f'[SAFE_LOOKUP]{context} Index out of bounds: min={min_idx} max={max_idx} table_size={table_size}. This would have triggered a CUDA device assert during gather.')
    return node_emb_table[x]

def get_dataset_metric(dataset_name, num_classes=None, is_multitask=None, metric_override=None):
    dataset_name = dataset_name.lower()
    if metric_override and metric_override != 'auto':
        return metric_override
    if 'chemhiv' in dataset_name or 'hiv' in dataset_name:
        return 'auc'
    elif 'chempcba' in dataset_name or 'pcba' in dataset_name:
        return ['ap', 'auc']
    elif 'mnist' in dataset_name:
        return 'accuracy'
    if num_classes is not None:
        if num_classes == 2:
            return 'auc'
        else:
            return 'accuracy'
    if is_multitask is not None:
        if is_multitask:
            return 'ap'
        else:
            return 'auc'
    return 'accuracy'

def calculate_multiple_metrics(predictions, labels, probabilities, metric_types):
    results = {}
    for metric_type in metric_types:
        results[metric_type] = calculate_metric(predictions, labels, probabilities, metric_type)
    return results

def aggregate_task_metrics(metric_values):
    if not metric_values:
        return 0.0

    def _mean_ignore_nan(values):
        filtered = [v for v in values if v == v]
        if not filtered:
            return 0.0
        return sum(filtered) / len(filtered)
    if all((isinstance(val, dict) for val in metric_values)):
        aggregated_metrics = {}
        for metric_name in metric_values[0].keys():
            metric_vals = [val[metric_name] for val in metric_values]
            aggregated_metrics[metric_name] = _mean_ignore_nan(metric_vals)
        return aggregated_metrics
    elif all((isinstance(val, (int, float)) for val in metric_values)):
        return _mean_ignore_nan(metric_values)
    else:
        primary_values = []
        for val in metric_values:
            if isinstance(val, dict):
                primary_val = val.get('auc', val.get('ap', next(iter(val.values()))))
                primary_values.append(primary_val)
            else:
                primary_values.append(val)
        return sum(primary_values) / len(primary_values)

def calculate_metric(predictions, labels, probabilities, metric_type):
    if metric_type == 'accuracy':
        accuracy = (predictions == labels).float().mean().item()
        return accuracy
    elif metric_type == 'auc':
        if probabilities.dim() > 1:
            if probabilities.shape[1] == 2:
                probs = probabilities[:, 1].cpu().numpy()
            elif probabilities.shape[1] == 1:
                probs = probabilities.view(-1).cpu().numpy()
            else:
                probs = probabilities.cpu().numpy()
        else:
            probs = probabilities.cpu().numpy()
        labels_np = labels.cpu().numpy()
        unique_labels = set(labels_np)
        if len(unique_labels) < 2:
            return float('nan')
        try:
            auc_score = roc_auc_score(labels_np, probs)
            if auc_score != auc_score:
                pass
                return 0.0
            return auc_score
        except ValueError as e:
            return float('nan')
    elif metric_type == 'ap':
        if probabilities.dim() > 1:
            if probabilities.shape[1] == 2:
                probs = probabilities[:, 1].cpu().numpy()
            elif probabilities.shape[1] == 1:
                probs = probabilities.view(-1).cpu().numpy()
            else:
                probs = probabilities.cpu().numpy()
        else:
            probs = probabilities.cpu().numpy()
        try:
            labels_np = labels.cpu().numpy()
            probs_np = probs
            ap_score = average_precision_score(labels_np, probs_np)
            return ap_score
        except ValueError as e:
            return float('nan')
    else:
        raise ValueError(f'Unknown metric type: {metric_type}')

def pool_graph_embeddings(node_embeddings, batch, pooling_method='mean', virtualnode_embeddings=None):
    if isinstance(node_embeddings, tuple):
        node_embeddings, virtualnode_embeddings = node_embeddings
    if virtualnode_embeddings is not None:
        return virtualnode_embeddings
    if pooling_method == 'mean':
        return global_mean_pool(node_embeddings, batch)
    elif pooling_method == 'max':
        return global_max_pool(node_embeddings, batch)
    elif pooling_method == 'sum':
        return global_add_pool(node_embeddings, batch)
    else:
        raise ValueError(f'Unsupported pooling method: {pooling_method}')

def _create_context_embeddings_computed(model, context_structure, dataset, task_idx, pooling_method, device, identity_projection, dataset_info):
    """Compute context embeddings normally (fallback for datasets without pre-computed embeddings)."""
    if task_idx not in context_structure:
        raise ValueError(f'Task {task_idx} not found in context structure')
    task_context = context_structure[task_idx]
    context_embeddings = []
    context_labels = []
    for class_label, context_data in task_context.items():
        if not context_data:
            continue
        if isinstance(context_data, dict) and 'graphs' in context_data and ('indices' in context_data):
            graphs = context_data['graphs']
            graph_indices = context_data['indices']
        else:
            graphs = context_data
            graph_indices = None
        batch_data = create_graph_batch(graphs, device)
        if graph_indices is not None:
            batch_data.original_graph_indices = torch.tensor(graph_indices, dtype=torch.long)
        node_emb_table = _get_node_embedding_table(dataset, task_idx, device, dataset_info)
        if node_emb_table is not None:
            batch_data.x = _safe_lookup_node_embeddings(node_emb_table, batch_data.x, context=f'context_task{task_idx}', batch_data=batch_data, dataset_info=dataset_info)
        else:
            raise ValueError('Expected node embedding table under unified setting')
        x_input = batch_data.x
        batch_data = _apply_gc_tta_edge_dropout_if_needed(batch_data, dataset_info)
        batch_data.adj_t = SparseTensor.from_edge_index(batch_data.edge_index, sparse_sizes=(batch_data.num_nodes, batch_data.num_nodes)).to_symmetric().coalesce()
        node_embeddings = model(x_input, batch_data.adj_t, batch_data.batch)
        graph_embeddings = pool_graph_embeddings(node_embeddings, batch_data.batch, pooling_method)
        context_embeddings.append(graph_embeddings)
        context_labels.extend([class_label] * len(graphs))
    if context_embeddings:
        context_embeddings = torch.cat(context_embeddings, dim=0)
        context_labels = torch.tensor(context_labels, device=device, dtype=torch.long)
    else:
        hidden_dim = 256
        context_embeddings = torch.empty(0, hidden_dim, device=device)
        context_labels = torch.empty(0, dtype=torch.long, device=device)
    return (context_embeddings, context_labels)

def _create_all_task_context_embeddings(model, context_structure, dataset, pooling_method, device, identity_projection, dataset_info, return_timing=False, sync_cuda=False, batch_size=None):
    if not context_structure:
        if return_timing:
            return ([], [], {'encode_time': 0.0, 'overhead_time': 0.0, 'concat_time': 0.0, 'total_time': 0.0, 'num_context_batches': 0})
        return ([], [])
    task_indices = sorted(context_structure.keys())
    num_tasks = max(task_indices) + 1 if task_indices else 0
    pos_indices_by_task = [[] for _ in range(num_tasks)]
    neg_indices_by_task = [[] for _ in range(num_tasks)]
    timing = {'encode_time': 0.0, 'overhead_time': 0.0, 'concat_time': 0.0, 'total_time': 0.0, 'num_context_batches': 0}
    total_start = time.perf_counter()
    use_sync = sync_cuda and torch.cuda.is_available()
    use_indices = True
    for task_context in context_structure.values():
        for context_data in task_context.values():
            if not context_data:
                continue
            if not (isinstance(context_data, dict) and 'graphs' in context_data and ('indices' in context_data)):
                use_indices = False
                break
        if not use_indices:
            break
    if dataset_info and 'feature_mapping' in dataset_info and (not use_indices):
        raise ValueError('Context indices required for feature mapping but not found in context_structure')
    prep_start = time.perf_counter()
    flat_graphs = []
    flat_indices = []
    index_to_pos = {}
    if use_indices:
        for task_idx in task_indices:
            task_context = context_structure[task_idx]
            for class_label, context_data in task_context.items():
                if not context_data:
                    continue
                graphs = context_data['graphs']
                indices = context_data['indices']
                positions = []
                for graph, idx in zip(graphs, indices):
                    idx_int = idx.item() if torch.is_tensor(idx) else int(idx)
                    if idx_int not in index_to_pos:
                        index_to_pos[idx_int] = len(flat_graphs)
                        flat_graphs.append(graph)
                        flat_indices.append(idx_int)
                    positions.append(index_to_pos[idx_int])
                if int(class_label) == 1:
                    pos_indices_by_task[task_idx] = positions
                else:
                    neg_indices_by_task[task_idx] = positions
    else:
        for task_idx in task_indices:
            task_context = context_structure[task_idx]
            for class_label, context_data in task_context.items():
                if not context_data:
                    continue
                if isinstance(context_data, dict) and 'graphs' in context_data:
                    graphs = context_data['graphs']
                else:
                    graphs = context_data
                start = len(flat_graphs)
                flat_graphs.extend(graphs)
                positions = list(range(start, start + len(graphs)))
                if int(class_label) == 1:
                    pos_indices_by_task[task_idx] = positions
                else:
                    neg_indices_by_task[task_idx] = positions
    prep_end = time.perf_counter()
    timing['overhead_time'] += prep_end - prep_start
    if not flat_graphs:
        empty_pos = [None] * num_tasks
        empty_neg = [None] * num_tasks
        timing['total_time'] = time.perf_counter() - total_start
        if return_timing:
            return (empty_pos, empty_neg, timing)
        return (empty_pos, empty_neg)
    if batch_size is None or batch_size <= 0:
        batch_size = len(flat_graphs)
    node_emb_table = _get_node_embedding_table(dataset, 0, device, dataset_info)
    if node_emb_table is None:
        raise ValueError('Expected node embedding table under unified setting')
    context_embeddings = []
    for start in range(0, len(flat_graphs), batch_size):
        end = min(start + batch_size, len(flat_graphs))
        step_start = time.perf_counter()
        batch_graphs = flat_graphs[start:end]
        batch_data = create_graph_batch(batch_graphs, device)
        if use_indices:
            batch_data.original_graph_indices = torch.tensor(flat_indices[start:end], dtype=torch.long)
        batch_data.x = _safe_lookup_node_embeddings(node_emb_table, batch_data.x, context='vec_context_all', batch_data=batch_data, dataset_info=dataset_info)
        x_input = batch_data.x
        batch_data = _apply_gc_tta_edge_dropout_if_needed(batch_data, dataset_info)
        batch_data.adj_t = SparseTensor.from_edge_index(batch_data.edge_index, sparse_sizes=(batch_data.num_nodes, batch_data.num_nodes)).to_symmetric().coalesce()
        if use_sync:
            torch.cuda.synchronize()
        encode_start = time.perf_counter()
        node_embeddings = model(x_input, batch_data.adj_t, batch_data.batch)
        graph_embeddings = pool_graph_embeddings(node_embeddings, batch_data.batch, pooling_method)
        if use_sync:
            torch.cuda.synchronize()
        encode_end = time.perf_counter()
        timing['encode_time'] += encode_end - encode_start
        timing['overhead_time'] += time.perf_counter() - step_start - (encode_end - encode_start)
        timing['num_context_batches'] += 1
        context_embeddings.append(graph_embeddings)
    concat_start = time.perf_counter()
    context_embeddings = torch.cat(context_embeddings, dim=0) if context_embeddings else torch.empty((0, 0), device=device)
    pos_embeddings_by_task = [None] * num_tasks
    neg_embeddings_by_task = [None] * num_tasks
    for t in range(num_tasks):
        pos_idx = pos_indices_by_task[t]
        neg_idx = neg_indices_by_task[t]
        if pos_idx:
            pos_tensor = torch.as_tensor(pos_idx, device=context_embeddings.device, dtype=torch.long)
            pos_embeddings_by_task[t] = context_embeddings.index_select(0, pos_tensor)
        if neg_idx:
            neg_tensor = torch.as_tensor(neg_idx, device=context_embeddings.device, dtype=torch.long)
            neg_embeddings_by_task[t] = context_embeddings.index_select(0, neg_tensor)
    timing['concat_time'] += time.perf_counter() - concat_start
    timing['total_time'] = time.perf_counter() - total_start
    if return_timing:
        return (pos_embeddings_by_task, neg_embeddings_by_task, timing)
    return (pos_embeddings_by_task, neg_embeddings_by_task)

def _vectorized_multitask_pfn_logits(predictor, target_embeddings, pos_embeds_by_task, neg_embeds_by_task, normalize_class_h=True, device=None, task_indices=None):
    """
    Compute per-task logits using PFN predictor. Runs PFN once per task.

    Returns:
        logits: [B, len(task_indices)] (or [B, T] when task_indices is None)
        task_context_mask: [len(task_indices)] bool (True if task has both pos/neg context)
    """
    if device is None:
        device = target_embeddings.device
    num_tasks = len(pos_embeds_by_task)
    if task_indices is None:
        task_indices = list(range(num_tasks))
    else:
        task_indices = list(task_indices)
    logits_by_task = []
    task_context_mask = torch.zeros(len(task_indices), dtype=torch.bool, device=device)
    for out_idx, t in enumerate(task_indices):
        if t < 0 or t >= num_tasks:
            raise IndexError(f'Task index {t} out of range [0, {num_tasks - 1}]')
        pos_e = pos_embeds_by_task[t]
        neg_e = neg_embeds_by_task[t]
        if pos_e is None or neg_e is None or pos_e.numel() == 0 or (neg_e.numel() == 0):
            logits_by_task.append(torch.zeros(target_embeddings.size(0), device=device))
            continue
        context_embeddings = torch.cat([pos_e, neg_e], dim=0)
        context_labels = torch.cat([torch.ones(pos_e.size(0), device=device), torch.zeros(neg_e.size(0), device=device)], dim=0).long()
        pfn_data = prepare_pfn_data_structure(context_embeddings, context_labels, num_classes=2, device=device)
        class_h = process_node_features(context_embeddings, pfn_data, degree_normalize=False, attention_pool_module=None, mlp_module=None, normalize=normalize_class_h)
        pred_output = predictor(pfn_data, context_embeddings, target_embeddings, context_labels, class_h, task_type='graph_classification')
        if isinstance(pred_output, (tuple, list)):
            scores = pred_output[0]
        else:
            scores = pred_output
        if scores.dim() == 2 and scores.size(1) == 2:
            logits_t = scores[:, 1] - scores[:, 0]
        else:
            logits_t = scores.view(-1)
        logits_by_task.append(logits_t)
        task_context_mask[out_idx] = True
    logits = torch.stack(logits_by_task, dim=1) if logits_by_task else torch.empty((target_embeddings.size(0), 0), device=device)
    return (logits, task_context_mask)

def train_graph_classification_multitask_vectorized(model, predictor, dataset_info, data_loaders, optimizer, pooling_method='mean', device='cuda', clip_grad=1.0, orthogonal_push=0.0, normalize_class_h=True, identity_projection=None, args=None, lambda_=1.0):
    model.train()
    predictor.train()
    use_supervised = bool(False)
    gc_supervised_head = dataset_info.get('gc_supervised_head')
    dataset = dataset_info['dataset']
    sample_graph = dataset[0]
    is_multitask = sample_graph.y.numel() > 1
    if not is_multitask:
        raise ValueError('Vectorized GC path expects a multi-task dataset (e.g., PCBA)')
    num_tasks = sample_graph.y.numel()
    start_time = time.perf_counter()
    total_loss = 0.0
    batch_count = 0
    batch_time_total = 0.0
    proto_time_total = 0.0
    proto_encode_time_total = 0.0
    proto_overhead_time_total = 0.0
    proto_concat_time_total = 0.0
    proto_context_batches_total = 0
    profile_context = False if args is not None else False
    sim_type = getattr(args, 'gc_sim', 'dot') if args is not None else 'dot'
    context_batch_size = getattr(args, 'gc_batch_size', 1024) if args is not None else 1024
    gc_vec_task_chunk_size = int(getattr(args, 'gc_vec_task_chunk_size', 0)) if args is not None else 0
    if gc_vec_task_chunk_size < 0:
        gc_vec_task_chunk_size = 0
    pos_weight = None
    edge_dropout_active = args is not None and getattr(args, 'edge_dropout_enabled', False) and (float(getattr(args, 'edge_dropout_rate', 0.0)) > 0)
    feature_dropout_active = args is not None and getattr(args, 'feature_dropout_enabled', False) and (float(getattr(args, 'feature_dropout_rate', 0.0)) > 0)
    cache_hard_blockers = []
    if edge_dropout_active:
        cache_hard_blockers.append('edge_dropout_enabled')
    if feature_dropout_active:
        cache_hard_blockers.append('feature_dropout_enabled')
    use_static_embedding_cache, gc_cache_mode, gc_cache_notes = _resolve_static_embedding_cache_policy(args, model, identity_projection=None, hard_blockers=cache_hard_blockers)
    supervised_debug_printed = False
    chunking_enabled = not False and (not False) and (gc_vec_task_chunk_size > 0) and (gc_vec_task_chunk_size < num_tasks)
    if chunking_enabled and use_static_embedding_cache:
        chunking_enabled = False
        gc_cache_notes.append('chunked_backward_disabled_with_static_cache')
    if chunking_enabled:
        print(f'[GC-VEC] task-chunked backward enabled: chunk_size={gc_vec_task_chunk_size}, num_tasks={num_tasks}')
    gc_cache_notes = list(dict.fromkeys(gc_cache_notes))
    dataset_name = getattr(dataset, 'name', 'unknown')
    if use_static_embedding_cache:
        print(f'[GC Cache] Enabled for {dataset_name} (mode={gc_cache_mode}): context embeddings cached per epoch.')
        if len(gc_cache_notes) > 0:
            print(f"[GC Cache] Note: {', '.join(gc_cache_notes)}")
    else:
        print(f"[GC Cache] Disabled for {dataset_name} (mode={gc_cache_mode}): {', '.join(gc_cache_notes)}")
    cached_context_embeddings = None
    cached_context_labels = None
    cached_context_masks = None
    cached_w = None
    cached_pos_embeds_by_task = None
    cached_neg_embeds_by_task = None
    if use_static_embedding_cache and (not False):
        proto_start = time.perf_counter()
        cached_pos_embeds_by_task, cached_neg_embeds_by_task = _create_all_task_context_embeddings(model, dataset_info['context_graphs'], dataset, pooling_method, device, None, dataset_info)
        proto_time_total += time.perf_counter() - proto_start
        cached_pos_embeds_by_task = [emb.detach() if emb is not None else None for emb in cached_pos_embeds_by_task]
        cached_neg_embeds_by_task = [emb.detach() if emb is not None else None for emb in cached_neg_embeds_by_task]
    for batch in data_loaders['train']:
        if use_static_embedding_cache:
            pos_embeds_by_task = cached_pos_embeds_by_task
            neg_embeds_by_task = cached_neg_embeds_by_task
        else:
            proto_start = time.perf_counter()
            pos_embeds_by_task, neg_embeds_by_task = _create_all_task_context_embeddings(model, dataset_info['context_graphs'], dataset, pooling_method, device, None, dataset_info)
            proto_time_total += time.perf_counter() - proto_start
        batch_start = time.perf_counter()
        optimizer.zero_grad()
        batch_data = batch.to(device)
        if batch_data.num_graphs == 0:
            continue
        node_emb_table = _get_node_embedding_table(dataset_info['dataset'], 0, device, dataset_info)
        if node_emb_table is not None:
            batch_data.x = _safe_lookup_node_embeddings(node_emb_table, batch_data.x, context='train_vec_multitask', batch_data=batch_data, dataset_info=dataset_info)
        else:
            raise ValueError('Expected node embedding table under unified setting')
        x_input = batch_data.x
        x_input = apply_feature_dropout_if_enabled(x_input, args, rank=0, training=model.training)
        if args is not None and hasattr(args, 'edge_dropout_enabled') and args.edge_dropout_enabled and hasattr(args, 'edge_dropout_rate'):
            batch_data = batch_edge_dropout(batch_data, args.edge_dropout_rate, training=model.training)
        batch_data.adj_t = SparseTensor.from_edge_index(batch_data.edge_index, sparse_sizes=(batch_data.num_nodes, batch_data.num_nodes)).to_symmetric().coalesce()
        node_embeddings = model(x_input, batch_data.adj_t, batch_data.batch)
        target_embeddings = pool_graph_embeddings(node_embeddings, batch_data.batch, pooling_method)
        batch_labels = batch_data.y
        if batch_labels.dim() == 1 and batch_labels.numel() == batch_data.num_graphs * num_tasks:
            batch_labels = batch_labels.view(batch_data.num_graphs, num_tasks)
        if hasattr(batch_data, 'task_mask') and batch_data.task_mask is not None:
            task_mask = batch_data.task_mask
            if task_mask.dim() == 1 and task_mask.numel() == batch_data.num_graphs * num_tasks:
                task_mask = task_mask.view(batch_data.num_graphs, num_tasks)
        elif batch_labels.dtype.is_floating_point:
            task_mask = ~torch.isnan(batch_labels)
        else:
            task_mask = torch.ones_like(batch_labels, dtype=torch.bool)
        task_mask = task_mask.bool()
        labels = batch_labels.float()
        labels = torch.where(task_mask, labels, torch.zeros_like(labels))
        valid = task_mask.float()
        global_valid = valid.sum().clamp_min(1.0)
        loss = torch.tensor(0.0, device=device)
        did_backward = False
        if chunking_enabled:
            predictor_params = [p for p in predictor.parameters() if p.requires_grad]
            grad_target = torch.zeros_like(target_embeddings) if target_embeddings.requires_grad else None
            grad_pos = {}
            grad_neg = {}
            chunk_loss_values = []
            for chunk_start in range(0, num_tasks, gc_vec_task_chunk_size):
                chunk_end = min(chunk_start + gc_vec_task_chunk_size, num_tasks)
                chunk_task_indices = list(range(chunk_start, chunk_end))
                chunk_logits, _ = _vectorized_multitask_pfn_logits(predictor, target_embeddings, pos_embeds_by_task, neg_embeds_by_task, normalize_class_h=normalize_class_h, device=device, task_indices=chunk_task_indices)
                if chunk_logits is None:
                    continue
                chunk_labels = labels[:, chunk_task_indices]
                chunk_valid = valid[:, chunk_task_indices]
                chunk_loss_matrix = F.binary_cross_entropy_with_logits(chunk_logits, chunk_labels, reduction='none')
                chunk_loss = (chunk_loss_matrix * chunk_valid).sum() / global_valid
                chunk_loss_values.append(chunk_loss.detach())
                if not chunk_loss.requires_grad:
                    continue
                grad_inputs = []
                grad_meta = []
                if target_embeddings.requires_grad:
                    grad_inputs.append(target_embeddings)
                    grad_meta.append(('target', None))
                for idx, p in enumerate(predictor_params):
                    grad_inputs.append(p)
                    grad_meta.append(('predictor', idx))
                for task_idx in chunk_task_indices:
                    pos_e = pos_embeds_by_task[task_idx]
                    neg_e = neg_embeds_by_task[task_idx]
                    if pos_e is not None and pos_e.numel() > 0 and pos_e.requires_grad:
                        grad_inputs.append(pos_e)
                        grad_meta.append(('pos', task_idx))
                    if neg_e is not None and neg_e.numel() > 0 and neg_e.requires_grad:
                        grad_inputs.append(neg_e)
                        grad_meta.append(('neg', task_idx))
                if not grad_inputs:
                    continue
                chunk_grads = torch.autograd.grad(chunk_loss, grad_inputs, allow_unused=True, retain_graph=False)
                for (kind, idx), grad in zip(grad_meta, chunk_grads):
                    if grad is None:
                        continue
                    if kind == 'target':
                        grad_target = grad_target + grad
                        did_backward = True
                    elif kind == 'predictor':
                        param = predictor_params[idx]
                        grad_detached = grad.detach()
                        if param.grad is None:
                            param.grad = grad_detached.clone()
                        else:
                            param.grad.add_(grad_detached)
                        did_backward = True
                    elif kind == 'pos':
                        if idx not in grad_pos:
                            grad_pos[idx] = grad
                        else:
                            grad_pos[idx] = grad_pos[idx] + grad
                        did_backward = True
                    elif kind == 'neg':
                        if idx not in grad_neg:
                            grad_neg[idx] = grad
                        else:
                            grad_neg[idx] = grad_neg[idx] + grad
                        did_backward = True
            if chunk_loss_values:
                loss = torch.stack(chunk_loss_values).sum()
            if did_backward:
                back_tensors = []
                back_grads = []
                if grad_target is not None:
                    back_tensors.append(target_embeddings)
                    back_grads.append(grad_target)
                for task_idx, grad in grad_pos.items():
                    pos_e = pos_embeds_by_task[task_idx]
                    if pos_e is not None and pos_e.requires_grad:
                        back_tensors.append(pos_e)
                        back_grads.append(grad)
                for task_idx, grad in grad_neg.items():
                    neg_e = neg_embeds_by_task[task_idx]
                    if neg_e is not None and neg_e.requires_grad:
                        back_tensors.append(neg_e)
                        back_grads.append(grad)
                if back_tensors:
                    torch.autograd.backward(back_tensors, back_grads)
        else:
            logits, _ = _vectorized_multitask_pfn_logits(predictor, target_embeddings, pos_embeds_by_task, neg_embeds_by_task, normalize_class_h=normalize_class_h, device=device)
            if logits is None:
                continue
            loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
            loss = (loss_matrix * valid).sum() / global_valid
            loss.backward()
            did_backward = True
        if not did_backward:
            continue
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
        optimizer.step()
        total_loss += loss.item()
        batch_count += 1
        batch_time_total += time.perf_counter() - batch_start
    total_time = time.perf_counter() - start_time
    avg_batch_time = batch_time_total / max(batch_count, 1)
    avg_proto_time = proto_time_total / max(batch_count, 1)
    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    method_name = 'pfn'
    print(f'[GC-VEC] train ({method_name}): avg_loss={avg_loss:.4f} ctx_avg={avg_proto_time:.2f}s batches={batch_count} avg_batch={avg_batch_time:.3f}s total={total_time:.2f}s')
    return avg_loss

@torch.no_grad()
def evaluate_graph_classification_multitask_vectorized(model, predictor, dataset_info, data_loaders, pooling_method='mean', device='cuda', normalize_class_h=True, dataset_name=None, identity_projection=None, args=None, return_logits=False):
    """
    Vectorized multi-task evaluation: compute logits [B, T] and AUC/AP with task_mask.
    """
    model.eval()
    predictor.eval()
    use_supervised = bool(False)
    gc_supervised_head = dataset_info.get('gc_supervised_head')
    dataset = dataset_info['dataset']
    sample_graph = dataset[0]
    is_multitask = sample_graph.y.numel() > 1
    if not is_multitask:
        raise ValueError('Vectorized GC eval expects a multi-task dataset (e.g., PCBA)')
    num_tasks = sample_graph.y.numel()
    metric_override = getattr(args, 'gc_metric', 'auto') if args is not None else None
    metric_type = get_dataset_metric(dataset_name, num_classes=dataset_info.get('num_classes', None), is_multitask=True, metric_override=metric_override)
    sim_type = getattr(args, 'gc_sim', 'dot') if args is not None else 'dot'
    context_batch_size = getattr(args, 'gc_test_batch_size', 4096) if args is not None else 4096
    max_eval_batches = getattr(args, 'gc_train_eval_max_batches', 0) if args is not None else 0
    eval_start = time.perf_counter()
    proto_start = time.perf_counter()
    pos_embeds_by_task, neg_embeds_by_task = _create_all_task_context_embeddings(model, dataset_info['context_graphs'], dataset, pooling_method, device, None, dataset_info)
    proto_time = time.perf_counter() - proto_start
    split_results = {}
    for split_name in ['train', 'val', 'test']:
        if split_name not in data_loaders:
            continue
        split_start = time.perf_counter()
        split_batch_time = 0.0
        all_logits = []
        all_labels = []
        all_masks = []
        for batch_idx, batch in enumerate(data_loaders[split_name]):
            if split_name == 'train' and max_eval_batches > 0 and (batch_idx >= max_eval_batches):
                break
            batch_start = time.perf_counter()
            batch_data = batch.to(device)
            if batch_data.num_graphs == 0:
                continue
            node_emb_table = _get_node_embedding_table(dataset_info['dataset'], 0, device, dataset_info)
            if node_emb_table is not None:
                batch_data.x = _safe_lookup_node_embeddings(node_emb_table, batch_data.x, context=f'eval_vec_{split_name}', batch_data=batch_data, dataset_info=dataset_info)
            else:
                raise ValueError('Expected node embedding table under unified setting')
            x_input = batch_data.x
            batch_data = _apply_gc_tta_edge_dropout_if_needed(batch_data, dataset_info)
            batch_data.adj_t = SparseTensor.from_edge_index(batch_data.edge_index, sparse_sizes=(batch_data.num_nodes, batch_data.num_nodes)).to_symmetric().coalesce()
            node_embeddings = model(x_input, batch_data.adj_t, batch_data.batch)
            target_embeddings = pool_graph_embeddings(node_embeddings, batch_data.batch, pooling_method)
            batch_labels = batch_data.y
            if batch_labels.dim() == 1 and batch_labels.numel() == batch_data.num_graphs * num_tasks:
                batch_labels = batch_labels.view(batch_data.num_graphs, num_tasks)
            if hasattr(batch_data, 'task_mask') and batch_data.task_mask is not None:
                task_mask = batch_data.task_mask
                if task_mask.dim() == 1 and task_mask.numel() == batch_data.num_graphs * num_tasks:
                    task_mask = task_mask.view(batch_data.num_graphs, num_tasks)
            elif batch_labels.dtype.is_floating_point:
                task_mask = ~torch.isnan(batch_labels)
            else:
                task_mask = torch.ones_like(batch_labels, dtype=torch.bool)
            task_mask = task_mask.bool()
            labels = batch_labels.float()
            labels = torch.where(task_mask, labels, torch.zeros_like(labels))
            logits, _ = _vectorized_multitask_pfn_logits(predictor, target_embeddings, pos_embeds_by_task, neg_embeds_by_task, normalize_class_h=normalize_class_h, device=device)
            if logits is None:
                continue
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_masks.append(task_mask.detach().cpu())
            split_batch_time += time.perf_counter() - batch_start
        if not all_logits:
            if isinstance(metric_type, list):
                split_results[split_name] = {metric_name: 0.0 for metric_name in metric_type}
            else:
                split_results[split_name] = 0.0
            continue
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_masks = torch.cat(all_masks, dim=0)
        print(f'[GC-VEC] {split_name} PFN logits: mean={all_logits.mean().item():.3f} std={all_logits.std().item():.3f} min={all_logits.min().item():.3f} max={all_logits.max().item():.3f}')
        task_metrics = []
        for t in range(num_tasks):
            mask_t = all_masks[:, t].bool()
            if mask_t.sum() == 0:
                continue
            labels_t = all_labels[mask_t, t]
            probs_t = torch.sigmoid(all_logits[mask_t, t]).view(-1)
            preds_t = (probs_t >= 0.5).long()
            if isinstance(metric_type, list):
                metrics_t = calculate_multiple_metrics(preds_t, labels_t.long(), probs_t, metric_type)
                task_metrics.append(metrics_t)
            else:
                metric_t = calculate_metric(preds_t, labels_t.long(), probs_t, metric_type)
                task_metrics.append(metric_t)
        split_results[split_name] = aggregate_task_metrics(task_metrics) if task_metrics else 0.0
        if return_logits:
            split_results[f'{split_name}_logits'] = all_logits
            split_results[f'{split_name}_labels'] = all_labels
            split_results[f'{split_name}_masks'] = all_masks
        split_time = time.perf_counter() - split_start
        avg_batch_time = split_batch_time / max(len(data_loaders[split_name]), 1)
        method_name = 'pfn'
        print(f'[GC-VEC] eval {split_name} ({method_name}): ctx={proto_time:.2f}s split_time={split_time:.2f}s avg_batch={avg_batch_time:.3f}s')
    total_time = time.perf_counter() - eval_start
    print(f'[GC-VEC] eval total: {total_time:.2f}s')
    return split_results

@torch.no_grad()
def evaluate_graph_classification_multitask_vectorized_with_tta(model, predictor, dataset_info, data_loaders, pooling_method='mean', device='cuda', normalize_class_h=True, dataset_name=None, identity_projection=None, args=None, num_augmentations=5, normalize_views=False, use_batchnorm=False, linear_projection=False, feature_projection=True, edge_dropout_rate=0.0):
    """
    Minimal vectorized GC TTA for multitask datasets such as PCBA.
    Uses the same node-embedding-table override as regular GC TTA and averages
    per-graph, per-task logits before metric computation.
    """
    dataset = dataset_info['dataset']
    original_override_present = '_gc_tta_node_embs_override' in dataset_info
    original_override = dataset_info.get('_gc_tta_node_embs_override', None)
    dataset_name = dataset_name or getattr(dataset, 'name', 'unknown')
    num_augmentations = 5 if num_augmentations is None else int(num_augmentations)
    edge_dropout_rate = min(max(float(edge_dropout_rate or 0.0), 0.0), 1.0)
    feature_projection = bool(feature_projection)
    edge_override_keys = ('_gc_tta_edge_dropout_rate', '_gc_tta_edge_dropout_seed', '_gc_tta_edge_dropout_counter')
    original_edge_overrides = {k: dataset_info[k] for k in edge_override_keys if k in dataset_info}

    def _select_source_table():
        if 'feature_mapping' in dataset_info and dataset_info['feature_mapping'] and ('node_embs' in dataset_info['feature_mapping']):
            return dataset_info['feature_mapping']['node_embs']
        if hasattr(dataset, 'node_embs') and dataset.node_embs is not None:
            return dataset.node_embs
        return None
    source_table = _select_source_table()
    if feature_projection and source_table is None:
        print(f'[GC_TTA_VEC] {dataset_name}: no node embedding table found, falling back to vectorized eval')
        return evaluate_graph_classification_multitask_vectorized(model, predictor, dataset_info, data_loaders, pooling_method=pooling_method, device=device, normalize_class_h=normalize_class_h, dataset_name=dataset_name, identity_projection=None, args=args)
    source_table = source_table.detach().cpu() if source_table is not None else None
    target_dim = source_table.size(1) if source_table is not None else None
    view_specs = [{'table': None, 'edge_rate': 0.0, 'seed': 0}]
    for aug_idx in range(num_augmentations):
        seed = 999000 + aug_idx
        table = None
        if feature_projection:
            aug_data = apply_random_projection_augmentation(Data(x=source_table), activation_pool=['identity'] if linear_projection else None, seed=seed, verbose=False, rank=0, max_hidden_dim=target_dim)
            q = min(target_dim, aug_data.x.size(0), aug_data.x.size(1))
            U, S, _ = torch.pca_lowrank(aug_data.x, q=q)
            x_aug = torch.mm(U, torch.diag(S))
            if x_aug.size(1) < target_dim:
                pad = torch.zeros(x_aug.size(0), target_dim - x_aug.size(1), dtype=x_aug.dtype)
                x_aug = torch.cat([x_aug, pad], dim=1)
            if normalize_views:
                if use_batchnorm:
                    x_aug = (x_aug - x_aug.mean(dim=0, keepdim=True)) / (x_aug.std(dim=0, keepdim=True, unbiased=False) + 1e-05)
                else:
                    x_aug = F.normalize(x_aug, p=2, dim=1)
            table = x_aug.detach().cpu()
        view_specs.append({'table': table, 'edge_rate': edge_dropout_rate, 'seed': seed})
    sample_graph = dataset[0]
    num_tasks = sample_graph.y.numel()
    metric_type = get_dataset_metric(dataset_name, num_classes=dataset_info.get('num_classes', None), is_multitask=True, metric_override=getattr(args, 'gc_metric', 'auto') if args is not None else None)
    try:
        view_results = []
        for view_idx, view_spec in enumerate(view_specs):
            table = view_spec['table']
            if table is None:
                dataset_info.pop('_gc_tta_node_embs_override', None)
            else:
                dataset_info['_gc_tta_node_embs_override'] = table
            dataset_info['_gc_tta_edge_dropout_rate'] = view_spec['edge_rate']
            dataset_info['_gc_tta_edge_dropout_seed'] = int(view_spec['seed']) + 7919 * view_idx
            dataset_info['_gc_tta_edge_dropout_counter'] = 0
            dataset_info['_gc_tta_edge_dropout_stats'] = []
            view_results.append(evaluate_graph_classification_multitask_vectorized(model, predictor, dataset_info, data_loaders, pooling_method=pooling_method, device=device, normalize_class_h=normalize_class_h, dataset_name=dataset_name, identity_projection=None, args=args, return_logits=True))
            stats = dataset_info.get('_gc_tta_edge_dropout_stats', [])
            if stats and view_spec['edge_rate'] > 0:
                pairs_total = sum((s[0] for s in stats))
                pairs_kept = sum((s[1] for s in stats))
                edges_total = sum((s[2] for s in stats))
                edges_kept = sum((s[3] for s in stats))
                if pairs_total > 0:
                    view_results[-1]['_edge_pair_keep_frac'] = pairs_kept / pairs_total
                if edges_total > 0:
                    view_results[-1]['_edge_entry_keep_frac'] = edges_kept / edges_total
        results = {}
        for split_name in ['train', 'val', 'test']:
            logits_key = f'{split_name}_logits'
            labels_key = f'{split_name}_labels'
            masks_key = f'{split_name}_masks'
            if logits_key not in view_results[0]:
                if split_name in view_results[0]:
                    results[split_name] = view_results[0][split_name]
                continue
            avg_logits = torch.stack([r[logits_key] for r in view_results], dim=0).mean(dim=0)
            labels = view_results[0][labels_key]
            masks = view_results[0][masks_key]
            task_metrics = []
            for t in range(num_tasks):
                mask_t = masks[:, t].bool()
                if mask_t.sum() == 0:
                    continue
                labels_t = labels[mask_t, t]
                probs_t = torch.sigmoid(avg_logits[mask_t, t]).view(-1)
                preds_t = (probs_t >= 0.5).long()
                if isinstance(metric_type, list):
                    task_metrics.append(calculate_multiple_metrics(preds_t, labels_t.long(), probs_t, metric_type))
                else:
                    task_metrics.append(calculate_metric(preds_t, labels_t.long(), probs_t, metric_type))
            results[split_name] = aggregate_task_metrics(task_metrics) if task_metrics else 0.0
        shown_split = 'test' if 'test' in results else next(iter(results.keys()), None)
        if shown_split is not None:
            view_metrics = [r.get(shown_split, 0.0) for r in view_results]
            display_metrics = [m.get('auc', m.get('ap', next(iter(m.values())))) if isinstance(m, dict) else m for m in view_metrics]
            agg_metric = results[shown_split]
            agg_display = agg_metric.get('auc', agg_metric.get('ap', next(iter(agg_metric.values())))) if isinstance(agg_metric, dict) else agg_metric
            edge_keeps = [r.get('_edge_pair_keep_frac') for r in view_results[1:] if r.get('_edge_pair_keep_frac') is not None]
            edge_keep_text = f' edge_pair_keep={sum(edge_keeps) / len(edge_keeps):.3f}' if edge_keeps else ''
            print(f'[GC_TTA_VEC] {dataset_name} split={shown_split}: original={display_metrics[0]:.4f} mean_view={sum(display_metrics) / len(display_metrics):.4f} agg={agg_display:.4f} views={len(view_results)} norm_views={normalize_views} linear_proj={linear_projection} feature_proj={feature_projection} edge_drop={edge_dropout_rate:.3f}{edge_keep_text}')
        return results
    finally:
        if original_override_present:
            dataset_info['_gc_tta_node_embs_override'] = original_override
        else:
            dataset_info.pop('_gc_tta_node_embs_override', None)
        for key in edge_override_keys:
            if key in original_edge_overrides:
                dataset_info[key] = original_edge_overrides[key]
            else:
                dataset_info.pop(key, None)
        dataset_info.pop('_gc_tta_edge_dropout_stats', None)

def prepare_pfn_data_structure(context_embeddings, context_labels, num_classes, device='cuda'):

    class GraphDataForPFN:

        def __init__(self, context_embeddings, context_labels, num_classes):
            self.context_sample = torch.arange(len(context_embeddings), device=device)
            self.y = context_labels
            self.num_classes = num_classes
            self.name = 'graph_classification_data'
    return GraphDataForPFN(context_embeddings, context_labels, num_classes)

def train_graph_classification_single_task(model, predictor, dataset_info, data_loaders, optimizer, task_idx, pooling_method='mean', device='cuda', clip_grad=1.0, orthogonal_push=0.0, normalize_class_h=True, identity_projection=None, context_k=None, args=None, lambda_=1.0):
    model.train()
    predictor.train()
    total_loss = 0
    num_batches = 0
    train_batches = list(data_loaders['train'])
    edge_dropout_active = args is not None and getattr(args, 'edge_dropout_enabled', False) and (float(getattr(args, 'edge_dropout_rate', 0.0)) > 0)
    feature_dropout_active = args is not None and getattr(args, 'feature_dropout_enabled', False) and (float(getattr(args, 'feature_dropout_rate', 0.0)) > 0)
    context_refresh_active = args is not None and int(getattr(args, 'context_batch_refresh_interval', 0)) > 0
    cache_hard_blockers = []
    if context_refresh_active:
        cache_hard_blockers.append('context_batch_refresh_enabled')
    if edge_dropout_active:
        cache_hard_blockers.append('edge_dropout_enabled')
    if feature_dropout_active:
        cache_hard_blockers.append('feature_dropout_enabled')
    use_static_embedding_cache, gc_cache_mode, gc_cache_notes = _resolve_static_embedding_cache_policy(args, model, identity_projection=None, hard_blockers=cache_hard_blockers)
    gc_cache_notes = list(dict.fromkeys(gc_cache_notes))
    dataset_name = getattr(dataset_info.get('dataset', None), 'name', 'unknown')
    if use_static_embedding_cache:
        print(f'[GC Cache] Enabled for {dataset_name}/task{task_idx} (mode={gc_cache_mode}): one context embedding pass per epoch.')
        if len(gc_cache_notes) > 0:
            print(f"[GC Cache] Note: {', '.join(gc_cache_notes)}")
    else:
        print(f"[GC Cache] Disabled for {dataset_name}/task{task_idx} (mode={gc_cache_mode}): {', '.join(gc_cache_notes)}")
    cached_context_embeddings = None
    cached_context_labels = None
    cached_pfn_data = None
    cached_class_h = None
    if use_static_embedding_cache:
        context_embeddings, context_labels = _create_context_embeddings_computed(model, dataset_info['context_graphs'], dataset_info['dataset'], task_idx, pooling_method, device, None, dataset_info)
        cached_context_embeddings = context_embeddings.detach()
        cached_context_labels = context_labels.detach()
        cached_pfn_data = prepare_pfn_data_structure(cached_context_embeddings, cached_context_labels, dataset_info['num_classes'], device)
        cached_class_h = process_node_features(cached_context_embeddings, cached_pfn_data, degree_normalize=False, attention_pool_module=None, mlp_module=None, normalize=normalize_class_h).detach()
    total_batches = len(train_batches)
    for batch_idx, batch_graphs in enumerate(train_batches):
        if args is not None:
            refresh_gc_context_if_needed(dataset_info, batch_idx, 0, args, device)
        print(f'\rTraining batch {batch_idx + 1}/{total_batches}', end='', flush=True)
        batch_data = batch_graphs.to(device)
        node_emb_table = _get_node_embedding_table(dataset_info['dataset'], task_idx, device, dataset_info)
        if node_emb_table is not None:
            batch_data.x = _safe_lookup_node_embeddings(node_emb_table, batch_data.x, context=f'train_task{task_idx}', batch_data=batch_data, dataset_info=dataset_info)
        else:
            raise ValueError('Expected node embedding table under unified setting')
        needs_proj = dataset_info.get('needs_identity_projection', False)
        has_proj = None is not None
        x_input = batch_data.x
        x_input = apply_feature_dropout_if_enabled(x_input, args, rank=0, training=model.training)
        if args is not None and hasattr(args, 'edge_dropout_enabled') and args.edge_dropout_enabled and hasattr(args, 'edge_dropout_rate'):
            batch_data = batch_edge_dropout(batch_data, args.edge_dropout_rate, training=model.training)
        batch_data.adj_t = SparseTensor.from_edge_index(batch_data.edge_index, sparse_sizes=(batch_data.num_nodes, batch_data.num_nodes)).to_symmetric().coalesce()
        node_embeddings = model(x_input, batch_data.adj_t, batch_data.batch)
        target_embeddings = pool_graph_embeddings(node_embeddings, batch_data.batch, pooling_method)
        batch_labels = batch_data.y
        batch_size = target_embeddings.size(0)
        if batch_size == 1:
            print(f'Warning: Skipping training batch with only 1 sample to avoid PFN predictor dimension issues')
            continue
        sample_graph = dataset_info['dataset'][0]
        is_multitask = sample_graph.y.numel() > 1
        if is_multitask:
            num_tasks = sample_graph.y.numel()
            if batch_labels.dim() == 1 and len(batch_labels) == batch_size * num_tasks:
                batch_labels = batch_labels.view(batch_size, num_tasks)
            batch_labels = batch_labels[:, task_idx]
        elif batch_labels.dim() > 1:
            batch_labels = batch_labels.squeeze()
        if batch_labels.dtype != torch.long:
            batch_labels = batch_labels.to(torch.long)
        if use_static_embedding_cache:
            context_embeddings = cached_context_embeddings
            context_labels = cached_context_labels
            pfn_data = cached_pfn_data
            class_h = cached_class_h
        else:
            context_embeddings, context_labels = _create_context_embeddings_computed(model, dataset_info['context_graphs'], dataset_info['dataset'], task_idx, pooling_method, device, None, dataset_info)
            pfn_data = prepare_pfn_data_structure(context_embeddings, context_labels, dataset_info['num_classes'], device)
            class_h = process_node_features(context_embeddings, pfn_data, degree_normalize=False, attention_pool_module=None, mlp_module=None, normalize=normalize_class_h)
        scores_raw, refined_class_h = predictor(pfn_data, context_embeddings, target_embeddings, context_labels, class_h, task_type='graph_classification')
        scores_log = F.log_softmax(scores_raw, dim=1)
        nll_loss = F.nll_loss(scores_log, batch_labels)
        if orthogonal_push > 0:
            refined_class_h_norm = F.normalize(refined_class_h, p=2, dim=1)
            class_matrix = refined_class_h_norm @ refined_class_h_norm.T
            mask = ~torch.eye(class_matrix.size(0), device=class_matrix.device, dtype=torch.bool)
            orthogonal_loss = torch.sum(class_matrix[mask] ** 2)
        else:
            orthogonal_loss = torch.tensor(0.0, device=device)
        loss = nll_loss + orthogonal_push * orthogonal_loss
        loss = loss * lambda_
        optimizer.zero_grad()
        loss.backward()
        total_grad_norm_before = 0.0
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f'\nTraining completed: {num_batches} batches, avg loss: {avg_loss:.4f}')
    return avg_loss

@torch.no_grad()
def evaluate_graph_classification_single_task(model, predictor, dataset_info, data_loaders, task_idx, pooling_method='mean', device='cuda', normalize_class_h=True, dataset_name=None, identity_projection=None, context_k=None, args=None, return_logits=False):
    model.eval()
    predictor.eval()
    results = {}
    total_start = time.perf_counter()
    context_start = time.perf_counter()
    context_embeddings, context_labels = _create_context_embeddings_computed(model, dataset_info['context_graphs'], dataset_info['dataset'], task_idx, pooling_method, device, None, dataset_info)
    context_time = time.perf_counter() - context_start
    pfn_prep_start = time.perf_counter()
    pfn_data = prepare_pfn_data_structure(context_embeddings, context_labels, dataset_info['num_classes'], device)
    class_h = process_node_features(context_embeddings, pfn_data, degree_normalize=False, attention_pool_module=None, mlp_module=None, normalize=normalize_class_h)
    pfn_prep_time = time.perf_counter() - pfn_prep_start
    sample_graph = dataset_info['dataset'][0]
    is_multitask = sample_graph.y.numel() > 1
    num_classes = dataset_info.get('num_classes', None)
    metric_override = getattr(args, 'gc_metric', 'auto') if args is not None else None
    metric_type = get_dataset_metric(dataset_name, num_classes=num_classes, is_multitask=is_multitask, metric_override=metric_override) if dataset_name else 'accuracy'
    use_graph_cs = False if args is not None else False
    all_test_embeddings = []
    all_test_logits = []
    all_test_labels = []
    split_sizes = {}
    split_times = {}
    for split_name, data_loader in data_loaders.items():
        split_start = time.perf_counter()
        if len(data_loader.dataset) == 0:
            results[split_name] = 0.0
            split_times[split_name] = 0.0
            continue
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_raw_logits = []
        split_embeddings = []
        split_logits = []
        batch_processing_start = time.perf_counter()
        total_samples = len(data_loader.dataset)
        max_eval_batches = getattr(args, 'gc_train_eval_max_batches', 0) if args is not None else 0
        for batch_idx, batch_graphs in enumerate(data_loader):
            if split_name == 'train' and max_eval_batches > 0 and (batch_idx >= max_eval_batches):
                break
            print(f'\rEvaluating batch {batch_idx + 1}...', end='', flush=True)
            batch_data = batch_graphs.to(device)
            node_emb_table = _get_node_embedding_table(dataset_info['dataset'], task_idx, device, dataset_info)
            if node_emb_table is not None:
                batch_data.x = _safe_lookup_node_embeddings(node_emb_table, batch_data.x, context=f'eval_task{task_idx}_batch{batch_idx}', batch_data=batch_data, dataset_info=dataset_info)
            else:
                raise ValueError('Expected node embedding table under unified setting')
            x_input = batch_data.x
            batch_data = _apply_gc_tta_edge_dropout_if_needed(batch_data, dataset_info)
            batch_data.adj_t = SparseTensor.from_edge_index(batch_data.edge_index, sparse_sizes=(batch_data.num_nodes, batch_data.num_nodes)).to_symmetric().coalesce()
            node_embeddings = model(x_input, batch_data.adj_t, batch_data.batch)
            target_embeddings = pool_graph_embeddings(node_embeddings, batch_data.batch, pooling_method)
            batch_labels = batch_data.y
            batch_size = target_embeddings.size(0)
            if batch_size == 1:
                print(f'Warning: Skipping batch with only 1 sample to avoid PFN predictor dimension issues')
                continue
            sample_graph = dataset_info['dataset'][0]
            is_multitask = sample_graph.y.numel() > 1
            if is_multitask:
                num_tasks = sample_graph.y.numel()
                if batch_labels.dim() == 1 and len(batch_labels) == batch_size * num_tasks:
                    batch_labels = batch_labels.view(batch_size, num_tasks)
                batch_labels = batch_labels[:, task_idx]
            elif batch_labels.dim() > 1:
                batch_labels = batch_labels.squeeze()
            if batch_labels.dtype != torch.long:
                batch_labels = batch_labels.to(torch.long)
            pred_output = predictor(pfn_data, context_embeddings, target_embeddings, context_labels, class_h, task_type='graph_classification')
            if len(pred_output) == 3:
                scores, _, _ = pred_output
            else:
                scores, _ = pred_output
            probabilities = F.softmax(scores, dim=1)
            predictions = scores.argmax(dim=1)
            all_predictions.append(predictions.cpu())
            all_labels.append(batch_labels.cpu())
            all_probabilities.append(probabilities.cpu())
            if return_logits:
                all_raw_logits.append(scores.detach().cpu())
            print(f'\rEvaluating batch {batch_idx + 1} completed ({len(predictions)} samples)', end='', flush=True)
        if all_predictions:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_probabilities = torch.cat(all_probabilities, dim=0)
            print(f'\n{split_name} evaluation: {len(all_predictions)} total samples processed', end='')
            if isinstance(metric_type, list):
                metric_values = calculate_multiple_metrics(all_predictions, all_labels, all_probabilities, metric_type)
                results[split_name] = metric_values
            else:
                metric_value = calculate_metric(all_predictions, all_labels, all_probabilities, metric_type)
                results[split_name] = metric_value
            if return_logits:
                results[f'{split_name}_logits'] = torch.cat(all_raw_logits, dim=0)
                results[f'{split_name}_labels'] = all_labels
        elif isinstance(metric_type, list):
            results[split_name] = {metric_name: 0.0 for metric_name in metric_type}
        else:
            results[split_name] = 0.0
        batch_processing_time = time.perf_counter() - batch_processing_start
        split_times[split_name] = time.perf_counter() - split_start
        print(f' | processing: {batch_processing_time:.3f}s')
    total_time = time.perf_counter() - total_start
    print(f'Task {task_idx} timing: context: {context_time:.3f}s, pfn_prep: {pfn_prep_time:.3f}s, total: {total_time:.3f}s')
    if split_times:
        split_breakdown = ', '.join([f'{split}: {t:.3f}s' for split, t in split_times.items()])
        print(f'  Split breakdown: {split_breakdown}')
    return results

@torch.no_grad()
def evaluate_graph_classification_single_task_with_tta(model, predictor, dataset_info, data_loaders, task_idx, pooling_method='mean', device='cuda', normalize_class_h=True, dataset_name=None, identity_projection=None, context_k=None, args=None, num_augmentations=5, normalize_views=False, use_batchnorm=False, linear_projection=False, feature_projection=True, edge_dropout_rate=0.0):
    """
    Minimal regular-path GC TTA for a single task.
    It overrides the dataset node-embedding table per view so context and target
    graphs are evaluated in the same augmented feature space, then averages logits.
    """
    dataset = dataset_info['dataset']
    original_override_present = '_gc_tta_node_embs_override' in dataset_info
    original_override = dataset_info.get('_gc_tta_node_embs_override', None)
    dataset_name = dataset_name or getattr(dataset, 'name', 'unknown')
    num_augmentations = 5 if num_augmentations is None else int(num_augmentations)
    edge_dropout_rate = min(max(float(edge_dropout_rate or 0.0), 0.0), 1.0)
    feature_projection = bool(feature_projection)
    edge_override_keys = ('_gc_tta_edge_dropout_rate', '_gc_tta_edge_dropout_seed', '_gc_tta_edge_dropout_counter')
    original_edge_overrides = {k: dataset_info[k] for k in edge_override_keys if k in dataset_info}

    def _select_source_table():
        if 'feature_mapping' in dataset_info and dataset_info['feature_mapping'] and ('node_embs' in dataset_info['feature_mapping']):
            return dataset_info['feature_mapping']['node_embs']
        if hasattr(dataset, 'node_embs') and dataset.node_embs is not None:
            return dataset.node_embs
        return None
    source_table = _select_source_table()
    if feature_projection and source_table is None:
        print(f'[GC_TTA] {dataset_name}: no node embedding table found, falling back to regular eval')
        return evaluate_graph_classification_single_task(model, predictor, dataset_info, data_loaders, task_idx, pooling_method=pooling_method, device=device, normalize_class_h=normalize_class_h, dataset_name=dataset_name, identity_projection=None, context_k=context_k, args=args)
    source_table = source_table.detach().cpu() if source_table is not None else None
    target_dim = source_table.size(1) if source_table is not None else None
    view_specs = [{'table': None, 'edge_rate': 0.0, 'seed': 0}]
    for aug_idx in range(num_augmentations):
        seed = 999000 + task_idx * 1000 + aug_idx
        table = None
        if feature_projection:
            aug_data = apply_random_projection_augmentation(Data(x=source_table), activation_pool=['identity'] if linear_projection else None, seed=seed, verbose=False, rank=0, max_hidden_dim=target_dim)
            q = min(target_dim, aug_data.x.size(0), aug_data.x.size(1))
            U, S, _ = torch.pca_lowrank(aug_data.x, q=q)
            x_aug = torch.mm(U, torch.diag(S))
            if x_aug.size(1) < target_dim:
                pad = torch.zeros(x_aug.size(0), target_dim - x_aug.size(1), dtype=x_aug.dtype)
                x_aug = torch.cat([x_aug, pad], dim=1)
            if normalize_views:
                if use_batchnorm:
                    x_aug = (x_aug - x_aug.mean(dim=0, keepdim=True)) / (x_aug.std(dim=0, keepdim=True, unbiased=False) + 1e-05)
                else:
                    x_aug = F.normalize(x_aug, p=2, dim=1)
            table = x_aug.detach().cpu()
        view_specs.append({'table': table, 'edge_rate': edge_dropout_rate, 'seed': seed})
    sample_graph = dataset[0]
    metric_type = get_dataset_metric(dataset_name, num_classes=dataset_info.get('num_classes', None), is_multitask=sample_graph.y.numel() > 1, metric_override=getattr(args, 'gc_metric', 'auto') if args is not None else None)
    try:
        view_results = []
        for view_idx, view_spec in enumerate(view_specs):
            table = view_spec['table']
            if table is None:
                dataset_info.pop('_gc_tta_node_embs_override', None)
            else:
                dataset_info['_gc_tta_node_embs_override'] = table
            dataset_info['_gc_tta_edge_dropout_rate'] = view_spec['edge_rate']
            dataset_info['_gc_tta_edge_dropout_seed'] = int(view_spec['seed']) + 7919 * view_idx
            dataset_info['_gc_tta_edge_dropout_counter'] = 0
            dataset_info['_gc_tta_edge_dropout_stats'] = []
            view_results.append(evaluate_graph_classification_single_task(model, predictor, dataset_info, data_loaders, task_idx, pooling_method=pooling_method, device=device, normalize_class_h=normalize_class_h, dataset_name=dataset_name, identity_projection=None, context_k=context_k, args=args, return_logits=True))
            stats = dataset_info.get('_gc_tta_edge_dropout_stats', [])
            if stats and view_spec['edge_rate'] > 0:
                pairs_total = sum((s[0] for s in stats))
                pairs_kept = sum((s[1] for s in stats))
                edges_total = sum((s[2] for s in stats))
                edges_kept = sum((s[3] for s in stats))
                if pairs_total > 0:
                    view_results[-1]['_edge_pair_keep_frac'] = pairs_kept / pairs_total
                if edges_total > 0:
                    view_results[-1]['_edge_entry_keep_frac'] = edges_kept / edges_total
        results = {}
        for split_name in data_loaders.keys():
            logits_key = f'{split_name}_logits'
            labels_key = f'{split_name}_labels'
            if logits_key not in view_results[0] or labels_key not in view_results[0]:
                results[split_name] = view_results[0].get(split_name, 0.0)
                continue
            avg_logits = torch.stack([r[logits_key] for r in view_results], dim=0).mean(dim=0)
            labels = view_results[0][labels_key]
            probabilities = F.softmax(avg_logits, dim=1)
            predictions = avg_logits.argmax(dim=1)
            if isinstance(metric_type, list):
                results[split_name] = calculate_multiple_metrics(predictions, labels, probabilities, metric_type)
            else:
                results[split_name] = calculate_metric(predictions, labels, probabilities, metric_type)
        shown_split = 'test' if 'test' in results else next(iter(results.keys()), None)
        if shown_split is not None:
            view_metrics = [r.get(shown_split, 0.0) for r in view_results]
            display_metrics = [m.get('auc', m.get('ap', next(iter(m.values())))) if isinstance(m, dict) else m for m in view_metrics]
            agg_metric = results[shown_split]
            agg_display = agg_metric.get('auc', agg_metric.get('ap', next(iter(agg_metric.values())))) if isinstance(agg_metric, dict) else agg_metric
            edge_keeps = [r.get('_edge_pair_keep_frac') for r in view_results[1:] if r.get('_edge_pair_keep_frac') is not None]
            edge_keep_text = f' edge_pair_keep={sum(edge_keeps) / len(edge_keeps):.3f}' if edge_keeps else ''
            print(f'[GC_TTA] {dataset_name} task={task_idx} split={shown_split}: original={display_metrics[0]:.4f} mean_view={sum(display_metrics) / len(display_metrics):.4f} agg={agg_display:.4f} views={len(view_results)} norm_views={normalize_views} linear_proj={linear_projection} feature_proj={feature_projection} edge_drop={edge_dropout_rate:.3f}{edge_keep_text}')
        return results
    finally:
        if original_override_present:
            dataset_info['_gc_tta_node_embs_override'] = original_override
        else:
            dataset_info.pop('_gc_tta_node_embs_override', None)
        for key in edge_override_keys:
            if key in original_edge_overrides:
                dataset_info[key] = original_edge_overrides[key]
            else:
                dataset_info.pop(key, None)
        dataset_info.pop('_gc_tta_edge_dropout_stats', None)
