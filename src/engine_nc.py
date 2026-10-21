import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import psutil
import os
from .utils import process_node_features, acc
from .data_utils import select_k_shot_context, edge_dropout_sparse_tensor, feature_dropout

def _sync_cuda_if_needed(device):
    if isinstance(device, torch.device) and device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize(device)

def _synced_perf_counter(device):
    _sync_cuda_if_needed(device)
    return time.perf_counter()

def _apply_tta_view_input_normalization(features, args):
    """
    Optionally mirror the main NC path's final input-normalization step for
    augmented TTA views after PCA/padding.

    This intentionally mirrors only the final `normalize_data/use_batchnorm`
    branch from `process_data()`. It does not replay the full preprocessing
    stack such as quantile normalization or whitening.
    """
    if args is None or not getattr(args, 'tta_apply_input_normalization_to_views', False):
        return features
    if not getattr(args, 'normalize_data', False):
        return features
    if getattr(args, 'use_batchnorm', False):
        batch_mean = features.mean(dim=0, keepdim=True)
        batch_std = features.std(dim=0, keepdim=True, unbiased=False)
        return (features - batch_mean) / (batch_std + 1e-05)
    return F.normalize(features, p=2, dim=1)

def _tta_true_class_probabilities(probs, y_true):
    idx = torch.arange(y_true.size(0), device=probs.device)
    return probs[idx, y_true]

def _tta_true_class_margins(probs, y_true):
    idx = torch.arange(y_true.size(0), device=probs.device)
    true_probs = probs[idx, y_true]
    masked = probs.clone()
    masked[idx, y_true] = float('-inf')
    max_other = masked.max(dim=-1).values
    return true_probs - max_other

def _tta_true_class_ranks(probs, y_true):
    true_probs = _tta_true_class_probabilities(probs, y_true)
    return (probs > true_probs.unsqueeze(-1)).sum(dim=-1) + 1

def _tta_subset_mean(values, mask):
    if mask.numel() == 0 or mask.sum().item() == 0:
        return None
    return values[mask].mean().item()

def _tta_subset_int_mean(values, mask):
    if mask.numel() == 0 or mask.sum().item() == 0:
        return None
    return values[mask].float().mean().item()

def _print_tta_distribution_debug(dataset_name, split_name, base_logits, all_view_logits, agg_logits, y_true):
    """
    Print a compact summary explaining how TTA aggregation changes raw class
    probability behavior for a given split.
    """
    if y_true.numel() == 0:
        return
    base_probs = F.softmax(base_logits, dim=-1)
    view_probs = torch.stack([F.softmax(logits, dim=-1) for logits in all_view_logits], dim=0)
    agg_probs = F.softmax(agg_logits, dim=-1)
    base_pred = base_probs.argmax(dim=-1)
    agg_pred = agg_probs.argmax(dim=-1)
    view_pred = view_probs.argmax(dim=-1)
    majority_pred = torch.mode(view_pred, dim=0).values
    corrected = (base_pred != y_true) & (agg_pred == y_true)
    broken = (base_pred == y_true) & (agg_pred != y_true)
    soft_only_corrected = corrected & (majority_pred != y_true)
    base_true_prob = _tta_true_class_probabilities(base_probs, y_true)
    mean_view_true_prob = view_probs.gather(2, y_true.view(1, -1, 1).expand(view_probs.size(0), -1, 1)).squeeze(-1).mean(dim=0)
    agg_true_prob = _tta_true_class_probabilities(agg_probs, y_true)
    base_margin = _tta_true_class_margins(base_probs, y_true)
    mean_view_margin = torch.stack([_tta_true_class_margins(view_probs[i], y_true) for i in range(view_probs.size(0))], dim=0).mean(dim=0)
    agg_margin = _tta_true_class_margins(agg_probs, y_true)
    base_rank = _tta_true_class_ranks(base_probs, y_true)
    agg_rank = _tta_true_class_ranks(agg_probs, y_true)
    vote_true_frac = (view_pred == y_true.unsqueeze(0)).float().mean(dim=0)
    unique_top1 = torch.tensor([view_pred[:, i].unique().numel() for i in range(view_pred.size(1))], device=view_pred.device, dtype=torch.float32)
    base_acc = (base_pred == y_true).float().mean().item()
    majority_acc = (majority_pred == y_true).float().mean().item()
    agg_acc = (agg_pred == y_true).float().mean().item()

    def _fmt(v):
        return 'n/a' if v is None else f'{v:.3f}'
    print(f'      [TTA_DIST:{split_name}] base={base_acc:.4f} vote={majority_acc:.4f} agg={agg_acc:.4f} corrected={int(corrected.sum().item())} broken={int(broken.sum().item())} soft_only={int(soft_only_corrected.sum().item())}')
    print(f'        p_true mean base/view/agg = {base_true_prob.mean().item():.3f}/{mean_view_true_prob.mean().item():.3f}/{agg_true_prob.mean().item():.3f}')
    print(f'        margin mean base/view/agg = {base_margin.mean().item():.3f}/{mean_view_margin.mean().item():.3f}/{agg_margin.mean().item():.3f}')
    print(f'        corrected: p_true base/view/agg={_fmt(_tta_subset_mean(base_true_prob, corrected))}/{_fmt(_tta_subset_mean(mean_view_true_prob, corrected))}/{_fmt(_tta_subset_mean(agg_true_prob, corrected))} rank base->agg={_fmt(_tta_subset_int_mean(base_rank, corrected))}->{_fmt(_tta_subset_int_mean(agg_rank, corrected))} vote_true_frac={_fmt(_tta_subset_mean(vote_true_frac, corrected))} unique_top1={_fmt(_tta_subset_mean(unique_top1, corrected))}')
    print(f'        broken:    p_true base/view/agg={_fmt(_tta_subset_mean(base_true_prob, broken))}/{_fmt(_tta_subset_mean(mean_view_true_prob, broken))}/{_fmt(_tta_subset_mean(agg_true_prob, broken))} rank base->agg={_fmt(_tta_subset_int_mean(base_rank, broken))}->{_fmt(_tta_subset_int_mean(agg_rank, broken))} vote_true_frac={_fmt(_tta_subset_mean(vote_true_frac, broken))} unique_top1={_fmt(_tta_subset_mean(unique_top1, broken))}')

def correct_and_smooth(adj, base_logits, context_idx, context_labels, num_classes, num_iters=50, alpha=0.5):
    device = adj.device()
    num_nodes = base_logits.size(0)
    deg = adj.sum(dim=1).to_dense()
    deg_with_selfloop = deg + 1
    deg_inv_sqrt = (deg_with_selfloop + 1e-09).pow(-0.5)
    Y = F.softmax(base_logits, dim=-1)
    Y_support = F.one_hot(context_labels.long(), num_classes=num_classes).float()
    for _ in range(num_iters):
        Y_new = deg_inv_sqrt.view(-1, 1) * Y
        Y_new = adj @ Y_new + Y_new
        Y_new = deg_inv_sqrt.view(-1, 1) * Y_new
        Y = (1 - alpha) * Y_new + alpha * Y
        Y[context_idx] = Y_support
    return Y

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

def refresh_dataset_context_if_needed(data, split_idx, batch_idx, epoch, args):
    if getattr(args, 'context_batch_refresh_interval', 0) <= 0:
        return
    if batch_idx > 0 and batch_idx % args.context_batch_refresh_interval == 0:
        refresh_seed = args.seed + epoch * 10000 + batch_idx
        torch.manual_seed(refresh_seed)
        if hasattr(data, 'context_sample') and data.context_sample is not None:
            current_context_size = len(data.context_sample) // len(data.y.unique())
            new_context_sample = select_k_shot_context(data, current_context_size, split_idx['train'])
            data.context_sample = new_context_sample.to(data.context_sample.device)
            print(f'🔄 Dataset {data.name} context refreshed at batch {batch_idx} ({len(new_context_sample)} samples)')

def train(model, data, train_idx, optimizer, pred, batch_size, degree=False, att=None, mlp=None, orthogonal_push=0.0, normalize_class_h=False, clip_grad=1.0, projector=None, rank=0, epoch=0, identity_projection=None, lambda_=1.0, args=None, external_embeddings=None):
    model.train()
    pred.train()
    device = next(pred.parameters()).device
    data = data.to(device)
    train_idx = train_idx.to(device)
    loader = DataLoader(range(train_idx.numel()), batch_size=batch_size, shuffle=True)
    if len(loader) == 0:
        raise ValueError(f'{data.name} has no training nodes')
    losses = []
    for batch_idx, indices in enumerate(loader):
        if args is not None:
            refresh_dataset_context_if_needed(data, {'train': train_idx}, batch_idx, epoch, args)
        query = train_idx[indices.to(device)]
        adj = data.adj_t
        if args is not None and args.edge_dropout_enabled and (args.edge_dropout_rate > 0):
            adj = edge_dropout_sparse_tensor(adj, args.edge_dropout_rate, training=True)
        features = apply_feature_dropout_if_enabled(data.x, args, rank=0, training=True)
        h = model(features, adj)
        support = h[data.context_sample]
        prototypes = process_node_features(support, data, normalize=normalize_class_h)
        logits, _ = pred(data, support, h[query], data.y[data.context_sample], prototypes)
        loss = F.nll_loss(F.log_softmax(logits, dim=1), data.y[query].reshape(-1))
        losses.append(loss.item())
        optimizer.zero_grad()
        (loss * lambda_).backward()
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            nn.utils.clip_grad_norm_(pred.parameters(), clip_grad)
        optimizer.step()
    mean = sum(losses) / len(losses)
    return {'total': mean, 'nll': mean, 'de': 0.0}

def train_all(model, data_list, split_idx_list, optimizer, pred, batch_size, degree=False, att=None, mlp=None, orthogonal_push=0.0, normalize_class_h=False, clip_grad=1.0, projector=None, rank=0, epoch=0, identity_projection=None, lambda_=1.0, args=None, external_embeddings_list=None):
    values = []
    for data, split in zip(data_list, split_idx_list):
        original_device = data.x.device
        try:
            value = train(model, data, split['train'], optimizer, pred, batch_size, normalize_class_h=normalize_class_h, clip_grad=clip_grad, epoch=epoch, lambda_=lambda_, args=args)['total']
            values.append(value)
            print(f'NC {data.name}: loss={value:.6f}')
        finally:
            data.to(original_device)
    if not values:
        raise ValueError('No NC source datasets were loaded')
    mean = sum(values) / len(values)
    return {'total': mean, 'nll': mean, 'de': 0.0}

@torch.no_grad()
def test(model, predictor, data, train_idx, valid_idx, test_idx, batch_size, degree=False, att=None, mlp=None, normalize_class_h=False, projector=None, rank=0, identity_projection=None, external_embeddings=None, args=None, use_cs=False, cs_num_iters=50, cs_alpha=0.5, epoch=0):
    st = time.time()
    model.eval()
    predictor.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        try:
            device = next(predictor.parameters()).device
        except StopIteration:
            device = data.x.device
    profile_active_compute = bool(False)
    projection_sec = 0.0
    gnn_sec = 0.0
    context_sec = 0.0
    predictor_sec = 0.0
    cs_sec = 0.0

    def _stage_start():
        if not profile_active_compute:
            return None
        return _synced_perf_counter(device)

    def _stage_stop(stage_start):
        if stage_start is None:
            return 0.0
        return _synced_perf_counter(device) - stage_start
    moved_to_device = False
    if data.x.device != device:
        data = data.to(device)
        train_idx = train_idx.to(device)
        valid_idx = valid_idx.to(device)
        test_idx = test_idx.to(device)
        if hasattr(data, 'context_sample') and data.context_sample is not None:
            data.context_sample = data.context_sample.to(device)
        moved_to_device = True
    active_compute_start = _stage_start()
    base_features = data.x
    projection_start = _stage_start()
    x_input = base_features
    projection_sec = _stage_stop(projection_start)
    gnn_start = _stage_start()
    h = model(x_input, data.adj_t)
    gnn_sec = _stage_stop(gnn_start)
    context_h = h[data.context_sample]
    context_y = data.y[data.context_sample]
    context_start = _stage_start()
    class_h = process_node_features(context_h, data, degree_normalize=degree, attention_pool_module=att, mlp_module=mlp, normalize=normalize_class_h)
    context_sec = _stage_stop(context_start)
    train_loader = DataLoader(range(train_idx.size(0)), batch_size, shuffle=False)
    valid_loader = DataLoader(range(valid_idx.size(0)), batch_size, shuffle=False)
    test_loader = DataLoader(range(test_idx.size(0)), batch_size, shuffle=False)
    predictor_start = _stage_start()
    valid_logits = []
    for idx in valid_loader:
        target_h = h[valid_idx[idx]]
        pred_output = predictor(data, context_h, target_h, context_y, class_h)
        if len(pred_output) == 3:
            out, _, _ = pred_output
        else:
            out, _ = pred_output
        valid_logits.append(out)
    valid_logits = torch.cat(valid_logits, dim=0)
    valid_score_base = valid_logits.argmax(dim=1)
    train_logits = []
    for idx in train_loader:
        target_h = h[train_idx[idx]]
        pred_output = predictor(data, context_h, target_h, context_y, class_h)
        if len(pred_output) == 3:
            out, _, _ = pred_output
        else:
            out, _ = pred_output
        train_logits.append(out)
    train_logits = torch.cat(train_logits, dim=0)
    train_score_base = train_logits.argmax(dim=1)
    test_logits = []
    for idx in test_loader:
        target_h = h[test_idx[idx]]
        pred_output = predictor(data, context_h, target_h, context_y, class_h)
        if len(pred_output) == 3:
            out, _, _ = pred_output
        else:
            out, _ = pred_output
        test_logits.append(out)
    test_logits = torch.cat(test_logits, dim=0)
    test_score_base = test_logits.argmax(dim=1)
    predictor_sec = _stage_stop(predictor_start)
    if use_cs:
        cs_start = _stage_start()
        all_logits = torch.zeros(data.num_nodes, train_logits.size(1), device=h.device)
        all_logits[train_idx] = train_logits
        all_logits[valid_idx] = valid_logits
        all_logits[test_idx] = test_logits
        num_classes = context_y.max().item() + 1
        cs_predictions = correct_and_smooth(data.adj_t, all_logits, data.context_sample, data.y[data.context_sample], num_classes, num_iters=cs_num_iters, alpha=cs_alpha)
        valid_y = data.y[valid_idx]
        valid_acc_base = acc(valid_y, valid_score_base)
        valid_acc_cs = acc(valid_y, cs_predictions[valid_idx].argmax(dim=1))
        if valid_acc_cs > valid_acc_base:
            train_score = cs_predictions[train_idx].argmax(dim=1)
            valid_score = cs_predictions[valid_idx].argmax(dim=1)
            test_score = cs_predictions[test_idx].argmax(dim=1)
            if rank == 0:
                dataset_name = getattr(data, 'name', 'unknown')
                print(f'[C&S] Using C&S for {dataset_name} (valid: {valid_acc_base:.4f} → {valid_acc_cs:.4f})')
        else:
            train_score = train_score_base
            valid_score = valid_score_base
            test_score = test_score_base
            if rank == 0:
                dataset_name = getattr(data, 'name', 'unknown')
                print(f'[C&S] Skipping C&S for {dataset_name} (valid: {valid_acc_base:.4f} ≥ {valid_acc_cs:.4f})')
        cs_sec = _stage_stop(cs_start)
    else:
        train_score = train_score_base
        valid_score = valid_score_base
        test_score = test_score_base
    active_compute_sec = _stage_stop(active_compute_start)
    valid_y = data.y[valid_idx]
    valid_results = acc(valid_y, valid_score)
    train_y = data.y[train_idx]
    train_results = acc(train_y, train_score)
    test_y = data.y[test_idx]
    test_results = acc(test_y, test_score)
    if profile_active_compute:
        total_elapsed_sec = time.time() - st
        excluded_sec = max(0.0, total_elapsed_sec - active_compute_sec)
        dataset_name = getattr(data, 'name', 'unknown')
        print(f'[NC_ACTIVE_COMPUTE] dataset={dataset_name} active_compute={active_compute_sec:.4f}s (projection={projection_sec:.4f}s gnn={gnn_sec:.4f}s context={context_sec:.4f}s predictor={predictor_sec:.4f}s cs={cs_sec:.4f}s) excluded={excluded_sec:.4f}s')
    if moved_to_device:
        data = data.cpu()
        if hasattr(data, 'context_sample') and data.context_sample is not None:
            data.context_sample = data.context_sample.cpu()
        train_idx = train_idx.cpu()
        valid_idx = valid_idx.cpu()
        test_idx = test_idx.cpu()
    if rank == 0:
        print(f'Test time: {time.time() - st}', flush=True)
    return (train_results, valid_results, test_results)

def test_all(model, predictor, data_list, split_idx_list, batch_size, degree=False, att=None, mlp=None, normalize_class_h=False, projector=None, rank=0, identity_projection=None, external_embeddings_list=None, use_cs=False, cs_num_iters=50, cs_alpha=0.5):
    tot_train_metric, tot_valid_metric, tot_test_metric = (1, 1, 1)
    for i, (data, split_idx) in enumerate(zip(data_list, split_idx_list)):
        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']
        external_embeddings = external_embeddings_list[i] if external_embeddings_list else None
        train_metric, valid_metric, test_metric = test(model, predictor, data, train_idx, valid_idx, test_idx, batch_size, degree, att, mlp, normalize_class_h, None, rank, None, None, args=None, use_cs=use_cs, cs_num_iters=cs_num_iters, cs_alpha=cs_alpha)
        if rank == 0:
            print(f'Dataset {data.name}')
            print(f'Train {train_metric}, Valid {valid_metric}, Test {test_metric}', flush=True)
        tot_train_metric *= train_metric
        tot_valid_metric *= valid_metric
        tot_test_metric *= test_metric
    return (tot_train_metric ** (1 / len(data_list)), tot_valid_metric ** (1 / len(data_list)), tot_test_metric ** (1 / len(data_list)))

def test_all_induct(model, predictor, data_list, split_idx_list, batch_size, degree=False, att=None, mlp=None, normalize_class_h=False, projector=None, rank=0, identity_projection=None, external_embeddings_list=None, use_cs=False, cs_num_iters=50, cs_alpha=0.5, args=None, epoch=0):
    import time
    train_metric_list, valid_metric_list, test_metric_list = ([], [], [])
    use_augmentation = False
    num_augmentations = 0
    num_original = len(data_list)
    num_augmented = 0
    original_valid_metrics = []
    augmented_valid_metrics = []
    for dataset_idx, (data, split_idx) in enumerate(zip(data_list, split_idx_list)):
        dataset_start_time = time.time()
        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']
        external_embeddings = external_embeddings_list[dataset_idx] if external_embeddings_list else None
        train_metric, valid_metric, test_metric = test(model, predictor, data, train_idx, valid_idx, test_idx, batch_size, degree, att, mlp, normalize_class_h, None, rank, None, None, args=args, use_cs=use_cs, cs_num_iters=cs_num_iters, cs_alpha=cs_alpha, epoch=epoch)
        dataset_time = time.time() - dataset_start_time
        is_augmented = dataset_idx >= num_original
        if is_augmented:
            augmented_valid_metrics.append(valid_metric)
        else:
            original_valid_metrics.append(valid_metric)
        if rank == 0:
            prefix = '[AUG]' if is_augmented else '[ORI]'
            print(f'    {prefix} Dataset {dataset_idx} ({data.name}): completed in {dataset_time:.2f}s')
            print(f'      Train {train_metric:.4f}, Valid {valid_metric:.4f}, Test {test_metric:.4f}', flush=True)
        train_metric_list.append(train_metric)
        valid_metric_list.append(valid_metric)
        test_metric_list.append(test_metric)
    if rank == 0 and num_augmented > 0:
        import numpy as np
        avg_original_valid = np.mean(original_valid_metrics) if original_valid_metrics else 0
        avg_augmented_valid = np.mean(augmented_valid_metrics) if augmented_valid_metrics else 0
        print(f"\n{'=' * 80}")
        print(f'[VALIDATION SUMMARY - Training Datasets]')
        print(f'  Original Graphs ({num_original}): Avg Valid Acc: {avg_original_valid:.4f}')
        print(f'  Augmented Graphs ({num_augmented}): Avg Valid Acc: {avg_augmented_valid:.4f}')
        print(f'  Difference (Aug - Ori): {avg_augmented_valid - avg_original_valid:+.4f}')
        print(f"{'=' * 80}\n")
    return (train_metric_list, valid_metric_list, test_metric_list)

def test_all_induct_with_tta(model, predictor, data_list, split_idx_list, batch_size, degree=False, att=None, mlp=None, normalize_class_h=False, projector=None, rank=0, identity_projection=None, external_embeddings_list=None, use_cs=False, cs_num_iters=50, cs_alpha=0.5, args=None):
    import time
    import torch
    import torch.nn.functional as F
    from src.data_utils import apply_random_projection_augmentation
    tta_num_augmentations = getattr(args, 'tta_num_augmentations', 5)
    tta_include_original = getattr(args, 'tta_include_original', True)
    tta_aggregation = getattr(args, 'tta_aggregation', 'logits')
    tta_gate_by_valid = getattr(args, 'tta_gate_by_valid', True)
    tta_apply_input_normalization_to_views = getattr(args, 'tta_apply_input_normalization_to_views', False)
    tta_linear_projection = getattr(args, 'tta_linear_projection', False)
    if rank == 0:
        print(f"\n{'=' * 80}")
        print(f'[TEST-TIME AUGMENTATION (TTA) - Unseen Test Datasets Only]')
        print(f'  Number of augmentations: {tta_num_augmentations}')
        print(f'  Include original: {tta_include_original}')
        print(f'  Aggregation strategy: {tta_aggregation}')
        print(f'  Gate by valid: {tta_gate_by_valid}')
        print(f'  Normalize augmented views like main path: {tta_apply_input_normalization_to_views}')
        print(f'  Linear projection only: {tta_linear_projection}')
        print(f'  Total versions per graph: {tta_num_augmentations + (1 if tta_include_original else 0)}')
        print(f"{'=' * 80}\n")
    try:
        inference_device = next(model.parameters()).device
    except StopIteration:
        try:
            inference_device = next(predictor.parameters()).device
        except StopIteration:
            inference_device = data_list[0].x.device if len(data_list) > 0 else torch.device('cpu')
    if rank == 0:
        print(f'  TTA inference device: {inference_device}')
    tta_profile_enabled = bool(False)
    tta_profile_view_interval = max(0, int(getattr(args, 'nc_eval_profile_view_interval', 0))) if args is not None else 0
    tta_profile_proc = psutil.Process(os.getpid()) if tta_profile_enabled else None

    def _tta_profile_snapshot():
        if not tta_profile_enabled:
            return None
        if inference_device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize(inference_device)
        cpu_times = tta_profile_proc.cpu_times()
        return (time.perf_counter(), cpu_times.user + cpu_times.system)

    def _tta_profile_elapsed(start_snapshot):
        if not tta_profile_enabled or start_snapshot is None:
            return (0.0, 0.0)
        end_snapshot = _tta_profile_snapshot()
        return (end_snapshot[0] - start_snapshot[0], end_snapshot[1] - start_snapshot[1])

    def _tta_profile_log(stage, start_snapshot=None, extra=''):
        if not tta_profile_enabled:
            return
        now_snapshot = _tta_profile_snapshot()
        wall_dt = 0.0
        cpu_dt = 0.0
        cpu_over_wall = 0.0
        if start_snapshot is not None:
            wall_dt = now_snapshot[0] - start_snapshot[0]
            cpu_dt = now_snapshot[1] - start_snapshot[1]
            if wall_dt > 1e-09:
                cpu_over_wall = 100.0 * cpu_dt / wall_dt
        sys_cpu = psutil.cpu_percent(interval=None)
        rss_gb = tta_profile_proc.memory_info().rss / 1024 ** 3
        gpu_alloc_gb = 0.0
        gpu_reserved_gb = 0.0
        if inference_device.type == 'cuda' and torch.cuda.is_available():
            dev_idx = inference_device.index if inference_device.index is not None else torch.cuda.current_device()
            gpu_alloc_gb = torch.cuda.memory_allocated(dev_idx) / 1024 ** 3
            gpu_reserved_gb = torch.cuda.memory_reserved(dev_idx) / 1024 ** 3
        extra_text = f' | {extra}' if extra else ''
        print(f'[NC_TTA_PROFILE] {stage} | wall={wall_dt:.3f}s cpu={cpu_dt:.3f}s cpu/wall={cpu_over_wall:.1f}% sys_cpu={sys_cpu:.1f}% rss={rss_gb:.2f}GB gpu_alloc={gpu_alloc_gb:.2f}GB gpu_reserved={gpu_reserved_gb:.2f}GB{extra_text}')
    if tta_profile_enabled:
        psutil.cpu_percent(interval=None)
        _tta_profile_log('enabled', extra=f'device={inference_device} num_augmentations={tta_num_augmentations} include_original={tta_include_original} view_interval={tta_profile_view_interval}')
    train_metric_list, valid_metric_list, test_metric_list = ([], [], [])
    for dataset_idx, (data, split_idx) in enumerate(zip(data_list, split_idx_list)):
        dataset_start_time = time.time()
        dataset_profile_start = _tta_profile_snapshot()
        data_to_device_wall = 0.0
        data_to_device_cpu = 0.0
        aug_wall = 0.0
        aug_cpu = 0.0
        pca_wall = 0.0
        pca_cpu = 0.0
        pca_lowrank_wall = 0.0
        pca_project_wall = 0.0
        pca_pad_wall = 0.0
        pca_unattributed_wall = 0.0
        infer_wall = 0.0
        infer_cpu = 0.0
        infer_inner_total_wall = 0.0
        infer_device_transfer_wall = 0.0
        infer_proj_wall = 0.0
        infer_gnn_wall = 0.0
        infer_context_gather_wall = 0.0
        infer_context_wall = 0.0
        infer_loader_setup_wall = 0.0
        infer_predictor_wall = 0.0
        infer_predictor_train_wall = 0.0
        infer_predictor_valid_wall = 0.0
        infer_predictor_test_wall = 0.0
        infer_restore_wall = 0.0
        infer_unattributed_wall = 0.0
        active_compute_wall = 0.0
        active_compute_cpu = 0.0
        if rank == 0:
            print(f'  [TTA] Processing unseen dataset {dataset_idx} ({data.name})...')
        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']
        external_embeddings = external_embeddings_list[dataset_idx] if external_embeddings_list else None
        data_to_device_start = _tta_profile_snapshot()
        data_device = data.clone().to(inference_device)
        if hasattr(data_device, 'x') and data_device.x is not None:
            data_device.x = data_device.x.to(inference_device)
        if hasattr(data_device, 'x_original') and data_device.x_original is not None:
            data_device.x_original = data_device.x_original.to(inference_device)
        if hasattr(data_device, 'y') and data_device.y is not None:
            data_device.y = data_device.y.to(inference_device)
        train_idx = train_idx.to(inference_device)
        valid_idx = valid_idx.to(inference_device)
        test_idx = test_idx.to(inference_device)
        if hasattr(data_device, 'context_sample') and data_device.context_sample is not None:
            data_device.context_sample = data_device.context_sample.to(inference_device)
        if hasattr(data_device, 'adj_t'):
            data_device.adj_t = data_device.adj_t.to(inference_device)
        data_to_device_wall, data_to_device_cpu = _tta_profile_elapsed(data_to_device_start)
        if tta_profile_enabled:
            _tta_profile_log('dataset_data_to_device', data_to_device_start, extra=f'dataset={dataset_idx} name={data.name}')
        active_compute_start = _tta_profile_snapshot()
        if hasattr(data, 'needs_identity_projection') and data.needs_identity_projection:
            pca_target_dim = args.projection_small_dim
        else:
            pca_target_dim = args.hidden
        all_train_logits = []
        all_valid_logits = []
        all_test_logits = []
        total_versions = tta_num_augmentations + (1 if tta_include_original else 0)
        if rank == 0:
            print(f'    Created {total_versions} versions (original + {tta_num_augmentations} augmented)')

        def _run_one_tta_view(data_version, view_type, view_idx):
            nonlocal infer_wall, infer_cpu
            nonlocal infer_inner_total_wall, infer_device_transfer_wall, infer_proj_wall, infer_gnn_wall
            nonlocal infer_context_gather_wall, infer_context_wall, infer_loader_setup_wall
            nonlocal infer_predictor_wall, infer_predictor_train_wall, infer_predictor_valid_wall
            nonlocal infer_predictor_test_wall, infer_restore_wall, infer_unattributed_wall
            if hasattr(data_version, 'x') and data_version.x is not None:
                data_version.x = data_version.x.to(inference_device)
            if hasattr(data_version, 'y') and data_version.y is not None:
                data_version.y = data_version.y.to(inference_device)
            if hasattr(data_version, 'context_sample') and data_version.context_sample is not None:
                data_version.context_sample = data_version.context_sample.to(inference_device)
            if hasattr(data_version, 'adj_t') and data_version.adj_t is not None:
                data_version.adj_t = data_version.adj_t.to(inference_device)
            infer_start = _tta_profile_snapshot()
            if tta_profile_enabled:
                train_logits, valid_logits, test_logits, infer_breakdown = test_single_with_logits(model, predictor, data_version, train_idx, valid_idx, test_idx, batch_size, degree, att, mlp, normalize_class_h, None, rank, None, None, args, use_cs, cs_num_iters, cs_alpha, return_timing=True, assume_on_device=True)
            else:
                train_logits, valid_logits, test_logits = test_single_with_logits(model, predictor, data_version, train_idx, valid_idx, test_idx, batch_size, degree, att, mlp, normalize_class_h, None, rank, None, None, args, use_cs, cs_num_iters, cs_alpha, assume_on_device=True)
                infer_breakdown = None
            if infer_breakdown is not None:
                infer_inner_total_wall += infer_breakdown.get('total', 0.0)
                infer_device_transfer_wall += infer_breakdown.get('device_transfer', 0.0)
                infer_proj_wall += infer_breakdown.get('projection', 0.0)
                infer_gnn_wall += infer_breakdown.get('gnn', 0.0)
                infer_context_gather_wall += infer_breakdown.get('context_gather', 0.0)
                infer_context_wall += infer_breakdown.get('context', 0.0)
                infer_loader_setup_wall += infer_breakdown.get('loader_setup', 0.0)
                infer_predictor_wall += infer_breakdown.get('predictor_total', 0.0)
                infer_predictor_train_wall += infer_breakdown.get('predictor_train', 0.0)
                infer_predictor_valid_wall += infer_breakdown.get('predictor_valid', 0.0)
                infer_predictor_test_wall += infer_breakdown.get('predictor_test', 0.0)
                infer_restore_wall += infer_breakdown.get('restore', 0.0)
                infer_unattributed_wall += infer_breakdown.get('unattributed', 0.0)
                if tta_profile_view_interval > 0:
                    is_last = view_idx == total_versions - 1
                    if view_idx == 0 or is_last or view_idx % tta_profile_view_interval == 0:
                        print(f"[NC_INFER_BREAKDOWN] dataset={dataset_idx} view={view_type} idx={view_idx} total={infer_breakdown.get('total', 0.0):.4f}s device_transfer={infer_breakdown.get('device_transfer', 0.0):.4f}s proj={infer_breakdown.get('projection', 0.0):.4f}s gnn={infer_breakdown.get('gnn', 0.0):.4f}s context_gather={infer_breakdown.get('context_gather', 0.0):.4f}s context={infer_breakdown.get('context', 0.0):.4f}s loader_setup={infer_breakdown.get('loader_setup', 0.0):.4f}s predictor={infer_breakdown.get('predictor_total', 0.0):.4f}s predictor_train={infer_breakdown.get('predictor_train', 0.0):.4f}s predictor_valid={infer_breakdown.get('predictor_valid', 0.0):.4f}s predictor_test={infer_breakdown.get('predictor_test', 0.0):.4f}s restore={infer_breakdown.get('restore', 0.0):.4f}s unattributed={infer_breakdown.get('unattributed', 0.0):.4f}s")
            infer_view_wall, infer_view_cpu = _tta_profile_elapsed(infer_start)
            infer_wall += infer_view_wall
            infer_cpu += infer_view_cpu
            if tta_profile_enabled and tta_profile_view_interval > 0:
                is_last = view_idx == total_versions - 1
                if view_idx == 0 or is_last or view_idx % tta_profile_view_interval == 0:
                    _tta_profile_log('tta_view_infer', infer_start, extra=f'dataset={dataset_idx} view={view_type} idx={view_idx}')
            all_train_logits.append(train_logits)
            all_valid_logits.append(valid_logits)
            all_test_logits.append(test_logits)
            if rank == 0:
                debug_device = train_logits.device
                y_view = data_version.y.to(debug_device)
                train_idx_view = train_idx.to(debug_device)
                valid_idx_view = valid_idx.to(debug_device)
                test_idx_view = test_idx.to(debug_device)
                train_pred = train_logits.argmax(dim=-1)
                valid_pred = valid_logits.argmax(dim=-1)
                test_pred = test_logits.argmax(dim=-1)
                train_acc = (train_pred == y_view[train_idx_view]).float().mean().item()
                valid_acc = (valid_pred == y_view[valid_idx_view]).float().mean().item()
                test_acc = (test_pred == y_view[test_idx_view]).float().mean().item()
                print(f'      [{view_type}] Train: {train_acc:.4f}, Valid: {valid_acc:.4f}, Test: {test_acc:.4f}')
        if tta_include_original:
            _run_one_tta_view(data_device, 'Original', 0)
        for aug_idx in range(tta_num_augmentations):
            seed = 999000 + dataset_idx * 1000 + aug_idx
            data_for_aug = data_device.clone()
            if hasattr(data_device, 'x_original'):
                data_for_aug.x = data_device.x_original.to(inference_device).clone()
            elif rank == 0 and aug_idx == 0:
                print(f"    WARNING: x_original not found for {getattr(data_device, 'name', 'unknown')}, using data.x (may be PCA-processed!)")
            if hasattr(data_for_aug, 'x') and data_for_aug.x is not None:
                data_for_aug.x = data_for_aug.x.to(inference_device)
            aug_start = _tta_profile_snapshot()
            data_aug = apply_random_projection_augmentation(data_for_aug, hidden_dim_range=None, activation_pool=['identity'] if tta_linear_projection else None, seed=seed, verbose=False, rank=rank, max_hidden_dim=getattr(args, 'augmentation_max_dim', None))
            if hasattr(data_aug, 'x') and data_aug.x is not None:
                data_aug.x = data_aug.x.to(inference_device)
            aug_view_wall, aug_view_cpu = _tta_profile_elapsed(aug_start)
            aug_wall += aug_view_wall
            aug_cpu += aug_view_cpu
            if tta_profile_enabled and tta_profile_view_interval > 0 and ((aug_idx + 1) % tta_profile_view_interval == 0):
                _tta_profile_log('tta_view_augment', aug_start, extra=f'dataset={dataset_idx} aug_idx={aug_idx + 1} seed={seed} out_dim={data_aug.x.shape[1]}')
            pca_start = _tta_profile_snapshot()
            q = min(pca_target_dim, data_aug.x.shape[0], data_aug.x.shape[1])
            pca_lowrank_start = _tta_profile_snapshot()
            U, S, _ = torch.pca_lowrank(data_aug.x, q=q)
            pca_lowrank_view_wall, _ = _tta_profile_elapsed(pca_lowrank_start)
            pca_lowrank_wall += pca_lowrank_view_wall
            pca_project_start = _tta_profile_snapshot()
            data_aug.x_pca = torch.mm(U, torch.diag(S))
            pca_project_view_wall, _ = _tta_profile_elapsed(pca_project_start)
            pca_project_wall += pca_project_view_wall
            pca_pad_start = _tta_profile_snapshot()
            if data_aug.x_pca.shape[1] < pca_target_dim:
                pad_size = pca_target_dim - data_aug.x_pca.shape[1]
                padding = torch.zeros(data_aug.x_pca.shape[0], pad_size, device=data_aug.x.device)
                data_aug.x_pca = torch.cat([data_aug.x_pca, padding], dim=1)
            pca_pad_view_wall, _ = _tta_profile_elapsed(pca_pad_start)
            pca_pad_wall += pca_pad_view_wall
            pca_view_wall, pca_view_cpu = _tta_profile_elapsed(pca_start)
            pca_wall += pca_view_wall
            pca_cpu += pca_view_cpu
            pca_unattributed_wall += max(0.0, pca_view_wall - pca_lowrank_view_wall - pca_project_view_wall - pca_pad_view_wall)
            if tta_profile_enabled and tta_profile_view_interval > 0 and ((aug_idx + 1) % tta_profile_view_interval == 0):
                pca_unattributed_view_wall = max(0.0, pca_view_wall - pca_lowrank_view_wall - pca_project_view_wall - pca_pad_view_wall)
                print(f'[NC_PCA_BREAKDOWN] dataset={dataset_idx} aug_idx={aug_idx + 1} total={pca_view_wall:.4f}s lowrank={pca_lowrank_view_wall:.4f}s project={pca_project_view_wall:.4f}s pad={pca_pad_view_wall:.4f}s unattributed={pca_unattributed_view_wall:.4f}s')
                _tta_profile_log('tta_view_pca', pca_start, extra=f'dataset={dataset_idx} aug_idx={aug_idx + 1} pca_dim={data_aug.x_pca.shape[1]}')
            data_aug.x = data_aug.x_pca
            data_aug.x = _apply_tta_view_input_normalization(data_aug.x, args)
            if hasattr(data_device, 'needs_identity_projection'):
                data_aug.needs_identity_projection = data_device.needs_identity_projection
            if hasattr(data_device, 'projection_target_dim'):
                data_aug.projection_target_dim = data_device.projection_target_dim
            if hasattr(data_device, 'needs_projection'):
                data_aug.needs_projection = data_device.needs_projection
            _run_one_tta_view(data_aug, f'Aug_{aug_idx + 1}', aug_idx + 1)
        if rank == 0:
            summary_device = all_test_logits[0].device
            y_all = data_device.y.to(summary_device)
            test_idx_dev = test_idx.to(summary_device)
            individual_test_accs = []
            for logits in all_test_logits:
                pred = logits.argmax(dim=-1)
                acc = (pred == y_all[test_idx_dev]).float().mean().item()
                individual_test_accs.append(acc)
            print(f'      Individual test accs: mean={sum(individual_test_accs) / len(individual_test_accs):.4f}, min={min(individual_test_accs):.4f}, max={max(individual_test_accs):.4f}')
        if tta_aggregation == 'logits':
            train_logits_agg = torch.stack(all_train_logits).mean(dim=0)
            valid_logits_agg = torch.stack(all_valid_logits).mean(dim=0)
            test_logits_agg = torch.stack(all_test_logits).mean(dim=0)
        elif tta_aggregation == 'probs':
            train_probs = torch.stack([F.softmax(logits, dim=-1) for logits in all_train_logits]).mean(dim=0)
            valid_probs = torch.stack([F.softmax(logits, dim=-1) for logits in all_valid_logits]).mean(dim=0)
            test_probs = torch.stack([F.softmax(logits, dim=-1) for logits in all_test_logits]).mean(dim=0)
            train_logits_agg = torch.log(train_probs + 1e-10)
            valid_logits_agg = torch.log(valid_probs + 1e-10)
            test_logits_agg = torch.log(test_probs + 1e-10)
            if rank == 0:
                y_agg = data_device.y.to(test_logits_agg.device)
                test_idx_dev = test_idx.to(test_logits_agg.device)
                agg_test_pred = test_logits_agg.argmax(dim=-1)
                agg_test_acc = (agg_test_pred == y_agg[test_idx_dev]).float().mean().item()
                print(f'      After aggregation: test_acc={agg_test_acc:.4f} (boost: +{agg_test_acc - sum(individual_test_accs) / len(individual_test_accs):.4f})')
        elif tta_aggregation == 'voting':
            train_preds = torch.stack([logits.argmax(dim=-1) for logits in all_train_logits])
            valid_preds = torch.stack([logits.argmax(dim=-1) for logits in all_valid_logits])
            test_preds = torch.stack([logits.argmax(dim=-1) for logits in all_test_logits])
            train_pred_mode = torch.mode(train_preds, dim=0).values
            valid_pred_mode = torch.mode(valid_preds, dim=0).values
            test_pred_mode = torch.mode(test_preds, dim=0).values
            num_classes = all_train_logits[0].size(1)
            train_logits_agg = F.one_hot(train_pred_mode, num_classes=num_classes).float() * 10.0
            valid_logits_agg = F.one_hot(valid_pred_mode, num_classes=num_classes).float() * 10.0
            test_logits_agg = F.one_hot(test_pred_mode, num_classes=num_classes).float() * 10.0
        metric_device = train_logits_agg.device
        y_device = data_device.y.to(metric_device)
        train_idx_dev = train_idx.to(metric_device)
        valid_idx_dev = valid_idx.to(metric_device)
        test_idx_dev = test_idx.to(metric_device)
        context_sample_dev = data_device.context_sample.to(metric_device)
        if rank == 0 and bool(args is not None and getattr(args, 'tta_distribution_debug', False)):
            _print_tta_distribution_debug(data.name, 'valid', all_valid_logits[0], all_valid_logits, valid_logits_agg, y_device[valid_idx_dev])
            _print_tta_distribution_debug(data.name, 'test', all_test_logits[0], all_test_logits, test_logits_agg, y_device[test_idx_dev])
        train_logits_sel = train_logits_agg
        valid_logits_sel = valid_logits_agg
        test_logits_sel = test_logits_agg
        if tta_gate_by_valid:
            if not tta_include_original:
                if rank == 0:
                    print('    [TTA] Gating skipped (tta_include_original=False).')
            else:
                train_logits_base = all_train_logits[0]
                valid_logits_base = all_valid_logits[0]
                test_logits_base = all_test_logits[0]
                valid_acc_base = (valid_logits_base.argmax(dim=-1) == y_device[valid_idx_dev]).float().mean().item()
                valid_acc_tta = (valid_logits_agg.argmax(dim=-1) == y_device[valid_idx_dev]).float().mean().item()
                if valid_acc_tta > valid_acc_base:
                    if rank == 0:
                        print(f'    [TTA] Valid improved: {valid_acc_base:.4f} → {valid_acc_tta:.4f} ✓')
                else:
                    train_logits_sel = train_logits_base
                    valid_logits_sel = valid_logits_base
                    test_logits_sel = test_logits_base
                    if rank == 0:
                        print(f'    [TTA] Valid not improved: {valid_acc_tta:.4f} ≤ {valid_acc_base:.4f} (using original)')
        train_score_base = train_logits_sel.argmax(dim=-1)
        valid_score_base = valid_logits_sel.argmax(dim=-1)
        test_score_base = test_logits_sel.argmax(dim=-1)
        if use_cs:
            if rank == 0:
                print(f'    Applying C&S to aggregated TTA logits...')
            all_logits = torch.zeros(data.num_nodes, train_logits_agg.size(1), device=metric_device)
            all_logits[train_idx_dev] = train_logits_sel
            all_logits[valid_idx_dev] = valid_logits_sel
            all_logits[test_idx_dev] = test_logits_sel
            context_y = y_device[context_sample_dev]
            num_classes = context_y.max().item() + 1
            cs_predictions = correct_and_smooth(data_device.adj_t.to(metric_device), all_logits, context_sample_dev, context_y, num_classes, cs_num_iters, cs_alpha)
            train_score_cs = cs_predictions[train_idx_dev].argmax(dim=-1)
            valid_score_cs = cs_predictions[valid_idx_dev].argmax(dim=-1)
            test_score_cs = cs_predictions[test_idx_dev].argmax(dim=-1)
            valid_acc_base = (valid_score_base == y_device[valid_idx_dev]).float().mean().item()
            valid_acc_cs = (valid_score_cs == y_device[valid_idx_dev]).float().mean().item()
            if valid_acc_cs > valid_acc_base:
                train_pred = train_score_cs
                valid_pred = valid_score_cs
                test_pred = test_score_cs
                if rank == 0:
                    print(f'    C&S improved validation: {valid_acc_base:.4f} → {valid_acc_cs:.4f} ✓')
            else:
                train_pred = train_score_base
                valid_pred = valid_score_base
                test_pred = test_score_base
                if rank == 0:
                    print(f'    C&S did not improve: {valid_acc_base:.4f} ≥ {valid_acc_cs:.4f} (using base)')
        else:
            train_pred = train_score_base
            valid_pred = valid_score_base
            test_pred = test_score_base
        active_compute_wall, active_compute_cpu = _tta_profile_elapsed(active_compute_start)
        train_metric = (train_pred == y_device[train_idx_dev]).float().mean().item()
        valid_metric = (valid_pred == y_device[valid_idx_dev]).float().mean().item()
        test_metric = (test_pred == y_device[test_idx_dev]).float().mean().item()
        dataset_time = time.time() - dataset_start_time
        if rank == 0:
            print(f'    [TTA] Dataset {dataset_idx} ({data.name}): completed in {dataset_time:.2f}s')
            print(f'      Train {train_metric:.4f}, Valid {valid_metric:.4f}, Test {test_metric:.4f}', flush=True)
            if tta_profile_enabled:
                infer_cpu_ratio = 100.0 * infer_cpu / infer_wall if infer_wall > 1e-09 else 0.0
                aug_cpu_ratio = 100.0 * aug_cpu / aug_wall if aug_wall > 1e-09 else 0.0
                pca_cpu_ratio = 100.0 * pca_cpu / pca_wall if pca_wall > 1e-09 else 0.0
                pca_lowrank_ratio = 100.0 * pca_lowrank_wall / pca_wall if pca_wall > 1e-09 else 0.0
                pca_project_ratio = 100.0 * pca_project_wall / pca_wall if pca_wall > 1e-09 else 0.0
                pca_pad_ratio = 100.0 * pca_pad_wall / pca_wall if pca_wall > 1e-09 else 0.0
                pca_unattributed_ratio = 100.0 * pca_unattributed_wall / pca_wall if pca_wall > 1e-09 else 0.0
                active_compute_cpu_ratio = 100.0 * active_compute_cpu / active_compute_wall if active_compute_wall > 1e-09 else 0.0
                excluded_wall = max(0.0, dataset_time - data_to_device_wall - active_compute_wall)
                infer_inner_total_ratio = 100.0 * infer_inner_total_wall / infer_wall if infer_wall > 1e-09 else 0.0
                infer_device_transfer_ratio = 100.0 * infer_device_transfer_wall / infer_inner_total_wall if infer_inner_total_wall > 1e-09 else 0.0
                infer_predictor_ratio = 100.0 * infer_predictor_wall / infer_inner_total_wall if infer_inner_total_wall > 1e-09 else 0.0
                infer_predictor_train_ratio = 100.0 * infer_predictor_train_wall / infer_inner_total_wall if infer_inner_total_wall > 1e-09 else 0.0
                infer_predictor_valid_ratio = 100.0 * infer_predictor_valid_wall / infer_inner_total_wall if infer_inner_total_wall > 1e-09 else 0.0
                infer_predictor_test_ratio = 100.0 * infer_predictor_test_wall / infer_inner_total_wall if infer_inner_total_wall > 1e-09 else 0.0
                infer_gnn_ratio = 100.0 * infer_gnn_wall / infer_inner_total_wall if infer_inner_total_wall > 1e-09 else 0.0
                infer_proj_ratio = 100.0 * infer_proj_wall / infer_inner_total_wall if infer_inner_total_wall > 1e-09 else 0.0
                infer_context_gather_ratio = 100.0 * infer_context_gather_wall / infer_inner_total_wall if infer_inner_total_wall > 1e-09 else 0.0
                infer_context_ratio = 100.0 * infer_context_wall / infer_inner_total_wall if infer_inner_total_wall > 1e-09 else 0.0
                infer_loader_ratio = 100.0 * infer_loader_setup_wall / infer_inner_total_wall if infer_inner_total_wall > 1e-09 else 0.0
                infer_restore_ratio = 100.0 * infer_restore_wall / infer_inner_total_wall if infer_inner_total_wall > 1e-09 else 0.0
                infer_unattributed_ratio = 100.0 * infer_unattributed_wall / infer_inner_total_wall if infer_inner_total_wall > 1e-09 else 0.0
                _tta_profile_log('dataset_summary', dataset_profile_start, extra=f'dataset={dataset_idx} name={data.name} versions={total_versions} data_to_device_wall={data_to_device_wall:.3f}s data_to_device_cpu={data_to_device_cpu:.3f}s active_compute_wall={active_compute_wall:.3f}s active_compute_cpu={active_compute_cpu:.3f}s active_compute_cpu_over_wall={active_compute_cpu_ratio:.1f}% excluded_wall={excluded_wall:.3f}s aug_wall={aug_wall:.3f}s aug_cpu={aug_cpu:.3f}s aug_cpu_over_wall={aug_cpu_ratio:.1f}% pca_wall={pca_wall:.3f}s pca_cpu={pca_cpu:.3f}s pca_cpu_over_wall={pca_cpu_ratio:.1f}% pca_lowrank_wall={pca_lowrank_wall:.3f}s pca_lowrank_share={pca_lowrank_ratio:.1f}% pca_project_wall={pca_project_wall:.3f}s pca_project_share={pca_project_ratio:.1f}% pca_pad_wall={pca_pad_wall:.3f}s pca_pad_share={pca_pad_ratio:.1f}% pca_unattributed_wall={pca_unattributed_wall:.3f}s pca_unattributed_share={pca_unattributed_ratio:.1f}% infer_wall={infer_wall:.3f}s infer_cpu={infer_cpu:.3f}s infer_cpu_over_wall={infer_cpu_ratio:.1f}% infer_inner_total_wall={infer_inner_total_wall:.3f}s infer_inner_total_over_infer={infer_inner_total_ratio:.1f}% infer_device_transfer_wall={infer_device_transfer_wall:.3f}s infer_device_transfer_share={infer_device_transfer_ratio:.1f}% infer_predictor_wall={infer_predictor_wall:.3f}s infer_predictor_share={infer_predictor_ratio:.1f}% infer_predictor_train_wall={infer_predictor_train_wall:.3f}s infer_predictor_train_share={infer_predictor_train_ratio:.1f}% infer_predictor_valid_wall={infer_predictor_valid_wall:.3f}s infer_predictor_valid_share={infer_predictor_valid_ratio:.1f}% infer_predictor_test_wall={infer_predictor_test_wall:.3f}s infer_predictor_test_share={infer_predictor_test_ratio:.1f}% infer_gnn_wall={infer_gnn_wall:.3f}s infer_gnn_share={infer_gnn_ratio:.1f}% infer_proj_wall={infer_proj_wall:.3f}s infer_proj_share={infer_proj_ratio:.1f}% infer_context_gather_wall={infer_context_gather_wall:.3f}s infer_context_gather_share={infer_context_gather_ratio:.1f}% infer_context_wall={infer_context_wall:.3f}s infer_context_share={infer_context_ratio:.1f}% infer_loader_setup_wall={infer_loader_setup_wall:.3f}s infer_loader_setup_share={infer_loader_ratio:.1f}% infer_restore_wall={infer_restore_wall:.3f}s infer_restore_share={infer_restore_ratio:.1f}% infer_unattributed_wall={infer_unattributed_wall:.3f}s infer_unattributed_share={infer_unattributed_ratio:.1f}%')
        train_metric_list.append(train_metric)
        valid_metric_list.append(valid_metric)
        test_metric_list.append(test_metric)
    return (train_metric_list, valid_metric_list, test_metric_list)

def test_single_with_logits(model, predictor, data, train_idx, valid_idx, test_idx, batch_size, degree=False, att=None, mlp=None, normalize_class_h=False, projector=None, rank=0, identity_projection=None, external_embeddings=None, args=None, use_cs=False, cs_num_iters=50, cs_alpha=0.5, return_timing=False, assume_on_device=False):
    """
    Helper function for TTA: Run inference on a single graph and return logits (not predictions).

    Returns:
        train_logits, valid_logits, test_logits: Raw logits for each split
    """
    from torch.utils.data import DataLoader
    model.eval()
    predictor.eval()
    try:
        device = next(model.parameters()).device
    except StopIteration:
        try:
            device = next(predictor.parameters()).device
        except StopIteration:
            device = data.x.device
    profile_infer_breakdown = bool(False)

    def _stage_start():
        if not profile_infer_breakdown:
            return None
        return _synced_perf_counter(device)

    def _stage_stop(stage_start):
        if stage_start is None:
            return 0.0
        return _synced_perf_counter(device) - stage_start
    infer_total_start = _stage_start()
    device_transfer_sec = 0.0
    projection_sec = 0.0
    gnn_sec = 0.0
    context_gather_sec = 0.0
    context_sec = 0.0
    loader_setup_sec = 0.0
    predictor_train_sec = 0.0
    predictor_valid_sec = 0.0
    predictor_test_sec = 0.0
    restore_sec = 0.0
    moved_to_device = False
    device_transfer_start = _stage_start()
    if not assume_on_device and (data.x.device != device or (hasattr(data, 'adj_t') and data.adj_t.device != device)):
        data = data.to(device)
        train_idx = train_idx.to(device)
        valid_idx = valid_idx.to(device)
        test_idx = test_idx.to(device)
        if hasattr(data, 'context_sample') and data.context_sample is not None:
            data.context_sample = data.context_sample.to(device)
        if hasattr(data, 'adj_t'):
            data.adj_t = data.adj_t.to(device)
        moved_to_device = True
    device_transfer_sec = _stage_stop(device_transfer_start)
    base_features = data.x
    projection_start = _stage_start()
    x_input = base_features
    projection_sec = _stage_stop(projection_start)
    gnn_start = _stage_start()
    with torch.no_grad():
        h = model(x_input, data.adj_t)
    gnn_sec = _stage_stop(gnn_start)
    context_gather_start = _stage_start()
    context_h = h[data.context_sample]
    context_y = data.y[data.context_sample]
    context_gather_sec = _stage_stop(context_gather_start)
    context_start = _stage_start()
    class_h = process_node_features(context_h, data, degree_normalize=degree, attention_pool_module=att, mlp_module=mlp, normalize=normalize_class_h)
    context_sec = _stage_stop(context_start)
    loader_setup_start = _stage_start()
    train_loader = DataLoader(range(train_idx.size(0)), batch_size, shuffle=False)
    valid_loader = DataLoader(range(valid_idx.size(0)), batch_size, shuffle=False)
    test_loader = DataLoader(range(test_idx.size(0)), batch_size, shuffle=False)
    loader_setup_sec = _stage_stop(loader_setup_start)
    train_logits = []
    predictor_train_start = _stage_start()
    with torch.no_grad():
        for idx in train_loader:
            target_h = h[train_idx[idx]]
            pred_output = predictor(data, context_h, target_h, context_y, class_h)
            if len(pred_output) == 3:
                out, _, _ = pred_output
            else:
                out, _ = pred_output
            train_logits.append(out)
    train_logits = torch.cat(train_logits, dim=0)
    predictor_train_sec = _stage_stop(predictor_train_start)
    valid_logits = []
    predictor_valid_start = _stage_start()
    with torch.no_grad():
        for idx in valid_loader:
            target_h = h[valid_idx[idx]]
            pred_output = predictor(data, context_h, target_h, context_y, class_h)
            if len(pred_output) == 3:
                out, _, _ = pred_output
            else:
                out, _ = pred_output
            valid_logits.append(out)
    valid_logits = torch.cat(valid_logits, dim=0)
    predictor_valid_sec = _stage_stop(predictor_valid_start)
    test_logits = []
    predictor_test_start = _stage_start()
    with torch.no_grad():
        for idx in test_loader:
            target_h = h[test_idx[idx]]
            pred_output = predictor(data, context_h, target_h, context_y, class_h)
            if len(pred_output) == 3:
                out, _, _ = pred_output
            else:
                out, _ = pred_output
            test_logits.append(out)
    test_logits = torch.cat(test_logits, dim=0)
    predictor_test_sec = _stage_stop(predictor_test_start)
    restore_start = _stage_start()
    if moved_to_device and (not assume_on_device):
        data = data.cpu()
        if hasattr(data, 'context_sample') and data.context_sample is not None:
            data.context_sample = data.context_sample.cpu()
        train_idx = train_idx.cpu()
        valid_idx = valid_idx.cpu()
        test_idx = test_idx.cpu()
        if hasattr(data, 'adj_t'):
            data.adj_t = data.adj_t.cpu()
    restore_sec = _stage_stop(restore_start)
    if return_timing:
        infer_total_sec = _stage_stop(infer_total_start)
        predictor_total_sec = predictor_train_sec + predictor_valid_sec + predictor_test_sec
        known_sec = device_transfer_sec + projection_sec + gnn_sec + context_gather_sec + context_sec + loader_setup_sec + predictor_total_sec + restore_sec
        infer_timing = {'total': infer_total_sec, 'device_transfer': device_transfer_sec, 'projection': projection_sec, 'gnn': gnn_sec, 'context_gather': context_gather_sec, 'context': context_sec, 'loader_setup': loader_setup_sec, 'predictor_train': predictor_train_sec, 'predictor_valid': predictor_valid_sec, 'predictor_test': predictor_test_sec, 'predictor_total': predictor_total_sec, 'restore': restore_sec, 'unattributed': max(0.0, infer_total_sec - known_sec)}
        return (train_logits, valid_logits, test_logits, infer_timing)
    return (train_logits, valid_logits, test_logits)
