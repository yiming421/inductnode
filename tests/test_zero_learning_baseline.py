"""
Zero-Learning Baseline: Pure GNN Structure Heuristic

Tests how well we can classify nodes WITHOUT any learning, using only:
1. GNN propagation (no learnable weights)
2. Mean pooling per class
3. Cosine similarity
"""

import argparse
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
import sys
import os
import wandb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data_nc import load_data, load_ogbn_data


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def normalize_features(x, method='none'):
    """
    Normalize node features.

    Args:
        method: 'none', 'row' (L2 per node), 'col' (standardize per feature), 'row+col'
    """
    if method == 'none':
        return x

    if 'col' in method:
        # Column normalization: standardize each feature dimension
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True) + 1e-9
        x = (x - mean) / std

    if 'row' in method:
        # Row normalization: L2 normalize each node
        x = F.normalize(x, p=2, dim=-1)

    return x


def gcn_propagate(x, adj, num_hops=2, method='residual', alpha=0.1, use_layer_norm=False):
    """
    GCN propagation without learnable weights.

    Args:
        method: 'residual' (original), 'appnp' (PPR-style), 'concat' (multi-hop concat), 'puregcn' (PureGCN-style)
        alpha: teleport probability for APPNP
        use_layer_norm: Apply LayerNorm after each propagation step
    """
    deg = adj.sum(dim=1).to_dense()

    # PureGCN uses deg + 1 for self-loop normalization
    deg_with_selfloop = deg + 1
    deg_inv_sqrt_pure = torch.rsqrt(deg_with_selfloop.clamp(min=1e-9)).view(-1, 1)

    # Standard GCN normalization
    deg_inv_sqrt = (deg + 1e-9).pow(-0.5)

    def propagate_once(h):
        h = deg_inv_sqrt.view(-1, 1) * h
        h = adj @ h
        h = deg_inv_sqrt.view(-1, 1) * h
        if use_layer_norm:
            # LayerNorm that adapts to arbitrary input dimension
            h = F.layer_norm(h, normalized_shape=(h.size(1),))
        return h

    def propagate_puregcn(h):
        """PureGCN-style propagation: norm * (A*x + x) * norm"""
        h = deg_inv_sqrt_pure * h
        h = adj @ h + h  # A*x + x (self-loop)
        h = deg_inv_sqrt_pure * h
        if use_layer_norm:
            # Post-LN: apply LayerNorm AFTER propagation
            h = F.layer_norm(h, normalized_shape=(h.size(1),))
        return h

    if method == 'puregcn':
        # PureGCN: directly iterate without external residual
        out = x
        for _ in range(num_hops):
            out = propagate_puregcn(out)
        return out

    elif method == 'residual':
        out = x
        for _ in range(num_hops):
            out = out + propagate_once(out)
        return out

    elif method == 'appnp':
        # APPNP: h^(k) = (1-α) * A * h^(k-1) + α * x
        out = x
        for _ in range(num_hops):
            out = (1 - alpha) * propagate_once(out) + alpha * x
        return out

    elif method == 'concat':
        # Concatenate features from all hops
        features = [x]
        h = x
        for _ in range(num_hops):
            h = propagate_once(h)
            features.append(h)
        return torch.cat(features, dim=-1)

    elif method == 'weighted':
        # Weighted sum of all hops (equal weights)
        out = x
        h = x
        for _ in range(num_hops):
            h = propagate_once(h)
            out = out + h
        return out / (num_hops + 1)

    else:
        raise ValueError(f"Unknown method: {method}")


def zero_learning_classify(x, context_indices, context_labels, target_indices, num_classes, sim='cos',
                           use_ridge=False, ridge_alpha=1.0):
    """
    Zero-learning classification with prototypical networks or ridge regression.

    Args:
        use_ridge: Use ridge regression (closed-form solution)
        ridge_alpha: Regularization strength for ridge regression
    """

    if use_ridge:
        # Ridge Regression: solve W = (X^T X + λI)^{-1} X^T Y
        support_x = x[context_indices]  # [n_support, dim]
        target_x = x[target_indices]    # [n_target, dim]

        # One-hot encode labels
        support_y = F.one_hot(context_labels.long(), num_classes=num_classes).float()  # [n_support, n_classes]

        # Solve ridge regression: W = (X^T X + λI)^{-1} X^T Y
        XtX = support_x.t() @ support_x  # [dim, dim]
        XtY = support_x.t() @ support_y  # [dim, n_classes]

        # Add regularization
        I = torch.eye(support_x.size(1), device=x.device)
        W = torch.linalg.solve(XtX + ridge_alpha * I, XtY)  # [dim, n_classes]

        # Predict
        logits = target_x @ W  # [n_target, n_classes]

        return logits.argmax(dim=1), logits

    else:
        # Prototypical Network: compute class prototypes then match
        prototypes = torch.zeros(num_classes, x.size(1), device=x.device)

        for c in range(num_classes):
            mask = context_labels == c
            if mask.any():
                class_features = x[context_indices[mask]]
                # Mean pooling
                prototypes[c] = class_features.mean(dim=0)

        target_x = x[target_indices]

        if sim == 'cos':
            target_norm = F.normalize(target_x, dim=-1)
            proto_norm = F.normalize(prototypes, dim=-1)
            logits = target_norm @ proto_norm.t()
        else:
            logits = target_x @ prototypes.t()

        return logits.argmax(dim=1), logits


def correct_and_smooth(adj, base_logits, train_idx, train_labels, num_classes,
                       num_iters=50, alpha=0.5):
    """
    Correct & Smooth: post-process feature-based predictions with label propagation.

    1. Start with base predictions from features (not zeros!)
    2. Propagate predictions through graph
    3. Clamp support set to ground truth at each step

    This combines feature information with graph structure.
    """
    device = adj.device()
    num_nodes = base_logits.size(0)

    # Compute normalized adjacency
    deg = adj.sum(dim=1).to_dense()
    deg_inv_sqrt = (deg + 1e-9).pow(-0.5)

    # Start with softmax of base logits (feature-based estimate)
    Y = F.softmax(base_logits, dim=-1)

    # Ground truth for support set
    Y_support = F.one_hot(train_labels.long(), num_classes=num_classes).float()

    # Propagate and clamp
    for _ in range(num_iters):
        # Propagate
        Y_new = deg_inv_sqrt.view(-1, 1) * Y
        Y_new = adj @ Y_new
        Y_new = deg_inv_sqrt.view(-1, 1) * Y_new

        # Blend with previous
        Y = (1 - alpha) * Y_new + alpha * Y

        # Clamp support set to ground truth (force truth to flow outward)
        Y[train_idx] = Y_support

    return Y


def apply_fast_pca_with_padding(features, target_dim, use_full_pca=False, preserve_norms=False):
    """
    Apply fast GPU PCA with padding (same logic as main pipeline).

    Args:
        features (torch.Tensor): Input features [N, D]
        target_dim (int): Target dimensionality after PCA
        use_full_pca (bool): Use full SVD instead of lowrank PCA
        preserve_norms (bool): Restore original L2 norms after PCA

    Returns:
        torch.Tensor: PCA-transformed and padded features [N, target_dim]
    """
    # Store original norms if preserving them
    if preserve_norms:
        original_norms = torch.norm(features, dim=1, p=2, keepdim=True)
    original_dim = features.size(1)
    num_nodes = features.size(0)
    max_pca_dim = min(num_nodes, original_dim)

    if original_dim >= target_dim:
        # Enough features, just PCA to target_dim
        pca_target_dim = min(target_dim, max_pca_dim)
    else:
        # Not enough features, PCA to all available then pad
        pca_target_dim = min(original_dim, max_pca_dim)

    # Apply PCA using same method as main pipeline
    if use_full_pca:
        U, S, V = torch.svd(features)
        U = U[:, :pca_target_dim]
        S = S[:pca_target_dim]
    else:
        U, S, V = torch.pca_lowrank(features, q=pca_target_dim)

    x_pca = torch.mm(U, torch.diag(S))

    # Padding if necessary (same logic as main pipeline)
    if x_pca.size(1) < target_dim:
        padding_size = target_dim - x_pca.size(1)
        # Use zero padding (can be extended to other strategies)
        padding = torch.zeros(x_pca.size(0), padding_size,
                            device=x_pca.device, dtype=x_pca.dtype)
        x_pca = torch.cat([x_pca, padding], dim=1)

    # Restore original norms if requested
    if preserve_norms:
        pca_norms = torch.norm(x_pca, dim=1, p=2, keepdim=True)
        pca_norms = pca_norms + 1e-9  # Avoid division by zero
        x_pca = x_pca * (original_norms / pca_norms)

    return x_pca


def few_shot_sample(labels, num_classes, k_shot, seed=42):
    """Sample k nodes per class for few-shot."""
    torch.manual_seed(seed)
    indices = []
    for c in range(num_classes):
        class_idx = (labels == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(class_idx))[:k_shot]
        indices.append(class_idx[perm])
    return torch.cat(indices)


def test_dataset(data, split_idx, num_classes, mode='full', k_shot=5, hops=2, sim='cos', device='cpu',
                 prop_method='residual', alpha=0.1, gcn_layer_norm=False, use_cs=False, cs_hops=50, cs_alpha=0.5,
                 use_ridge=False, ridge_alpha=1.0, feature_norm='none', dataset_name='',
                 use_pca=False, pca_dim=128, use_full_pca=False, pca_preserve_norms=False, num_runs=1, seeds=None):
    """Test zero-learning baseline on a dataset with multiple runs for averaging.

    Args:
        use_pca: Whether to apply PCA with padding for dimension unification
        pca_dim: Target dimension for PCA
        use_full_pca: Use full SVD instead of lowrank PCA
        pca_preserve_norms: Restore original L2 norms after PCA transformation
        num_runs: Number of runs with different random seeds
        seeds: List of seeds to use (if None, will use range based on first seed)
    """

    if seeds is None:
        # Default: use consecutive seeds starting from current random state
        import numpy as np
        base_seed = int(torch.initial_seed() % (2**31))
        seeds = [base_seed + i for i in range(num_runs)]

    # Store results from all runs
    all_results = []

    # Data loading and preprocessing (done once, independent of seed)
    # Ensure adj_t exists
    if not hasattr(data, 'adj_t') or data.adj_t is None:
        adj = SparseTensor.from_edge_index(
            data.edge_index,
            sparse_sizes=(data.num_nodes, data.num_nodes)
        ).to_symmetric()
    else:
        adj = data.adj_t

    test_idx = split_idx['test']
    x = data.x.to(device)
    adj = adj.to(device)
    test_idx_device = test_idx.to(device)
    target_y = data.y[test_idx].to(device)

    # Apply feature normalization (deterministic, done once)
    x = normalize_features(x, method=feature_norm)

    # Apply PCA BEFORE GCN propagation (matches main pipeline: PCA is during preprocessing, not after GCN)
    if use_pca:
        print(f'  Before PCA: {x.shape}')
        x = apply_fast_pca_with_padding(x, target_dim=pca_dim, use_full_pca=use_full_pca, preserve_norms=pca_preserve_norms)
        print(f'  After PCA+padding: {x.shape}')

        # IMPORTANT: Normalize after PCA (matches main pipeline behavior)
        batch_mean = x.mean(dim=0, keepdim=True)
        batch_std = x.std(dim=0, keepdim=True, unbiased=False)
        x = (x - batch_mean) / (batch_std + 1e-5)
        print(f'  Applied BatchNorm-style normalization after PCA')

    # GCN propagation (deterministic, done once outside loop)
    if hops == 0:
        x_prop_base = x
    else:
        x_prop_base = gcn_propagate(x, adj, num_hops=hops, method=prop_method, alpha=alpha,
                                    use_layer_norm=gcn_layer_norm)

    # Loop over different seeds
    for run_idx, seed in enumerate(seeds):
        torch.manual_seed(seed)
        import numpy as np
        np.random.seed(seed)

        # Get train indices for this seed (few-shot sampling)
        if mode == 'full':
            train_idx = split_idx['train']
        else:  # few-shot
            # Sample k_shot per class from train
            all_train = split_idx['train']
            train_labels = data.y[all_train]
            # Sample k_shot per class
            sampled = []
            for c in range(num_classes):
                class_mask = train_labels == c
                class_idx = all_train[class_mask]
                if len(class_idx) >= k_shot:
                    perm = torch.randperm(len(class_idx))[:k_shot]
                    sampled.append(class_idx[perm])
                else:
                    sampled.append(class_idx)  # Use all if not enough
            train_idx = torch.cat(sampled)

        context_y = data.y[train_idx].to(device)
        train_idx_device = train_idx.to(device)

        if run_idx == 0:  # Print info only for first run
            print(f'  Train: {len(train_idx)}, Test: {len(test_idx)}')
            print(f'  Samples per class: {[(context_y == c).sum().item() for c in range(num_classes)]}')
            if num_runs > 1:
                print(f'  Running {num_runs} times with seeds: {seeds}')

        # PCA is now applied before GCN propagation (see above), so just use the propagated features
        x_prop = x_prop_base

        # Get base predictions
        preds, _ = zero_learning_classify(x_prop, train_idx_device, context_y, test_idx_device, num_classes, sim,
                                          use_ridge=use_ridge,
                                          ridge_alpha=ridge_alpha)
        base_acc = (preds == target_y).float().mean().item()

        # If C&S is enabled, try it and use if better
        if use_cs:
            # Get logits for all nodes (needed for C&S)
            _, base_logits = zero_learning_classify(x_prop, train_idx_device, context_y,
                                                     torch.arange(data.num_nodes, device=device),
                                                     num_classes, sim,
                                                     use_ridge=use_ridge,
                                                     ridge_alpha=ridge_alpha)
            # Apply Correct & Smooth
            Y = correct_and_smooth(adj, base_logits, train_idx_device, context_y, num_classes,
                                   num_iters=cs_hops, alpha=cs_alpha)
            cs_preds = Y[test_idx_device].argmax(dim=1)
            cs_acc = (cs_preds == target_y).float().mean().item()

            # Use C&S if it improves performance
            if cs_acc > base_acc:
                final_acc = cs_acc
                used_cs = True
            else:
                final_acc = base_acc
                used_cs = False
            
            all_results.append({
                'base_acc': base_acc,
                'cs_acc': cs_acc,
                'final_acc': final_acc,
                'used_cs': used_cs
            })
        else:
            final_acc = base_acc
            all_results.append({
                'base_acc': base_acc,
                'final_acc': final_acc
            })

        if run_idx == 0 or num_runs <= 5:  # Print individual results for first run or if few runs
            method_str = f'{hops}-hop {prop_method} {sim.upper()}'
            if use_ridge:
                method_str += f' + ridge(α={ridge_alpha})'
            
            if use_cs:
                if all_results[-1]['used_cs']:
                    method_str += f' + C&S ({cs_hops} iters, α={cs_alpha}) [SELECTED]'
                    print(f'    Run {run_idx+1}: {method_str}: {final_acc:.4f} ({final_acc*100:.2f}%) [Base: {base_acc:.4f}]')
                else:
                    print(f'    Run {run_idx+1}: {method_str}: {final_acc:.4f} ({final_acc*100:.2f}%) [C&S: {cs_acc:.4f}, base better]')
            else:
                print(f'    Run {run_idx+1}: {method_str}: {final_acc:.4f} ({final_acc*100:.2f}%)')

    # Compute and display averaged results
    base_accs = [r['base_acc'] for r in all_results]
    final_accs = [r['final_acc'] for r in all_results]
    
    avg_base = sum(base_accs) / len(base_accs)
    avg_final = sum(final_accs) / len(final_accs)
    
    method_str = f'{hops}-hop {prop_method} {sim.upper()}'
    if use_ridge:
        method_str += f' + ridge(α={ridge_alpha})'
    
    if use_cs:
        cs_accs = [r.get('cs_acc', 0) for r in all_results]
        avg_cs = sum(cs_accs) / len(cs_accs)
        cs_used_count = sum(1 for r in all_results if r.get('used_cs', False))
        method_str += f' + C&S ({cs_hops} iters, α={cs_alpha})'
        print(f'  Average: {method_str}: {avg_final:.4f} ({avg_final*100:.2f}%) [Base: {avg_base:.4f}, C&S: {avg_cs:.4f}, C&S used: {cs_used_count}/{num_runs}]')
    else:
        print(f'  Average: {method_str}: {avg_final:.4f} ({avg_final*100:.2f}%)')

    return avg_final


def main():
    parser = argparse.ArgumentParser(description='Zero-Learning Baseline Test')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['Actor', 'AirBrazil', 'AirEU', 'AirUS', 'AmzComp', 'AmzPhoto', 'AmzRatings',
                                 'BlogCatalog', 'Chameleon', 'Citeseer', 'CoCS', 'CoPhysics', 'Cora', 'Cornell',
                                 'DBLP', 'Deezer', 'LastFMAsia', 'Minesweeper', 'Pubmed', 'Questions', 'Reddit',
                                 'Roman', 'Squirrel', 'Texas', 'Tolokers', 'Wiki', 'Wisconsin', 'WikiCS',
                                 'ogbn-arxiv', 'ogbn-products', 'FullCora'],
                        help='Datasets to test')
    parser.add_argument('--mode', type=str, choices=['full', 'few-shot'], default='few-shot',
                        help='full-shot or few-shot mode')
    parser.add_argument('--k_shot', type=int, default=5,
                        help='Number of shots per class (for few-shot mode)')
    parser.add_argument('--hops', type=int, default=3,
                        help='Number of GCN propagation hops')
    parser.add_argument('--sim', type=str, choices=['cos', 'dot'], default='cos',
                        help='Similarity function')
    parser.add_argument('--prop_method', type=str, choices=['puregcn', 'residual', 'appnp', 'concat', 'weighted'], default='residual',
                        help='Feature propagation method (puregcn matches main pipeline)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Teleport probability for APPNP')
    parser.add_argument('--gcn_layer_norm', type=str2bool, default=False,
                        help='Apply LayerNorm after each GCN propagation step')
    parser.add_argument('--use_cs', type=str2bool, default=True,
                        help='Try Correct & Smooth and use if it improves performance')
    parser.add_argument('--cs_hops', type=int, default=50,
                        help='Number of iterations for C&S')
    parser.add_argument('--cs_alpha', type=float, default=0.5,
                        help='Alpha for C&S')
    parser.add_argument('--ridge', type=str2bool, default=False,
                        help='Use ridge regression (closed-form linear classifier)')
    parser.add_argument('--ridge_alpha', type=float, default=1.0,
                        help='Regularization strength for ridge regression')
    parser.add_argument('--feature_norm', type=str, choices=['none', 'row', 'col', 'row+col'], default='none',
                        help='Feature normalization (row=L2 per node, col=standardize per feature)')
    parser.add_argument('--use_pca', type=str2bool, default=False,
                        help='Apply PCA with padding for dimension unification')
    parser.add_argument('--pca_dim', type=int, default=128,
                        help='Target dimension for PCA (with zero-padding if needed)')
    parser.add_argument('--use_full_pca', type=str2bool, default=False,
                        help='Use full SVD instead of lowrank PCA')
    parser.add_argument('--pca_preserve_norms', type=str2bool, default=False,
                        help='Restore original L2 norms after PCA transformation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for few-shot sampling')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device number (use -1 for CPU)')
    parser.add_argument('--wandb', type=str2bool, default=True,
                        help='Enable Weights & Biases logging')
    parser.add_argument('--num_runs', type=int, default=3,
                        help='Number of runs with different seeds for averaging')
    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'

    # Initialize wandb if enabled
    if args.wandb:
        # Create run name based on configuration
        method_parts = [args.prop_method, f'{args.hops}hop', args.sim]
        if args.use_pca:
            pca_type = 'fullPCA' if args.use_full_pca else 'PCA'
            pca_name = f'{pca_type}{args.pca_dim}'
            if args.pca_preserve_norms:
                pca_name += 'NormPres'
            method_parts.append(pca_name)
        if args.ridge:
            method_parts.append(f'ridge{args.ridge_alpha}')
        if args.feature_norm != 'none':
            method_parts.append(f'norm-{args.feature_norm}')

        run_name = f"{args.mode}_{'_'.join(method_parts)}"

        wandb.init(
            project='zero-learning-baseline',
            name=run_name,
            config=vars(args)
        )

    print('='*60)
    print('ZERO-LEARNING BASELINE: Pure GNN Structure Heuristic')
    print(f'Mode: {args.mode}' + (f' (k={args.k_shot})' if args.mode == 'few-shot' else ''))
    print(f'Similarity: {args.sim.upper()}, Device: {device}')
    print('='*60)

    torch.manual_seed(args.seed)
    all_results = {}

    for dataset_name in args.datasets:
        print(f'\n{"="*60}')
        print(f'Dataset: {dataset_name}')
        print('='*60)

        try:
            # Handle OGB datasets separately
            if dataset_name.startswith('ogbn-'):
                data, split_idx = load_ogbn_data(dataset_name)
            else:
                data, split_idx = load_data(dataset_name)
            num_classes = data.y.max().item() + 1
            print(f'Nodes: {data.num_nodes}, Features: {data.num_features}, Classes: {num_classes}')

            acc = test_dataset(
                data, split_idx, num_classes,
                mode=args.mode, k_shot=args.k_shot,
                hops=args.hops, sim=args.sim, device=device,
                prop_method=args.prop_method, alpha=args.alpha,
                gcn_layer_norm=args.gcn_layer_norm,
                use_cs=args.use_cs, cs_hops=args.cs_hops, cs_alpha=args.cs_alpha,
                use_ridge=args.ridge, ridge_alpha=args.ridge_alpha,
                feature_norm=args.feature_norm,
                dataset_name=dataset_name,
                use_pca=args.use_pca, pca_dim=args.pca_dim, use_full_pca=args.use_full_pca,
                pca_preserve_norms=args.pca_preserve_norms,
                num_runs=args.num_runs
            )
            all_results[dataset_name] = acc

            # Log individual dataset result to wandb
            if args.wandb:
                wandb.log({f'acc/{dataset_name}': acc})

        except Exception as e:
            print(f'  Error loading {dataset_name}: {e}')
            continue

    # Summary table
    if all_results:
        print(f'\n{"="*60}')
        method_desc = []
        if args.feature_norm != 'none':
            method_desc.append(f'feat-{args.feature_norm}')
        method_desc.append(f'{args.prop_method}')
        if args.use_pca:
            pca_type = 'fullPCA' if args.use_full_pca else 'PCA'
            pca_desc = f'{pca_type}{args.pca_dim}'
            if args.pca_preserve_norms:
                pca_desc += '+NormPres'
            method_desc.append(pca_desc)
        if args.use_cs:
            method_desc.append('C&S')
        if args.ridge:
            method_desc.append(f'ridge({args.ridge_alpha})')
        print(f'SUMMARY ({"+".join(method_desc)}, {args.hops}-hop, {args.sim.upper()}, {args.mode})')
        print('='*60)
        print(f'{"Dataset":<15} {"Accuracy":>10}')
        print('-'*25)
        for name, acc in all_results.items():
            print(f'{name:<15} {acc*100:>9.2f}%')
        print('-'*25)
        avg_acc = sum(all_results.values()) / len(all_results)
        print(f'{"Average":<15} {avg_acc*100:>9.2f}%')

        # Log average accuracy to wandb
        if args.wandb:
            wandb.log({'acc/average': avg_acc})
            wandb.finish()


if __name__ == '__main__':
    main()
