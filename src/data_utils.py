import torch
import torch.nn.functional as F
import time
import numpy as np
from sklearn.decomposition import IncrementalPCA
from torch_geometric.utils import negative_sampling
from torch_sparse import SparseTensor

def apply_incremental_pca_cpu(data_tensor, target_dim, batch_size=10000, sign_normalize=False, rank=0):
    if rank == 0:
        print(f'[Incremental PCA] Starting CPU-based PCA: {data_tensor.shape} -> {target_dim} components')
    start_time = time.time()
    n_samples, n_features = data_tensor.shape
    if data_tensor.is_cuda:
        gpu_to_cpu_start = time.time()
        data_cpu_tensor = data_tensor.cpu()
        gpu_to_cpu_time = time.time() - gpu_to_cpu_start
        if rank == 0:
            print(f'[Incremental PCA] GPU→CPU transfer: {gpu_to_cpu_time:.2f}s')
    else:
        data_cpu_tensor = data_tensor
        gpu_to_cpu_time = 0
    data_numpy = data_cpu_tensor.numpy()
    if rank == 0:
        print(f'[Incremental PCA] Memory usage: {data_numpy.nbytes / 1000000000.0:.2f}GB')
    pca = IncrementalPCA(n_components=target_dim, batch_size=batch_size)
    fit_start = time.time()
    n_batches = (n_samples + batch_size - 1) // batch_size
    if rank == 0:
        print(f'[Incremental PCA] Fitting PCA with {n_batches} batches of size {batch_size}')
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch = data_numpy[start_idx:end_idx]
        pca.partial_fit(batch)
        if rank == 0 and (batch_idx + 1) % max(1, n_batches // 10) == 0:
            progress = (batch_idx + 1) / n_batches * 100
            print(f'[Incremental PCA] Fit progress: {progress:.1f}% ({batch_idx + 1}/{n_batches})')
    fit_time = time.time() - fit_start
    transform_start = time.time()
    transformed_batches = []
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch = data_numpy[start_idx:end_idx]
        transformed_batch = pca.transform(batch)
        transformed_batches.append(transformed_batch)
    data_transformed = np.concatenate(transformed_batches, axis=0)
    transform_time = time.time() - transform_start
    if sign_normalize:
        sign_norm_start = time.time()
        components = pca.components_
        singular_values = np.sqrt(pca.explained_variance_)
        for i in range(target_dim):
            feature_vector = components[i] * singular_values[i]
            max_idx = np.argmax(np.abs(feature_vector))
            if feature_vector[max_idx] < 0:
                data_transformed[:, i] = -data_transformed[:, i]
        sign_norm_time = time.time() - sign_norm_start
        if rank == 0:
            print(f'[Incremental PCA] Sign normalization: {sign_norm_time:.2f}s')
    result_tensor = torch.from_numpy(data_transformed).to(dtype=data_tensor.dtype)
    total_time = time.time() - start_time
    if rank == 0:
        print(f'[Incremental PCA] Performance Summary:')
        if gpu_to_cpu_time > 0:
            print(f'  - GPU→CPU: {gpu_to_cpu_time:.2f}s')
        print(f'  - PCA Fit: {fit_time:.2f}s ({n_batches} batches)')
        print(f'  - PCA Transform: {transform_time:.2f}s')
        print(f'  - Total: {total_time:.2f}s')
        print(f'  - Result shape: {result_tensor.shape} (CPU)')
        print(f'  - Explained variance ratio: {pca.explained_variance_ratio_[:min(5, target_dim)].sum():.4f} (first {min(5, target_dim)} components)')
    return result_tensor

def select_k_shot_context(data, k, index):
    classes = data.y.unique()
    context_samples = []
    mask = torch.zeros(data.y.size(0), dtype=torch.bool, device=data.y.device)
    mask[index] = True
    for c in classes:
        class_mask = (data.y == c) & mask
        class_indices = torch.nonzero(class_mask, as_tuple=False).squeeze()
        if class_indices.numel() == 0:
            print(f'Warning: Class {c} has no training examples, skipping...')
            continue
        if class_indices.dim() == 0:
            class_indices = class_indices.unsqueeze(0)
        num_available = class_indices.size(0)
        num_to_select = min(k, num_available)
        selected_indices = torch.randperm(num_available)[:num_to_select]
        selected_indices = class_indices[selected_indices]
        context_samples.append(selected_indices)
    if len(context_samples) == 0:
        return torch.tensor([], dtype=torch.long, device=data.y.device)
    context_samples = torch.cat(context_samples)
    return context_samples

def process_data(data, split_idx, hidden, context_num, sign_normalize=False, use_full_pca=False, normalize_data=False, rank=0, use_batchnorm=False, pca_device='gpu', incremental_pca_batch_size=10000, process_test_only=False, is_augmented_data=False):
    device = data.x.device
    split_idx['train'] = split_idx['train'].to(device)
    split_idx['valid'] = split_idx['valid'].to(device)
    split_idx['test'] = split_idx['test'].to(device)
    if len(split_idx['train']) > 0:
        context_source_split = split_idx['train']
    else:
        context_source_split = split_idx['test']
    data.context_sample = select_k_shot_context(data, context_num, context_source_split)
    data.context_sample = data.context_sample.to(device)
    st = time.time()
    data.is_augmented_data = bool(is_augmented_data)
    if process_test_only:
        context_indices = data.context_sample
        test_indices = split_idx['test']
        process_indices = torch.unique(torch.cat([context_indices, test_indices]))
        if rank == 0:
            print(f'[Test-Only Mode] Processing {len(process_indices)} nodes (context={len(context_indices)}, test={len(test_indices)})')
            print(f'[Test-Only Mode] Skipping {data.x.size(0) - len(process_indices)} nodes (train/valid)')
    else:
        process_indices = None
    full_features = data.x
    if process_indices is not None:
        input_features = full_features[process_indices]
        if rank == 0:
            print(f'[Test-Only Mode] Extracted subset: {full_features.shape} -> {input_features.shape}')
    else:
        input_features = full_features
    data.uses_fug_embeddings = False
    original_dim = input_features.size(1)
    num_nodes = input_features.size(0)
    max_pca_dim = min(num_nodes, original_dim)
    if original_dim >= hidden:
        pca_target_dim = min(hidden, max_pca_dim)
        data.needs_projection = False
        if rank == 0:
            print(f'Dataset {data.name}: Sufficient features ({original_dim} >= {hidden}), PCA to {pca_target_dim}')
    else:
        pca_target_dim = min(original_dim, max_pca_dim)
        data.needs_projection = False
        if rank == 0:
            print(f'Dataset {data.name}: Using zero padding ({original_dim} -> zero-pad to {hidden})')
    data.x_original = input_features.clone()
    if pca_device == 'cpu':
        pca_start = time.time()
        data.x_pca = apply_incremental_pca_cpu(input_features, target_dim=pca_target_dim, batch_size=incremental_pca_batch_size, sign_normalize=sign_normalize, rank=rank)
        pca_time = time.time() - pca_start
        if rank == 0:
            print(f'Dataset {data.name}: Incremental PCA completed in {pca_time:.2f}s (result on CPU)')
    else:
        if use_full_pca:
            U, S, V = torch.svd(input_features)
            U = U[:, :pca_target_dim]
            S = S[:pca_target_dim]
        else:
            U, S, V = torch.pca_lowrank(input_features, q=pca_target_dim)
        if sign_normalize:
            for i in range(pca_target_dim):
                feature_vector = U[:, i] * S[i]
                max_idx = torch.argmax(torch.abs(feature_vector))
                if feature_vector[max_idx] < 0:
                    U[:, i] = -U[:, i]
        data.x_pca = torch.mm(U, torch.diag(S))
    if pca_device != 'cpu':
        data.x_pca = data.x_pca.to(device)
    elif rank == 0:
        print(f'Dataset {data.name}: Keeping Incremental PCA result on CPU ({data.x_pca.shape})')
    if data.needs_projection:
        data.x = data.x_pca
        data.needs_final_pca = True
    else:
        if data.x_pca.size(1) < hidden:
            padding_size = hidden - data.x_pca.size(1)
            pca_device = data.x_pca.device
            padding = torch.zeros(data.x_pca.size(0), padding_size, device=pca_device)
            data.x = torch.cat([data.x_pca, padding], dim=1)
            if rank == 0:
                print(f'Dataset {data.name}: Applied zero padding ({data.x_pca.size(1)} -> {hidden}) on {pca_device}')
        else:
            data.x = data.x_pca
        data.needs_final_pca = False
    if process_indices is not None:
        full_processed = torch.zeros(full_features.size(0), data.x.size(1), device=device, dtype=data.x.dtype)
        full_processed[process_indices] = data.x
        data.x = full_processed
        if rank == 0:
            print(f'[Test-Only Mode] Scattered processed features back to full array: {data.x.shape}')
    if normalize_data:
        if use_batchnorm:
            batch_mean = data.x.mean(dim=0, keepdim=True)
            batch_std = data.x.std(dim=0, keepdim=True, unbiased=False)
            data.x = (data.x - batch_mean) / (batch_std + 1e-05)
            if rank == 0:
                print(f'Dataset {data.name}: Applied BatchNorm-style normalization')
        else:
            data.x = F.normalize(data.x, p=2, dim=1)
            if rank == 0:
                print(f'Dataset {data.name}: Applied LayerNorm-style normalization')
    if rank == 0:
        print(f'PCA time: {time.time() - st:.2f}s', flush=True)

def prepare_link_data(data, split_idx, negative_ratio=1.0):
    device = data.x.device
    result = {}
    if 'train' in split_idx and 'edge' in split_idx['train']:
        pos_train_edge = split_idx['train']['edge'].to(device)
        num_pos_train = pos_train_edge.size(0)
        num_neg_train = int(num_pos_train * negative_ratio)
        row, col, _ = data.adj_t.to_symmetric().coo()
        edge_index = torch.stack([row, col], dim=0)
        try:
            neg_train_edge = negative_sampling(edge_index=edge_index, num_nodes=data.num_nodes, num_neg_samples=num_neg_train, method='sparse').t().to(device)
            train_edges = torch.cat([pos_train_edge, neg_train_edge], dim=0)
            train_labels = torch.cat([torch.ones(num_pos_train, device=device, dtype=torch.long), torch.zeros(neg_train_edge.size(0), device=device, dtype=torch.long)], dim=0)
            result['train'] = {'edge_pairs': train_edges, 'labels': train_labels}
        except Exception as e:
            print(f'ERROR in prepare_link_data during negative sampling: {e}')
            raise e
    for split in ['valid', 'test']:
        if split in split_idx and 'edge' in split_idx[split] and ('edge_neg' in split_idx[split]):
            pos_edge = split_idx[split]['edge'].to(device)
            neg_edge = split_idx[split]['edge_neg'].to(device)
            edges = torch.cat([pos_edge, neg_edge], dim=0)
            labels = torch.cat([torch.ones(pos_edge.size(0), device=device, dtype=torch.long), torch.zeros(neg_edge.size(0), device=device, dtype=torch.long)], dim=0)
            result[split] = {'edge_pairs': edges, 'labels': labels}
    return result

def select_link_context(train_link_data, k, context_neg_ratio=1.0, remove_from_train=False):
    """
    Select a balanced subset of positive and negative edges for the context set.
    """
    edge_pairs = train_link_data['edge_pairs']
    labels = train_link_data['labels']
    pos_indices = torch.where(labels == 1)[0]
    neg_indices = torch.where(labels == 0)[0]
    num_pos_to_select = min(k, len(pos_indices))
    if num_pos_to_select == 0:
        print(f'WARNING: No positive edges available for context selection!')
        return ({'edge_pairs': torch.empty((0, 2), dtype=edge_pairs.dtype, device=edge_pairs.device), 'labels': torch.empty((0,), dtype=labels.dtype, device=labels.device)}, torch.ones_like(labels, dtype=torch.bool))
    selected_pos_indices = pos_indices[torch.randperm(len(pos_indices))[:num_pos_to_select]]
    num_neg_to_select = min(int(k * context_neg_ratio), len(neg_indices))
    if num_neg_to_select == 0:
        print(f'WARNING: No negative edges available for context selection!')
        return ({'edge_pairs': torch.empty((0, 2), dtype=edge_pairs.dtype, device=edge_pairs.device), 'labels': torch.empty((0,), dtype=labels.dtype, device=labels.device)}, torch.ones_like(labels, dtype=torch.bool))
    selected_neg_indices = neg_indices[torch.randperm(len(neg_indices))[:num_neg_to_select]]
    context_indices = torch.cat([selected_pos_indices, selected_neg_indices])
    context_data = {'edge_pairs': edge_pairs[context_indices], 'labels': labels[context_indices]}
    train_mask = torch.ones_like(labels, dtype=torch.bool)
    if remove_from_train:
        train_mask[context_indices] = False
    return (context_data, train_mask)

def process_link_data(data, args, rank=0):
    """
    Dedicated data processing for link prediction.
    Applies the same feature transformations (PCA, projection, padding, normalization)
    as the node classification script, but avoids node-specific logic like context sampling.
    """
    device = data.x.device
    st = time.time()
    hidden_dim = args.hidden
    use_identity_projection = False
    projection_small_dim = args.projection_small_dim
    projection_large_dim = args.projection_large_dim
    use_full_pca = args.use_full_pca
    normalize_data = args.normalize_data
    use_batchnorm = args.use_batchnorm
    padding_strategy = args.padding_strategy
    use_random_orthogonal = False
    use_quantile_normalization = False
    quantile_norm_before_padding = getattr(args, 'quantile_norm_before_padding', True)
    lp_orthogonal_noise_dim = max(1, int(getattr(args, 'lp_orthogonal_noise_dim', 128)))
    force_lp_orthogonal_noise = bool(getattr(data, 'lp_force_orthogonal_noise', False))
    force_noise_dim = getattr(data, 'lp_force_orthogonal_noise_dim', None)
    if force_noise_dim is not None:
        lp_orthogonal_noise_dim = max(1, int(force_noise_dim))
    original_dim = data.x.size(1)
    if original_dim >= hidden_dim:
        pca_target_dim = hidden_dim
    else:
        pca_target_dim = original_dim
    if use_full_pca:
        U, S, V = torch.svd(data.x)
        U = U[:, :pca_target_dim]
        S = S[:pca_target_dim]
        data.x_pca = torch.mm(U, torch.diag(S)).to(device)
    else:
        U, S, V = torch.pca_lowrank(data.x, q=pca_target_dim)
        U = U[:, :pca_target_dim]
        S = S[:pca_target_dim]
        data.x_pca = torch.mm(U, torch.diag(S)).to(device)
    if data.x_pca.size(1) < hidden_dim:
        padding_size = hidden_dim - data.x_pca.size(1)
        padding = torch.zeros(data.x_pca.size(0), padding_size, device=device)
        data.x = torch.cat([data.x_pca, padding], dim=1)
        if rank == 0:
            print(f'Dataset {data.name}: Applied zero padding ({data.x_pca.size(1)} -> {hidden_dim})')
    else:
        data.x = data.x_pca
    if normalize_data and True:
        if use_batchnorm:
            batch_mean = data.x.mean(dim=0, keepdim=True)
            batch_std = data.x.std(dim=0, keepdim=True, unbiased=False)
            data.x = (data.x - batch_mean) / (batch_std + 1e-05)
        else:
            data.x = F.normalize(data.x, p=2, dim=1)
    if rank == 0:
        print(f'Feature processing time: {time.time() - st:.2f}s', flush=True)

def edge_dropout_sparse_tensor(adj_t, dropout_rate, training=True, verbose=False):
    if not training or dropout_rate <= 0.0:
        return adj_t
    if verbose:
        start_time = time.time()
    row, col, edge_attr = adj_t.coo()
    num_edges = row.size(0)
    keep_mask = torch.rand(num_edges, device=row.device) > dropout_rate
    row_kept = row[keep_mask]
    col_kept = col[keep_mask]
    edge_attr_kept = edge_attr[keep_mask] if edge_attr is not None else None
    result = SparseTensor(row=row_kept, col=col_kept, value=edge_attr_kept, sparse_sizes=adj_t.sparse_sizes())
    if verbose:
        elapsed = (time.time() - start_time) * 1000
        edges_removed = num_edges - row_kept.size(0)
        print(f'[Edge Dropout] {edges_removed}/{num_edges} edges removed ({dropout_rate:.1%}) in {elapsed:.2f}ms')
    return result

def edge_dropout_edge_index(edge_index, edge_attr=None, dropout_rate=0.1, training=True):
    if not training or dropout_rate <= 0.0:
        return (edge_index, edge_attr)
    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges, device=edge_index.device) > dropout_rate
    edge_index_dropped = edge_index[:, keep_mask]
    edge_attr_dropped = None
    if edge_attr is not None:
        edge_attr_dropped = edge_attr[keep_mask]
    return (edge_index_dropped, edge_attr_dropped)

def batch_edge_dropout(batch, dropout_rate, training=True):
    if not training or dropout_rate <= 0.0:
        return batch
    batch_dropped = batch.clone()
    edge_index_dropped, edge_attr_dropped = edge_dropout_edge_index(batch.edge_index, getattr(batch, 'edge_attr', None), dropout_rate, training)
    batch_dropped.edge_index = edge_index_dropped
    if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
        batch_dropped.edge_attr = edge_attr_dropped
    return batch_dropped

def feature_dropout(x, dropout_rate, training=True, dropout_type='element_wise', verbose=False):
    if not training or dropout_rate <= 0.0:
        return x
    if verbose:
        start_time = time.time()
    if dropout_type == 'element_wise':
        result = torch.nn.functional.dropout(x, p=dropout_rate, training=True)
    elif dropout_type == 'channel_wise':
        num_features = x.size(1)
        keep_mask = torch.rand(num_features, device=x.device) > dropout_rate
        result = x * keep_mask.unsqueeze(0)
    elif dropout_type == 'gaussian_noise':
        if x.numel() > 0:
            noise_std = dropout_rate * x.std()
            noise = torch.randn_like(x) * noise_std
            result = x + noise
        else:
            result = x
    else:
        raise ValueError(f"Unknown dropout_type: {dropout_type}. Choose 'element_wise', 'channel_wise', or 'gaussian_noise'")
    if verbose:
        elapsed = (time.time() - start_time) * 1000
        if dropout_type == 'gaussian_noise':
            noise_std = dropout_rate * x.std() if x.numel() > 0 else 0.0
            print(f'[Feature Dropout] {dropout_type}: noise_std={noise_std:.4f} in {elapsed:.2f}ms')
        elif dropout_type == 'channel_wise':
            num_features = x.size(1)
            keep_mask = torch.rand(num_features, device=x.device) > dropout_rate
            features_kept = keep_mask.float().mean().item()
            print(f'[Feature Dropout] {dropout_type}: {features_kept:.1%} feature dims kept ({dropout_rate:.1%} dropout) in {elapsed:.2f}ms')
        else:
            non_zero_ratio = (result != 0).float().mean().item()
            print(f'[Feature Dropout] {dropout_type}: {non_zero_ratio:.1%} elements kept in {elapsed:.2f}ms')
    return result

def apply_random_projection_augmentation(data, hidden_dim_range=None, activation_pool=None, seed=None, verbose=False, rank=0, max_depth=1, dropout_rate=0.0, max_hidden_dim=None):
    if activation_pool is None:
        activation_pool = ['relu', 'gelu', 'silu', 'elu', 'tanh', 'sigmoid', 'softplus', 'softsign', 'leaky_relu', 'sin', 'cos', 'squared', 'abs']
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    device = data.x.device
    num_nodes, input_dim = data.x.shape
    if max_depth > 1:
        depth = np.random.randint(1, max_depth + 1)
    else:
        depth = 1
    if hidden_dim_range is None:
        min_dim = max(int(input_dim * 0.5), 32)
        max_dim = max(int(input_dim * 2.0), min_dim)
    else:
        min_dim, max_dim = hidden_dim_range
        if max_dim < min_dim:
            raise ValueError(f'hidden_dim_range must satisfy min_dim <= max_dim, got ({min_dim}, {max_dim})')
    if max_hidden_dim is not None and max_hidden_dim > 0:
        if max_dim > max_hidden_dim:
            max_dim = max_hidden_dim
        if min_dim > max_dim:
            min_dim = max_dim
    if rank == 0 and verbose:
        if depth == 1:
            print(f'[Random Projection Augmentation] Applying single layer σ(WX + b)')
        else:
            print(f'[Random Projection Augmentation] Applying {depth}-layer MLP')
        print(f'  Input shape: {data.x.shape}')
    x_current = data.x
    current_dim = input_dim
    activation_names = []
    for layer_idx in range(depth):
        if layer_idx == depth - 1:
            hidden_dim = np.random.randint(min_dim, max_dim + 1)
        else:
            hidden_dim = np.random.randint(min_dim, max_dim + 1)
        if dropout_rate > 0.0:
            dropout_mask = torch.rand_like(x_current) > dropout_rate
            x_current = x_current * dropout_mask / (1.0 - dropout_rate)
        W = torch.randn(current_dim, hidden_dim, device=device, dtype=data.x.dtype)
        std = np.sqrt(2.0 / (current_dim + hidden_dim))
        W = W * std
        b = torch.randn(hidden_dim, device=device, dtype=data.x.dtype) * 0.1
        x_projected = torch.matmul(x_current, W) + b
        activation_name = np.random.choice(activation_pool)
        activation_names.append(activation_name)
        if activation_name == 'relu':
            x_current = F.relu(x_projected)
        elif activation_name == 'gelu':
            x_current = F.gelu(x_projected)
        elif activation_name == 'tanh':
            x_current = torch.tanh(x_projected)
        elif activation_name == 'silu':
            x_current = F.silu(x_projected)
        elif activation_name == 'elu':
            x_current = F.elu(x_projected)
        elif activation_name == 'sigmoid':
            x_current = torch.sigmoid(x_projected)
        elif activation_name == 'softplus':
            x_current = F.softplus(x_projected)
        elif activation_name == 'softsign':
            x_current = F.softsign(x_projected)
        elif activation_name == 'leaky_relu':
            negative_slope = 0.01 + torch.rand(1).item() * 0.29
            x_current = F.leaky_relu(x_projected, negative_slope=negative_slope)
        elif activation_name == 'sin':
            x_current = torch.sin(x_projected)
        elif activation_name == 'cos':
            x_current = torch.cos(x_projected)
        elif activation_name == 'squared':
            x_current = x_projected ** 2
        elif activation_name == 'abs':
            x_current = torch.abs(x_projected)
        elif activation_name == 'identity':
            x_current = x_projected
        else:
            raise ValueError(f'Unsupported activation function: {activation_name}')
        current_dim = hidden_dim
        if rank == 0 and verbose:
            dropout_info = f', dropout={dropout_rate}' if dropout_rate > 0.0 else ''
            print(f'  Layer {layer_idx + 1}: {activation_name}, dim={hidden_dim}{dropout_info}')
    x_augmented = x_current
    if rank == 0 and verbose and (depth > 1):
        print(f"  Activation sequence: {' -> '.join(activation_names)}")
    data_augmented = data.clone()
    data_augmented.x = x_augmented
    if rank == 0 and verbose:
        print(f'  Output shape: {x_augmented.shape}')
        print(f'  Original feature stats: mean={data.x.mean():.4f}, std={data.x.std():.4f}')
        print(f'  Augmented feature stats: mean={x_augmented.mean():.4f}, std={x_augmented.std():.4f}')
    return data_augmented
