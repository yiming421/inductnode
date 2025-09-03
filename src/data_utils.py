import torch
import torch.nn.functional as F
import time
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from torch_geometric.utils import negative_sampling

def apply_incremental_pca_cpu(data_tensor, target_dim, batch_size=10000, sign_normalize=False, rank=0):
    """
    Apply Incremental PCA on CPU with performance monitoring.
    Keeps everything on CPU to avoid wasteful GPU↔CPU transfers.
    
    Args:
        data_tensor (torch.Tensor): Input data tensor [n_samples, n_features] (can be GPU or CPU)
        target_dim (int): Target number of PCA components
        batch_size (int): Batch size for incremental processing
        sign_normalize (bool): Whether to normalize eigenvector signs
        rank (int): Process rank for logging
    
    Returns:
        torch.Tensor: PCA-transformed data [n_samples, target_dim] on CPU
    """
    if rank == 0:
        print(f"[Incremental PCA] Starting CPU-based PCA: {data_tensor.shape} -> {target_dim} components")
    
    start_time = time.time()
    n_samples, n_features = data_tensor.shape
    
    # Move to CPU if needed (only once)
    if data_tensor.is_cuda:
        gpu_to_cpu_start = time.time()
        data_cpu_tensor = data_tensor.cpu()
        gpu_to_cpu_time = time.time() - gpu_to_cpu_start
        if rank == 0:
            print(f"[Incremental PCA] GPU→CPU transfer: {gpu_to_cpu_time:.2f}s")
    else:
        data_cpu_tensor = data_tensor
        gpu_to_cpu_time = 0
    
    # Convert to numpy for sklearn
    data_numpy = data_cpu_tensor.numpy()
    if rank == 0:
        print(f"[Incremental PCA] Memory usage: {data_numpy.nbytes / 1e9:.2f}GB")
    
    # Initialize Incremental PCA
    pca = IncrementalPCA(n_components=target_dim, batch_size=batch_size)
    
    # Fit PCA in batches
    fit_start = time.time()
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    if rank == 0:
        print(f"[Incremental PCA] Fitting PCA with {n_batches} batches of size {batch_size}")
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch = data_numpy[start_idx:end_idx]
        
        pca.partial_fit(batch)
        
        if rank == 0 and (batch_idx + 1) % max(1, n_batches // 10) == 0:
            progress = (batch_idx + 1) / n_batches * 100
            print(f"[Incremental PCA] Fit progress: {progress:.1f}% ({batch_idx + 1}/{n_batches})")
    
    fit_time = time.time() - fit_start
    
    # Transform in batches
    transform_start = time.time()
    transformed_batches = []
    
    for batch_idx in range(n_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, n_samples)
        batch = data_numpy[start_idx:end_idx]
        
        transformed_batch = pca.transform(batch)
        transformed_batches.append(transformed_batch)
    
    # Concatenate all transformed batches
    data_transformed = np.concatenate(transformed_batches, axis=0)
    transform_time = time.time() - transform_start
    
    # Apply sign normalization if requested
    if sign_normalize:
        sign_norm_start = time.time()
        components = pca.components_  # Shape: [target_dim, n_features]
        singular_values = np.sqrt(pca.explained_variance_)
        
        for i in range(target_dim):
            # Reconstruct the feature vector: component * singular_value
            feature_vector = components[i] * singular_values[i]  
            max_idx = np.argmax(np.abs(feature_vector))
            if feature_vector[max_idx] < 0:
                # Flip the sign of the transformed data for this component
                data_transformed[:, i] = -data_transformed[:, i]
        
        sign_norm_time = time.time() - sign_norm_start
        if rank == 0:
            print(f"[Incremental PCA] Sign normalization: {sign_norm_time:.2f}s")
    
    # Convert back to CPU torch tensor (no GPU transfer!)
    result_tensor = torch.from_numpy(data_transformed).to(dtype=data_tensor.dtype)
    
    total_time = time.time() - start_time
    
    if rank == 0:
        print(f"[Incremental PCA] Performance Summary:")
        if gpu_to_cpu_time > 0:
            print(f"  - GPU→CPU: {gpu_to_cpu_time:.2f}s")  
        print(f"  - PCA Fit: {fit_time:.2f}s ({n_batches} batches)")
        print(f"  - PCA Transform: {transform_time:.2f}s")
        print(f"  - Total: {total_time:.2f}s")
        print(f"  - Result shape: {result_tensor.shape} (CPU)")
        print(f"  - Explained variance ratio: {pca.explained_variance_ratio_[:min(5, target_dim)].sum():.4f} (first {min(5, target_dim)} components)")
    
    return result_tensor

def select_k_shot_context(data, k, index):
    classes = data.y.unique()
    context_samples = []
    mask = torch.zeros(data.y.size(0), dtype=torch.bool, device=data.y.device)
    mask[index] = True  # only select context from training set
    for c in classes:
        class_mask = (data.y == c) & mask
        class_indices = torch.nonzero(class_mask, as_tuple=False).squeeze()
        
        # Handle case where class has no training examples
        if class_indices.numel() == 0:
            print(f"Warning: Class {c} has no training examples, skipping...")
            continue
        
        # Handle case where class_indices is 0-dimensional (single element)
        if class_indices.dim() == 0:
            class_indices = class_indices.unsqueeze(0)
        
        # Select k samples (or all if fewer than k available)
        num_available = class_indices.size(0)
        num_to_select = min(k, num_available)
        selected_indices = torch.randperm(num_available)[:num_to_select]
        selected_indices = class_indices[selected_indices]
        context_samples.append(selected_indices)
    
    if len(context_samples) == 0:
        # If no classes have training examples, return empty tensor
        return torch.tensor([], dtype=torch.long, device=data.y.device)
    
    context_samples = torch.cat(context_samples)
    return context_samples

def process_data(data, split_idx, hidden, context_num, sign_normalize=False, use_full_pca=False, 
                 normalize_data=False, use_projector=False, min_pca_dim=32, rank=0, 
                 padding_strategy='zero', use_batchnorm=False, use_identity_projection=False, 
                 projection_small_dim=128, projection_large_dim=256, pca_device='gpu',
                 incremental_pca_batch_size=10000):
    device = data.x.device
    split_idx['train'] = split_idx['train'].to(device)
    split_idx['valid'] = split_idx['valid'].to(device)
    split_idx['test'] = split_idx['test'].to(device)
    data.context_sample = select_k_shot_context(data, context_num, split_idx['train'])
    data.context_sample = data.context_sample.to(device)

    st = time.time()
    
    # Identity projection approach - PCA to small_dim, then project to large_dim
    if use_identity_projection:
        original_dim = data.x.size(1)
        if rank == 0:
            print(f"Dataset {data.name}: Identity projection ({original_dim}D -> PCA to {projection_small_dim}D -> Project to {projection_large_dim}D)")
        
        # Step 1: PCA to small dimension (NO padding needed!)
        if original_dim >= projection_small_dim:
            # Standard PCA to small_dim
            if use_full_pca:
                U, S, V = torch.svd(data.x)
                U = U[:, :projection_small_dim]
                S = S[:projection_small_dim]
            else:
                U, S, V = torch.pca_lowrank(data.x, q=projection_small_dim)
            
            # Sign normalization if requested
            if sign_normalize:
                for i in range(projection_small_dim):
                    feature_vector = U[:, i] * S[i]
                    max_idx = torch.argmax(torch.abs(feature_vector))
                    if feature_vector[max_idx] < 0:
                        U[:, i] = -U[:, i]
            
            data.x_pca = torch.mm(U, torch.diag(S))
        else:
            # For very small datasets, use all dimensions + minimal padding if needed
            data.x_pca = data.x
            if data.x_pca.size(1) < projection_small_dim:
                pad_size = projection_small_dim - data.x_pca.size(1)
                padding = torch.zeros(data.x_pca.size(0), pad_size, device=device)
                data.x_pca = torch.cat([data.x_pca, padding], dim=1)
                if rank == 0:
                    print(f"Dataset {data.name}: Minimal padding {data.x.size(1)} -> {projection_small_dim}")
        
        # Step 2: Set up for identity projection during forward pass
        data.x = data.x_pca  # Input to identity projection layer
        data.needs_identity_projection = True
        data.projection_target_dim = projection_large_dim
        
        # Apply normalization if requested
        if normalize_data:
            if use_batchnorm:
                batch_mean = data.x.mean(dim=0, keepdim=True)
                batch_std = data.x.std(dim=0, keepdim=True, unbiased=False)
                data.x = (data.x - batch_mean) / (batch_std + 1e-5)
                if rank == 0:
                    print(f"Dataset {data.name}: Applied BatchNorm-style normalization")
            else:
                data.x = F.normalize(data.x, p=2, dim=1)
                if rank == 0:
                    print(f"Dataset {data.name}: Applied LayerNorm-style normalization")
        
        if rank == 0:
            print(f"Identity projection preprocessing time: {time.time()-st:.2f}s", flush=True)
        return
    
    # Original PCA-based approach
    # Determine PCA target dimension
    original_dim = data.x.size(1)
    
    if original_dim >= hidden:
        # Case 1: Enough features, just PCA to hidden (no padding/projection needed)
        pca_target_dim = hidden
        data.needs_projection = False
        if rank == 0:
            print(f"Dataset {data.name}: Sufficient features ({original_dim} >= {hidden}), PCA to {hidden}")
    elif use_projector:
        # Case 2a: Not enough features, use projector pathway
        pca_target_dim = min(original_dim, min_pca_dim)
        data.needs_projection = True
        if rank == 0:
            print(f"Dataset {data.name}: Using projector pathway ({original_dim} -> {pca_target_dim} -> {hidden})")
    else:
        # Case 2b: Not enough features, use zero padding
        pca_target_dim = original_dim  # Use all available features
        data.needs_projection = False
        if rank == 0:
            print(f"Dataset {data.name}: Using zero padding ({original_dim} -> zero-pad to {hidden})")
    
    # Apply PCA - choose method based on configuration
    if pca_device == 'cpu':
        # Use CPU-based Incremental PCA (stays on CPU)
        pca_start = time.time()
        data.x_pca = apply_incremental_pca_cpu(
            data.x, 
            target_dim=pca_target_dim, 
            batch_size=incremental_pca_batch_size,
            sign_normalize=sign_normalize,
            rank=rank
        )
        pca_time = time.time() - pca_start
        if rank == 0:
            print(f"Dataset {data.name}: Incremental PCA completed in {pca_time:.2f}s (result on CPU)")
    else:
        # Use original GPU-based PCA methods
        if use_full_pca:
            U, S, V = torch.svd(data.x)
            U = U[:, :pca_target_dim]
            S = S[:pca_target_dim]
        else:
            U, S, V = torch.pca_lowrank(data.x, q=pca_target_dim)

        # normalize the eigenvectors direction
        if sign_normalize:
            for i in range(pca_target_dim):
                feature_vector = U[:, i] * S[i]
                max_idx = torch.argmax(torch.abs(feature_vector))
                if feature_vector[max_idx] < 0:
                    U[:, i] = -U[:, i]

        data.x_pca = torch.mm(U, torch.diag(S))
    
    # Handle device placement for PCA results
    if pca_device != 'cpu':
        # GPU PCA result -> move to device as usual
        data.x_pca = data.x_pca.to(device)
    else:
        # CPU Incremental PCA result -> keep on CPU, will move to GPU per-dataset during training
        if rank == 0:
            print(f"Dataset {data.name}: Keeping Incremental PCA result on CPU ({data.x_pca.shape})")
    
    # Handle different pathways
    if data.needs_projection:
        # Will be projected by MLP projector during forward pass, then PCA again
        data.x = data.x_pca  # Keep PCA features for projector input
        data.needs_final_pca = True  # Flag to apply PCA after projection
    else:
        # Either sufficient features or zero padding needed
        if data.x_pca.size(1) < hidden:
            padding_size = hidden - data.x_pca.size(1)
            
            # Determine device for padding operations
            pca_device = data.x_pca.device
            
            if padding_strategy == 'zero':
                # Zero padding (can harm performance due to distribution mismatch)
                padding = torch.zeros(data.x_pca.size(0), padding_size, device=pca_device)
                data.x = torch.cat([data.x_pca, padding], dim=1)
                if rank == 0:
                    print(f"Dataset {data.name}: Applied zero padding ({data.x_pca.size(1)} -> {hidden}) on {pca_device}")
            elif padding_strategy == 'random':
                # Random padding from same distribution as real features
                real_std = data.x_pca.std(dim=0, keepdim=True)
                real_mean = data.x_pca.mean(dim=0, keepdim=True)
                random_padding = torch.randn(data.x_pca.size(0), padding_size, device=pca_device) * real_std.mean() + real_mean.mean()
                data.x = torch.cat([data.x_pca, random_padding], dim=1)
                if rank == 0:
                    print(f"Dataset {data.name}: Applied random padding ({data.x_pca.size(1)} -> {hidden}) on {pca_device}")
            elif padding_strategy == 'repeat':
                # Feature repetition
                data.x = data.x_pca
                while data.x.size(1) < hidden:
                    remaining = hidden - data.x.size(1)
                    repeat_size = min(remaining, data.x_pca.size(1))
                    data.x = torch.cat([data.x, data.x_pca[:, :repeat_size]], dim=1)
                if rank == 0:
                    print(f"Dataset {data.name}: Applied feature repetition padding")
            else:
                raise ValueError(f"Unknown padding strategy: {padding_strategy}")
        else:
            # Sufficient features, use as-is
            data.x = data.x_pca
        data.needs_final_pca = False
                                
    # normalize the data
    if normalize_data:
        if use_batchnorm:
            # BatchNorm-style: normalize each feature across batch
            batch_mean = data.x.mean(dim=0, keepdim=True)
            batch_std = data.x.std(dim=0, keepdim=True, unbiased=False)
            data.x = (data.x - batch_mean) / (batch_std + 1e-5)
            if rank == 0:
                print(f"Dataset {data.name}: Applied BatchNorm-style normalization")
        else:
            # LayerNorm-style: L2 normalization per sample across features
            data.x = F.normalize(data.x, p=2, dim=1)
            if rank == 0:
                print(f"Dataset {data.name}: Applied LayerNorm-style normalization")

    if rank == 0:
        print(f"PCA time: {time.time()-st:.2f}s", flush=True) 

def prepare_link_data(data, split_idx, negative_ratio=1.0):
    """
    Prepare link prediction data. For the training split, it performs negative
    sampling. For validation and test splits, it uses the provided negative edges.
    
    Args:
        data: Graph data object.
        split_idx: Dictionary containing train/valid/test edge indices.
        negative_ratio: Ratio of negative to positive samples for training.
    
    Returns:
        Dictionary with 'edge_pairs' and 'labels' for each split.
    """
    device = data.x.device
    result = {}
    
    # Training split: perform negative sampling
    if 'train' in split_idx and 'edge' in split_idx['train']:
        pos_train_edge = split_idx['train']['edge'].to(device)
        num_pos_train = pos_train_edge.size(0)
        num_neg_train = int(num_pos_train * negative_ratio)
        
        # Convert SparseTensor to edge_index for negative_sampling
        row, col, _ = data.adj_t.to_symmetric().coo()
        edge_index = torch.stack([row, col], dim=0)

        try:
            # Use PyG's negative_sampling utility
            neg_train_edge = negative_sampling(
                edge_index=edge_index, # All known edges
                num_nodes=data.num_nodes,
                num_neg_samples=num_neg_train,
                method='sparse'
            ).t().to(device)

            train_edges = torch.cat([pos_train_edge, neg_train_edge], dim=0)
            train_labels = torch.cat([
                torch.ones(num_pos_train, device=device, dtype=torch.long),
                torch.zeros(neg_train_edge.size(0), device=device, dtype=torch.long)
            ], dim=0)
            
            result['train'] = {'edge_pairs': train_edges, 'labels': train_labels}
        except Exception as e:
            print(f"ERROR in prepare_link_data during negative sampling: {e}")
            raise e

    # Validation and Test splits: use provided negative edges
    for split in ['valid', 'test']:
        if split in split_idx and 'edge' in split_idx[split] and 'edge_neg' in split_idx[split]:
            pos_edge = split_idx[split]['edge'].to(device)
            neg_edge = split_idx[split]['edge_neg'].to(device)
            
            edges = torch.cat([pos_edge, neg_edge], dim=0)
            labels = torch.cat([
                torch.ones(pos_edge.size(0), device=device, dtype=torch.long),
                torch.zeros(neg_edge.size(0), device=device, dtype=torch.long)
            ], dim=0)
            
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

    # Sample k positive edges
    num_pos_to_select = min(k, len(pos_indices))
    if num_pos_to_select == 0:
        print(f"WARNING: No positive edges available for context selection!")
        return {'edge_pairs': torch.empty((0, 2), dtype=edge_pairs.dtype, device=edge_pairs.device),
                'labels': torch.empty((0,), dtype=labels.dtype, device=labels.device)}, torch.ones_like(labels, dtype=torch.bool)
    
    selected_pos_indices = pos_indices[torch.randperm(len(pos_indices))[:num_pos_to_select]]

    # Sample k * ratio negative edges
    num_neg_to_select = min(int(k * context_neg_ratio), len(neg_indices))
    if num_neg_to_select == 0:
        print(f"WARNING: No negative edges available for context selection!")
        return {'edge_pairs': torch.empty((0, 2), dtype=edge_pairs.dtype, device=edge_pairs.device),
                'labels': torch.empty((0,), dtype=labels.dtype, device=labels.device)}, torch.ones_like(labels, dtype=torch.bool)
    
    selected_neg_indices = neg_indices[torch.randperm(len(neg_indices))[:num_neg_to_select]]

    context_indices = torch.cat([selected_pos_indices, selected_neg_indices])
    
    context_data = {
        'edge_pairs': edge_pairs[context_indices],
        'labels': labels[context_indices]
    }
    
    # Create a mask indicating which samples are for training vs. context
    train_mask = torch.ones_like(labels, dtype=torch.bool)
    if remove_from_train:
        train_mask[context_indices] = False
    
    # Return the context data and the mask for the training samples
    return context_data, train_mask

def process_link_data(data, args, rank=0):
    """
    Dedicated data processing for link prediction.
    Applies the same feature transformations (PCA, projection, padding, normalization)
    as the node classification script, but avoids node-specific logic like context sampling.
    """
    device = data.x.device
    st = time.time()
    
    # Use args from the run/checkpoint to drive the logic
    hidden_dim = args.hidden
    use_identity_projection = args.use_identity_projection
    projection_small_dim = args.projection_small_dim
    projection_large_dim = args.projection_large_dim
    use_full_pca = args.use_full_pca
    normalize_data = args.normalize_data
    use_batchnorm = args.use_batchnorm
    padding_strategy = args.padding_strategy
    
    # Identity projection approach
    if use_identity_projection:
        original_dim = data.x.size(1)
        if rank == 0:
            print(f"Dataset {data.name}: Identity projection ({original_dim}D -> PCA to {projection_small_dim}D -> Project to {projection_large_dim}D)")
        
        if original_dim >= projection_small_dim:
            if use_full_pca:
                # Full PCA to small_dim
                U, S, V = torch.svd(data.x)
                U = U[:, :projection_small_dim]
                S = S[:projection_small_dim]
            else:
                # Low-rank PCA to small_dim
                U, S, V = torch.pca_lowrank(data.x, q=projection_small_dim)
            data.x_pca = torch.mm(U, torch.diag(S))
        else:
            raise ValueError(f"Original dimension {original_dim} is less than projection_small_dim {projection_small_dim}. Cannot apply identity projection.")
        
        data.x = data.x_pca
        data.needs_identity_projection = True
        data.projection_target_dim = projection_large_dim
        
    # Original PCA-based approach
    else:
        original_dim = data.x.size(1)
        if original_dim >= hidden_dim:
            pca_target_dim = hidden_dim
        else:
            pca_target_dim = original_dim
        
        if use_full_pca:
            U, S, V = torch.svd(data.x)
        else:
            U, S, V = torch.pca_lowrank(data.x, q=pca_target_dim)
        
        U = U[:, :pca_target_dim]
        S = S[:pca_target_dim]
        
        data.x_pca = torch.mm(U, torch.diag(S)).to(device)
        
        if data.x_pca.size(1) < hidden_dim:
            padding_size = hidden_dim - data.x_pca.size(1)
            if padding_strategy == 'zero':
                                # Zero padding (can harm performance due to distribution mismatch)
                padding = torch.zeros(data.x_pca.size(0), padding_size, device=device)
                data.x = torch.cat([data.x_pca, padding], dim=1)
                if rank == 0:
                    print(f"Dataset {data.name}: Applied zero padding ({data.x_pca.size(1)} -> {hidden_dim})")
            elif padding_strategy == 'random':
                # Random padding from same distribution as real features
                real_std = data.x_pca.std(dim=0, keepdim=True)
                real_mean = data.x_pca.mean(dim=0, keepdim=True)
                random_padding = torch.randn(data.x_pca.size(0), padding_size, device=device) * real_std.mean() + real_mean.mean()
                data.x = torch.cat([data.x_pca, random_padding], dim=1)
                if rank == 0:
                    print(f"Dataset {data.name}: Applied random padding ({data.x_pca.size(1)} -> {hidden_dim})")
            elif padding_strategy == 'repeat':
                # Feature repetition
                data.x = data.x_pca
                while data.x.size(1) < hidden_dim:
                    remaining = hidden_dim - data.x.size(1)
                    repeat_size = min(remaining, data.x_pca.size(1))
                    data.x = torch.cat([data.x, data.x_pca[:, :repeat_size]], dim=1)
                if rank == 0:
                    print(f"Dataset {data.name}: Applied feature repetition padding")
            else:
                raise ValueError(f"Unknown padding strategy: {padding_strategy}")
        else:
            data.x = data.x_pca
            
    # Final normalization
    if normalize_data:
        if use_batchnorm:
            batch_mean = data.x.mean(dim=0, keepdim=True)
            batch_std = data.x.std(dim=0, keepdim=True, unbiased=False)
            data.x = (data.x - batch_mean) / (batch_std + 1e-5)
        else:
            data.x = F.normalize(data.x, p=2, dim=1)
            
    if rank == 0:
        print(f"Feature processing time: {time.time()-st:.2f}s", flush=True) 