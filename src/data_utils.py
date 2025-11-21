import torch
import torch.nn.functional as F
import time
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from torch_geometric.utils import negative_sampling
from torch_sparse import SparseTensor
import os

# Import ablation study modules
from src.projection_methods import apply_random_orthogonal_projection, apply_sparse_random_projection
from src.feature_analysis import compute_feature_statistics, print_feature_statistics, plot_tsne_features

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

# Note: apply_random_orthogonal_projection, apply_sparse_random_projection,
# compute_feature_statistics, print_feature_statistics, and plot_tsne_features
# have been moved to src/projection_methods.py and src/feature_analysis.py

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
                 incremental_pca_batch_size=10000, external_embeddings=None, use_random_orthogonal=False,
                 use_sparse_random=False, sparse_random_density=0.1,
                 plot_tsne=False, tsne_save_dir='./tsne_plots', use_pca_whitening=False, whitening_epsilon=0.01,
                 use_fug_embeddings=False, use_dynamic_encoder=False):
    device = data.x.device
    split_idx['train'] = split_idx['train'].to(device)
    split_idx['valid'] = split_idx['valid'].to(device)
    split_idx['test'] = split_idx['test'].to(device)
    data.context_sample = select_k_shot_context(data, context_num, split_idx['train'])
    data.context_sample = data.context_sample.to(device)

    st = time.time()

    # Use external embeddings if provided, otherwise use original data.x
    if external_embeddings is not None:
        # Move external embeddings to the correct device and replace data.x
        input_features = external_embeddings.to(device)
        if rank == 0:
            print(f"Dataset {data.name}: Using external embeddings ({input_features.shape}) instead of data.x ({data.x.shape})")
    else:
        input_features = data.x

    # FUG embeddings mode: Skip PCA/identity projection, just use features directly
    # IMPORTANT: FUG embeddings should NOT be combined with GPSE/LapPE/RWSE to maintain uniform 1024-dim
    # The projector MLP in engine_nc.py will handle 1024 -> hidden projection
    if use_fug_embeddings and external_embeddings is not None:
        data.uses_fug_embeddings = True
        data.x = input_features
        data.needs_projection = False
        data.needs_identity_projection = False

        # Apply normalization to FUG embeddings (same as regular inputs)
        if normalize_data:
            if use_batchnorm:
                batch_mean = data.x.mean(dim=0, keepdim=True)
                batch_std = data.x.std(dim=0, keepdim=True, unbiased=False)
                data.x = (data.x - batch_mean) / (batch_std + 1e-5)
                if rank == 0:
                    print(f"Dataset {data.name}: Applied BatchNorm-style normalization to FUG embeddings")
            else:
                data.x = F.normalize(data.x, p=2, dim=1)
                if rank == 0:
                    print(f"Dataset {data.name}: Applied L2 normalization to FUG embeddings")

        if rank == 0:
            print(f"Dataset {data.name}: FUG embeddings mode - using features directly ({input_features.size(1)}D)")
            print(f"Dataset {data.name}: Skipping GPSE/LapPE/RWSE concatenation to preserve FUG's uniform dimension")
            print(f"Dataset {data.name}: Projector MLP will handle {input_features.size(1)}D -> hidden projection")
            print(f"Data preprocessing time: {time.time()-st:.2f}s", flush=True)
        return

    # Not using FUG embeddings
    data.uses_fug_embeddings = False

    # Dynamic Encoder mode: Skip PCA, keep raw features for end-to-end learning
    # DE will learn the projection matrix during training
    if use_dynamic_encoder:
        if rank == 0:
            print(f"Dataset {data.name}: Using Dynamic Encoder (skip PCA preprocessing)")
            print(f"  - Original features: {input_features.shape}")

        # GPSE/LapPE/RWSE Enhancement: Concatenate positional encodings if available
        # (Apply these BEFORE DE so DE learns on the augmented features)
        if hasattr(data, 'gpse_embeddings') and data.gpse_embeddings is not None:
            gpse_emb = data.gpse_embeddings.to(device)
            original_dim = input_features.size(1)
            input_features = torch.cat([input_features, gpse_emb], dim=1)
            if rank == 0:
                print(f"  - Concatenated GPSE: {original_dim}D + {gpse_emb.size(1)}D = {input_features.size(1)}D")

        if hasattr(data, 'lappe_embeddings') and data.lappe_embeddings is not None:
            lappe_emb = data.lappe_embeddings.to(device)
            original_dim = input_features.size(1)
            input_features = torch.cat([input_features, lappe_emb], dim=1)
            if rank == 0:
                print(f"  - Concatenated LapPE: {original_dim}D + {lappe_emb.size(1)}D = {input_features.size(1)}D")

        if hasattr(data, 'rwse_embeddings') and data.rwse_embeddings is not None:
            rwse_emb = data.rwse_embeddings.to(device)
            original_dim = input_features.size(1)
            input_features = torch.cat([input_features, rwse_emb], dim=1)
            if rank == 0:
                print(f"  - Concatenated RWSE: {original_dim}D + {rwse_emb.size(1)}D = {input_features.size(1)}D")

        # Store raw features (DE will project them during forward pass)
        data.x = input_features
        data.uses_dynamic_encoder = True
        data.needs_projection = False
        data.needs_identity_projection = False
        data.de_original_dim = input_features.size(1)  # Store for DE initialization

        # Optional normalization (standardization helps DE training)
        if normalize_data:
            if use_batchnorm:
                batch_mean = data.x.mean(dim=0, keepdim=True)
                batch_std = data.x.std(dim=0, keepdim=True, unbiased=False)
                data.x = (data.x - batch_mean) / (batch_std + 1e-5)
                if rank == 0:
                    print(f"  - Applied BatchNorm-style normalization")
            else:
                data.x = F.normalize(data.x, p=2, dim=1)
                if rank == 0:
                    print(f"  - Applied L2 normalization")

        if rank == 0:
            print(f"  - Final feature dimension: {data.x.size(1)}D")
            print(f"  - DE will project to {hidden}D during forward pass")
            print(f"Data preprocessing time: {time.time()-st:.2f}s", flush=True)
        return

    # GPSE Enhancement: Concatenate GPSE embeddings if available (only when not using FUG)
    if hasattr(data, 'gpse_embeddings') and data.gpse_embeddings is not None:
        gpse_emb = data.gpse_embeddings.to(device)
        original_dim = input_features.size(1)
        input_features = torch.cat([input_features, gpse_emb], dim=1)
        if rank == 0:
            print(f"Dataset {data.name}: Concatenated GPSE embeddings ({original_dim}D + {gpse_emb.size(1)}D = {input_features.size(1)}D)")

    # LapPE Enhancement: Concatenate Laplacian PE if available (only when not using FUG)
    if hasattr(data, 'lappe_embeddings') and data.lappe_embeddings is not None:
        lappe_emb = data.lappe_embeddings.to(device)
        original_dim = input_features.size(1)
        input_features = torch.cat([input_features, lappe_emb], dim=1)
        if rank == 0:
            print(f"Dataset {data.name}: Concatenated LapPE embeddings ({original_dim}D + {lappe_emb.size(1)}D = {input_features.size(1)}D)")

    # RWSE Enhancement: Concatenate Random Walk SE if available (only when not using FUG)
    if hasattr(data, 'rwse_embeddings') and data.rwse_embeddings is not None:
        rwse_emb = data.rwse_embeddings.to(device)
        original_dim = input_features.size(1)
        input_features = torch.cat([input_features, rwse_emb], dim=1)
        if rank == 0:
            print(f"Dataset {data.name}: Concatenated RWSE embeddings ({original_dim}D + {rwse_emb.size(1)}D = {input_features.size(1)}D)")

    # Identity projection approach - PCA to small_dim, then project to large_dim
    if use_identity_projection:
        original_dim = input_features.size(1)
        if rank == 0:
            print(f"Dataset {data.name}: Identity projection ({original_dim}D -> PCA to {projection_small_dim}D -> Project to {projection_large_dim}D)")
        
        # Step 1: PCA or random projection to small dimension (NO padding needed!)
        if original_dim >= projection_small_dim:
            if use_sparse_random:
                # Use sparse random projection instead of PCA
                data.x_pca = apply_sparse_random_projection(
                    input_features, input_dim=original_dim, target_dim=projection_small_dim,
                    density=sparse_random_density, seed=42, rank=rank
                )
            elif use_random_orthogonal:
                # Use dense random orthogonal projection instead of PCA
                data.x_pca = apply_random_orthogonal_projection(
                    input_features, input_dim=original_dim, target_dim=projection_small_dim, seed=42, rank=rank
                )
            elif use_full_pca:
                # Standard full PCA to small_dim
                U, S, V = torch.svd(input_features)
                U = U[:, :projection_small_dim]
                S = S[:projection_small_dim]

                # Sign normalization if requested
                if sign_normalize:
                    for i in range(projection_small_dim):
                        feature_vector = U[:, i] * S[i]
                        max_idx = torch.argmax(torch.abs(feature_vector))
                        if feature_vector[max_idx] < 0:
                            U[:, i] = -U[:, i]

                if use_pca_whitening:
                    # Whitening: divide by sqrt(eigenvalues + epsilon)
                    S_whitened = torch.sqrt(S + whitening_epsilon)
                    data.x_pca = torch.mm(U, torch.diag(S / S_whitened))
                    if rank == 0:
                        print(f"Dataset {data.name}: Applied PCA whitening (epsilon={whitening_epsilon})")
                else:
                    data.x_pca = torch.mm(U, torch.diag(S))
            else:
                # Low-rank PCA to small_dim
                U, S, V = torch.pca_lowrank(input_features, q=projection_small_dim)

                # Sign normalization if requested
                if sign_normalize:
                    for i in range(projection_small_dim):
                        feature_vector = U[:, i] * S[i]
                        max_idx = torch.argmax(torch.abs(feature_vector))
                        if feature_vector[max_idx] < 0:
                            U[:, i] = -U[:, i]

                if use_pca_whitening:
                    # Whitening: divide by sqrt(eigenvalues + epsilon)
                    S_whitened = torch.sqrt(S + whitening_epsilon)
                    data.x_pca = torch.mm(U, torch.diag(S / S_whitened))
                    if rank == 0:
                        print(f"Dataset {data.name}: Applied PCA whitening (epsilon={whitening_epsilon})")
                else:
                    data.x_pca = torch.mm(U, torch.diag(S))
        else:
            # For very small datasets, use all dimensions + minimal padding if needed
            data.x_pca = input_features
            if data.x_pca.size(1) < projection_small_dim:
                pad_size = projection_small_dim - data.x_pca.size(1)
                padding = torch.zeros(data.x_pca.size(0), pad_size, device=device)
                data.x_pca = torch.cat([data.x_pca, padding], dim=1)
                if rank == 0:
                    print(f"Dataset {data.name}: Minimal padding {input_features.size(1)} -> {projection_small_dim}")

        # Step 2: Set up for identity projection during forward pass
        data.x = data.x_pca  # Input to identity projection layer
        data.needs_identity_projection = True
        data.projection_target_dim = projection_large_dim

        # Save features before normalization for t-SNE visualization
        features_before_norm = data.x.clone() if plot_tsne and rank == 0 else None

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

        # Save features after normalization too
        features_after_norm = data.x.clone() if plot_tsne and rank == 0 else None

        if rank == 0:
            print(f"Identity projection preprocessing time: {time.time()-st:.2f}s", flush=True)

        # Plot t-SNE if requested (identity projection path)
        # Compare BEFORE and AFTER normalization to see BatchNorm's effect
        if plot_tsne and rank == 0:
            if use_sparse_random:
                method_name = "Sparse Random Projection"
            elif use_random_orthogonal:
                method_name = "Dense Random Orthogonal Projection"
            else:
                method_name = "PCA"
                if use_pca_whitening:
                    method_name += " + Whitening"

            # Plot BEFORE BatchNorm
            print(f"\n{'='*70}")
            print(f"BEFORE BatchNorm:")
            print(f"{'='*70}")
            plot_tsne_features(
                original_features=input_features,
                processed_features=features_before_norm,
                labels=data.y,
                dataset_name=data.name,
                method_name=method_name + " (Before BN)",
                save_dir=tsne_save_dir,
                rank=rank
            )

            # Plot AFTER BatchNorm (if normalization was applied)
            if normalize_data and features_after_norm is not None:
                print(f"\n{'='*70}")
                print(f"AFTER BatchNorm:")
                print(f"{'='*70}")
                plot_tsne_features(
                    original_features=input_features,
                    processed_features=features_after_norm,
                    labels=data.y,
                    dataset_name=data.name,
                    method_name=method_name + " (After BN)",
                    save_dir=tsne_save_dir,
                    rank=rank
                )
        return
    
    # Original PCA-based approach
    # Determine PCA target dimension
    original_dim = input_features.size(1)
    num_nodes = input_features.size(0)
    max_pca_dim = min(num_nodes, original_dim)  # Maximum valid PCA dimension

    if original_dim >= hidden:
        # Case 1: Enough features, just PCA to hidden (no padding/projection needed)
        pca_target_dim = min(hidden, max_pca_dim)  # Ensure we don't exceed matrix limits
        data.needs_projection = False
        if rank == 0:
            print(f"Dataset {data.name}: Sufficient features ({original_dim} >= {hidden}), PCA to {pca_target_dim}")
    elif use_projector:
        # Case 2a: Not enough features, use projector pathway
        pca_target_dim = min(original_dim, min_pca_dim, max_pca_dim)
        data.needs_projection = True
        if rank == 0:
            print(f"Dataset {data.name}: Using projector pathway ({original_dim} -> {pca_target_dim} -> {hidden})")
    else:
        # Case 2b: Not enough features, use zero padding
        pca_target_dim = min(original_dim, max_pca_dim)  # Use all available features but respect limits
        data.needs_projection = False
        if rank == 0:
            print(f"Dataset {data.name}: Using zero padding ({original_dim} -> zero-pad to {hidden})")
    
    # Apply PCA or Random Orthogonal Projection - choose method based on configuration
    if use_random_orthogonal:
        # Use random orthogonal projection instead of PCA (ablation study)
        projection_start = time.time()
        data.x_pca = apply_random_orthogonal_projection(
            input_features,
            input_dim=original_dim,
            target_dim=pca_target_dim,
            seed=42,
            rank=rank
        )
        projection_time = time.time() - projection_start
        if rank == 0:
            print(f"Dataset {data.name}: Random orthogonal projection completed in {projection_time:.2f}s")
    elif pca_device == 'cpu':
        # Use CPU-based Incremental PCA (stays on CPU)
        pca_start = time.time()
        data.x_pca = apply_incremental_pca_cpu(
            input_features,
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
            U, S, V = torch.svd(input_features)
            U = U[:, :pca_target_dim]
            S = S[:pca_target_dim]
        else:
            U, S, V = torch.pca_lowrank(input_features, q=pca_target_dim)

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

    # Plot t-SNE if requested
    if plot_tsne and rank == 0:
        method_name = "Random Orthogonal Projection" if use_random_orthogonal else "PCA"
        plot_tsne_features(
            original_features=input_features,
            processed_features=data.x,
            labels=data.y,
            dataset_name=data.name,
            method_name=method_name,
            save_dir=tsne_save_dir,
            rank=rank
        )

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
    use_random_orthogonal = getattr(args, 'use_random_orthogonal', False)
    
    # Identity projection approach
    if use_identity_projection:
        original_dim = data.x.size(1)
        if rank == 0:
            print(f"Dataset {data.name}: Identity projection ({original_dim}D -> PCA to {projection_small_dim}D -> Project to {projection_large_dim}D)")
        
        if original_dim >= projection_small_dim:
            if use_random_orthogonal:
                # Use random orthogonal projection instead of PCA
                data.x_pca = apply_random_orthogonal_projection(
                    data.x, input_dim=original_dim, target_dim=projection_small_dim, seed=42, rank=rank
                )
            elif use_full_pca:
                # Full PCA to small_dim
                U, S, V = torch.svd(data.x)
                U = U[:, :projection_small_dim]
                S = S[:projection_small_dim]
                data.x_pca = torch.mm(U, torch.diag(S))
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

        if use_random_orthogonal:
            # Use random orthogonal projection instead of PCA
            data.x_pca = apply_random_orthogonal_projection(
                data.x, input_dim=original_dim, target_dim=pca_target_dim, seed=42, rank=rank
            ).to(device)
        elif use_full_pca:
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


# =============================================================================
# Edge Dropout Augmentation Functions
# =============================================================================

def edge_dropout_sparse_tensor(adj_t, dropout_rate, training=True, verbose=False):
    """
    Apply edge dropout to SparseTensor (for single large graphs).

    Args:
        adj_t (SparseTensor): Input adjacency tensor
        dropout_rate (float): Probability of dropping each edge (0.0-1.0)
        training (bool): Only apply dropout during training
        verbose (bool): Print timing information

    Returns:
        SparseTensor: Adjacency tensor with dropped edges
    """
    if not training or dropout_rate <= 0.0:
        return adj_t

    if verbose:
        start_time = time.time()

    # Get COO format - already on correct device
    row, col, edge_attr = adj_t.coo()

    # Fast GPU random generation + masking
    num_edges = row.size(0)
    keep_mask = torch.rand(num_edges, device=row.device) > dropout_rate

    # Apply mask to all components
    row_kept = row[keep_mask]
    col_kept = col[keep_mask]
    edge_attr_kept = edge_attr[keep_mask] if edge_attr is not None else None

    # Reconstruct SparseTensor
    result = SparseTensor(
        row=row_kept,
        col=col_kept,
        value=edge_attr_kept,
        sparse_sizes=adj_t.sparse_sizes()
    )

    if verbose:
        elapsed = (time.time() - start_time) * 1000  # Convert to ms
        edges_removed = num_edges - row_kept.size(0)
        print(f"[Edge Dropout] {edges_removed}/{num_edges} edges removed ({dropout_rate:.1%}) in {elapsed:.2f}ms")

    return result


def edge_dropout_edge_index(edge_index, edge_attr=None, dropout_rate=0.1, training=True):
    """
    Apply edge dropout to edge_index format (for graph datasets).

    Args:
        edge_index (torch.Tensor): Edge indices [2, num_edges]
        edge_attr (torch.Tensor, optional): Edge attributes [num_edges, num_features]
        dropout_rate (float): Probability of dropping each edge (0.0-1.0)
        training (bool): Only apply dropout during training

    Returns:
        tuple: (edge_index_dropped, edge_attr_dropped)
    """
    if not training or dropout_rate <= 0.0:
        return edge_index, edge_attr

    num_edges = edge_index.size(1)
    keep_mask = torch.rand(num_edges, device=edge_index.device) > dropout_rate

    # Apply mask to edge_index
    edge_index_dropped = edge_index[:, keep_mask]

    # Apply mask to edge_attr if present
    edge_attr_dropped = None
    if edge_attr is not None:
        edge_attr_dropped = edge_attr[keep_mask]

    return edge_index_dropped, edge_attr_dropped


def batch_edge_dropout(batch, dropout_rate, training=True):
    """
    Apply edge dropout to a PyTorch Geometric batch (for graph classification).

    Args:
        batch: PyTorch Geometric batch object
        dropout_rate (float): Probability of dropping each edge (0.0-1.0)
        training (bool): Only apply dropout during training

    Returns:
        batch: Modified batch with dropped edges
    """
    if not training or dropout_rate <= 0.0:
        return batch

    # Clone batch to avoid modifying original
    batch_dropped = batch.clone()

    # Apply edge dropout to the entire batch
    edge_index_dropped, edge_attr_dropped = edge_dropout_edge_index(
        batch.edge_index,
        getattr(batch, 'edge_attr', None),
        dropout_rate,
        training
    )

    # Update batch with dropped edges
    batch_dropped.edge_index = edge_index_dropped
    if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
        batch_dropped.edge_attr = edge_attr_dropped

    return batch_dropped


def apply_edge_dropout(data, dropout_rate, training=True, data_type='auto'):
    """
    Universal edge dropout function that handles different data formats.

    Args:
        data: Input data (SparseTensor, PyG Data, or PyG Batch)
        dropout_rate (float): Probability of dropping each edge (0.0-1.0)
        training (bool): Only apply dropout during training
        data_type (str): Data type hint ('sparse_tensor', 'edge_index', 'batch', 'auto')

    Returns:
        Modified data with edge dropout applied
    """
    if not training or dropout_rate <= 0.0:
        return data

    # Auto-detect data type
    if data_type == 'auto':
        if isinstance(data, SparseTensor):
            data_type = 'sparse_tensor'
        elif hasattr(data, 'batch'):  # PyG Batch
            data_type = 'batch'
        elif hasattr(data, 'edge_index'):  # PyG Data
            data_type = 'edge_index'
        else:
            raise ValueError(f"Cannot auto-detect data type for: {type(data)}")

    # Apply appropriate dropout function
    if data_type == 'sparse_tensor':
        return edge_dropout_sparse_tensor(data, dropout_rate, training)
    elif data_type == 'batch':
        return batch_edge_dropout(data, dropout_rate, training)
    elif data_type == 'edge_index':
        # For PyG Data objects
        edge_index_dropped, edge_attr_dropped = edge_dropout_edge_index(
            data.edge_index,
            getattr(data, 'edge_attr', None),
            dropout_rate,
            training
        )
        data_dropped = data.clone()
        data_dropped.edge_index = edge_index_dropped
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data_dropped.edge_attr = edge_attr_dropped
        return data_dropped
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")


def get_edge_dropout_stats(original_data, dropped_data, data_type='auto'):
    """
    Get statistics about edge dropout operation.

    Args:
        original_data: Original data before dropout
        dropped_data: Data after dropout
        data_type (str): Data type hint

    Returns:
        dict: Statistics about the dropout operation
    """
    # Count edges in original data
    if isinstance(original_data, SparseTensor):
        original_edges = original_data.nnz()
    elif hasattr(original_data, 'edge_index'):
        original_edges = original_data.edge_index.size(1)
    else:
        original_edges = 0

    # Count edges in dropped data
    if isinstance(dropped_data, SparseTensor):
        dropped_edges = dropped_data.nnz()
    elif hasattr(dropped_data, 'edge_index'):
        dropped_edges = dropped_data.edge_index.size(1)
    else:
        dropped_edges = 0

    # Calculate statistics
    edges_removed = original_edges - dropped_edges
    actual_dropout_rate = edges_removed / original_edges if original_edges > 0 else 0.0

    return {
        'original_edges': original_edges,
        'remaining_edges': dropped_edges,
        'edges_removed': edges_removed,
        'actual_dropout_rate': actual_dropout_rate
    }


# =============================================================================
# Feature Dropout Augmentation Functions
# =============================================================================

def feature_dropout(x, dropout_rate, training=True, dropout_type='element_wise', verbose=False):
    """
    Apply feature dropout to node features with different strategies.

    Args:
        x (torch.Tensor): Input features [num_nodes, num_features]
        dropout_rate (float): Probability of dropping features (0.0-1.0)
        training (bool): Only apply dropout during training
        dropout_type (str): Type of dropout ('element_wise', 'channel_wise', 'gaussian_noise')
        verbose (bool): Print timing information

    Returns:
        torch.Tensor: Features with dropout applied
    """
    if not training or dropout_rate <= 0.0:
        return x

    if verbose:
        start_time = time.time()

    if dropout_type == 'element_wise':
        # Standard element-wise dropout - randomly zero out individual features
        result = torch.nn.functional.dropout(x, p=dropout_rate, training=True)

    elif dropout_type == 'channel_wise':
        # Channel-wise dropout - drop entire feature dimensions across all nodes
        num_features = x.size(1)
        keep_mask = torch.rand(num_features, device=x.device) > dropout_rate
        result = x * keep_mask.unsqueeze(0)  # Broadcast to all nodes

    elif dropout_type == 'gaussian_noise':
        # Add Gaussian noise instead of zeroing (softer augmentation)
        if x.numel() > 0:
            noise_std = dropout_rate * x.std()
            noise = torch.randn_like(x) * noise_std
            result = x + noise
        else:
            result = x
    else:
        raise ValueError(f"Unknown dropout_type: {dropout_type}. Choose 'element_wise', 'channel_wise', or 'gaussian_noise'")

    if verbose:
        elapsed = (time.time() - start_time) * 1000  # Convert to ms
        if dropout_type == 'gaussian_noise':
            noise_std = dropout_rate * x.std() if x.numel() > 0 else 0.0
            print(f"[Feature Dropout] {dropout_type}: noise_std={noise_std:.4f} in {elapsed:.2f}ms")
        elif dropout_type == 'channel_wise':
            num_features = x.size(1)
            keep_mask = torch.rand(num_features, device=x.device) > dropout_rate
            features_kept = keep_mask.float().mean().item()
            print(f"[Feature Dropout] {dropout_type}: {features_kept:.1%} feature dims kept ({dropout_rate:.1%} dropout) in {elapsed:.2f}ms")
        else:  # element_wise
            non_zero_ratio = (result != 0).float().mean().item()
            print(f"[Feature Dropout] {dropout_type}: {non_zero_ratio:.1%} elements kept in {elapsed:.2f}ms")

    return result


def batch_feature_dropout(batch, dropout_rate, training=True, dropout_type='element_wise', verbose=False):
    """
    Apply feature dropout to a PyTorch Geometric batch.

    Args:
        batch: PyTorch Geometric batch object
        dropout_rate (float): Probability of dropping features (0.0-1.0)
        training (bool): Only apply dropout during training
        dropout_type (str): Type of dropout ('element_wise', 'channel_wise', 'gaussian_noise')
        verbose (bool): Print timing information

    Returns:
        batch: Modified batch with feature dropout applied
    """
    if not training or dropout_rate <= 0.0:
        return batch

    # Clone batch to avoid modifying original
    batch_dropped = batch.clone()

    # Apply feature dropout to node features
    batch_dropped.x = feature_dropout(batch.x, dropout_rate, training, dropout_type, verbose)

    return batch_dropped


def apply_feature_dropout(data, dropout_rate, training=True, dropout_type='element_wise',
                         data_type='auto', verbose=False):
    """
    Universal feature dropout function that handles different data formats.

    Args:
        data: Input data (PyG Data, PyG Batch, or raw tensor)
        dropout_rate (float): Probability of dropping features (0.0-1.0)
        training (bool): Only apply dropout during training
        dropout_type (str): Type of dropout ('element_wise', 'channel_wise', 'gaussian_noise')
        data_type (str): Data type hint ('tensor', 'data', 'batch', 'auto')
        verbose (bool): Print timing information

    Returns:
        Modified data with feature dropout applied
    """
    if not training or dropout_rate <= 0.0:
        return data

    # Auto-detect data type
    if data_type == 'auto':
        if hasattr(data, 'batch'):  # PyG Batch
            data_type = 'batch'
        elif hasattr(data, 'x'):  # PyG Data
            data_type = 'data'
        elif isinstance(data, torch.Tensor):  # Raw tensor
            data_type = 'tensor'
        else:
            raise ValueError(f"Cannot auto-detect data type for: {type(data)}")

    # Apply appropriate dropout function
    if data_type == 'batch':
        return batch_feature_dropout(data, dropout_rate, training, dropout_type, verbose)
    elif data_type == 'data':
        # For PyG Data objects
        data_dropped = data.clone()
        data_dropped.x = feature_dropout(data.x, dropout_rate, training, dropout_type, verbose)
        return data_dropped
    elif data_type == 'tensor':
        # For raw tensors
        return feature_dropout(data, dropout_rate, training, dropout_type, verbose)
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")


def get_feature_dropout_stats(original_features, dropped_features, dropout_type='element_wise'):
    """
    Get statistics about feature dropout operation.

    Args:
        original_features (torch.Tensor): Original features before dropout
        dropped_features (torch.Tensor): Features after dropout
        dropout_type (str): Type of dropout used

    Returns:
        dict: Statistics about the dropout operation
    """
    if dropout_type == 'gaussian_noise':
        # For Gaussian noise, compute noise magnitude
        if original_features.numel() > 0:
            noise_magnitude = (dropped_features - original_features).abs().mean().item()
            feature_magnitude = original_features.abs().mean().item()
            relative_noise = noise_magnitude / feature_magnitude if feature_magnitude > 0 else 0.0
        else:
            noise_magnitude = 0.0
            feature_magnitude = 0.0
            relative_noise = 0.0

        return {
            'original_feature_magnitude': feature_magnitude,
            'noise_magnitude': noise_magnitude,
            'relative_noise_ratio': relative_noise,
            'features_changed': 1.0  # All features have noise added
        }
    else:
        # For masking-based dropout, compute fraction of features kept
        if original_features.numel() > 0:
            features_kept = (dropped_features != 0).float().mean().item()
            features_dropped = 1.0 - features_kept
        else:
            features_kept = 1.0
            features_dropped = 0.0

        return {
            'original_features': original_features.numel(),
            'features_kept': features_kept,
            'features_dropped': features_dropped,
            'actual_dropout_rate': features_dropped
        } 