import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import psutil
import os
from .utils import process_node_features, acc, apply_final_pca
from .data_utils import select_k_shot_context, edge_dropout_sparse_tensor, feature_dropout


def compute_separability_metrics(features, labels, sample_size=2000):
    """
    Compute class separability metrics for projected features.

    Metrics:
    - Fisher's ratio: Between-class variance / Within-class variance
    - Mean inter-class distance
    - Mean intra-class distance
    - Silhouette-like score

    Args:
        features: (N, d) feature tensor
        labels: (N,) label tensor
        sample_size: Max nodes to sample for efficiency

    Returns:
        dict of metrics
    """
    with torch.no_grad():
        # Sample for efficiency
        N = features.size(0)
        if N > sample_size:
            idx = torch.randperm(N, device=features.device)[:sample_size]
            features = features[idx]
            labels = labels[idx]

        labels = labels.squeeze()
        unique_labels = labels.unique()
        num_classes = len(unique_labels)

        if num_classes < 2:
            return {'fisher_ratio': 0.0, 'inter_class_dist': 0.0, 'intra_class_dist': 0.0}

        # Compute class centroids
        centroids = []
        intra_dists = []
        class_counts = []

        for c in unique_labels:
            mask = labels == c
            class_features = features[mask]
            if class_features.size(0) == 0:
                continue
            centroid = class_features.mean(dim=0)
            centroids.append(centroid)
            class_counts.append(class_features.size(0))

            # Intra-class distance: mean distance to centroid
            dists_to_centroid = torch.norm(class_features - centroid, dim=1)
            intra_dists.append(dists_to_centroid.mean().item())

        if len(centroids) < 2:
            return {'fisher_ratio': 0.0, 'inter_class_dist': 0.0, 'intra_class_dist': 0.0}

        centroids = torch.stack(centroids)

        # Inter-class distance: mean pairwise distance between centroids
        inter_dists = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                inter_dists.append(torch.norm(centroids[i] - centroids[j]).item())

        mean_inter = sum(inter_dists) / len(inter_dists) if inter_dists else 0.0
        mean_intra = sum(intra_dists) / len(intra_dists) if intra_dists else 1e-6

        # Fisher's ratio approximation
        fisher_ratio = mean_inter / (mean_intra + 1e-6)

        return {
            'fisher_ratio': fisher_ratio,
            'inter_class_dist': mean_inter,
            'intra_class_dist': mean_intra,
            'num_classes': num_classes
        }


def apply_feature_dropout_if_enabled(x, args, rank=0, training=True):
    """
    Apply feature dropout if enabled in args (after projection only).

    Args:
        x (torch.Tensor): Input features
        args: Arguments containing feature dropout configuration
        rank (int): Process rank for logging
        training (bool): Whether the model is in training mode

    Returns:
        torch.Tensor: Features with dropout applied
    """
    if (args is not None and
        hasattr(args, 'feature_dropout_enabled') and args.feature_dropout_enabled and
        hasattr(args, 'feature_dropout_rate') and args.feature_dropout_rate > 0):

        dropout_type = getattr(args, 'feature_dropout_type', 'element_wise')
        verbose = getattr(args, 'verbose_feature_dropout', False) and rank == 0

        return feature_dropout(x, args.feature_dropout_rate, training=training,
                             dropout_type=dropout_type, verbose=verbose)
    return x


def refresh_dataset_context_if_needed(data, split_idx, batch_idx, epoch, args):
    """
    Simple dataset-specific context refresh for batch-level updates.
    
    Args:
        data: Single dataset object
        split_idx: Split indices for this dataset
        batch_idx: Current batch index within this dataset
        epoch: Current epoch
        args: Arguments with refresh settings
    """
    # Check if batch refresh is enabled and it's time to refresh
    if getattr(args, 'context_batch_refresh_interval', 0) <= 0:
        return
        
    if batch_idx > 0 and batch_idx % args.context_batch_refresh_interval == 0:
        # Refresh context for this specific dataset
        refresh_seed = args.seed + epoch * 10000 + batch_idx
        torch.manual_seed(refresh_seed)
        
        # Simple approach: just use basic context refresh without complex imports
        # Get current context size
        if hasattr(data, 'context_sample') and data.context_sample is not None:
            current_context_size = len(data.context_sample) // len(data.y.unique())
            
            # Resample context
            new_context_sample = select_k_shot_context(data, current_context_size, split_idx['train'])
            data.context_sample = new_context_sample.to(data.context_sample.device)
            
            print(f"🔄 Dataset {data.name} context refreshed at batch {batch_idx} ({len(new_context_sample)} samples)")

def train(model, data, train_idx, optimizer, pred, batch_size, degree=False, att=None, mlp=None,
          orthogonal_push=0.0, normalize_class_h=False, clip_grad=1.0, projector=None, rank=0, epoch=0,
          identity_projection=None, lambda_=1.0, args=None, external_embeddings=None):
    st = time.time()
    print(f"[RANK {rank}] Starting training epoch {epoch} on device cuda:{rank}", flush=True)
    
    # Show detailed memory breakdown for first epoch and every 10th epoch
    show_detailed = (epoch == 0 or epoch % 10 == 0)

    model.train()
    if att is not None:
        att.train()
    if mlp is not None:
        mlp.train()
    if projector is not None:
        projector.train()

    # Use distributed sampler for DDP
    if dist.is_initialized():
        indices = torch.arange(train_idx.size(0))
        sampler = DistributedSampler(indices, shuffle=True)
        sampler.set_epoch(epoch)  # Set epoch for proper shuffling
        dataloader = DataLoader(indices, batch_size=batch_size, sampler=sampler)
    else:
        dataloader = DataLoader(range(train_idx.size(0)), batch_size, shuffle=True)

    total_loss = 0
    total_nll_loss = 0
    total_de_loss = 0
    for batch_idx, perm in enumerate(dataloader):
        # Batch-level context refresh for this dataset
        if args is not None:
            refresh_dataset_context_if_needed(data, {'train': train_idx}, batch_idx, epoch, args)
        
        if isinstance(perm, torch.Tensor):
            perm = perm.tolist()
        train_perm_idx = train_idx[perm]
        
        base_features = data.x

        # Note: GPSE embeddings are already concatenated in process_data (data_utils.py)
        # No need to concatenate again here

        # Apply different projection strategies
        # Priority: Dynamic Encoder > FUG embeddings > identity projection > standard projection > raw features
        if hasattr(data, 'uses_dynamic_encoder') and data.uses_dynamic_encoder:
            # Dynamic Encoder: use raw features directly, DE will project inside model forward
            x_input = base_features
        elif hasattr(data, 'uses_fug_embeddings') and data.uses_fug_embeddings and projector is not None:
            # FUG embeddings are uniform 1024-dim, just use simple MLP projection to hidden
            # No need for PCA or identity projection since FUG already provides consistent embeddings
            x_input = projector(base_features)
        elif hasattr(data, 'needs_identity_projection') and data.needs_identity_projection and identity_projection is not None:
            x_input = identity_projection(base_features)
        elif hasattr(data, 'needs_projection') and data.needs_projection and projector is not None:
            projected_features = projector(base_features)
            # Apply final PCA to get features in proper PCA form
            if hasattr(data, 'needs_final_pca') and data.needs_final_pca:
                x_input = apply_final_pca(projected_features, projected_features.size(1))
            else:
                x_input = projected_features
        else:
            x_input = base_features

        # Apply feature dropout AFTER projection
        x_input = apply_feature_dropout_if_enabled(x_input, args, rank, training=model.training)

        # Apply edge dropout if enabled
        adj_t_input = data.adj_t
        if args is not None and hasattr(args, 'edge_dropout_enabled') and args.edge_dropout_enabled and hasattr(args, 'edge_dropout_rate'):
            verbose_dropout = getattr(args, 'verbose_edge_dropout', False) and rank == 0
            adj_t_input = edge_dropout_sparse_tensor(data.adj_t, args.edge_dropout_rate, training=model.training, verbose=verbose_dropout)

        # Memory-optimized forward pass with gradient checkpointing and chunking
        # GNN forward pass
        if hasattr(data, 'use_gradient_checkpointing') and data.use_gradient_checkpointing:
            # Use gradient checkpointing to reduce memory
            h = torch.utils.checkpoint.checkpoint(model, x_input, adj_t_input)
        else:
            # Standard forward pass
            h = model(x_input, adj_t_input)


        # Extract DE loss if model has Dynamic Encoder
        de_loss = torch.tensor(0.0, device=h.device)
        if hasattr(model, 'get_de_loss'):
            de_loss = model.get_de_loss()

        # Compute separability metrics for DE projected features (periodically)
        if batch_idx == 0 and epoch % 5 == 0 and hasattr(model, 'de') and hasattr(model, '_debug_x_proj'):
            x_proj = model._debug_x_proj
            if x_proj is not None:
                sep_metrics = compute_separability_metrics(x_proj.detach(), data.y)
                print(f"    [DE Separability @ Epoch {epoch}] "
                      f"Fisher: {sep_metrics['fisher_ratio']:.3f}, "
                      f"Inter: {sep_metrics['inter_class_dist']:.3f}, "
                      f"Intra: {sep_metrics['intra_class_dist']:.3f}")

        # Memory optimization: Extract needed embeddings immediately and delete large tensor
        context_h = h[data.context_sample]
        context_y = data.y[data.context_sample]
        
        # Extract target embeddings for this batch
        target_h = h[train_perm_idx]
        
        # Fix type safety by properly handling None values
        class_h = process_node_features(
            context_h, data, 
            degree_normalize=degree,
            attention_pool_module=att if att is not None else None, 
            mlp_module=mlp if mlp is not None else None, 
            normalize=normalize_class_h
        )

        target_y = data.y[train_perm_idx]
        score, class_h = pred(data, context_h, target_h, context_y, class_h)
        score = F.log_softmax(score, dim=1)
        label = data.y[train_perm_idx].squeeze()

        # Compute orthogonal loss with better numerical stability
        if orthogonal_push > 0:
            class_h_norm = F.normalize(class_h, p=2, dim=1)
            class_matrix = class_h_norm @ class_h_norm.T
            # Remove diagonal elements
            mask = ~torch.eye(class_matrix.size(0), device=class_matrix.device, dtype=torch.bool)
            orthogonal_loss = torch.sum(class_matrix[mask]**2)
        else:
            orthogonal_loss = torch.tensor(0.0, device=label.device)

        nll_loss = F.nll_loss(score, label)
        loss = nll_loss + orthogonal_push * orthogonal_loss + de_loss

        # Track separate loss components BEFORE lambda scaling
        total_nll_loss += nll_loss.item()
        total_de_loss += de_loss.item() if isinstance(de_loss, torch.Tensor) else de_loss
        total_loss += loss.item()  # Total before scaling

        # DEBUG: Test gradient contribution from NLL vs DE loss separately (first batch only)
        if batch_idx == 0 and epoch == 0 and hasattr(model, 'de'):
            print(f"      [Gradient Source Test]")
            # Store current state
            de_params_before = [p.grad.clone() if p.grad is not None else None for p in model.de.parameters()]

            # Test 1: Backward with ONLY NLL loss
            optimizer.zero_grad()
            (nll_loss * lambda_).backward(retain_graph=True)
            de_grad_from_nll = model.de.lin_in.weight.grad.norm().item() if model.de.lin_in.weight.grad is not None else 0.0
            print(f"        DE grad from NLL loss only: {de_grad_from_nll:.6f}")

            # Test 2: Backward with ONLY DE loss
            optimizer.zero_grad()
            (de_loss * lambda_).backward(retain_graph=True)
            de_grad_from_de_loss = model.de.lin_in.weight.grad.norm().item() if model.de.lin_in.weight.grad is not None else 0.0
            print(f"        DE grad from DE loss only: {de_grad_from_de_loss:.6f}")
            print(f"        Ratio (DE_loss / NLL_loss): {de_grad_from_de_loss / (de_grad_from_nll + 1e-10):.2f}x")

            # Clear for actual backward
            optimizer.zero_grad()

        # Apply lambda scaling for gradient update
        loss = loss * lambda_

        # DEBUG: Check if x_proj has gradient enabled (only first batch)
        if batch_idx == 0 and epoch == 0 and hasattr(model, 'de'):
            print(f"    [Gradient Debug] x_proj requires_grad: {x_proj.requires_grad if 'x_proj' in locals() else 'N/A'}")
            print(f"    [Gradient Debug] h requires_grad: {h.requires_grad}")
            print(f"    [Gradient Debug] score requires_grad: {score.requires_grad}")
            print(f"    [Gradient Debug] nll_loss requires_grad: {nll_loss.requires_grad}")
            print(f"    [Gradient Debug] de_loss requires_grad: {de_loss.requires_grad if isinstance(de_loss, torch.Tensor) else 'scalar'}")

        # DEBUG: Retain gradients on key tensors to trace flow (only first batch)
        if batch_idx == 0 and epoch == 0 and hasattr(model, 'de'):
            h.retain_grad()
            score.retain_grad()
            if hasattr(pred, 'class_h'):
                pred.class_h.retain_grad()

        # Only perform optimization if optimizer is provided (for joint training compatibility)
        optimizer.zero_grad()
        loss.backward()

        # DEBUG: Check gradient magnitudes after backward (first few batches of first epoch)
        if batch_idx <= 2 and epoch == 0 and hasattr(model, 'de'):
            print(f"    [Gradient Debug After Backward - Batch {batch_idx}]")

            # Trace gradient flow from loss backwards
            print(f"      [Gradient Flow Trace]")
            print(f"        nll_loss: {nll_loss.item():.6f}")

            if score.grad is not None:
                print(f"        score.grad norm: {score.grad.norm().item():.6f}")

            # Check h gradient (GNN output = Transformer input)
            if h.grad is not None:
                print(f"        h.grad norm: {h.grad.norm().item():.6f} ← Gradient FROM predictor TO GNN")
                print(f"        h.grad mean: {h.grad.mean().item():.6e}, max: {h.grad.abs().max().item():.6e}")

                # Calculate gradient reduction through predictor
                if score.grad is not None:
                    reduction_factor = score.grad.norm().item() / (h.grad.norm().item() + 1e-10)
                    print(f"        Gradient reduction through predictor: {reduction_factor:.2f}x")

                # TRACE GRADIENT EXPLOSION: h → x_proj → projection_matrix → DE
                if hasattr(model, '_debug_x_proj') and model._debug_x_proj is not None:
                    print(f"\n      [Gradient Explosion Trace: h → DE]")

                    # Step 1: h → x_proj (through GNN backward)
                    if model._debug_x_proj.grad is not None:
                        x_proj_grad_norm = model._debug_x_proj.grad.norm().item()
                        print(f"        1. h.grad: {h.grad.norm().item():.6f}")
                        print(f"           ↓ GNN backward")
                        print(f"           x_proj.grad: {x_proj_grad_norm:.6f}")
                        gnn_amplification = x_proj_grad_norm / (h.grad.norm().item() + 1e-10)
                        print(f"           Amplification through GNN: {gnn_amplification:.2f}x")

                        # Check x_proj statistics
                        print(f"           x_proj norm: {model._debug_x_proj.norm().item():.2f}")
                        print(f"           x_proj mean: {model._debug_x_proj.mean().item():.6f}, std: {model._debug_x_proj.std().item():.6f}")
                    else:
                        print(f"        x_proj has NO gradient!")

                    # Step 2: Check input features x
                    print(f"\n        2. Input features x:")
                    print(f"           x.shape: {model._debug_x.shape}")
                    print(f"           x norm: {model._debug_x.norm().item():.2f}")
                    print(f"           x mean: {model._debug_x.mean().item():.6f}, std: {model._debug_x.std().item():.6f}")

                    # Step 3: x_proj → projection_matrix → DE
                    de_grad_norm = model.de.lin_in.weight.grad.norm().item() if model.de.lin_in.weight.grad is not None else 0.0
                    if model._debug_x_proj.grad is not None:
                        print(f"\n        3. x_proj.grad: {model._debug_x_proj.grad.norm().item():.6f}")
                        print(f"           ↓ Projection backward (x.T @ grad_x_proj)")
                        print(f"           DE param grad: {de_grad_norm:.6f}")
                        projection_amplification = de_grad_norm / (model._debug_x_proj.grad.norm().item() + 1e-10)
                        print(f"           Amplification through projection: {projection_amplification:.2f}x")

                        # Total amplification
                        total_amplification = de_grad_norm / (h.grad.norm().item() + 1e-10)
                        print(f"\n        TOTAL: h ({h.grad.norm().item():.6f}) → DE ({de_grad_norm:.6f})")
                        print(f"        Total amplification: {total_amplification:.2f}x")
            else:
                print(f"        h has NO grad (retain_grad not called or failed)")

            # Check if we can manually compute gradient on projection matrix
            if hasattr(model, 'current_projection_matrix') and model.current_projection_matrix is not None:
                proj = model.current_projection_matrix
                print(f"      projection_matrix.requires_grad: {proj.requires_grad}")
                print(f"      projection_matrix.grad_fn: {proj.grad_fn}")
                print(f"      projection_matrix.is_leaf: {proj.is_leaf}")
                if proj.grad is not None:
                    print(f"      projection_matrix.grad norm: {proj.grad.norm().item():.6f}")
                else:
                    print(f"      projection_matrix.grad is None")
                    # If it has grad_fn but no grad, it's a non-leaf tensor (expected)
                    # Gradients only accumulate on leaf tensors with requires_grad=True
                    if proj.grad_fn is not None:
                        print(f"      (This is OK - it's a non-leaf tensor, grad flows through grad_fn)")

            # Check DE parameter gradients (these ARE leaf tensors)
            print(f"      DE parameter gradients:")
            for name, param in list(model.de.named_parameters())[:3]:  # First 3 params
                if param.grad is not None:
                    print(f"        {name}: grad_norm={param.grad.norm().item():.6f}")
                else:
                    print(f"        {name}: NO GRADIENT!")


            # Check GNN parameter gradients
            print(f"      GNN parameter gradients:")
            for name, param in list(model.gnn.named_parameters())[:3]:  # First 3 params
                if param.grad is not None:
                    print(f"        {name}: grad_norm={param.grad.norm().item():.6f}")
                else:
                    print(f"        {name}: NO GRADIENT!")

            # Check Predictor (Transformer) parameter gradients
            print(f"      Predictor (Transformer) parameter gradients:")
            pred_grad_norms = []
            for name, param in list(pred.named_parameters())[:5]:  # First 5 params
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    pred_grad_norms.append(grad_norm)
                    print(f"        {name}: grad_norm={grad_norm:.6f}")
                else:
                    print(f"        {name}: NO GRADIENT!")
            if pred_grad_norms:
                print(f"        Avg predictor grad norm: {sum(pred_grad_norms)/len(pred_grad_norms):.6f}")

            # Most importantly: check the gradient norm of the SCORE
            print(f"      Score (predictor output) gradient check:")
            print(f"        score.requires_grad: {score.requires_grad}")
            print(f"        score.grad_fn: {score.grad_fn}")
            print(f"        nll_loss value: {nll_loss.item():.6f}")
            print(f"        score mean: {score.mean().item():.6f}, std: {score.std().item():.6f}")

        # Apply gradient clipping
        if clip_grad > 0:
            # DEBUG: Check gradient norms before and after clipping (first batch only)
            if batch_idx == 0 and epoch == 0 and hasattr(model, 'de'):
                total_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                pred_norm_before = torch.nn.utils.clip_grad_norm_(pred.parameters(), float('inf'))
                print(f"      [Gradient Clipping]")
                print(f"        Model grad norm before clip: {total_norm_before:.6f}")
                print(f"        Pred grad norm before clip: {pred_norm_before:.6f}")
                print(f"        Clip threshold: {clip_grad}")

            # Handle DDP wrapped models
            model_norm_after = nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            pred_norm_after = nn.utils.clip_grad_norm_(pred.parameters(), clip_grad)

            if batch_idx == 0 and epoch == 0 and hasattr(model, 'de'):
                print(f"        Model grad norm after clip: {model_norm_after:.6f}")
                print(f"        Pred grad norm after clip: {pred_norm_after:.6f}")
                if model_norm_after >= clip_grad * 0.99:
                    print(f"        ⚠️  MODEL GRADIENTS WERE CLIPPED! (reduced from {total_norm_before:.6f})")
                if pred_norm_after >= clip_grad * 0.99:
                    print(f"        ⚠️  PREDICTOR GRADIENTS WERE CLIPPED! (reduced from {pred_norm_before:.6f})")
            if att is not None:
                nn.utils.clip_grad_norm_(att.parameters(), clip_grad)
            if mlp is not None:
                nn.utils.clip_grad_norm_(mlp.parameters(), clip_grad)
            if projector is not None:
                nn.utils.clip_grad_norm_(projector.parameters(), clip_grad)
            if identity_projection is not None:
                nn.utils.clip_grad_norm_(identity_projection.parameters(), clip_grad)

        optimizer.step()
        
    en = time.time()
    if rank == 0:
        print(f"Train time: {en-st:.2f}s", flush=True)

    # Compute average losses
    avg_total_loss = total_loss / len(dataloader)
    avg_nll_loss = total_nll_loss / len(dataloader)
    avg_de_loss = total_de_loss / len(dataloader)

    # Print breakdown if DE is being used
    if rank == 0 and avg_de_loss > 0:
        print(f"[RANK {rank}] Epoch {epoch} - Loss breakdown:")
        print(f"  Total: {avg_total_loss:.4f}")
        print(f"  NLL: {avg_nll_loss:.4f}")
        print(f"  DE: {avg_de_loss:.6f} (scale: {avg_de_loss/avg_nll_loss*100:.2f}% of NLL)")

        # Get DE diagnostics if available
        if hasattr(model, 'get_de_diagnostics'):
            diag = model.get_de_diagnostics()
            if diag:
                print(f"  [DE Diagnostics]")
                if diag.get('proj_has_nan', False) or diag.get('proj_has_inf', False):
                    print(f"    ⚠️  WARNING: Projection has NaN={diag.get('proj_has_nan')} Inf={diag.get('proj_has_inf')}")
                print(f"    Projection: mean={diag.get('proj_mean', 0):.4f} std={diag.get('proj_std', 0):.4f} norm={diag.get('proj_norm', 0):.2f}")
                print(f"    Basis norms: {diag.get('basis_norm_mean', 0):.4f}±{diag.get('basis_norm_std', 0):.4f} (min={diag.get('basis_norm_min', 0):.4f} max={diag.get('basis_norm_max', 0):.4f})")
                print(f"    DE params: norm_mean={diag.get('de_param_norm_mean', 0):.4f} norm_max={diag.get('de_param_norm_max', 0):.4f}")
                if diag.get('de_grad_has_none', True):
                    print(f"    ⚠️  WARNING: DE has no gradients!")
                else:
                    print(f"    DE grads: norm_mean={diag.get('de_grad_norm_mean', 0):.6f} norm_max={diag.get('de_grad_norm_max', 0):.6f}")
                print(f"    Uniformity: mean_vec_norm={diag.get('mean_vec_norm', 0):.6f} loss={diag.get('uniformity_loss', 0):.6f}")
    else:
        loss_str = f"{avg_total_loss:.4f}" if optimizer is not None else "tensor"
        print(f"[RANK {rank}] Completed training epoch {epoch}, loss: {loss_str}", flush=True)

    # Return dictionary with breakdown for logging
    return {
        'total': avg_total_loss,
        'nll': avg_nll_loss,
        'de': avg_de_loss
    }

def train_all(model, data_list, split_idx_list, optimizer, pred, batch_size, degree=False, att=None,
              mlp=None, orthogonal_push=0.0, normalize_class_h=False, clip_grad=1.0, projector=None,
              rank=0, epoch=0, identity_projection=None, lambda_=1.0, args=None, external_embeddings_list=None):
    tot_loss = 0
    tot_nll_loss = 0
    tot_de_loss = 0

    for i, (data, split_idx) in enumerate(zip(data_list, split_idx_list)):
        train_idx = split_idx['train']
        external_embeddings = external_embeddings_list[i] if external_embeddings_list else None
        loss_dict = train(model, data, train_idx, optimizer, pred, batch_size, degree, att, mlp,
                          orthogonal_push, normalize_class_h, clip_grad, projector, rank, epoch,
                          identity_projection, lambda_, args, external_embeddings)

        # Handle both dict and scalar returns (backward compatibility)
        if isinstance(loss_dict, dict):
            loss_val = loss_dict['total']
            tot_loss += loss_val
            tot_nll_loss += loss_dict['nll']
            tot_de_loss += loss_dict['de']
            if rank == 0:
                print(f"Dataset {data.name} - Total: {loss_val:.4f}, NLL: {loss_dict['nll']:.4f}, DE: {loss_dict['de']:.6f}", flush=True)
        else:
            tot_loss += loss_dict
            if rank == 0:
                print(f"Dataset {data.name} Loss: {loss_dict}", flush=True)

    # Return dictionary with averages
    return {
        'total': tot_loss / len(data_list),
        'nll': tot_nll_loss / len(data_list),
        'de': tot_de_loss / len(data_list)
    }

@torch.no_grad()
def test(model, predictor, data, train_idx, valid_idx, test_idx, batch_size, degree=False,
         att=None, mlp=None, normalize_class_h=False, projector=None, rank=0, identity_projection=None, external_embeddings=None, args=None):
    st = time.time()
    model.eval()
    predictor.eval()
    if projector is not None:
        projector.eval()
    if identity_projection is not None:
        identity_projection.eval()

    base_features = data.x

    # Note: GPSE embeddings are already concatenated in process_data (data_utils.py)
    # No need to concatenate again here

    # Apply different projection strategies
    # Priority: FUG embeddings > identity projection > standard projection > raw features
    if hasattr(data, 'uses_fug_embeddings') and data.uses_fug_embeddings and projector is not None:
        # FUG embeddings are uniform 1024-dim, just use simple MLP projection to hidden
        # No need for PCA or identity projection since FUG already provides consistent embeddings
        x_input = projector(base_features)
    elif hasattr(data, 'needs_identity_projection') and data.needs_identity_projection and identity_projection is not None:
        # Apply identity projection
        x_input = identity_projection(base_features)
    elif hasattr(data, 'needs_projection') and data.needs_projection and projector is not None:
        projected_features = projector(base_features)
        # Apply final PCA to get features in proper PCA form
        if hasattr(data, 'needs_final_pca') and data.needs_final_pca:
            x_input = apply_final_pca(projected_features, projected_features.size(1))
        else:
            x_input = projected_features
    else:
        x_input = base_features

    # GNN forward pass
    h = model(x_input, data.adj_t)

    context_h = h[data.context_sample]
    context_y = data.y[data.context_sample]

    class_h = process_node_features(context_h, data, degree_normalize=degree, attention_pool_module=att, 
                                    mlp_module=mlp, normalize=normalize_class_h)

    # predict
    # break into mini-batches for large edge sets
    train_loader = DataLoader(range(train_idx.size(0)), batch_size, shuffle=False)
    valid_loader = DataLoader(range(valid_idx.size(0)), batch_size, shuffle=False)
    test_loader = DataLoader(range(test_idx.size(0)), batch_size, shuffle=False)

    valid_score = []
    for idx in valid_loader:
        target_h = h[valid_idx[idx]]
        pred_output = predictor(data, context_h, target_h, context_y, class_h)
        if len(pred_output) == 3:  # MoE case with auxiliary loss
            out, _, _ = pred_output  # Discard auxiliary loss during evaluation
        else:  # Standard case
            out, _ = pred_output
        out = out.argmax(dim=1).flatten()
        valid_score.append(out)
    valid_score = torch.cat(valid_score, dim=0)

    train_score = []
    for idx in train_loader:
        target_h = h[train_idx[idx]]
        pred_output = predictor(data, context_h, target_h, context_y, class_h)
        if len(pred_output) == 3:  # MoE case with auxiliary loss
            out, _, _ = pred_output  # Discard auxiliary loss during evaluation
        else:  # Standard case
            out, _ = pred_output
        out = out.argmax(dim=1).flatten()
        train_score.append(out)
    train_score = torch.cat(train_score, dim=0)

    test_score = []
    for idx in test_loader:
        target_h = h[test_idx[idx]]
        pred_output = predictor(data, context_h, target_h, context_y, class_h)
        if len(pred_output) == 3:  # MoE case with auxiliary loss
            out, _, _ = pred_output  # Discard auxiliary loss during evaluation
        else:  # Standard case
            out, _ = pred_output
        out = out.argmax(dim=1).flatten()
        test_score.append(out)
    test_score = torch.cat(test_score, dim=0)

    # calculate valid metric
    valid_y = data.y[valid_idx]
    valid_results = acc(valid_y, valid_score)
    train_y = data.y[train_idx]
    train_results = acc(train_y, train_score)
    test_y = data.y[test_idx]
    test_results = acc(test_y, test_score)

    if rank == 0:
        print(f"Test time: {time.time()-st}", flush=True)
    return train_results, valid_results, test_results

def test_all(model, predictor, data_list, split_idx_list, batch_size, degree=False, att=None,
             mlp=None, normalize_class_h=False, projector=None, rank=0, identity_projection=None, external_embeddings_list=None):
    tot_train_metric, tot_valid_metric, tot_test_metric = 1, 1, 1
    for i, (data, split_idx) in enumerate(zip(data_list, split_idx_list)):
        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']
        external_embeddings = external_embeddings_list[i] if external_embeddings_list else None

        train_metric, valid_metric, test_metric = \
        test(model, predictor, data, train_idx, valid_idx, test_idx, batch_size, degree, att, mlp,
             normalize_class_h, projector, rank, identity_projection, external_embeddings)
        if rank == 0:
            print(f"Dataset {data.name}")
            print(f"Train {train_metric}, Valid {valid_metric}, Test {test_metric}", flush=True)
        tot_train_metric *= train_metric
        tot_valid_metric *= valid_metric
        tot_test_metric *= test_metric
    return tot_train_metric ** (1/(len(data_list))), tot_valid_metric ** (1/(len(data_list))), \
           tot_test_metric ** (1/(len(data_list)))

def test_all_induct(model, predictor, data_list, split_idx_list, batch_size, degree=False,
                    att=None, mlp=None, normalize_class_h=False, projector=None, rank=0, identity_projection=None, external_embeddings_list=None):
    import time

    train_metric_list, valid_metric_list, test_metric_list = [], [], []
    for dataset_idx, (data, split_idx) in enumerate(zip(data_list, split_idx_list)):
        dataset_start_time = time.time()

        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']

        external_embeddings = external_embeddings_list[dataset_idx] if external_embeddings_list else None
        train_metric, valid_metric, test_metric = \
        test(model, predictor, data, train_idx, valid_idx, test_idx, batch_size, degree, att, mlp,
             normalize_class_h, projector, rank, identity_projection, external_embeddings)

        dataset_time = time.time() - dataset_start_time

        if rank == 0:
            print(f"    Dataset {dataset_idx} ({data.name}): completed in {dataset_time:.2f}s")
            print(f"      Train {train_metric:.4f}, Valid {valid_metric:.4f}, Test {test_metric:.4f}", flush=True)
        train_metric_list.append(train_metric)
        valid_metric_list.append(valid_metric)
        test_metric_list.append(test_metric)
    return train_metric_list, valid_metric_list, test_metric_list 