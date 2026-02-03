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


def correct_and_smooth(adj, base_logits, context_idx, context_labels, num_classes,
                       num_iters=50, alpha=0.5):
    """
    Correct & Smooth: post-process feature-based predictions with label propagation.

    1. Start with base predictions from features (not zeros!)
    2. Propagate predictions through graph
    3. Clamp context/support set to ground truth at each step

    This combines feature information with graph structure.

    IMPORTANT: context_idx should be the FEW-SHOT context samples (e.g., data.context_sample),
    NOT the full training split, to avoid label leakage in few-shot evaluation!

    Args:
        adj: SparseTensor adjacency matrix
        base_logits: [num_nodes, num_classes] initial predictions from model
        context_idx: Tensor of context/support node indices (FEW-SHOT samples only!)
        context_labels: Tensor of context node labels
        num_classes: Number of classes
        num_iters: Number of propagation iterations (default: 50)
        alpha: Blending factor (default: 0.5)

    Returns:
        Y: [num_nodes, num_classes] refined predictions after C&S
    """
    device = adj.device()
    num_nodes = base_logits.size(0)

    # Compute normalized adjacency WITH self-loops: D^{-1/2} (A + I) D^{-1/2}
    deg = adj.sum(dim=1).to_dense()
    deg_with_selfloop = deg + 1  # Add 1 for self-loop
    deg_inv_sqrt = (deg_with_selfloop + 1e-9).pow(-0.5)

    # Start with softmax of base logits (feature-based estimate)
    Y = F.softmax(base_logits, dim=-1)

    # Ground truth for context/support set (few-shot samples only!)
    Y_support = F.one_hot(context_labels.long(), num_classes=num_classes).float()

    # Propagate and clamp
    for _ in range(num_iters):
        # Propagate with self-loop: D^{-1/2} (A*Y + Y) D^{-1/2}
        Y_new = deg_inv_sqrt.view(-1, 1) * Y
        Y_new = adj @ Y_new + Y_new  # A*Y + Y (implicit self-loop)
        Y_new = deg_inv_sqrt.view(-1, 1) * Y_new

        # Blend with previous
        Y = (1 - alpha) * Y_new + alpha * Y

        # Clamp context set to ground truth (force truth to flow outward from few-shot samples)
        Y[context_idx] = Y_support

    return Y


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

    # Refresh Bank of Tags permutation if enabled and it's time to refresh
    if (args is not None and
        hasattr(args, 'use_bank_of_tags') and args.use_bank_of_tags and
        hasattr(args, 'bank_of_tags_refresh_interval') and args.bank_of_tags_refresh_interval > 0 and
        hasattr(pred, 'use_bank_of_tags') and pred.use_bank_of_tags and
        epoch % args.bank_of_tags_refresh_interval == 0):

        # Get number of classes for this dataset
        num_classes = int(data.y.max().item() + 1)

        # Refresh permutation with epoch-dependent seed for reproducibility
        # Use dataset name hash for additional variety if available
        dataset_hash = hash(getattr(data, 'name', 'default')) % 10000
        refresh_seed = args.seed + epoch * 1000 + dataset_hash
        pred.bank_of_tags.refresh_permutation(num_classes=num_classes, seed=refresh_seed)

        if rank == 0:
            dataset_name = getattr(data, 'name', 'unknown')
            print(f"[BankOfTags] Refreshed permutation for {dataset_name} (epoch {epoch}, {num_classes} classes)")

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
    total_contrastive_loss = 0
    total_grad_norm = 0.0
    max_grad_norm = 0.0
    grad_norm_count = 0

    # Train-time TTA (full-batch only): precompute augmented feature views once per epoch
    use_train_tta = args is not None and getattr(args, 'use_train_time_augmentation', False)
    train_tta_num_augmentations = getattr(args, 'train_tta_num_augmentations', 1) if args is not None else 0
    train_tta_include_original = getattr(args, 'train_tta_include_original', True) if args is not None else True
    tta_base_features_list = None

    if use_train_tta:
        # Avoid degenerate config
        if train_tta_num_augmentations <= 0 and not train_tta_include_original:
            if rank == 0:
                print("[Train-TTA] Disabled (no views configured).")
            use_train_tta = False
        else:
            from src.data_utils import apply_random_projection_augmentation

            # Determine PCA target dimension based on args (same as test-time TTA)
            if hasattr(data, 'needs_identity_projection') and data.needs_identity_projection:
                pca_target_dim = args.projection_small_dim
            else:
                pca_target_dim = args.hidden

            # Start from raw features if available (same as eval TTA)
            raw_features = data.x_original if hasattr(data, 'x_original') else data.x
            if not hasattr(data, 'x_original') and rank == 0:
                print(f"[Train-TTA] WARNING: x_original not found for {getattr(data, 'name', 'unknown')}, using data.x")

            tta_base_features_list = []
            if train_tta_include_original:
                tta_base_features_list.append(data.x)

            # Seed offset to vary per epoch/dataset
            dataset_hash = hash(getattr(data, 'name', 'default')) % 1000
            for aug_idx in range(train_tta_num_augmentations):
                seed = epoch * 100000 + aug_idx * 1000 + dataset_hash
                data_aug = data.clone()
                data_aug.x = raw_features.clone()

                # Apply random projection augmentation (same as eval TTA)
                data_aug = apply_random_projection_augmentation(
                    data_aug,
                    hidden_dim_range=None,
                    activation_pool=None,
                    seed=seed,
                    verbose=False,
                    rank=rank
                )

                # Apply PCA to target dimension (or pad if needed)
                if data_aug.x.shape[1] >= pca_target_dim:
                    U, S, V = torch.pca_lowrank(data_aug.x, q=pca_target_dim)
                    data_aug.x_pca = torch.mm(U, torch.diag(S))
                else:
                    U, S, V = torch.pca_lowrank(data_aug.x, q=data_aug.x.shape[1])
                    data_aug.x_pca = torch.mm(U, torch.diag(S))
                    pad_size = pca_target_dim - data_aug.x_pca.shape[1]
                    padding = torch.zeros(data_aug.x_pca.shape[0], pad_size, device=data_aug.x.device)
                    data_aug.x_pca = torch.cat([data_aug.x_pca, padding], dim=1)

                tta_base_features_list.append(data_aug.x_pca)

            if rank == 0:
                print(f"[Train-TTA] Enabled: {len(tta_base_features_list)} view(s) "
                      f"(include_original={train_tta_include_original}, num_aug={train_tta_num_augmentations})")
    for batch_idx, perm in enumerate(dataloader):
        # Batch-level context refresh for this dataset
        if args is not None:
            refresh_dataset_context_if_needed(data, {'train': train_idx}, batch_idx, epoch, args)
        
        if isinstance(perm, torch.Tensor):
            perm = perm.tolist()
        train_perm_idx = train_idx[perm]
        
        base_features = data.x
        context_y = data.y[data.context_sample]
        device = base_features.device

        # Note: GPSE embeddings are already concatenated in process_data (data_utils.py)
        # No need to concatenate again here

        # Apply edge dropout if enabled (shared across views)
        adj_t_input = data.adj_t
        if args is not None and hasattr(args, 'edge_dropout_enabled') and args.edge_dropout_enabled and hasattr(args, 'edge_dropout_rate'):
            verbose_dropout = getattr(args, 'verbose_edge_dropout', False) and rank == 0
            adj_t_input = edge_dropout_sparse_tensor(data.adj_t, args.edge_dropout_rate, training=model.training, verbose=verbose_dropout)

        # Helper: forward pass for a given base feature tensor
        def _forward_with_features(features):
            # Apply different projection strategies
            # Priority: Dynamic Encoder > FUG embeddings > identity projection > standard projection > raw features
            if hasattr(data, 'uses_dynamic_encoder') and data.uses_dynamic_encoder:
                x_input = features
            elif hasattr(data, 'uses_fug_embeddings') and data.uses_fug_embeddings and projector is not None:
                x_input = projector(features)
            elif hasattr(data, 'needs_identity_projection') and data.needs_identity_projection and identity_projection is not None:
                x_input = identity_projection(features)
            elif hasattr(data, 'needs_projection') and data.needs_projection and projector is not None:
                projected_features = projector(features)
                if hasattr(data, 'needs_final_pca') and data.needs_final_pca:
                    x_input = apply_final_pca(projected_features, projected_features.size(1))
                else:
                    x_input = projected_features
            else:
                x_input = features

            # Apply feature dropout AFTER projection
            x_input = apply_feature_dropout_if_enabled(x_input, args, rank, training=model.training)

            # GNN forward pass
            if hasattr(data, 'use_gradient_checkpointing') and data.use_gradient_checkpointing:
                h = torch.utils.checkpoint.checkpoint(model, x_input, adj_t_input)
            else:
                h = model(x_input, adj_t_input)

            return h, x_input

        if use_train_tta and tta_base_features_list is not None:
            view_logits = []
            saved_context_h = None
            saved_target_h = None
            saved_class_h = None
            saved_x_input = None

            for view_idx, view_features in enumerate(tta_base_features_list):
                h, x_input = _forward_with_features(view_features)
                context_h = h[data.context_sample]
                target_h = h[train_perm_idx]

                class_h = process_node_features(
                    context_h, data,
                    degree_normalize=degree,
                    attention_pool_module=att if att is not None else None,
                    mlp_module=mlp if mlp is not None else None,
                    normalize=normalize_class_h
                )

                pred_output = pred(data, context_h, target_h, context_y, class_h)
                if len(pred_output) == 3:
                    out, _, _ = pred_output
                else:
                    out, _ = pred_output
                view_logits.append(out)

                if view_idx == 0:
                    saved_context_h = context_h
                    saved_target_h = target_h
                    saved_class_h = class_h
                    saved_x_input = x_input

            score = torch.stack(view_logits).mean(dim=0)
            context_h = saved_context_h
            target_h = saved_target_h
            class_h = saved_class_h
            x_input = saved_x_input
        else:
            h, x_input = _forward_with_features(base_features)

        # Extract DE loss if model has Dynamic Encoder
        de_loss = torch.tensor(0.0, device=device)
        if hasattr(model, 'get_de_loss'):
            de_loss = model.get_de_loss()

        # Memory optimization: Extract needed embeddings immediately and delete large tensor
        if not use_train_tta:
            context_h = h[data.context_sample]
            target_h = h[train_perm_idx]

            class_h = process_node_features(
                context_h, data,
                degree_normalize=degree,
                attention_pool_module=att if att is not None else None,
                mlp_module=mlp if mlp is not None else None,
                normalize=normalize_class_h
            )

            # Enable debug printing in predictor for first batch
            if rank == 0 and batch_idx == 0:
                pred._debug_print = True

            score, class_h = pred(data, context_h, target_h, context_y, class_h)

        target_y = data.y[train_perm_idx]

        # Compute contrastive augmentation loss if enabled
        # This compares Transformer-generated embeddings from original vs augmented features
        contrastive_loss = torch.tensor(0.0, device=score.device)

        if (args is not None and
            hasattr(args, 'use_contrastive_augmentation_loss') and
            args.use_contrastive_augmentation_loss and
            hasattr(data, 'x_pca_original') and data.x_pca_original is not None and
            hasattr(pred, 'get_target_embeddings')):

            # Use x_pca_original directly - it's already PCA-projected with correct dimensions
            # Get GNN embeddings from original PCA features
            h_original = model(data.x_pca_original, adj_t_input)
            target_h_original = h_original[train_perm_idx]

            # Enable Transformer layer debug on first batch of epoch 0
            if rank == 0 and batch_idx == 0 and epoch == 0:
                for layer in pred.transformer_row:
                    layer._debug_collapse_step = True

            # Get Transformer-generated embeddings for both augmented and original
            target_emb_aug = pred.get_target_embeddings(context_h, target_h, context_y, class_h)
            target_emb_orig = pred.get_target_embeddings(context_h, target_h_original, context_y, class_h)

            # Disable debug after
            if rank == 0 and batch_idx == 0 and epoch == 0:
                for layer in pred.transformer_row:
                    layer._debug_collapse_step = False

            # Compute cosine similarity at different stages
            if rank == 0 and batch_idx == 0:
                # Stage 1: Input features
                input_aug = x_input[train_perm_idx]
                input_orig = data.x_pca_original[train_perm_idx]
                input_aug_norm = F.normalize(input_aug, p=2, dim=1)
                input_orig_norm = F.normalize(input_orig, p=2, dim=1)
                input_cosine_sim = (input_aug_norm * input_orig_norm).sum(dim=1).mean()

                # Stage 2: After GNN
                gnn_aug_norm = F.normalize(target_h, p=2, dim=1)
                gnn_orig_norm = F.normalize(target_h_original, p=2, dim=1)
                gnn_cosine_sim = (gnn_aug_norm * gnn_orig_norm).sum(dim=1).mean()

                # Stage 3: After Transformer
                transformer_aug_norm = F.normalize(target_emb_aug, p=2, dim=1)
                transformer_orig_norm = F.normalize(target_emb_orig, p=2, dim=1)
                transformer_cosine_sim = (transformer_aug_norm * transformer_orig_norm).sum(dim=1).mean()

                print(f"\n[DEBUG CL] Cosine Similarity at Each Stage:")
                print(f"  Stage 1 - Input Features:     {input_cosine_sim.item():.6f}")
                print(f"  Stage 2 - After GNN:           {gnn_cosine_sim.item():.6f}")
                print(f"  Stage 3 - After Transformer:   {transformer_cosine_sim.item():.6f}")

                # Check if Transformer is collapsing outputs from the start
                print(f"\n[TRANSFORMER DIAGNOSIS]:")
                # Check diversity within augmented outputs (should be high if not collapsing)
                if target_emb_aug.size(0) > 1:
                    aug_sim_matrix = F.cosine_similarity(
                        target_emb_aug.unsqueeze(1), target_emb_aug.unsqueeze(0), dim=2
                    )
                    mask = ~torch.eye(aug_sim_matrix.size(0), dtype=torch.bool, device=aug_sim_matrix.device)
                    intra_aug_sim = aug_sim_matrix[mask].mean()
                    print(f"  Diversity check - Intra-sample similarity: {intra_aug_sim.item():.6f}")
                    print(f"    (Low = diverse/healthy, High = collapsed)")

                # Check if context is dominating
                # Compare target outputs to context
                if context_h.size(0) > 0:
                    # Average context embedding
                    context_mean = context_h.mean(dim=0, keepdim=True)
                    # Similarity of target outputs to context mean
                    target_to_context_sim = F.cosine_similarity(target_emb_aug, context_mean.expand_as(target_emb_aug), dim=1).mean()
                    print(f"  Context domination check - Target vs Context similarity: {target_to_context_sim.item():.6f}")
                    print(f"    (High = context is dominating target outputs)")

                # Check if Transformer is acting like identity (residual too strong)
                input_to_output_sim = F.cosine_similarity(target_h, target_emb_aug, dim=1).mean()
                print(f"\n[TRANSFORMER BEHAVIOR]:")
                print(f"  Input→Output similarity (GNN out vs Transformer out): {input_to_output_sim.item():.6f}")
                print(f"    (High ~1.0 = Transformer is acting like identity, not transforming)")

                # Check how much the input changes after Transformer
                input_norm = target_h.norm(dim=1).mean()
                output_norm = target_emb_aug.norm(dim=1).mean()
                norm_ratio = output_norm / (input_norm + 1e-10)
                print(f"  Norm ratio (output/input): {norm_ratio.item():.4f}")
                print(f"    (Close to 1.0 = minimal transformation)")

                # Check context diversity (if context itself is collapsed, outputs will be too)
                if context_h.size(0) > 1:
                    context_sim_matrix = F.cosine_similarity(
                        context_h.unsqueeze(1), context_h.unsqueeze(0), dim=2
                    )
                    mask = ~torch.eye(context_sim_matrix.size(0), dtype=torch.bool, device=context_sim_matrix.device)
                    context_diversity = context_sim_matrix[mask].mean()
                    print(f"  Context diversity (intra-context similarity): {context_diversity.item():.6f}")
                    print(f"    (High = context nodes are similar → limited info to distinguish targets)")

                # Test the "Common Vector" / Anisotropy theory
                # Theory: y_i = x_i + c, where c is a large common vector added to all samples
                print(f"\n[ANISOTROPY TEST - Common Vector Theory]:")

                # Calculate the mean vector (center of gravity) of outputs
                mean_output = target_emb_aug.mean(dim=0, keepdim=True)  # [1, hidden_dim]

                # Mean Dominance: How similar is each sample to this mean vector?
                sim_to_mean = F.cosine_similarity(target_emb_aug, mean_output.expand_as(target_emb_aug), dim=1)
                mean_dominance = sim_to_mean.mean()

                print(f"  Mean Dominance (similarity to mean vector μ): {mean_dominance.item():.6f}")
                print(f"    ^ High value (>0.4) = All outputs point toward a common vector!")

                # Compare to input mean dominance (baseline)
                mean_input = target_h.mean(dim=0, keepdim=True)
                sim_to_mean_input = F.cosine_similarity(target_h, mean_input.expand_as(target_h), dim=1)
                mean_dominance_input = sim_to_mean_input.mean()

                print(f"  Input Mean Dominance (baseline): {mean_dominance_input.item():.6f}")
                print(f"  Anisotropy Increase: {mean_dominance_input.item():.6f} → {mean_dominance.item():.6f}")

                if mean_dominance > 0.4 and mean_dominance > mean_dominance_input + 0.1:
                    print(f"\n  ⚠ ANISOTROPY CONFIRMED!")
                    print(f"    Transformer is adding a common vector c to all samples:")
                    print(f"    y_i = x_i + c")
                    print(f"    Where c is a strong vector (likely from uniform cross-attention)")
                    print(f"    All outputs cluster around this mean vector μ ≈ c")

                # Verify the "orthogonal transformation" part: x ⊥ y but y ≈ μ
                # If y_i = x_i + c and c is large, then x_i ⊥ y_i but y_i ≈ c
                print(f"\n  Theory check:")
                print(f"    x ⊥ y (input vs output): {input_to_output_sim.item():.6f} ✓" if abs(input_to_output_sim.item()) < 0.1 else f"    x ⊥ y: {input_to_output_sim.item():.6f} ✗")
                print(f"    y ≈ μ (output vs mean):  {mean_dominance.item():.6f} ✓" if mean_dominance > 0.4 else f"    y ≈ μ: {mean_dominance.item():.6f} ✗")
                print(f"    y_i ≈ y_j (output collapse): {intra_aug_sim.item():.6f} ✓" if intra_aug_sim > 0.4 else f"    y_i ≈ y_j: {intra_aug_sim.item():.6f} ✗")

                # Calculate what the "common vector c" magnitude is
                # If y = x + c, then c ≈ mean(y) - mean(x) (approximately)
                estimated_c = mean_output - mean_input
                c_norm = estimated_c.norm().item()
                x_norm = mean_input.norm().item()
                y_norm = mean_output.norm().item()

                print(f"\n  Common vector c analysis:")
                print(f"    ||c|| (common vector magnitude): {c_norm:.4f}")
                print(f"    ||mean(x)|| (input mean norm): {x_norm:.4f}")
                print(f"    ||mean(y)|| (output mean norm): {y_norm:.4f}")
                print(f"    Ratio ||c|| / ||mean(x)||: {c_norm/(x_norm+1e-10):.4f}")
                if c_norm > x_norm * 0.5:
                    print(f"    → Common vector c is LARGE (>{x_norm*0.5:.2f}), dominating the output!")

                # Why does residual fail? Check magnitude of transformation Δ
                # If y = x + Δ, then Δ = y - x
                delta = target_emb_aug - target_h
                delta_norm_avg = delta.norm(dim=1).mean().item()
                x_norm_avg = target_h.norm(dim=1).mean().item()
                y_norm_avg = target_emb_aug.norm(dim=1).mean().item()

                print(f"\n  Residual branch analysis (y = x + Δ):")
                print(f"    Average ||x|| (input): {x_norm_avg:.4f}")
                print(f"    Average ||Δ|| (transformation): {delta_norm_avg:.4f}")
                print(f"    Average ||y|| (output): {y_norm_avg:.4f}")
                print(f"    Ratio ||Δ|| / ||x||: {delta_norm_avg/(x_norm_avg+1e-10):.4f}")
                if delta_norm_avg > x_norm_avg * 2:
                    print(f"    ⚠ RESIDUAL OVERWHELMED: ||Δ|| >> ||x||!")
                    print(f"    The transformation is {delta_norm_avg/x_norm_avg:.1f}x larger than input.")
                    print(f"    Even with residual connection, Δ dominates: y ≈ Δ")
                    print(f"\n  ROOT CAUSE: Transformer layers initialized with TOO LARGE weights!")
                    print(f"    At init, cross-attention/FFN should produce small Δ, not large Δ.")

            # Compute cosine similarity loss
            # Normalize embeddings
            emb_aug_norm = F.normalize(target_emb_aug, p=2, dim=1)
            emb_orig_norm = F.normalize(target_emb_orig, p=2, dim=1)

            # Cosine similarity: dot product of normalized vectors
            cosine_sim = (emb_aug_norm * emb_orig_norm).sum(dim=1).mean()

            # Loss: minimize distance (maximize similarity)
            contrastive_loss = (1.0 - cosine_sim) * args.contrastive_loss_weight

        # DEBUG: Compute accuracy for different methods if debug info available
        if hasattr(pred, '_debug_preds') and pred._debug_preds is not None:
            true_labels = data.y[train_perm_idx].squeeze()
            print(f"\n=== ACCURACY COMPARISON (batch) ===")
            for method, preds in pred._debug_preds.items():
                acc = (preds == true_labels).float().mean().item()
                print(f"  {method}: {acc:.4f} ({acc*100:.2f}%)")
            pred._debug_preds = None  # Clear after printing
            print()

        score = F.log_softmax(score, dim=1)
        label = data.y[train_perm_idx].squeeze()

        # Compute orthogonal loss with better numerical stability
        if orthogonal_push > 0 and class_h is not None:
            class_h_norm = F.normalize(class_h, p=2, dim=1)
            class_matrix = class_h_norm @ class_h_norm.T
            # Remove diagonal elements
            mask = ~torch.eye(class_matrix.size(0), device=class_matrix.device, dtype=torch.bool)
            orthogonal_loss = torch.sum(class_matrix[mask]**2)
        else:
            orthogonal_loss = torch.tensor(0.0, device=label.device)

        nll_loss = F.nll_loss(score, label)
        loss = nll_loss + orthogonal_push * orthogonal_loss + de_loss + contrastive_loss

        # Track separate loss components BEFORE lambda scaling
        total_nll_loss += nll_loss.item()
        total_de_loss += de_loss.item() if isinstance(de_loss, torch.Tensor) else de_loss
        total_contrastive_loss += contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else contrastive_loss
        total_loss += loss.item()  # Total before scaling

        # Apply lambda scaling for gradient update
        loss = loss * lambda_

        optimizer.zero_grad()
        loss.backward()

        # Track gradient norm before clipping (NC only)
        grad_norm_sq = 0.0
        for module in (model, pred, att, mlp, projector, identity_projection):
            if module is None:
                continue
            for p in module.parameters():
                if p.grad is not None:
                    grad_norm_sq += p.grad.data.norm(2).item() ** 2
        batch_grad_norm = grad_norm_sq ** 0.5
        total_grad_norm += batch_grad_norm
        if batch_grad_norm > max_grad_norm:
            max_grad_norm = batch_grad_norm
        grad_norm_count += 1

        # Apply gradient clipping
        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            nn.utils.clip_grad_norm_(pred.parameters(), clip_grad)
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
    avg_contrastive_loss = total_contrastive_loss / len(dataloader)
    avg_grad_norm = total_grad_norm / max(grad_norm_count, 1)

    # Print breakdown if DE or contrastive loss is being used
    if rank == 0 and (avg_de_loss > 0 or avg_contrastive_loss > 0):
        print(f"[RANK {rank}] Epoch {epoch} - Loss breakdown:")
        print(f"  Total: {avg_total_loss:.4f}")
        print(f"  NLL: {avg_nll_loss:.4f}")
        print(f"  GradNorm: avg={avg_grad_norm:.4f} max={max_grad_norm:.4f}")
        if avg_de_loss > 0:
            print(f"  DE: {avg_de_loss:.6f} (scale: {avg_de_loss/avg_nll_loss*100:.2f}% of NLL)")
        if avg_contrastive_loss > 0:
            print(f"  Contrastive: {avg_contrastive_loss:.6f} (scale: {avg_contrastive_loss/avg_nll_loss*100:.2f}% of NLL)")

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
        print(f"[RANK {rank}] Completed training epoch {epoch}, loss: {loss_str}, "
              f"grad_norm(avg/max): {avg_grad_norm:.4f}/{max_grad_norm:.4f}", flush=True)

    # Return dictionary with breakdown for logging
    return {
        'total': avg_total_loss,
        'nll': avg_nll_loss,
        'de': avg_de_loss,
        'contrastive': avg_contrastive_loss,
        'grad_norm_avg': avg_grad_norm,
        'grad_norm_max': max_grad_norm
    }

def train_all(model, data_list, split_idx_list, optimizer, pred, batch_size, degree=False, att=None,
              mlp=None, orthogonal_push=0.0, normalize_class_h=False, clip_grad=1.0, projector=None,
              rank=0, epoch=0, identity_projection=None, lambda_=1.0, args=None, external_embeddings_list=None):
    tot_loss = 0
    tot_nll_loss = 0
    tot_de_loss = 0
    tot_contrastive_loss = 0

    # Track augmented vs original losses separately
    original_loss = 0
    original_nll_loss = 0
    augmented_loss = 0
    augmented_nll_loss = 0

    # =============================================================================
    # WARNING: POTENTIALLY DEAD CODE PATH
    # =============================================================================
    # This train_all() function may NOT be called when MiniBatchNCLoader is used.
    # In train.py, nc_loaders are ALWAYS created (line 1743-1749), so the code
    # takes the minibatch path (train.py line 1884) instead of calling train_all().
    #
    # However, we're keeping this code because:
    # 1. Not sure if some code paths still use it
    # 2. MiniBatchNCLoader has fullbatch mode, unclear if it delegates to train_all
    # 3. Better to keep and comment than delete prematurely
    #
    # If you see this warning printed, this code IS being executed (not dead).
    # If you don't see it, this code is likely dead for your configuration.
    # =============================================================================

    # Check augmentation settings
    use_augmentation = args is not None and hasattr(args, 'use_random_projection_augmentation') and args.use_random_projection_augmentation
    augmentation_mode = getattr(args, 'augmentation_mode', 'preprocessing') if use_augmentation else None
    num_augmentations = getattr(args, 'num_augmentations', 1) if use_augmentation else 0
    include_original = getattr(args, 'augmentation_include_original', True) if use_augmentation else True

    # DEBUG: Check if this code path is actually executed
    if rank == 0:
        print(f"\n{'='*80}")
        print(f"[WARNING] train_all() in engine_nc.py IS BEING CALLED")
        print(f"  This path may be DEAD CODE if MiniBatchNCLoader is used")
        print(f"  Epoch {epoch}, use_augmentation={use_augmentation}, mode={augmentation_mode}")
        print(f"{'='*80}\n")

    # For per-epoch mode, create augmented data on-the-fly at specified intervals
    augmentation_interval = getattr(args, 'augmentation_regenerate_interval', 1)
    should_regenerate = (augmentation_mode == 'per_epoch' and epoch % augmentation_interval == 0)

    if use_augmentation and should_regenerate:
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"[Augmentation Regeneration] Epoch {epoch}: Regenerating augmentations (interval={augmentation_interval})")
            print(f"{'='*80}\n")

        from src.data_utils import apply_random_projection_augmentation, process_data

        # Extract only original graphs (data_list contains original + old augmented from loading)
        num_original_graphs = len(data_list) // (1 + num_augmentations)
        original_data_list = data_list[:num_original_graphs]
        original_split_idx_list = split_idx_list[:num_original_graphs]

        # DEBUG: Check what we're starting with
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"[DEBUG Per-Epoch Augmentation] Epoch {epoch}")
            print(f"  Total data_list length: {len(data_list)}")
            print(f"  Extracted {num_original_graphs} original graphs")
            print(f"  original_data_list[0].x (PCA-processed): shape={original_data_list[0].x.shape}, mean={original_data_list[0].x.mean().item():.6f}")
            if hasattr(original_data_list[0], 'x_original'):
                print(f"  original_data_list[0].x_original (RAW): shape={original_data_list[0].x_original.shape}, mean={original_data_list[0].x_original.mean().item():.6f}")
                print(f"  Will augment from RAW features and redo PCA")
            else:
                print(f"  WARNING: x_original not found! Cannot do proper per-epoch augmentation")
            print(f"{'='*80}\n")

        # Get activation setting
        augmentation_activation = getattr(args, 'augmentation_activation', 'random')
        augmentation_max_depth = getattr(args, 'augmentation_max_depth', 1)

        if augmentation_activation == 'random':
            activation_pool_to_use = None  # Will use default diverse pool
            activation_desc = "Random (sampling from diverse pool)"
        else:
            activation_pool_to_use = [augmentation_activation]  # Fixed activation
            activation_desc = f"Fixed ({augmentation_activation})"

        if augmentation_max_depth == 1:
            depth_desc = "Single layer σ(WX+b)"
        else:
            depth_desc = f"Random depth in [1, {augmentation_max_depth}] layers"

        if rank == 0:
            print(f"\n[Per-Epoch Augmentation] Epoch {epoch}: Creating {num_augmentations} augmented copy/copies")
            print(f"  Activation: {activation_desc}")
            print(f"  MLP Depth: {depth_desc}")

        # Create augmented data with epoch-dependent seed for variety
        augmented_data_list = []
        for copy_idx in range(num_augmentations):
            for graph_idx, data in enumerate(original_data_list):
                # Use epoch in seed to get different augmentations each epoch
                seed = epoch * 100000 + copy_idx * 10000 + graph_idx * 100 + 42

                # Get augmentation settings
                augmentation_use_random_noise = getattr(args, 'augmentation_use_random_noise', False)
                augmentation_dropout_rate = getattr(args, 'augmentation_dropout_rate', 0.0)
                augmentation_use_feature_mixing = getattr(args, 'augmentation_use_feature_mixing', False)
                augmentation_mix_ratio = getattr(args, 'augmentation_mix_ratio', 0.3)
                augmentation_mix_alpha = getattr(args, 'augmentation_mix_alpha', 0.5)

                # Create a temporary data object with ORIGINAL raw features for augmentation
                data_for_aug = data.clone()
                if hasattr(data, 'x_original'):
                    data_for_aug.x = data.x_original.clone()  # Use RAW features

                    # DEBUG: First graph only
                    if graph_idx == 0 and copy_idx == 0 and rank == 0:
                        print(f"  [DEBUG] About to augment:")
                        print(f"    data_for_aug.x (RAW): shape={data_for_aug.x.shape}, mean={data_for_aug.x.mean().item():.6f}")
                else:
                    # Fallback: use current data.x (not ideal)
                    if graph_idx == 0 and copy_idx == 0 and rank == 0:
                        print(f"  [WARNING] x_original not available, using PCA-processed data.x")

                # Apply augmentation to RAW features
                data_aug = apply_random_projection_augmentation(
                    data_for_aug,
                    hidden_dim_range=None,
                    activation_pool=activation_pool_to_use,
                    seed=seed,
                    verbose=False,
                    rank=rank,
                    use_random_noise=augmentation_use_random_noise,
                    max_depth=augmentation_max_depth,
                    dropout_rate=augmentation_dropout_rate,
                    use_feature_mixing=augmentation_use_feature_mixing,
                    mix_ratio=augmentation_mix_ratio,
                    mix_alpha=augmentation_mix_alpha
                )

                # DEBUG: Check augmented features (first graph of first copy only)
                if graph_idx == 0 and copy_idx == 0:
                    print(f"  [DEBUG] After augmentation (rank={rank}):")
                    print(f"    data_aug.x (augmented RAW): shape={data_aug.x.shape}, mean={data_aug.x.mean().item():.6f}")

                    # Check similarity before PCA
                    aug_raw = data_aug.x
                    orig_raw = data.x_original
                    aug_raw_norm = F.normalize(aug_raw, p=2, dim=1)
                    orig_raw_norm = F.normalize(orig_raw, p=2, dim=1)
                    raw_cosine_sim = (aug_raw_norm * orig_raw_norm).sum(dim=1).mean()
                    print(f"    Raw features cosine similarity (BEFORE PCA): {raw_cosine_sim.item():.6f}")

                # Now apply process_data to do PCA/padding on the augmented features
                split_idx = original_split_idx_list[graph_idx]
                external_emb = external_embeddings_list[graph_idx] if external_embeddings_list else None

                # Call process_data to apply PCA/padding/normalization
                process_data(
                    data_aug, split_idx, args.hidden, args.context_num, False, args.use_full_pca,
                    args.normalize_data, False, 32, rank, args.padding_strategy,
                    args.use_batchnorm, args.use_identity_projection,
                    args.projection_small_dim, args.projection_large_dim, args.pca_device,
                    args.incremental_pca_batch_size, external_emb, args.use_random_orthogonal,
                    args.use_sparse_random, args.sparse_random_density,
                    False, './tsne_plots', args.use_pca_whitening, args.whitening_epsilon,
                    args.use_quantile_normalization, args.quantile_norm_before_padding,
                    getattr(args, 'use_external_embeddings_nc', False),
                    args.use_dynamic_encoder, False, args.use_orthogonal_noise
                )

                # DEBUG: Check after PCA
                if graph_idx == 0 and copy_idx == 0 and rank == 0:
                    print(f"  [DEBUG] After process_data (PCA/padding):")
                    print(f"    data_aug.x (final): shape={data_aug.x.shape}, mean={data_aug.x.mean().item():.6f}")
                    print(f"{'='*80}\n")

                augmented_data_list.append(data_aug)

        # Combine original and augmented based on include_original setting
        if include_original:
            training_data_list = original_data_list + augmented_data_list
            training_split_idx_list = original_split_idx_list * (1 + num_augmentations)
            num_original = len(original_data_list)
            num_augmented = len(augmented_data_list)
        else:
            # Train only on augmented graphs
            training_data_list = augmented_data_list
            training_split_idx_list = original_split_idx_list * num_augmentations
            num_original = 0
            num_augmented = len(augmented_data_list)

        if rank == 0:
            if include_original:
                print(f"  Created {num_augmented} augmented graphs (total: {len(training_data_list)} with original)")
            else:
                print(f"  Created {num_augmented} augmented graphs (training only on augmented, no original)")

    elif use_augmentation and augmentation_mode == 'preprocessing':
        # Preprocessing mode: augmented data already in data_list
        if include_original:
            training_data_list = data_list
            training_split_idx_list = split_idx_list
            num_original = len(data_list) // (1 + num_augmentations)
            num_augmented = len(data_list) - num_original

            if rank == 0 and epoch == 0:
                print(f"[DEBUG] Preprocessing mode: num_original={num_original}, num_augmented={num_augmented}")
        else:
            # Extract only augmented graphs (they're at the end of data_list)
            num_original_graphs = len(data_list) // (1 + num_augmentations)
            training_data_list = data_list[num_original_graphs:]
            training_split_idx_list = split_idx_list[num_original_graphs:]
            num_original = 0
            num_augmented = len(training_data_list)

            if rank == 0 and epoch == 0:
                print(f"\n[Augmentation] Training only on {num_augmented} augmented graphs (excluding {num_original_graphs} original)\n")
    else:
        # No augmentation
        training_data_list = data_list
        training_split_idx_list = split_idx_list
        num_original = len(data_list)
        num_augmented = 0

    # Shuffle training data if requested (mixes original and augmented graphs)
    augmentation_shuffle = getattr(args, 'augmentation_shuffle', False) if args is not None else False
    if augmentation_shuffle:
        import random
        # Track which graphs are augmented before shuffling
        is_augmented_list = [i >= num_original for i in range(len(training_data_list))]

        # Include external_embeddings_list in shuffle if it exists
        if external_embeddings_list:
            paired_data = list(zip(training_data_list, training_split_idx_list, is_augmented_list, external_embeddings_list))
        else:
            paired_data = list(zip(training_data_list, training_split_idx_list, is_augmented_list))

        # Use epoch-dependent seed for reproducibility but different order each epoch
        shuffle_seed = epoch * 12345 + 67890
        random.Random(shuffle_seed).shuffle(paired_data)

        if external_embeddings_list:
            training_data_list, training_split_idx_list, is_augmented_list, external_embeddings_list = zip(*paired_data)
            external_embeddings_list = list(external_embeddings_list)
        else:
            training_data_list, training_split_idx_list, is_augmented_list = zip(*paired_data)

        training_data_list = list(training_data_list)
        training_split_idx_list = list(training_split_idx_list)
        is_augmented_list = list(is_augmented_list)

        if rank == 0 and epoch == 0:
            print(f"[Shuffling] Training data shuffled with epoch-dependent seed (epoch={epoch}, seed={shuffle_seed})")
    else:
        # No shuffling - use index-based detection
        is_augmented_list = None

    for i, (data, split_idx) in enumerate(zip(training_data_list, training_split_idx_list)):
        train_idx = split_idx['train']
        external_embeddings = external_embeddings_list[i] if external_embeddings_list else None
        loss_dict = train(model, data, train_idx, optimizer, pred, batch_size, degree, att, mlp,
                          orthogonal_push, normalize_class_h, clip_grad, projector, rank, epoch,
                          identity_projection, lambda_, args, external_embeddings)

        # Determine if this is an original or augmented graph
        if is_augmented_list is not None:
            is_augmented = is_augmented_list[i]
        else:
            is_augmented = i >= num_original

        # Handle both dict and scalar returns (backward compatibility)
        if isinstance(loss_dict, dict):
            loss_val = loss_dict['total']
            tot_loss += loss_val
            tot_nll_loss += loss_dict['nll']
            tot_de_loss += loss_dict['de']
            tot_contrastive_loss += loss_dict.get('contrastive', 0)

            # Track augmented vs original separately
            if is_augmented:
                augmented_loss += loss_val
                augmented_nll_loss += loss_dict['nll']
            else:
                original_loss += loss_val
                original_nll_loss += loss_dict['nll']

            if rank == 0:
                prefix = "[AUG]" if is_augmented else "[ORI]"
                contrastive_str = f", Contrastive: {loss_dict.get('contrastive', 0):.6f}" if loss_dict.get('contrastive', 0) > 0 else ""
                print(f"{prefix} Dataset {data.name} - Total: {loss_val:.4f}, NLL: {loss_dict['nll']:.4f}, DE: {loss_dict['de']:.6f}{contrastive_str}", flush=True)
        else:
            tot_loss += loss_dict

            # Track augmented vs original separately
            if is_augmented:
                augmented_loss += loss_dict
            else:
                original_loss += loss_dict

            if rank == 0:
                prefix = "[AUG]" if is_augmented else "[ORI]"
                print(f"{prefix} Dataset {data.name} Loss: {loss_dict}", flush=True)

    # Calculate averages
    avg_total = tot_loss / len(data_list)
    avg_nll = tot_nll_loss / len(data_list)
    avg_de = tot_de_loss / len(data_list)

    # Calculate separate averages for original and augmented
    avg_original_total = original_loss / num_original if num_original > 0 else 0
    avg_original_nll = original_nll_loss / num_original if num_original > 0 else 0
    avg_augmented_total = augmented_loss / num_augmented if num_augmented > 0 else 0
    avg_augmented_nll = augmented_nll_loss / num_augmented if num_augmented > 0 else 0

    # Print summary if augmentation is used
    if rank == 0 and num_augmented > 0:
        print(f"\n{'='*80}")
        print(f"[LOSS SUMMARY] Epoch {epoch}")
        print(f"  Overall Avg Loss: {avg_total:.4f} (NLL: {avg_nll:.4f})")
        print(f"  Original Graphs ({num_original}): Avg Loss: {avg_original_total:.4f} (NLL: {avg_original_nll:.4f})")
        print(f"  Augmented Graphs ({num_augmented}): Avg Loss: {avg_augmented_total:.4f} (NLL: {avg_augmented_nll:.4f})")
        print(f"  Difference (Aug - Ori): {avg_augmented_total - avg_original_total:+.4f} (NLL: {avg_augmented_nll - avg_original_nll:+.4f})")
        print(f"{'='*80}\n")

    # Return dictionary with averages
    return {
        'total': avg_total,
        'nll': avg_nll,
        'de': avg_de,
        'original_total': avg_original_total,
        'original_nll': avg_original_nll,
        'augmented_total': avg_augmented_total,
        'augmented_nll': avg_augmented_nll,
        'num_original': num_original,
        'num_augmented': num_augmented
    }

@torch.no_grad()
def test(model, predictor, data, train_idx, valid_idx, test_idx, batch_size, degree=False,
         att=None, mlp=None, normalize_class_h=False, projector=None, rank=0, identity_projection=None,
         external_embeddings=None, args=None, use_cs=False, cs_num_iters=50, cs_alpha=0.5, epoch=0):
    st = time.time()
    model.eval()
    predictor.eval()
    if projector is not None:
        projector.eval()
    if identity_projection is not None:
        identity_projection.eval()

    # Refresh Bank of Tags permutation to match training (use same seed as training for this epoch)
    # This ensures test evaluation uses the same tag mapping as training
    if (args is not None and
        hasattr(args, 'use_bank_of_tags') and args.use_bank_of_tags and
        hasattr(predictor, 'use_bank_of_tags') and predictor.use_bank_of_tags):

        # Get number of classes for this dataset
        num_classes = int(data.y.max().item() + 1)

        # Use same seed as training to ensure consistency
        dataset_hash = hash(getattr(data, 'name', 'default')) % 10000
        refresh_seed = args.seed + epoch * 1000 + dataset_hash
        predictor.bank_of_tags.refresh_permutation(num_classes=num_classes, seed=refresh_seed)

    device = next(model.parameters()).device
    moved_to_device = False
    if data.x.device != device:
        data = data.to(device)
        train_idx = train_idx.to(device)
        valid_idx = valid_idx.to(device)
        test_idx = test_idx.to(device)
        if hasattr(data, 'context_sample') and data.context_sample is not None:
            data.context_sample = data.context_sample.to(device)
        moved_to_device = True

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

    # Get base predictions for all splits
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

    # Apply C&S if enabled with validation-based selection
    # Decision: Only use C&S if it improves VALIDATION performance (no test leakage)
    if use_cs:
        # Build full logits tensor for all nodes
        all_logits = torch.zeros(data.num_nodes, train_logits.size(1), device=h.device)
        all_logits[train_idx] = train_logits
        all_logits[valid_idx] = valid_logits
        all_logits[test_idx] = test_logits

        # Apply Correct & Smooth
        # IMPORTANT: Use only context samples, not full training split, to avoid label leakage!
        num_classes = context_y.max().item() + 1
        cs_predictions = correct_and_smooth(
            data.adj_t, all_logits, data.context_sample, data.y[data.context_sample],
            num_classes, num_iters=cs_num_iters, alpha=cs_alpha
        )

        # Validation-based selection: Compare base vs C&S on validation set only
        valid_y = data.y[valid_idx]
        valid_acc_base = acc(valid_y, valid_score_base)
        valid_acc_cs = acc(valid_y, cs_predictions[valid_idx].argmax(dim=1))

        # Decision: Use C&S if it improves validation performance
        if valid_acc_cs > valid_acc_base:
            train_score = cs_predictions[train_idx].argmax(dim=1)
            valid_score = cs_predictions[valid_idx].argmax(dim=1)
            test_score = cs_predictions[test_idx].argmax(dim=1)
            if rank == 0:
                dataset_name = getattr(data, 'name', 'unknown')
                print(f"[C&S] Using C&S for {dataset_name} (valid: {valid_acc_base:.4f} → {valid_acc_cs:.4f})")
        else:
            # Keep base predictions
            train_score = train_score_base
            valid_score = valid_score_base
            test_score = test_score_base
            if rank == 0:
                dataset_name = getattr(data, 'name', 'unknown')
                print(f"[C&S] Skipping C&S for {dataset_name} (valid: {valid_acc_base:.4f} ≥ {valid_acc_cs:.4f})")
    else:
        # Use base predictions
        train_score = train_score_base
        valid_score = valid_score_base
        test_score = test_score_base

    # calculate valid metric
    valid_y = data.y[valid_idx]
    valid_results = acc(valid_y, valid_score)
    train_y = data.y[train_idx]
    train_results = acc(train_y, train_score)
    test_y = data.y[test_idx]
    test_results = acc(test_y, test_score)

    if moved_to_device:
        data = data.cpu()
        if hasattr(data, 'context_sample') and data.context_sample is not None:
            data.context_sample = data.context_sample.cpu()
        train_idx = train_idx.cpu()
        valid_idx = valid_idx.cpu()
        test_idx = test_idx.cpu()

    if rank == 0:
        print(f"Test time: {time.time()-st}", flush=True)
    return train_results, valid_results, test_results

def test_all(model, predictor, data_list, split_idx_list, batch_size, degree=False, att=None,
             mlp=None, normalize_class_h=False, projector=None, rank=0, identity_projection=None,
             external_embeddings_list=None, use_cs=False, cs_num_iters=50, cs_alpha=0.5):
    tot_train_metric, tot_valid_metric, tot_test_metric = 1, 1, 1
    for i, (data, split_idx) in enumerate(zip(data_list, split_idx_list)):
        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']
        external_embeddings = external_embeddings_list[i] if external_embeddings_list else None

        train_metric, valid_metric, test_metric = \
        test(model, predictor, data, train_idx, valid_idx, test_idx, batch_size, degree, att, mlp,
             normalize_class_h, projector, rank, identity_projection, external_embeddings, args=None,
             use_cs=use_cs, cs_num_iters=cs_num_iters, cs_alpha=cs_alpha)
        if rank == 0:
            print(f"Dataset {data.name}")
            print(f"Train {train_metric}, Valid {valid_metric}, Test {test_metric}", flush=True)
        tot_train_metric *= train_metric
        tot_valid_metric *= valid_metric
        tot_test_metric *= test_metric
    return tot_train_metric ** (1/(len(data_list))), tot_valid_metric ** (1/(len(data_list))), \
           tot_test_metric ** (1/(len(data_list)))

def test_all_induct(model, predictor, data_list, split_idx_list, batch_size, degree=False,
                    att=None, mlp=None, normalize_class_h=False, projector=None, rank=0, identity_projection=None,
                    external_embeddings_list=None, use_cs=False, cs_num_iters=50, cs_alpha=0.5, args=None, epoch=0):
    import time

    train_metric_list, valid_metric_list, test_metric_list = [], [], []

    # Track augmented vs original metrics separately - ONLY for training datasets
    use_augmentation = args is not None and hasattr(args, 'use_random_projection_augmentation') and args.use_random_projection_augmentation
    num_augmentations = getattr(args, 'num_augmentations', 1) if use_augmentation else 0

    # Calculate number of original graphs
    if use_augmentation and num_augmentations > 0:
        num_original = len(data_list) // (1 + num_augmentations)
        num_augmented = len(data_list) - num_original
    else:
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
        train_metric, valid_metric, test_metric = \
        test(model, predictor, data, train_idx, valid_idx, test_idx, batch_size, degree, att, mlp,
             normalize_class_h, projector, rank, identity_projection, external_embeddings, args=args,
             use_cs=use_cs, cs_num_iters=cs_num_iters, cs_alpha=cs_alpha, epoch=epoch)

        dataset_time = time.time() - dataset_start_time

        # Track augmented vs original separately ONLY if this is a training dataset with augmentation
        is_augmented = dataset_idx >= num_original
        if is_augmented:
            augmented_valid_metrics.append(valid_metric)
        else:
            original_valid_metrics.append(valid_metric)

        if rank == 0:
            prefix = "[AUG]" if is_augmented else "[ORI]"
            print(f"    {prefix} Dataset {dataset_idx} ({data.name}): completed in {dataset_time:.2f}s")
            print(f"      Train {train_metric:.4f}, Valid {valid_metric:.4f}, Test {test_metric:.4f}", flush=True)
        train_metric_list.append(train_metric)
        valid_metric_list.append(valid_metric)
        test_metric_list.append(test_metric)

    # Print summary if augmentation is used
    if rank == 0 and num_augmented > 0:
        import numpy as np
        avg_original_valid = np.mean(original_valid_metrics) if original_valid_metrics else 0
        avg_augmented_valid = np.mean(augmented_valid_metrics) if augmented_valid_metrics else 0
        print(f"\n{'='*80}")
        print(f"[VALIDATION SUMMARY - Training Datasets]")
        print(f"  Original Graphs ({num_original}): Avg Valid Acc: {avg_original_valid:.4f}")
        print(f"  Augmented Graphs ({num_augmented}): Avg Valid Acc: {avg_augmented_valid:.4f}")
        print(f"  Difference (Aug - Ori): {avg_augmented_valid - avg_original_valid:+.4f}")
        print(f"{'='*80}\n")

    return train_metric_list, valid_metric_list, test_metric_list


def test_all_induct_with_tta(model, predictor, data_list, split_idx_list, batch_size, degree=False,
                              att=None, mlp=None, normalize_class_h=False, projector=None, rank=0, identity_projection=None,
                              external_embeddings_list=None, use_cs=False, cs_num_iters=50, cs_alpha=0.5, args=None):
    """
    Test UNSEEN datasets with Test-Time Augmentation (TTA).

    For each unseen test graph:
    1. Create K augmented versions using random projection
    2. Run inference on all versions (+ original if configured)
    3. Aggregate predictions by averaging logits

    This should ONLY be used for unseen test datasets, not training datasets.

    Args:
        Same as test_all_induct, plus:
        args: Should contain TTA settings:
            - tta_num_augmentations: Number of augmented versions to create
            - tta_include_original: Whether to include original graph in aggregation
            - tta_aggregation: 'logits' (default), 'probs', or 'voting'

    Returns:
        train_metric_list, valid_metric_list, test_metric_list: Same as test_all_induct
    """
    import time
    import torch
    import torch.nn.functional as F
    from src.data_utils import apply_random_projection_augmentation

    # TTA configuration
    tta_num_augmentations = getattr(args, 'tta_num_augmentations', 5)
    tta_include_original = getattr(args, 'tta_include_original', True)
    tta_aggregation = getattr(args, 'tta_aggregation', 'logits')
    tta_gate_by_valid = getattr(args, 'tta_gate_by_valid', True)

    if rank == 0:
        print(f"\n{'='*80}")
        print(f"[TEST-TIME AUGMENTATION (TTA) - Unseen Test Datasets Only]")
        print(f"  Number of augmentations: {tta_num_augmentations}")
        print(f"  Include original: {tta_include_original}")
        print(f"  Aggregation strategy: {tta_aggregation}")
        print(f"  Gate by valid: {tta_gate_by_valid}")
        print(f"  Total versions per graph: {tta_num_augmentations + (1 if tta_include_original else 0)}")
        print(f"{'='*80}\n")

    train_metric_list, valid_metric_list, test_metric_list = [], [], []

    for dataset_idx, (data, split_idx) in enumerate(zip(data_list, split_idx_list)):
        dataset_start_time = time.time()

        if rank == 0:
            print(f"  [TTA] Processing unseen dataset {dataset_idx} ({data.name})...")

        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']
        external_embeddings = external_embeddings_list[dataset_idx] if external_embeddings_list else None

        # Determine PCA target dimension based on args (same as in process_data)
        if hasattr(data, 'needs_identity_projection') and data.needs_identity_projection:
            pca_target_dim = args.projection_small_dim
        else:
            pca_target_dim = args.hidden

        # Create augmented versions
        augmented_data_list = []
        if tta_include_original:
            augmented_data_list.append(data)

        for aug_idx in range(tta_num_augmentations):
            # Use deterministic seed for reproducibility: base 999000 + graph_idx * 1000 + aug_idx
            seed = 999000 + dataset_idx * 1000 + aug_idx

            # TTA chain: ORIGINAL → AUG → PCA → MODEL
            # Step 1: Start with original features (before PCA)
            data_aug = data.clone()
            if hasattr(data, 'x_original'):
                data_aug.x = data.x_original.clone()
            else:
                if rank == 0 and aug_idx == 0:
                    print(f"    WARNING: x_original not found for {data.name}, using data.x (may be PCA-processed!)")

            # Step 2: Apply augmentation (σ(WX+b))
            data_aug = apply_random_projection_augmentation(
                data_aug,
                hidden_dim_range=None,  # Use default (0.5x to 2.0x)
                activation_pool=None,   # Use default diverse pool
                seed=seed,
                verbose=False,
                rank=rank
            )

            # Step 3: Apply PCA to augmented features
            # Apply PCA to target dimension (or less if we don't have enough features)
            if data_aug.x.shape[1] >= pca_target_dim:
                # PCA to target dimension
                U, S, V = torch.pca_lowrank(data_aug.x, q=pca_target_dim)
                data_aug.x_pca = torch.mm(U, torch.diag(S))
            else:
                # PCA to whatever we have, then pad
                U, S, V = torch.pca_lowrank(data_aug.x, q=data_aug.x.shape[1])
                data_aug.x_pca = torch.mm(U, torch.diag(S))
                # Pad to target dimension
                pad_size = pca_target_dim - data_aug.x_pca.shape[1]
                padding = torch.zeros(data_aug.x_pca.shape[0], pad_size, device=data_aug.x.device)
                data_aug.x_pca = torch.cat([data_aug.x_pca, padding], dim=1)

            # Set the processed features as the main features
            data_aug.x = data_aug.x_pca

            # Copy other necessary attributes from original data
            if hasattr(data, 'needs_identity_projection'):
                data_aug.needs_identity_projection = data.needs_identity_projection
            if hasattr(data, 'projection_target_dim'):
                data_aug.projection_target_dim = data.projection_target_dim
            if hasattr(data, 'needs_projection'):
                data_aug.needs_projection = data.needs_projection

            augmented_data_list.append(data_aug)

        if rank == 0:
            print(f"    Created {len(augmented_data_list)} versions (original + {tta_num_augmentations} augmented)")

        # Collect logits from all augmented versions
        all_train_logits = []
        all_valid_logits = []
        all_test_logits = []

        for version_idx, data_version in enumerate(augmented_data_list):
            # Run inference on this version
            train_logits, valid_logits, test_logits = test_single_with_logits(
                model, predictor, data_version, train_idx, valid_idx, test_idx,
                batch_size, degree, att, mlp, normalize_class_h, projector, rank,
                identity_projection, external_embeddings, args, use_cs, cs_num_iters, cs_alpha
            )

            all_train_logits.append(train_logits)
            all_valid_logits.append(valid_logits)
            all_test_logits.append(test_logits)

            # DEBUG: Print accuracy of each view
            if rank == 0:
                y_view = data.y.to(train_logits.device)
                train_idx_view = train_idx.to(train_logits.device)
                valid_idx_view = valid_idx.to(train_logits.device)
                test_idx_view = test_idx.to(train_logits.device)

                train_pred = train_logits.argmax(dim=-1)
                valid_pred = valid_logits.argmax(dim=-1)
                test_pred = test_logits.argmax(dim=-1)

                train_acc = (train_pred == y_view[train_idx_view]).float().mean().item()
                valid_acc = (valid_pred == y_view[valid_idx_view]).float().mean().item()
                test_acc = (test_pred == y_view[test_idx_view]).float().mean().item()

                view_type = "Original" if version_idx == 0 and tta_include_original else f"Aug_{version_idx}"
                print(f"      [{view_type}] Train: {train_acc:.4f}, Valid: {valid_acc:.4f}, Test: {test_acc:.4f}")

        # DEBUG: Print individual view accuracies for comparison
        if rank == 0:
            y_all = data.y.to(all_test_logits[0].device)
            test_idx_dev = test_idx.to(all_test_logits[0].device)
            individual_test_accs = []
            for logits in all_test_logits:
                pred = logits.argmax(dim=-1)
                acc = (pred == y_all[test_idx_dev]).float().mean().item()
                individual_test_accs.append(acc)
            print(f"      Individual test accs: mean={sum(individual_test_accs)/len(individual_test_accs):.4f}, "
                  f"min={min(individual_test_accs):.4f}, max={max(individual_test_accs):.4f}")

        # Aggregate logits across all TTA versions
        if tta_aggregation == 'logits':
            # Average logits (recommended)
            train_logits_agg = torch.stack(all_train_logits).mean(dim=0)
            valid_logits_agg = torch.stack(all_valid_logits).mean(dim=0)
            test_logits_agg = torch.stack(all_test_logits).mean(dim=0)

        elif tta_aggregation == 'probs':
            # Average probabilities, convert back to logits
            train_probs = torch.stack([F.softmax(logits, dim=-1) for logits in all_train_logits]).mean(dim=0)
            valid_probs = torch.stack([F.softmax(logits, dim=-1) for logits in all_valid_logits]).mean(dim=0)
            test_probs = torch.stack([F.softmax(logits, dim=-1) for logits in all_test_logits]).mean(dim=0)

            train_logits_agg = torch.log(train_probs + 1e-10)
            valid_logits_agg = torch.log(valid_probs + 1e-10)
            test_logits_agg = torch.log(test_probs + 1e-10)

            # DEBUG: Verify aggregation
            if rank == 0:
                y_agg = data.y.to(test_logits_agg.device)
                test_idx_dev = test_idx.to(test_logits_agg.device)
                agg_test_pred = test_logits_agg.argmax(dim=-1)
                agg_test_acc = (agg_test_pred == y_agg[test_idx_dev]).float().mean().item()
                print(f"      After aggregation: test_acc={agg_test_acc:.4f} (boost: +{agg_test_acc - sum(individual_test_accs)/len(individual_test_accs):.4f})")

        elif tta_aggregation == 'voting':
            # Majority voting - get most common prediction
            train_preds = torch.stack([logits.argmax(dim=-1) for logits in all_train_logits])
            valid_preds = torch.stack([logits.argmax(dim=-1) for logits in all_valid_logits])
            test_preds = torch.stack([logits.argmax(dim=-1) for logits in all_test_logits])

            train_pred_mode = torch.mode(train_preds, dim=0).values
            valid_pred_mode = torch.mode(valid_preds, dim=0).values
            test_pred_mode = torch.mode(test_preds, dim=0).values

            # Convert to one-hot "logits" (hard labels)
            num_classes = all_train_logits[0].size(1)
            train_logits_agg = F.one_hot(train_pred_mode, num_classes=num_classes).float() * 10.0  # Scale for numerical stability
            valid_logits_agg = F.one_hot(valid_pred_mode, num_classes=num_classes).float() * 10.0
            test_logits_agg = F.one_hot(test_pred_mode, num_classes=num_classes).float() * 10.0


        metric_device = train_logits_agg.device
        y_device = data.y.to(metric_device)
        train_idx_dev = train_idx.to(metric_device)
        valid_idx_dev = valid_idx.to(metric_device)
        test_idx_dev = test_idx.to(metric_device)
        context_sample_dev = data.context_sample.to(metric_device)

        # Optionally gate TTA by validation performance (per dataset)
        train_logits_sel = train_logits_agg
        valid_logits_sel = valid_logits_agg
        test_logits_sel = test_logits_agg

        if tta_gate_by_valid:
            if not tta_include_original:
                if rank == 0:
                    print("    [TTA] Gating skipped (tta_include_original=False).")
            else:
                train_logits_base = all_train_logits[0]
                valid_logits_base = all_valid_logits[0]
                test_logits_base = all_test_logits[0]

                valid_acc_base = (valid_logits_base.argmax(dim=-1) == y_device[valid_idx_dev]).float().mean().item()
                valid_acc_tta = (valid_logits_agg.argmax(dim=-1) == y_device[valid_idx_dev]).float().mean().item()

                if valid_acc_tta > valid_acc_base:
                    if rank == 0:
                        print(f"    [TTA] Valid improved: {valid_acc_base:.4f} → {valid_acc_tta:.4f} ✓")
                else:
                    train_logits_sel = train_logits_base
                    valid_logits_sel = valid_logits_base
                    test_logits_sel = test_logits_base
                    if rank == 0:
                        print(f"    [TTA] Valid not improved: {valid_acc_tta:.4f} ≤ {valid_acc_base:.4f} (using original)")

        # Get base predictions from selected logits
        train_score_base = train_logits_sel.argmax(dim=-1)
        valid_score_base = valid_logits_sel.argmax(dim=-1)
        test_score_base = test_logits_sel.argmax(dim=-1)

        # Apply C&S if enabled (Post-TTA aggregation)
        if use_cs:
            if rank == 0:
                print(f"    Applying C&S to aggregated TTA logits...")

            # Build full logits tensor for all nodes
            all_logits = torch.zeros(data.num_nodes, train_logits_agg.size(1), device=metric_device)
            all_logits[train_idx_dev] = train_logits_sel
            all_logits[valid_idx_dev] = valid_logits_sel
            all_logits[test_idx_dev] = test_logits_sel

            # Get context for C&S
            context_y = y_device[context_sample_dev]
            num_classes = context_y.max().item() + 1

            cs_predictions = correct_and_smooth(
                data.adj_t.to(metric_device), all_logits, context_sample_dev, context_y,
                num_classes, cs_num_iters, cs_alpha
            )

            # Get C&S predictions
            train_score_cs = cs_predictions[train_idx_dev].argmax(dim=-1)
            valid_score_cs = cs_predictions[valid_idx_dev].argmax(dim=-1)
            test_score_cs = cs_predictions[test_idx_dev].argmax(dim=-1)

            # Evaluate both and choose based on validation
            valid_acc_base = (valid_score_base == y_device[valid_idx_dev]).float().mean().item()
            valid_acc_cs = (valid_score_cs == y_device[valid_idx_dev]).float().mean().item()

            if valid_acc_cs > valid_acc_base:
                train_pred = train_score_cs
                valid_pred = valid_score_cs
                test_pred = test_score_cs
                if rank == 0:
                    print(f"    C&S improved validation: {valid_acc_base:.4f} → {valid_acc_cs:.4f} ✓")
            else:
                train_pred = train_score_base
                valid_pred = valid_score_base
                test_pred = test_score_base
                if rank == 0:
                    print(f"    C&S did not improve: {valid_acc_base:.4f} ≥ {valid_acc_cs:.4f} (using base)")
        else:
            # No C&S, use base aggregated predictions
            train_pred = train_score_base
            valid_pred = valid_score_base
            test_pred = test_score_base

        # Compute final metrics
        train_metric = (train_pred == y_device[train_idx_dev]).float().mean().item()
        valid_metric = (valid_pred == y_device[valid_idx_dev]).float().mean().item()
        test_metric = (test_pred == y_device[test_idx_dev]).float().mean().item()

        dataset_time = time.time() - dataset_start_time

        if rank == 0:
            print(f"    [TTA] Dataset {dataset_idx} ({data.name}): completed in {dataset_time:.2f}s")
            print(f"      Train {train_metric:.4f}, Valid {valid_metric:.4f}, Test {test_metric:.4f}", flush=True)

        train_metric_list.append(train_metric)
        valid_metric_list.append(valid_metric)
        test_metric_list.append(test_metric)

    return train_metric_list, valid_metric_list, test_metric_list


def test_single_with_logits(model, predictor, data, train_idx, valid_idx, test_idx, batch_size, degree=False,
                             att=None, mlp=None, normalize_class_h=False, projector=None, rank=0, identity_projection=None,
                             external_embeddings=None, args=None, use_cs=False, cs_num_iters=50, cs_alpha=0.5):
    """
    Helper function for TTA: Run inference on a single graph and return logits (not predictions).

    Returns:
        train_logits, valid_logits, test_logits: Raw logits for each split
    """
    from torch.utils.data import DataLoader

    model.eval()
    predictor.eval()
    if projector is not None:
        projector.eval()
    if identity_projection is not None:
        identity_projection.eval()

    device = next(model.parameters()).device
    moved_to_device = False
    if data.x.device != device or (hasattr(data, 'adj_t') and data.adj_t.device != device):
        data = data.to(device)
        train_idx = train_idx.to(device)
        valid_idx = valid_idx.to(device)
        test_idx = test_idx.to(device)
        if hasattr(data, 'context_sample') and data.context_sample is not None:
            data.context_sample = data.context_sample.to(device)
        if hasattr(data, 'adj_t'):
            data.adj_t = data.adj_t.to(device)
        moved_to_device = True

    base_features = data.x

    # Apply projection strategies (same as test())
    if hasattr(data, 'uses_fug_embeddings') and data.uses_fug_embeddings and projector is not None:
        x_input = projector(base_features)
    elif hasattr(data, 'needs_identity_projection') and data.needs_identity_projection and identity_projection is not None:
        x_input = identity_projection(base_features)
    elif hasattr(data, 'needs_projection') and data.needs_projection and projector is not None:
        projected_features = projector(base_features)
        if hasattr(data, 'needs_final_pca') and data.needs_final_pca:
            from src.projection_methods import apply_final_pca
            x_input = apply_final_pca(projected_features, projected_features.size(1))
        else:
            x_input = projected_features
    else:
        x_input = base_features

    # GNN forward pass
    with torch.no_grad():
        h = model(x_input, data.adj_t)

    context_h = h[data.context_sample]
    context_y = data.y[data.context_sample]

    class_h = process_node_features(context_h, data, degree_normalize=degree, attention_pool_module=att,
                                    mlp_module=mlp, normalize=normalize_class_h)

    # Get logits for all splits
    train_loader = DataLoader(range(train_idx.size(0)), batch_size, shuffle=False)
    valid_loader = DataLoader(range(valid_idx.size(0)), batch_size, shuffle=False)
    test_loader = DataLoader(range(test_idx.size(0)), batch_size, shuffle=False)

    train_logits = []
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

    valid_logits = []
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

    test_logits = []
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

    if moved_to_device:
        data = data.cpu()
        if hasattr(data, 'context_sample') and data.context_sample is not None:
            data.context_sample = data.context_sample.cpu()
        train_idx = train_idx.cpu()
        valid_idx = valid_idx.cpu()
        test_idx = test_idx.cpu()
        if hasattr(data, 'adj_t'):
            data.adj_t = data.adj_t.cpu()

    # Note: For TTA, we don't apply C&S because we want to aggregate raw model predictions
    # C&S can be applied after aggregation if needed

    return train_logits, valid_logits, test_logits 
