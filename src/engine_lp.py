import torch
import math
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from ogb.linkproppred import Evaluator
from torch_sparse import SparseTensor
import copy
import time
import psutil
import os
from .data_utils import edge_dropout_sparse_tensor, feature_dropout, apply_random_projection_augmentation


def refresh_lp_context_if_needed(data, batch_idx, epoch, args, context_edges, train_mask, train_edges):
    """
    Proper dataset-specific context refresh for LP batch-level updates.
    """
    # Check if batch refresh is enabled and it's time to refresh
    if getattr(args, 'context_batch_refresh_interval', 0) <= 0:
        return context_edges, train_mask
        
    if batch_idx > 0 and batch_idx % args.context_batch_refresh_interval == 0:
        # Refresh LP context for this specific dataset
        refresh_seed = args.seed + epoch * 10000 + batch_idx
        torch.manual_seed(refresh_seed)
        
        try:
            # Import here to avoid circular imports
            import sys
            import os
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            if project_root not in sys.path:
                sys.path.insert(0, project_root)

            from train import resolve_context_shots
            from src.data_utils import select_link_context
            
            # Get dynamic context shots using the same logic as epoch refresh
            context_shots = resolve_context_shots(data.name, 'lp', args, epoch)
            
            # Regenerate context using the train_edges (which is the full training data)
            if train_edges['edge_pairs'].size(0) > 0:
                new_context_data, new_train_mask = select_link_context(
                    train_edges, context_shots, args.context_neg_ratio,
                    args.remove_context_from_train
                )
                
                print(f"🔄 LP Dataset {data.name} context refreshed at batch {batch_idx} ({context_shots} context shots)")
                return new_context_data, new_train_mask
                
        except Exception as e:
            print(f"🔄 LP Dataset {data.name} context refresh failed at batch {batch_idx}: {e}")
    
    return context_edges, train_mask

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


def _has_trainable_parameters(module):
    """Return True if module has any trainable parameter."""
    if module is None:
        return False
    return any(p.requires_grad for p in module.parameters())


def _is_puregcn_v1_parameter_free(model_module):
    """
    Conservative check for parameter-free PureGCN_v1 forward path.

    Returns:
        (is_safe, reason)
    """
    if model_module is None:
        return False, "model_missing"

    if model_module.__class__.__name__ != 'PureGCN_v1':
        return False, "model_not_puregcn_v1"

    if not isinstance(getattr(model_module, 'lin', None), nn.Identity):
        return False, "input_projection_not_identity"

    if getattr(model_module, 'use_virtual_node', False):
        return False, "virtual_node_enabled"

    if float(getattr(model_module, 'dp', 0.0)) > 0:
        return False, "model_dropout_enabled"

    if getattr(model_module, 'norm', False):
        norms = getattr(model_module, 'norms', None)
        if norms is None:
            return False, "missing_norm_layers"
        for layer_norm in norms:
            if getattr(layer_norm, 'elementwise_affine', False):
                return False, "gnn_norm_affine_enabled"

    if _has_trainable_parameters(model_module):
        return False, "model_has_trainable_params"

    return True, None


def get_node_embeddings(model, data, projector=None, identity_projection=None, use_full_adj=False, args=None, rank=0):
    """
    Get node embeddings using the same model and preprocessing as node classification.

    Args:
        model: Trained GNN model
        data: Graph data
        projector: Optional projector module
        identity_projection: Optional identity projection module
        use_full_adj: Whether to use full_adj_t if available (for test evaluation)
        args: Training arguments (for edge dropout configuration)
        rank: Process rank for logging

    Returns:
        Node embeddings [num_nodes, hidden_dim]
    """
    # Apply different projection strategies
    if hasattr(data, 'needs_identity_projection') and data.needs_identity_projection and identity_projection is not None:
        x_input = identity_projection(data.x)
    elif hasattr(data, 'needs_projection') and data.needs_projection and projector is not None:
        projected_features = projector(data.x)
        # Apply final PCA to get features in proper PCA form
        if hasattr(data, 'needs_final_pca') and data.needs_final_pca:
            from .utils import apply_final_pca
            x_input = apply_final_pca(projected_features, projected_features.size(1))
        else:
            x_input = projected_features
    else:
        x_input = data.x

    # Apply feature dropout AFTER projection
    x_input = apply_feature_dropout_if_enabled(x_input, args, rank, training=model.training)

    # Choose adjacency matrix: use full_adj_t for test evaluation if available
    if use_full_adj and hasattr(data, 'full_adj_t') and data.full_adj_t is not None:
        adj_matrix = data.full_adj_t
    else:
        adj_matrix = data.adj_t

    # Apply edge dropout if enabled (only during training)
    if args is not None and hasattr(args, 'edge_dropout_enabled') and args.edge_dropout_enabled and hasattr(args, 'edge_dropout_rate'):
        verbose_dropout = getattr(args, 'verbose_edge_dropout', False) and rank == 0
        adj_matrix = edge_dropout_sparse_tensor(adj_matrix, args.edge_dropout_rate, training=model.training, verbose=verbose_dropout)

    # Get node embeddings
    node_embeddings = model(x_input, adj_matrix)
    return node_embeddings

def get_link_prototypes(node_embeddings, context_data, att_pool, mlp_pool, normalize=False):
    """Generates 'link' and 'no-link' prototypes using the context set."""
    context_edges = context_data['edge_pairs']
    context_labels = context_data['labels']
    # print(f"Context edges: {context_edges}, Context labels: {context_labels}")
    
    # Ensure all tensors are on the same device as node_embeddings
    device = node_embeddings.device
    context_edges = context_edges.to(device)
    context_labels = context_labels.to(device)

    # Validate context data
    if context_edges.size(0) == 0:
        return None
        
    pos_mask = context_labels == 1
    neg_mask = context_labels == 0
    
    # Ensure we have both positive and negative examples
    if not pos_mask.any() or not neg_mask.any():
        return None

    src_embeds = node_embeddings[context_edges[:, 0]]
    dst_embeds = node_embeddings[context_edges[:, 1]]

    # Combine node embeddings to get edge embeddings (Hadamard product is a good choice)
    edge_embeddings = src_embeds * dst_embeds

    pos_edge_embeddings = edge_embeddings[pos_mask]
    neg_edge_embeddings = edge_embeddings[neg_mask]

    # Use AttentionPool and MLP to get prototypes, similar to node classification
    if att_pool:
        # For link prediction, pool each set (pos/neg) into a single prototype.
        # We treat each set as having a single class (label 0).
        pos_labels = torch.zeros(pos_edge_embeddings.size(0), dtype=torch.long, device=device)
        neg_labels = torch.zeros(neg_edge_embeddings.size(0), dtype=torch.long, device=device)
        
        pos_prototype = att_pool(pos_edge_embeddings, pos_labels, num_classes=1).squeeze(0)
        neg_prototype = att_pool(neg_edge_embeddings, neg_labels, num_classes=1).squeeze(0)
    else:
        # Fallback to mean pooling if no attention pool is provided
        pos_prototype = pos_edge_embeddings.mean(dim=0)
        neg_prototype = neg_edge_embeddings.mean(dim=0)
        
    if mlp_pool:
        pos_prototype = mlp_pool(pos_prototype)
        neg_prototype = mlp_pool(neg_prototype)
        
    # Stack prototypes: 0 for neg, 1 for pos
    link_prototypes = torch.stack([neg_prototype, pos_prototype], dim=0)

    if normalize:
        link_prototypes = F.normalize(link_prototypes, p=2, dim=1)

    return link_prototypes


def train_link_prediction(model, predictor, data, train_edges, context_edges, train_mask, optimizer, 
                          batch_size, att=None, mlp=None, projector=None, identity_projection=None, 
                          clip_grad=1.0, rank=0, orthogonal_push=0.0, normalize_class_h=False, 
                          epoch=0, mask_target_edges=False, degree=False, lambda_=1.0, args=None):
    """
    Train link prediction using the PFN methodology.
    """
    
    try:
        print(f"Rank {rank}: Starting link prediction training for epoch {epoch}")
        model.train()
        predictor.train()
        if att: att.train()
        if mlp: mlp.train()
        if projector: projector.train()
        if identity_projection: identity_projection.train()
        device = data.x.device
        train_mask = train_mask.to(device)
        head_type = getattr(predictor, 'lp_head_type', '')
        use_lp_cn = getattr(args, 'lp_concat_common_neighbors', False) and head_type == 'standard'
        cache_mask_target_for_mplp_only = (
            args is not None and
            getattr(args, 'lp_cache_mask_target_only_for_mplp', False) and
            bool(mask_target_edges) and
            head_type == 'mplp'
        )

        edge_pairs = train_edges['edge_pairs'].to(device)
        labels = train_edges['labels'].to(device)
        
        # The dataloader iterates over indices of the FULL training set
        indices = torch.arange(edge_pairs.size(0))  # Keep on CPU initially
    
        # Use DistributedSampler if in DDP mode
        sampler = None
        if dist.is_initialized() and dist.get_world_size() > 1:
            try:
                sampler = DistributedSampler(indices, shuffle=True)
                sampler.set_epoch(epoch)
            except Exception as e:
                print(f"[ERROR] Rank {rank}: Failed to create DistributedSampler: {e}")
                raise

        # Standard DataLoader with DistributedSampler
        try:
            dataloader = DataLoader(indices, batch_size=batch_size, sampler=sampler, 
                                   shuffle=(sampler is None))
        except Exception as e:
            print(f"[ERROR] Rank {rank}: Failed to create DataLoader: {e}")
            raise

        # Correctly pre-filter for positive edges, as in the reference
        pos_train_mask = (labels == 1)
        pos_train_edges = edge_pairs[pos_train_mask]
        # The mask should be on the positive edges only
        pos_adjmask = torch.ones(pos_train_edges.size(0), dtype=torch.bool, device=device)
        
        # Map original indices to their position in the positive-only list
        pos_original_indices = torch.where(pos_train_mask)[0]
        pos_indices_map = {orig_idx.item(): pos_idx for pos_idx, orig_idx in enumerate(pos_original_indices)}

        def _build_masked_adj_for_batch(batch_indices_local):
            """Mask current-batch positive target edges from train adjacency."""
            adj_local = data.adj_t
            batch_labels_check = labels[batch_indices_local]
            batch_pos_mask = batch_labels_check == 1

            if not batch_pos_mask.any():
                return adj_local

            batch_pos_indices = batch_indices_local[batch_pos_mask]

            indices_to_mask_in_pos_list = []
            for batch_pos_idx in batch_pos_indices:
                mapped_pos_idx = pos_indices_map.get(batch_pos_idx.item())
                if mapped_pos_idx is not None:
                    indices_to_mask_in_pos_list.append(mapped_pos_idx)

            if len(indices_to_mask_in_pos_list) == 0:
                return adj_local

            pos_adjmask[indices_to_mask_in_pos_list] = False
            try:
                edge = pos_train_edges[pos_adjmask].t()
                adj_local = SparseTensor.from_edge_index(
                    edge, sparse_sizes=(data.num_nodes, data.num_nodes)
                ).to(device)
                adj_local = adj_local.to_symmetric().coalesce()
            finally:
                pos_adjmask[indices_to_mask_in_pos_list] = True

            return adj_local

        lp_cache_mode = getattr(args, 'static_embedding_cache', 'auto') if args is not None else 'auto'
        if lp_cache_mode not in ('off', 'auto', 'force'):
            lp_cache_mode = 'auto'

        model_module = model.module if hasattr(model, 'module') else model
        model_has_trainable_params = _has_trainable_parameters(model_module)
        projector_has_trainable_params = _has_trainable_parameters(projector)
        identity_has_trainable_params = _has_trainable_parameters(identity_projection)
        puregcn_cacheable, puregcn_reason = _is_puregcn_v1_parameter_free(model_module)

        edge_dropout_active = (
            args is not None and
            getattr(args, 'edge_dropout_enabled', False) and
            float(getattr(args, 'edge_dropout_rate', 0.0)) > 0
        )
        feature_dropout_active = (
            args is not None and
            getattr(args, 'feature_dropout_enabled', False) and
            float(getattr(args, 'feature_dropout_rate', 0.0)) > 0
        )

        cache_hard_blockers = []
        if mask_target_edges and not cache_mask_target_for_mplp_only:
            cache_hard_blockers.append('mask_target_edges_enabled')
        if edge_dropout_active:
            cache_hard_blockers.append('edge_dropout_enabled')
        if feature_dropout_active:
            cache_hard_blockers.append('feature_dropout_enabled')

        use_static_embedding_cache = False
        cache_notes = []
        if lp_cache_mode == 'off':
            cache_notes.append('policy_off')
        elif len(cache_hard_blockers) > 0:
            cache_notes.extend(cache_hard_blockers)
        elif lp_cache_mode == 'auto':
            if not puregcn_cacheable:
                cache_notes.append(puregcn_reason or 'model_not_cacheable')
            if model_has_trainable_params:
                cache_notes.append('model_has_trainable_params')
            if projector_has_trainable_params:
                cache_notes.append('projector_has_trainable_params')
            if identity_has_trainable_params:
                cache_notes.append('identity_projection_has_trainable_params')
            use_static_embedding_cache = len(cache_notes) == 0
        else:  # force
            if model_has_trainable_params:
                cache_notes.append('force_rejected_model_has_trainable_params')
            if projector_has_trainable_params:
                cache_notes.append('force_rejected_projector_has_trainable_params')
            if identity_has_trainable_params:
                cache_notes.append('force_rejected_identity_projection_has_trainable_params')
            use_static_embedding_cache = len(cache_notes) == 0
            if use_static_embedding_cache and not puregcn_cacheable:
                cache_notes.append(f"forced_without_strict_check:{puregcn_reason or 'unknown'}")

        cache_notes = list(dict.fromkeys(cache_notes))
        if rank == 0:
            dataset_name = getattr(data, 'name', 'unknown')
            if use_static_embedding_cache:
                print(f"[LP Cache] Enabled for {dataset_name} (mode={lp_cache_mode}): one full-graph GNN forward per epoch.")
                if len(cache_notes) > 0:
                    print(f"[LP Cache] Note: {', '.join(cache_notes)}")
            else:
                print(f"[LP Cache] Disabled for {dataset_name} (mode={lp_cache_mode}): {', '.join(cache_notes)}")

        static_node_embeddings = None
        static_data_for_gnn = None
        if use_static_embedding_cache:
            static_data_for_gnn = copy.copy(data)
            static_data_for_gnn.adj_t = data.adj_t
            with torch.no_grad():
                static_node_embeddings = get_node_embeddings(
                    model, static_data_for_gnn, projector, identity_projection,
                    use_full_adj=False, args=args, rank=rank
                )
            static_node_embeddings = static_node_embeddings.detach()
        
        total_loss = 0
        batch_count = 0
        gate_sum = 0.0
        gate_count = 0
        calib_sum = 0.0
        calib_count = 0
        hybrid_w_std_sum = 0.0
        hybrid_w_mplp_sum = 0.0
        hybrid_w_ncn_sum = 0.0
        hybrid_w_count = 0
        struct_sum = 0.0
        struct_sumsq = 0.0
        struct_count = 0
        feat_sum = 0.0
        feat_sumsq = 0.0
        feat_count = 0
        logit_sum = 0.0
        logit_sumsq = 0.0
        logit_count = 0
        struct_loss_sum = 0.0
        struct_loss_count = 0
        
        for batch_indices in dataloader:
            # Batch-level context refresh for this LP dataset
            if args is not None:
                context_edges, train_mask = refresh_lp_context_if_needed(data, batch_count, epoch, args, context_edges, train_mask, train_edges)
            
            st = time.time()
            try:
                # Move batch indices to the same device as edge_pairs/labels for indexing
                if batch_indices.device != device:
                    batch_indices = batch_indices.to(device)
                
                # Only zero gradients if optimizer is provided (for joint training compatibility)
                if optimizer is not None:
                    optimizer.zero_grad()

                if use_static_embedding_cache:
                    # Keep cached node embeddings fixed; optionally use masked adjacency
                    # only for MPLP structural scoring.
                    if cache_mask_target_for_mplp_only:
                        adj_for_gnn = _build_masked_adj_for_batch(batch_indices)
                    else:
                        adj_for_gnn = data.adj_t
                    data_for_gnn = static_data_for_gnn
                    node_embeddings = static_node_embeddings
                else:
                    # --- Optional: Masking Target Edges ---
                    adj_for_gnn = data.adj_t
                    if mask_target_edges:
                        adj_for_gnn = _build_masked_adj_for_batch(batch_indices)

                    # Recompute embeddings and prototypes for each batch to maintain the computation graph
                    data_for_gnn = copy.copy(data)
                    data_for_gnn.adj_t = adj_for_gnn
                    node_embeddings = get_node_embeddings(
                        model, data_for_gnn, projector, identity_projection,
                        use_full_adj=False, args=args, rank=rank
                    )

                # -----------------------------------------

                # Get context edge embeddings for PFN predictor
                context_edge_pairs = context_edges['edge_pairs'].to(device)
                context_labels = context_edges['labels'].to(device)
                context_src_embeds = node_embeddings[context_edge_pairs[:, 0]]
                context_dst_embeds = node_embeddings[context_edge_pairs[:, 1]]
                context_edge_embeds = context_src_embeds * context_dst_embeds
                cn_context = None
                if use_lp_cn:
                    cn_context = _common_neighbor_count(adj_for_gnn, context_edge_pairs)

                batch_labels = labels[batch_indices]
                batch_edges = edge_pairs[batch_indices]

                # Get embeddings for target edges
                src_embeds = node_embeddings[batch_edges[:, 0]]
                dst_embeds = node_embeddings[batch_edges[:, 1]]
                target_edge_embeds = src_embeds * dst_embeds
                cn_target = None
                if use_lp_cn:
                    cn_target = _common_neighbor_count(adj_for_gnn, batch_edges)

                # Get link prototypes (binary class embeddings)
                link_prototypes = get_link_prototypes(node_embeddings, context_edges, att, mlp, normalize_class_h)
                if link_prototypes is None:
                    if rank == 0:
                        print("Warning: Could not form link prototypes. Skipping batch.")
                    continue

                # Use unified PFNPredictorNodeCls for link prediction
                if head_type in ('mplp', 'hybrid3'):
                    scores, link_prototypes = predictor(
                        data_for_gnn,
                        context_edge_embeds,
                        target_edge_embeds,
                        context_labels.long(),
                        link_prototypes,
                        "link_prediction",
                        adj_t=adj_for_gnn,
                        lp_edges=batch_edges.t(),
                        node_emb=node_embeddings,
                        lp_context_edges=context_edge_pairs.t()
                    )
                    if head_type == 'mplp' and getattr(predictor, 'lp_head', None) is not None:
                        gate_val = getattr(predictor.lp_head, 'last_gate_mean', None)
                        if gate_val is not None:
                            gate_sum += float(gate_val.item())
                            gate_count += 1
                        calib_val = getattr(predictor.lp_head, 'last_gate_calib_ms', None)
                        if calib_val is not None:
                            calib_sum += float(calib_val)
                            calib_count += 1
                    elif head_type == 'hybrid3' and getattr(predictor, 'lp_head', None) is not None:
                        fusion_w = getattr(predictor.lp_head, 'last_fusion_weights', None)
                        if fusion_w is not None and fusion_w.numel() >= 3:
                            hybrid_w_std_sum += float(fusion_w[0].item())
                            hybrid_w_mplp_sum += float(fusion_w[1].item())
                            hybrid_w_ncn_sum += float(fusion_w[2].item())
                            hybrid_w_count += 1
                elif head_type == 'ncn':
                    scores, link_prototypes = predictor(
                        data_for_gnn,
                        context_edge_embeds,
                        target_edge_embeds,
                        context_labels.long(),
                        link_prototypes,
                        "link_prediction",
                        adj_t=adj_for_gnn,
                        lp_edges=batch_edges.t(),
                        node_emb=node_embeddings
                    )
                else:
                    scores, link_prototypes = predictor(
                        data_for_gnn,
                        context_edge_embeds,
                        target_edge_embeds,
                        context_labels.long(),
                        link_prototypes,
                        "link_prediction",
                        lp_cn_context=cn_context,
                        lp_cn_target=cn_target
                    )

                # Use the train_mask to ensure loss is only calculated on non-context edges
                # Make sure the mask is properly aligned with batch indices
                if train_mask.size(0) != edge_pairs.size(0):
                    if rank == 0:
                        print(f"Warning: train_mask size {train_mask.size(0)} doesn't match edge_pairs size {edge_pairs.size(0)}")
                    # Create a default mask that includes all edges in the batch
                    mask_for_loss = torch.ones(batch_indices.size(0), dtype=torch.bool, device=device)
                else:
                    mask_for_loss = train_mask[batch_indices]

                if scores.dim() == 1:
                    # Use BCEWithLogitsLoss for binary classification (link vs no-link)
                    mask_scores = scores[mask_for_loss]
                    if mask_scores.dim() > 1:
                        mask_scores = mask_scores.squeeze(-1)
                    if mask_scores.numel() > 0:
                        logit_sum += mask_scores.sum().item()
                        logit_sumsq += (mask_scores ** 2).sum().item()
                        logit_count += mask_scores.numel()
                    nll_loss = F.binary_cross_entropy_with_logits(
                        mask_scores, batch_labels[mask_for_loss].float()
                    )
                    # Optional struct-only debug loss (no backprop)
                    if head_type == 'mplp' and getattr(predictor, 'lp_head', None) is not None:
                        struct_scores = getattr(predictor.lp_head, 'last_struct_score', None)
                        if struct_scores is not None:
                            struct_scores = struct_scores.to(mask_scores.device)
                            struct_scores = struct_scores[mask_for_loss]
                            struct_sum += struct_scores.sum().item()
                            struct_sumsq += (struct_scores ** 2).sum().item()
                            struct_count += struct_scores.numel()
                            struct_loss = F.binary_cross_entropy_with_logits(
                                struct_scores, batch_labels[mask_for_loss].float()
                            )
                            struct_loss_sum += float(struct_loss.item())
                            struct_loss_count += 1
                        feat_scores = getattr(predictor.lp_head, 'last_feat_score', None)
                        if feat_scores is not None:
                            feat_scores = feat_scores.to(mask_scores.device)
                            feat_scores = feat_scores[mask_for_loss]
                            feat_sum += feat_scores.sum().item()
                            feat_sumsq += (feat_scores ** 2).sum().item()
                            feat_count += feat_scores.numel()
                else:
                    # Fallback to original two-class loss when head is disabled
                    nll_loss = F.cross_entropy(scores[mask_for_loss], batch_labels[mask_for_loss].long())

                # Compute optional orthogonal loss on prototypes
                if orthogonal_push > 0:
                    proto_norm = F.normalize(link_prototypes, p=2, dim=1)
                    proto_matrix = proto_norm @ proto_norm.T
                    mask = ~torch.eye(proto_matrix.size(0), device=device, dtype=torch.bool)
                    orthogonal_loss = torch.sum(proto_matrix[mask]**2)
                else:
                    orthogonal_loss = torch.tensor(0.0, device=device)

                loss = nll_loss + orthogonal_push * orthogonal_loss
                loss = loss * lambda_  # Apply lambda scaling
                
                try:
                    loss.backward()
                except Exception as e:
                    print(f"[ERROR] Rank {rank}: Exception during loss.backward(): {e}")
                    import traceback
                    print(f"[ERROR] Rank {rank}: Traceback: {traceback.format_exc()}")
                finally:
                    total_loss += loss.item()
                
                # Update weights
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
                    if att: torch.nn.utils.clip_grad_norm_(att.parameters(), clip_grad)
                    if mlp: torch.nn.utils.clip_grad_norm_(mlp.parameters(), clip_grad)
                    if projector: torch.nn.utils.clip_grad_norm_(projector.parameters(), clip_grad)
                    if identity_projection: torch.nn.utils.clip_grad_norm_(identity_projection.parameters(), clip_grad)
                
                optimizer.step()
                
                batch_count += 1
                    
                if rank == 0 and batch_count % 100 == 0:  # Only print every 100 batches
                    print(f'Rank: {rank}, Batch: {batch_count}, Batch time: {time.time() - st:.2f}, Loss: {loss.item():.4f}', flush=True)

            except Exception as e:
                print(f"[ERROR] Rank {rank}: Exception during training batch {batch_count}: {e}")
                import traceback
                print(f"[ERROR] Rank {rank}: Traceback: {traceback.format_exc()}")
                if rank == 0:
                    print(f"Error in training batch: {e}")
                continue

        if rank == 0:
            loss_str = f"{total_loss:.4f}" if optimizer is not None else "tensor"
            print(f"Rank {rank}: Epoch {epoch} training complete. Total loss: {loss_str}, Batch count: {batch_count}")

        if getattr(predictor, 'lp_head', None) is not None:
            if gate_count > 0:
                predictor.lp_head.last_gate_mean_train = gate_sum / gate_count
            else:
                predictor.lp_head.last_gate_mean_train = None
            if calib_count > 0:
                predictor.lp_head.last_gate_calib_ms_train = calib_sum / calib_count
            else:
                predictor.lp_head.last_gate_calib_ms_train = None
            if struct_loss_count > 0:
                predictor.lp_head.last_struct_loss_train = struct_loss_sum / struct_loss_count
            else:
                predictor.lp_head.last_struct_loss_train = None
            if hybrid_w_count > 0:
                predictor.lp_head.last_hybrid_w_std_train = hybrid_w_std_sum / hybrid_w_count
                predictor.lp_head.last_hybrid_w_mplp_train = hybrid_w_mplp_sum / hybrid_w_count
                predictor.lp_head.last_hybrid_w_ncn_train = hybrid_w_ncn_sum / hybrid_w_count
            else:
                predictor.lp_head.last_hybrid_w_std_train = None
                predictor.lp_head.last_hybrid_w_mplp_train = None
                predictor.lp_head.last_hybrid_w_ncn_train = None
            if struct_count > 0:
                mean = struct_sum / struct_count
                var = max(struct_sumsq / struct_count - mean * mean, 0.0)
                predictor.lp_head.last_struct_score_mean_train = mean
                predictor.lp_head.last_struct_score_std_train = math.sqrt(var)
            else:
                predictor.lp_head.last_struct_score_mean_train = None
                predictor.lp_head.last_struct_score_std_train = None
            if feat_count > 0:
                mean = feat_sum / feat_count
                var = max(feat_sumsq / feat_count - mean * mean, 0.0)
                predictor.lp_head.last_feat_score_mean_train = mean
                predictor.lp_head.last_feat_score_std_train = math.sqrt(var)
            else:
                predictor.lp_head.last_feat_score_mean_train = None
                predictor.lp_head.last_feat_score_std_train = None
            if logit_count > 0:
                mean = logit_sum / logit_count
                var = max(logit_sumsq / logit_count - mean * mean, 0.0)
                predictor.lp_head.last_logit_mean_train = mean
                predictor.lp_head.last_logit_std_train = math.sqrt(var)
            else:
                predictor.lp_head.last_logit_mean_train = None
                predictor.lp_head.last_logit_std_train = None
        
        # Final synchronization to ensure all processes complete training
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()

        # Move persistent tensors back to CPU to free GPU memory
        train_edges['edge_pairs'] = train_edges['edge_pairs'].cpu()
        train_edges['labels'] = train_edges['labels'].cpu()
        context_edges['edge_pairs'] = context_edges['edge_pairs'].cpu()
        context_edges['labels'] = context_edges['labels'].cpu()
        train_mask = train_mask.cpu()
            
        return total_loss / max(batch_count, 1)

    except Exception as e:
        print(f"[ERROR] Rank {rank}: Fatal exception in train_link_prediction: {e}")
        import traceback
        print(f"[ERROR] Rank {rank}: Traceback: {traceback.format_exc()}")
        raise

def get_dataset_default_metric(dataset_name):
    """Get the default metric for each dataset."""
    if dataset_name == 'ogbl-ddi':
        return 'hits@20'
    elif dataset_name == 'ogbl-collab':
        return 'hits@50'
    elif dataset_name == 'ogbl-citation2':
        return 'mrr'
    elif dataset_name == 'ogbl-ppa':
        return 'hits@100'
    else:
        return 'hits@100'  # Default metric

def get_evaluation_metric(dataset_name, lp_metric='auto'):
    """Get the metric to use for evaluation based on user preference."""
    if lp_metric == 'auto':
        return get_dataset_default_metric(dataset_name)
    else:
        return lp_metric


def _common_neighbor_count(adj_t, edge_pairs):
    """
    Compute common neighbor counts for each edge (u, v) in edge_pairs.
    Vectorized GPU-friendly implementation using CSR adjacency lists.

    Args:
        adj_t: SparseTensor adjacency.
        edge_pairs: Tensor [E, 2] of edge endpoints.
    Returns:
        Tensor [E, 1] of common neighbor counts (float, on edge_pairs.device).
    """
    if edge_pairs.numel() == 0:
        return torch.zeros((0, 1), device=edge_pairs.device)

    if edge_pairs.dim() == 2 and edge_pairs.size(0) == 2:
        edge_pairs = edge_pairs.t()

    device = edge_pairs.device
    num_nodes = adj_t.size(0)
    rowptr, col, _ = adj_t.csr()
    rowptr = rowptr.to(device)
    col = col.to(device)

    edges = edge_pairs.to(device)
    E = edges.size(0)

    def _gather_neighbors(nodes):
        starts = rowptr[nodes]
        ends = rowptr[nodes + 1]
        counts = (ends - starts).to(torch.long)
        total = int(counts.sum().item())
        if total == 0:
            return (torch.empty(0, device=device, dtype=col.dtype),
                    torch.empty(0, device=device, dtype=torch.long))

        prefix = torch.cumsum(counts, dim=0)
        base = prefix - counts
        base_rep = torch.repeat_interleave(base, counts)
        rel = torch.arange(total, device=device) - base_rep
        idx = torch.repeat_interleave(starts, counts) + rel
        neighbors = col[idx]
        edge_idx = torch.repeat_interleave(torch.arange(E, device=device), counts)
        return neighbors, edge_idx

    u_neighbors, u_edge_idx = _gather_neighbors(edges[:, 0])
    v_neighbors, v_edge_idx = _gather_neighbors(edges[:, 1])

    if u_neighbors.numel() == 0 or v_neighbors.numel() == 0:
        return torch.zeros((E, 1), device=device)

    keys_u = u_edge_idx.to(torch.int64) * num_nodes + u_neighbors.to(torch.int64)
    keys_v = v_edge_idx.to(torch.int64) * num_nodes + v_neighbors.to(torch.int64)
    keys = torch.cat([keys_u, keys_v], dim=0)

    uniq, cnt = torch.unique(keys, return_counts=True)
    edge_idx = (uniq // num_nodes).to(torch.long)

    cn = torch.zeros(E, device=device, dtype=torch.float32)
    cn.scatter_add_(0, edge_idx, (cnt == 2).to(torch.float32))
    return cn.unsqueeze(1)

def compute_mrr_citation2(pos_scores, neg_scores):
    """
    Compute MRR for ogbl-citation2 dataset with special format requirement.
    Each positive sample should be evaluated against 1000 negative samples.
    """
    try:
        num_pos = pos_scores.size(0)
        expected_neg_size = num_pos * 1000
        neg_scores_reshaped = neg_scores[:expected_neg_size].view(num_pos, 1000)
        
        # Use OGB evaluator for MRR calculation
        evaluator = Evaluator(name='ogbl-citation2')
        result = evaluator.eval({
            'y_pred_pos': pos_scores.cpu(),
            'y_pred_neg': neg_scores_reshaped.cpu(),
        })
        
        return result['mrr_list'].mean().item()
        
    except Exception as e:
        print(f"Error computing citation2 MRR: {e}")
        return 0.0


def _select_debug_indices(num_items, max_samples, device):
    if num_items <= max_samples:
        return torch.arange(num_items, device=device)
    return torch.linspace(0, num_items - 1, steps=max_samples, device=device).long()


def _print_lp_edge_probe_debug(node_embeddings, context_edge_pairs, context_labels,
                               pos_edges, neg_edges,
                               pos_scores, neg_scores, rank=0, max_samples=512,
                               dataset_name='unknown', debug_label='eval',
                               ridge_alpha=1e-2, k_values=(20, 50, 100)):
    """
    Fit same-view ridge probes on several edge feature maps:
    had=h_u*h_v, inv=[dot,cos,dist,norms], sym=[h_u+h_v, |h_u-h_v|], dot=scalar dot.
    This separates node-embedding geometry loss from Hadamard-specific loss.
    """
    if rank != 0 or max_samples <= 0:
        return
    if pos_edges.numel() == 0 or neg_edges.numel() == 0:
        return

    try:
        pos_idx = _select_debug_indices(pos_edges.size(0), max_samples, pos_edges.device)
        neg_idx = _select_debug_indices(neg_edges.size(0), max_samples, neg_edges.device)
        sample_pos = pos_edges[pos_idx]
        sample_neg = neg_edges[neg_idx]
        if sample_pos.size(0) < 4 or sample_neg.size(0) < 4:
            return

        split_pos = sample_pos.size(0) // 2
        split_neg = sample_neg.size(0) // 2
        train_pos = sample_pos[:split_pos]
        train_neg = sample_neg[:split_neg]
        test_pos = sample_pos[split_pos:]
        test_neg = sample_neg[split_neg:]

        def _edge_features(edges, mode):
            hu = node_embeddings[edges[:, 0]].float()
            hv = node_embeddings[edges[:, 1]].float()
            if mode == 'had':
                return hu * hv
            if mode == 'sym':
                return torch.cat([hu + hv, (hu - hv).abs()], dim=1)
            dot = (hu * hv).sum(dim=1, keepdim=True)
            if mode == 'dot':
                return dot
            if mode == 'inv':
                nu = hu.norm(dim=1, keepdim=True)
                nv = hv.norm(dim=1, keepdim=True)
                cos = dot / (nu * nv + 1e-8)
                dist = (hu - hv).norm(dim=1, keepdim=True)
                return torch.cat([dot, cos, dist, nu, nv], dim=1)
            raise ValueError(f"unknown edge feature mode: {mode}")

        X_train_by_mode = {
            mode: torch.cat([_edge_features(train_pos, mode), _edge_features(train_neg, mode)], dim=0)
            for mode in ('had', 'inv', 'sym', 'dot')
        }
        y_train = torch.cat([
            torch.ones(train_pos.size(0), device=node_embeddings.device),
            torch.zeros(train_neg.size(0), device=node_embeddings.device),
        ]).float()
        X_test_by_mode = {
            mode: torch.cat([_edge_features(test_pos, mode), _edge_features(test_neg, mode)], dim=0)
            for mode in ('had', 'inv', 'sym', 'dot')
        }
        y_test = torch.cat([
            torch.ones(test_pos.size(0), device=node_embeddings.device),
            torch.zeros(test_neg.size(0), device=node_embeddings.device),
        ]).float()

        pos_scores_sample = pos_scores[pos_idx.detach().cpu()]
        neg_scores_sample = neg_scores[neg_idx.detach().cpu()]
        fixed_scores = torch.cat([pos_scores_sample[split_pos:], neg_scores_sample[split_neg:]], dim=0).float()

        from sklearn.metrics import roc_auc_score, average_precision_score
        labels_np = y_test.detach().cpu().numpy()
        fixed_np = fixed_scores.detach().cpu().numpy()
        fixed_auc = roc_auc_score(labels_np, fixed_np)
        fixed_ap = average_precision_score(labels_np, fixed_np)
        fixed_sep = (fixed_scores[:test_pos.size(0)].mean() - fixed_scores[test_pos.size(0):].mean()).item()

        probe_auc = {}
        probe_ap = {}
        probe_sep = {}
        y_centered = y_train * 2.0 - 1.0
        for mode, X_train in X_train_by_mode.items():
            X_test = X_test_by_mode[mode]
            mean = X_train.mean(dim=0, keepdim=True)
            std = X_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
            X_train_n = (X_train - mean) / std
            X_test_n = (X_test - mean) / std
            X_train_n = torch.cat([X_train_n, torch.ones(X_train_n.size(0), 1, device=X_train_n.device)], dim=1)
            X_test_n = torch.cat([X_test_n, torch.ones(X_test_n.size(0), 1, device=X_test_n.device)], dim=1)
            eye = torch.eye(X_train_n.size(1), device=X_train_n.device, dtype=X_train_n.dtype)
            eye[-1, -1] = 0.0
            w = torch.linalg.solve(X_train_n.t() @ X_train_n + ridge_alpha * eye, X_train_n.t() @ y_centered)
            scores = X_test_n @ w
            scores_np = scores.detach().cpu().numpy()
            probe_auc[mode] = roc_auc_score(labels_np, scores_np)
            probe_ap[mode] = average_precision_score(labels_np, scores_np)
            probe_sep[mode] = (scores[:test_pos.size(0)].mean() - scores[test_pos.size(0):].mean()).item()

        print(
            f"[LP_EDGE_PROBE] {dataset_name}/{debug_label}: "
            f"fixed_auc={fixed_auc:.4f} fixed_ap={fixed_ap:.4f} fixed_sep={fixed_sep:.4f} "
            f"had_auc={probe_auc['had']:.4f} inv_auc={probe_auc['inv']:.4f} "
            f"sym_auc={probe_auc['sym']:.4f} dot_auc={probe_auc['dot']:.4f} "
            f"had_ap={probe_ap['had']:.4f} inv_ap={probe_ap['inv']:.4f} "
            f"sym_ap={probe_ap['sym']:.4f} dot_ap={probe_ap['dot']:.4f} "
            f"train={y_train.numel()} test={y_test.numel()}"
        )

        if context_edge_pairs is not None and context_labels is not None and len(k_values) > 0:
            context_labels = context_labels.float().to(node_embeddings.device)
            pos_context = int((context_labels == 1).sum().item())
            neg_context = int((context_labels == 0).sum().item())
            if context_edge_pairs.size(0) >= 4 and pos_context > 0 and neg_context > 0:
                X_context = _edge_features(context_edge_pairs.to(node_embeddings.device), 'inv').float()
                y_context = context_labels * 2.0 - 1.0
                inv_mean = X_context.mean(dim=0, keepdim=True)
                inv_std = X_context.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
                X_context = (X_context - inv_mean) / inv_std
                X_context = torch.cat([
                    X_context,
                    torch.ones(X_context.size(0), 1, device=X_context.device)
                ], dim=1)
                eye = torch.eye(X_context.size(1), device=X_context.device, dtype=X_context.dtype)
                eye[-1, -1] = 0.0
                inv_w = torch.linalg.solve(
                    X_context.t() @ X_context + ridge_alpha * eye,
                    X_context.t() @ y_context
                )

                def _score_full_edges(edges, mode, batch_size=65536):
                    scores = []
                    for start in range(0, edges.size(0), batch_size):
                        batch_edges = edges[start:start + batch_size]
                        feats = _edge_features(batch_edges, mode).float()
                        if mode == 'dot':
                            batch_scores = feats.squeeze(1)
                        elif mode == 'inv':
                            feats = (feats - inv_mean) / inv_std
                            feats = torch.cat([
                                feats,
                                torch.ones(feats.size(0), 1, device=feats.device)
                            ], dim=1)
                            batch_scores = feats @ inv_w
                        else:
                            raise ValueError(f"unsupported official probe mode: {mode}")
                        scores.append(batch_scores.detach().cpu())
                    return torch.cat(scores, dim=0)

                dot_pos = _score_full_edges(pos_edges, 'dot')
                dot_neg = _score_full_edges(neg_edges, 'dot')
                inv_pos = _score_full_edges(pos_edges, 'inv')
                inv_neg = _score_full_edges(neg_edges, 'inv')

                evaluator_name = dataset_name if isinstance(dataset_name, str) and dataset_name.startswith('ogbl-') else 'ogbl-ppa'
                official_evaluator = Evaluator(name=evaluator_name)
                official_parts = []
                for k in k_values:
                    official_evaluator.K = int(k)
                    fixed_hits = official_evaluator.eval({
                        'y_pred_pos': pos_scores.cpu(),
                        'y_pred_neg': neg_scores.cpu(),
                    })[f'hits@{int(k)}']
                    dot_hits = official_evaluator.eval({
                        'y_pred_pos': dot_pos,
                        'y_pred_neg': dot_neg,
                    })[f'hits@{int(k)}']
                    inv_hits = official_evaluator.eval({
                        'y_pred_pos': inv_pos,
                        'y_pred_neg': inv_neg,
                    })[f'hits@{int(k)}']
                    official_parts.append(
                        f"hits@{int(k)} fixed={fixed_hits:.4f} dot={dot_hits:.4f} inv_ctx={inv_hits:.4f}"
                    )
                print(
                    f"[LP_EDGE_OFFICIAL] {dataset_name}/{debug_label}: "
                    + " | ".join(official_parts)
                    + f" context_pos={pos_context} context_neg={neg_context}"
                )
    except Exception as e:
        print(f"[LP_EDGE_PROBE] {dataset_name}/{debug_label}: skipped ({type(e).__name__}: {e})")


@torch.no_grad()
def evaluate_link_prediction(model, predictor, data, test_edges, context_edges, batch_size,
                             att, mlp, projector=None, identity_projection=None, rank=0,
                             normalize_class_h=False, degree=False, evaluator=None,
                             neg_edges=None, k_values=[20, 50, 100], use_full_adj_for_test=True,
                             lp_metric='auto', lp_concat_common_neighbors=False,
                             return_scores=False, lp_edge_probe_debug=False,
                             lp_edge_probe_samples=512, lp_edge_probe_label='eval',
                             return_context_dot_scores=False):
    """
    Evaluate link prediction using the PFN methodology with Hits@K metric.
    
    Args:
        use_full_adj_for_test: If True, use full_adj_t (train+valid edges) for test evaluation
                              when available. This is required for OGB standards (e.g., ogbl-collab).
    """
    try:
        model.eval()
        predictor.eval()
        if att: att.eval()
        if mlp: mlp.eval()
        if projector: projector.eval()
        if identity_projection: identity_projection.eval()
        
        device = data.x.device
        head_type = getattr(predictor, 'lp_head_type', '')
        use_lp_cn = lp_concat_common_neighbors and head_type == 'standard'
        ncn_overlap_fn = None
        if head_type == 'hybrid3':
            try:
                from .model import _ncn_adjoverlap as ncn_overlap_fn
            except Exception:
                ncn_overlap_fn = None
        
        # Get node embeddings - use full_adj_t for test evaluation if available
        node_embeddings = get_node_embeddings(model, data, projector, identity_projection, use_full_adj_for_test, args=None, rank=rank)
        
        # Choose adjacency for LP feature computation
        adj_for_lp = data.full_adj_t if (use_full_adj_for_test and hasattr(data, 'full_adj_t') and data.full_adj_t is not None) else data.adj_t

        # Get context edge embeddings for PFN predictor
        context_edge_pairs = context_edges['edge_pairs'].to(device)
        context_labels = context_edges['labels'].to(device)
        context_src_embeds = node_embeddings[context_edge_pairs[:, 0]]
        context_dst_embeds = node_embeddings[context_edge_pairs[:, 1]]
        context_edge_embeds = context_src_embeds * context_dst_embeds
        cn_context = None
        if use_lp_cn:
            cn_context = _common_neighbor_count(adj_for_lp, context_edge_pairs)
        
        # Generate link prototypes
        link_prototypes = get_link_prototypes(node_embeddings, context_edges, att, mlp, normalize_class_h)
        if link_prototypes is None:
            if rank == 0:
                print("Warning: Could not form link prototypes during testing. Returning default results")
            return {f'hits@{k}': 0.0 for k in k_values}

        # Separate positive and negative edges
        edge_pairs = test_edges['edge_pairs'].to(device)
        labels = test_edges['labels'].to(device)
        
        # Validate test data
        if edge_pairs.size(0) == 0:
            if rank == 0:
                print("Warning: No test edges provided")
            return {f'hits@{k}': 0.0 for k in k_values}
        
        # Split into positive and negative edges
        pos_mask = labels == 1
        neg_mask = labels == 0
        pos_edges = edge_pairs[pos_mask]
        neg_edges_from_test = edge_pairs[neg_mask]
        
        # Use provided negative edges if available, otherwise use negatives from test set
        if neg_edges is not None:
            neg_edges_to_use = neg_edges.to(device)
        else:
            neg_edges_to_use = neg_edges_from_test
        
        if pos_edges.size(0) == 0 or neg_edges_to_use.size(0) == 0:
            if rank == 0:
                print("Warning: Insufficient positive or negative edges for evaluation")
            return {f'hits@{k}': 0.0 for k in k_values}

        gate_sum = 0.0
        gate_count = 0
        calib_sum = 0.0
        calib_count = 0
        gate_values = []
        hybrid_w_std_sum = 0.0
        hybrid_w_mplp_sum = 0.0
        hybrid_w_ncn_sum = 0.0
        hybrid_w_count = 0
        ncn_neg_overlap_nonzero = 0
        ncn_neg_overlap_total = 0
        hybrid_std_pos_scores = []
        hybrid_std_neg_scores = []
        hybrid_mplp_pos_scores = []
        hybrid_mplp_neg_scores = []
        hybrid_ncn_pos_scores = []
        hybrid_ncn_neg_scores = []
        struct_pos_scores = []
        struct_neg_scores = []
        feat_pos_scores = []
        feat_neg_scores = []
        struct_sum = 0.0
        struct_sumsq = 0.0
        struct_count = 0
        feat_sum = 0.0
        feat_sumsq = 0.0
        feat_count = 0
        gate_struct_abs_sum = 0.0
        gate_struct_abs_count = 0
        feat_abs_sum = 0.0
        feat_abs_count = 0
        try:
            # Compute predictions for positive edges
            pos_scores = []
            pos_dataloader = DataLoader(range(pos_edges.size(0)), batch_size, shuffle=False)
            for batch_idx in pos_dataloader:
                batch_edges = pos_edges[batch_idx]

                src_embeds = node_embeddings[batch_edges[:, 0]]
                dst_embeds = node_embeddings[batch_edges[:, 1]]
                target_edge_embeds = src_embeds * dst_embeds
                cn_target = None
                if use_lp_cn:
                    cn_target = _common_neighbor_count(adj_for_lp, batch_edges)

                # Use the unified predictor for link prediction
                if head_type in ('mplp', 'hybrid3'):
                    pred_output = predictor(
                        data,
                        context_edge_embeds,
                        target_edge_embeds,
                        context_labels.long(),
                        link_prototypes,
                        "link_prediction",
                        adj_t=adj_for_lp,
                        lp_edges=batch_edges.t(),
                        node_emb=node_embeddings,
                        lp_context_edges=context_edge_pairs.t()
                    )
                    if head_type == 'mplp' and getattr(predictor, 'lp_head', None) is not None:
                        gate_val = getattr(predictor.lp_head, 'last_gate_mean', None)
                        gate_weight = getattr(predictor.lp_head, 'last_gate_value', None)
                        if gate_val is not None:
                            gate_sum += float(gate_val.item())
                            gate_count += 1
                            gate_values.append(float(gate_val.item()))
                        calib_val = getattr(predictor.lp_head, 'last_gate_calib_ms', None)
                        if calib_val is not None:
                            calib_sum += float(calib_val)
                            calib_count += 1
                        struct_scores = getattr(predictor.lp_head, 'last_struct_score', None)
                        feat_scores = getattr(predictor.lp_head, 'last_feat_score', None)
                        if (struct_scores is None or feat_scores is None) and getattr(predictor.lp_head, 'struct_score', None) is not None:
                            struct_scores, feat_scores = predictor.lp_head.score_components(
                                target_edge_embeds, adj_for_lp, batch_edges.t(), node_emb=node_embeddings
                            )
                        if struct_scores is not None:
                            struct_pos_scores.append(struct_scores.detach().cpu())
                            struct_sum += struct_scores.sum().item()
                            struct_sumsq += (struct_scores ** 2).sum().item()
                            struct_count += struct_scores.numel()
                        if feat_scores is not None:
                            feat_sum += feat_scores.sum().item()
                            feat_sumsq += (feat_scores ** 2).sum().item()
                            feat_count += feat_scores.numel()
                            feat_pos_scores.append(feat_scores.detach().cpu())
                            feat_abs_sum += feat_scores.abs().sum().item()
                            feat_abs_count += feat_scores.numel()
                        if gate_weight is None:
                            gate_weight = gate_val
                        if gate_weight is not None and struct_scores is not None:
                            gate_struct_abs_sum += (struct_scores.abs() * gate_weight.abs()).sum().item()
                            gate_struct_abs_count += struct_scores.numel()
                    elif head_type == 'hybrid3' and getattr(predictor, 'lp_head', None) is not None:
                        fusion_w = getattr(predictor.lp_head, 'last_fusion_weights', None)
                        if fusion_w is not None and fusion_w.numel() >= 3:
                            hybrid_w_std_sum += float(fusion_w[0].item())
                            hybrid_w_mplp_sum += float(fusion_w[1].item())
                            hybrid_w_ncn_sum += float(fusion_w[2].item())
                            hybrid_w_count += 1
                        std_scores = getattr(predictor.lp_head, 'last_std_score', None)
                        mplp_scores = getattr(predictor.lp_head, 'last_mplp_struct_score', None)
                        ncn_scores = getattr(predictor.lp_head, 'last_ncn_score', None)
                        if std_scores is not None:
                            hybrid_std_pos_scores.append(std_scores.detach().cpu())
                        if mplp_scores is not None:
                            hybrid_mplp_pos_scores.append(mplp_scores.detach().cpu())
                        if ncn_scores is not None:
                            hybrid_ncn_pos_scores.append(ncn_scores.detach().cpu())
                elif head_type == 'ncn':
                    pred_output = predictor(
                        data,
                        context_edge_embeds,
                        target_edge_embeds,
                        context_labels.long(),
                        link_prototypes,
                        "link_prediction",
                        adj_t=adj_for_lp,
                        lp_edges=batch_edges.t(),
                        node_emb=node_embeddings,
                        lp_context_edges=context_edge_pairs.t()
                    )
                else:
                    pred_output = predictor(
                        data,
                        context_edge_embeds,
                        target_edge_embeds,
                        context_labels.long(),
                        link_prototypes,
                        "link_prediction",
                        lp_cn_context=cn_context,
                        lp_cn_target=cn_target
                    )
                if len(pred_output) == 3:  # MoE case with auxiliary loss
                    batch_scores, _, _ = pred_output  # Discard auxiliary loss during evaluation
                else:  # Standard case
                    batch_scores, _ = pred_output
                if batch_scores.dim() > 1:
                    batch_scores = batch_scores[:, 1]
                pos_scores.append(batch_scores.squeeze(-1).cpu())

            pos_scores = torch.cat(pos_scores, dim=0)

            # Compute predictions for negative edges
            neg_scores = []
            neg_dataloader = DataLoader(range(neg_edges_to_use.size(0)), batch_size, shuffle=False)
            for batch_idx in neg_dataloader:
                batch_edges = neg_edges_to_use[batch_idx]

                src_embeds = node_embeddings[batch_edges[:, 0]]
                dst_embeds = node_embeddings[batch_edges[:, 1]]
                target_edge_embeds = src_embeds * dst_embeds
                cn_target = None
                if use_lp_cn:
                    cn_target = _common_neighbor_count(adj_for_lp, batch_edges)

                # Use the unified predictor for link prediction
                if head_type in ('mplp', 'hybrid3'):
                    pred_output = predictor(
                        data,
                        context_edge_embeds,
                        target_edge_embeds,
                        context_labels.long(),
                        link_prototypes,
                        "link_prediction",
                        adj_t=adj_for_lp,
                        lp_edges=batch_edges.t(),
                        node_emb=node_embeddings,
                        lp_context_edges=context_edge_pairs.t()
                    )
                    if head_type == 'mplp' and getattr(predictor, 'lp_head', None) is not None:
                        gate_val = getattr(predictor.lp_head, 'last_gate_mean', None)
                        gate_weight = getattr(predictor.lp_head, 'last_gate_value', None)
                        if gate_val is not None:
                            gate_sum += float(gate_val.item())
                            gate_count += 1
                            gate_values.append(float(gate_val.item()))
                        calib_val = getattr(predictor.lp_head, 'last_gate_calib_ms', None)
                        if calib_val is not None:
                            calib_sum += float(calib_val)
                            calib_count += 1
                        struct_scores = getattr(predictor.lp_head, 'last_struct_score', None)
                        feat_scores = getattr(predictor.lp_head, 'last_feat_score', None)
                        if (struct_scores is None or feat_scores is None) and getattr(predictor.lp_head, 'struct_score', None) is not None:
                            struct_scores, feat_scores = predictor.lp_head.score_components(
                                target_edge_embeds, adj_for_lp, batch_edges.t(), node_emb=node_embeddings
                            )
                        if struct_scores is not None:
                            struct_neg_scores.append(struct_scores.detach().cpu())
                            struct_sum += struct_scores.sum().item()
                            struct_sumsq += (struct_scores ** 2).sum().item()
                            struct_count += struct_scores.numel()
                        if feat_scores is not None:
                            feat_sum += feat_scores.sum().item()
                            feat_sumsq += (feat_scores ** 2).sum().item()
                            feat_count += feat_scores.numel()
                            feat_neg_scores.append(feat_scores.detach().cpu())
                            feat_abs_sum += feat_scores.abs().sum().item()
                            feat_abs_count += feat_scores.numel()
                        if gate_weight is None:
                            gate_weight = gate_val
                        if gate_weight is not None and struct_scores is not None:
                            gate_struct_abs_sum += (struct_scores.abs() * gate_weight.abs()).sum().item()
                            gate_struct_abs_count += struct_scores.numel()
                    elif head_type == 'hybrid3' and getattr(predictor, 'lp_head', None) is not None:
                        fusion_w = getattr(predictor.lp_head, 'last_fusion_weights', None)
                        if fusion_w is not None and fusion_w.numel() >= 3:
                            hybrid_w_std_sum += float(fusion_w[0].item())
                            hybrid_w_mplp_sum += float(fusion_w[1].item())
                            hybrid_w_ncn_sum += float(fusion_w[2].item())
                            hybrid_w_count += 1
                        if ncn_overlap_fn is not None and getattr(predictor.lp_head, 'ncn_cn_branch', None) is not None:
                            cn_overlap = ncn_overlap_fn(
                                adj_for_lp,
                                adj_for_lp,
                                batch_edges.t(),
                                cnsampledeg=getattr(predictor.lp_head.ncn_cn_branch, 'cndeg', -1)
                            )
                            rowcount = cn_overlap.storage.rowcount()
                            ncn_neg_overlap_nonzero += int((rowcount > 0).sum().item())
                            ncn_neg_overlap_total += int(rowcount.numel())
                        std_scores = getattr(predictor.lp_head, 'last_std_score', None)
                        mplp_scores = getattr(predictor.lp_head, 'last_mplp_struct_score', None)
                        ncn_scores = getattr(predictor.lp_head, 'last_ncn_score', None)
                        if std_scores is not None:
                            hybrid_std_neg_scores.append(std_scores.detach().cpu())
                        if mplp_scores is not None:
                            hybrid_mplp_neg_scores.append(mplp_scores.detach().cpu())
                        if ncn_scores is not None:
                            hybrid_ncn_neg_scores.append(ncn_scores.detach().cpu())
                elif head_type == 'ncn':
                    pred_output = predictor(
                        data,
                        context_edge_embeds,
                        target_edge_embeds,
                        context_labels.long(),
                        link_prototypes,
                        "link_prediction",
                        adj_t=adj_for_lp,
                        lp_edges=batch_edges.t(),
                        node_emb=node_embeddings
                    )
                else:
                    pred_output = predictor(
                        data,
                        context_edge_embeds,
                        target_edge_embeds,
                        context_labels.long(),
                        link_prototypes,
                        "link_prediction",
                        lp_cn_context=cn_context,
                        lp_cn_target=cn_target
                    )
                if len(pred_output) == 3:  # MoE case with auxiliary loss
                    batch_scores, _, _ = pred_output  # Discard auxiliary loss during evaluation
                else:  # Standard case
                    batch_scores, _ = pred_output
                if batch_scores.dim() > 1:
                    batch_scores = batch_scores[:, 1]
                neg_scores.append(batch_scores.squeeze(-1).cpu())

            neg_scores = torch.cat(neg_scores, dim=0)
            if lp_edge_probe_debug:
                _print_lp_edge_probe_debug(
                    node_embeddings, context_edge_pairs, context_labels,
                    pos_edges, neg_edges_to_use, pos_scores, neg_scores,
                    rank=rank,
                    max_samples=int(lp_edge_probe_samples),
                    dataset_name=getattr(data, 'name', 'unknown'),
                    debug_label=lp_edge_probe_label,
                    k_values=k_values
                )
        finally:
            pass

        # Compute Hits@K using OGB evaluator
        dataset_name = getattr(data, 'name', 'unknown')
        # Use user-specified metric or dataset default
        evaluation_metric = get_evaluation_metric(dataset_name, lp_metric)
        default_metric = get_dataset_default_metric(dataset_name)
        
        if evaluator is None:
            # Choose evaluator based on dataset
            if isinstance(dataset_name, str) and dataset_name.startswith('ogbl-'):
                evaluator = Evaluator(name=dataset_name)
            else:
                evaluator = Evaluator(name='ogbl-ppa')  # Use as default

        def _has_scores(scores):
            if isinstance(scores, torch.Tensor):
                return scores.numel() > 0
            return len(scores) > 0

        struct_results = {}
        if _has_scores(struct_pos_scores) and _has_scores(struct_neg_scores):
            if not isinstance(struct_pos_scores, torch.Tensor):
                struct_pos_scores = torch.cat(struct_pos_scores, dim=0)
            if not isinstance(struct_neg_scores, torch.Tensor):
                struct_neg_scores = torch.cat(struct_neg_scores, dim=0)
            for k in k_values:
                evaluator.K = k
                hits_k = evaluator.eval({
                    'y_pred_pos': struct_pos_scores.cpu(),
                    'y_pred_neg': struct_neg_scores.cpu(),
                })[f'hits@{k}']
                struct_results[f'hits@{k}'] = hits_k
            try:
                from sklearn.metrics import roc_auc_score, accuracy_score
                pos_labels = torch.ones(struct_pos_scores.size(0))
                neg_labels = torch.zeros(struct_neg_scores.size(0))
                all_labels = torch.cat([pos_labels, neg_labels]).cpu().numpy()
                all_scores = torch.cat([struct_pos_scores, struct_neg_scores])
                all_probs = torch.sigmoid(all_scores).cpu().numpy()
                struct_results['auc'] = roc_auc_score(all_labels, all_probs)
                struct_results['acc'] = accuracy_score(all_labels, (all_probs > 0.5).astype(int))
            except Exception:
                pass

            if dataset_name == 'ogbl-citation2':
                struct_results['mrr'] = compute_mrr_citation2(struct_pos_scores, struct_neg_scores)

        feat_results = {}
        if _has_scores(feat_pos_scores) and _has_scores(feat_neg_scores):
            if not isinstance(feat_pos_scores, torch.Tensor):
                feat_pos_scores = torch.cat(feat_pos_scores, dim=0)
            if not isinstance(feat_neg_scores, torch.Tensor):
                feat_neg_scores = torch.cat(feat_neg_scores, dim=0)
            for k in k_values:
                evaluator.K = k
                hits_k = evaluator.eval({
                    'y_pred_pos': feat_pos_scores.cpu(),
                    'y_pred_neg': feat_neg_scores.cpu(),
                })[f'hits@{k}']
                feat_results[f'hits@{k}'] = hits_k
            try:
                from sklearn.metrics import roc_auc_score, accuracy_score
                pos_labels = torch.ones(feat_pos_scores.size(0))
                neg_labels = torch.zeros(feat_neg_scores.size(0))
                all_labels = torch.cat([pos_labels, neg_labels]).cpu().numpy()
                all_scores = torch.cat([feat_pos_scores, feat_neg_scores])
                all_probs = torch.sigmoid(all_scores).cpu().numpy()
                feat_results['auc'] = roc_auc_score(all_labels, all_probs)
                feat_results['acc'] = accuracy_score(all_labels, (all_probs > 0.5).astype(int))
            except Exception:
                pass
            if dataset_name == 'ogbl-citation2':
                feat_results['mrr'] = compute_mrr_citation2(feat_pos_scores, feat_neg_scores)

        def _compute_branch_results(pos_list, neg_list):
            branch_results = {}
            if not (_has_scores(pos_list) and _has_scores(neg_list)):
                return branch_results
            pos_scores_branch = pos_list if isinstance(pos_list, torch.Tensor) else torch.cat(pos_list, dim=0)
            neg_scores_branch = neg_list if isinstance(neg_list, torch.Tensor) else torch.cat(neg_list, dim=0)
            for k in k_values:
                evaluator.K = k
                hits_k = evaluator.eval({
                    'y_pred_pos': pos_scores_branch.cpu(),
                    'y_pred_neg': neg_scores_branch.cpu(),
                })[f'hits@{k}']
                branch_results[f'hits@{k}'] = hits_k
            try:
                from sklearn.metrics import roc_auc_score, accuracy_score
                pos_labels = torch.ones(pos_scores_branch.size(0))
                neg_labels = torch.zeros(neg_scores_branch.size(0))
                all_labels = torch.cat([pos_labels, neg_labels]).cpu().numpy()
                all_scores = torch.cat([pos_scores_branch, neg_scores_branch])
                all_probs = torch.sigmoid(all_scores).cpu().numpy()
                branch_results['auc'] = roc_auc_score(all_labels, all_probs)
                branch_results['acc'] = accuracy_score(all_labels, (all_probs > 0.5).astype(int))
            except Exception:
                pass
            if dataset_name == 'ogbl-citation2':
                branch_results['mrr'] = compute_mrr_citation2(pos_scores_branch, neg_scores_branch)
            return branch_results

        hybrid_std_results = _compute_branch_results(hybrid_std_pos_scores, hybrid_std_neg_scores)
        hybrid_mplp_results = _compute_branch_results(hybrid_mplp_pos_scores, hybrid_mplp_neg_scores)
        hybrid_ncn_results = _compute_branch_results(hybrid_ncn_pos_scores, hybrid_ncn_neg_scores)
        
        results = {}
        for k in k_values:
            evaluator.K = k
            hits_k = evaluator.eval({
                'y_pred_pos': pos_scores.cpu(),
                'y_pred_neg': neg_scores.cpu(),
            })[f'hits@{k}']
            results[f'hits@{k}'] = hits_k
        
        # Compute AUC and accuracy metrics
        try:
            from sklearn.metrics import roc_auc_score, accuracy_score
            import numpy as np

            # Prepare labels and scores for AUC/accuracy computation
            pos_labels = torch.ones(pos_scores.size(0))
            neg_labels = torch.zeros(neg_scores.size(0))
            all_labels = torch.cat([pos_labels, neg_labels]).cpu().numpy()
            all_scores = torch.cat([pos_scores, neg_scores])
            all_probs = torch.sigmoid(all_scores).cpu().numpy()

            # Compute AUC
            auc_score = roc_auc_score(all_labels, all_probs)
            results['auc'] = auc_score

            # Compute accuracy (using 0.5 as threshold)
            predictions = (all_probs > 0.5).astype(int)
            acc_score = accuracy_score(all_labels, predictions)
            results['acc'] = acc_score

            if rank == 0 and evaluation_metric in ['auc', 'acc']:
                print(f"AUC: {auc_score:.4f}, ACC: {acc_score:.4f}")

        except ImportError:
            if rank == 0 and evaluation_metric in ['auc', 'acc']:
                print("Warning: sklearn not available, cannot compute AUC/accuracy metrics")
        except Exception as e:
            if rank == 0:
                print(f"Error computing AUC/accuracy: {e}")

        # Compute special metrics for specific datasets
        if dataset_name == 'ogbl-citation2':
            # Special MRR calculation for citation2
            if rank == 0:
                print(f"Computing special MRR for {dataset_name}")
            mrr_value = compute_mrr_citation2(pos_scores, neg_scores)
            results['mrr'] = mrr_value
            if rank == 0:
                print(f"Citation2 MRR: {mrr_value:.4f}")
        
        # Add the evaluation metric as a convenience key
        if evaluation_metric in results:
            results['default_metric'] = results[evaluation_metric]
            results['default_metric_name'] = evaluation_metric
        elif default_metric in results:
            # Fallback to default metric if evaluation metric not available
            results['default_metric'] = results[default_metric]
            results['default_metric_name'] = default_metric
            if rank == 0 and evaluation_metric != 'auto':
                print(f"Warning: Requested metric '{evaluation_metric}' not available, using '{default_metric}' instead")
        else:
            # Emergency fallback - use hits@100 if available
            fallback_metric = 'hits@100'
            if fallback_metric in results:
                results['default_metric'] = results[fallback_metric]
                results['default_metric_name'] = fallback_metric
                if rank == 0:
                    print(f"Warning: Neither '{evaluation_metric}' nor '{default_metric}' available, using '{fallback_metric}'")
            else:
                results['default_metric'] = 0.0
                results['default_metric_name'] = 'unavailable'
                if rank == 0:
                    print(f"Warning: No suitable metric available, setting to 0.0")

        if struct_results:
            struct_metric_name = None
            if evaluation_metric in struct_results:
                struct_metric_name = evaluation_metric
            elif default_metric in struct_results:
                struct_metric_name = default_metric
            elif 'hits@100' in struct_results:
                struct_metric_name = 'hits@100'
            if struct_metric_name is not None:
                results['mplp_struct_only_metric'] = struct_results[struct_metric_name]
                results['mplp_struct_only_metric_name'] = struct_metric_name
            for k in k_values:
                key = f'hits@{k}'
                if key in struct_results:
                    results[f'mplp_struct_only_{key}'] = struct_results[key]
        else:
            results['mplp_struct_only_metric'] = None
            results['mplp_struct_only_metric_name'] = None

        if feat_results:
            feat_metric_name = None
            if evaluation_metric in feat_results:
                feat_metric_name = evaluation_metric
            elif default_metric in feat_results:
                feat_metric_name = default_metric
            elif 'hits@100' in feat_results:
                feat_metric_name = 'hits@100'
            if feat_metric_name is not None:
                results['mplp_feat_only_metric'] = feat_results[feat_metric_name]
                results['mplp_feat_only_metric_name'] = feat_metric_name
            for k in k_values:
                key = f'hits@{k}'
                if key in feat_results:
                    results[f'mplp_feat_only_{key}'] = feat_results[key]
        else:
            results['mplp_feat_only_metric'] = None
            results['mplp_feat_only_metric_name'] = None

        if struct_count > 0:
            mean = struct_sum / struct_count
            var = max(struct_sumsq / struct_count - mean * mean, 0.0)
            results['mplp_struct_score_mean'] = mean
            results['mplp_struct_score_std'] = math.sqrt(var)
        else:
            results['mplp_struct_score_mean'] = None
            results['mplp_struct_score_std'] = None

        if feat_count > 0:
            mean = feat_sum / feat_count
            var = max(feat_sumsq / feat_count - mean * mean, 0.0)
            results['mplp_feat_score_mean'] = mean
            results['mplp_feat_score_std'] = math.sqrt(var)
        else:
            results['mplp_feat_score_mean'] = None
            results['mplp_feat_score_std'] = None

        if results.get('mplp_struct_score_std') is not None and results.get('mplp_feat_score_std') is not None:
            results['mplp_struct_feat_std_ratio'] = results['mplp_struct_score_std'] / (results['mplp_feat_score_std'] + 1e-8)
        else:
            results['mplp_struct_feat_std_ratio'] = None

        if results.get('mplp_struct_score_mean') is not None and results.get('mplp_feat_score_mean') is not None:
            results['mplp_struct_feat_absmean_ratio'] = abs(results['mplp_struct_score_mean']) / (abs(results['mplp_feat_score_mean']) + 1e-8)
        else:
            results['mplp_struct_feat_absmean_ratio'] = None

        if gate_count > 0:
            results['mplp_gate_mean'] = gate_sum / gate_count
        else:
            results['mplp_gate_mean'] = None
        if calib_count > 0:
            results['mplp_gate_calib_ms'] = calib_sum / calib_count
        else:
            results['mplp_gate_calib_ms'] = None
        if gate_struct_abs_count > 0:
            results['mplp_gate_abs_struct_mean'] = gate_struct_abs_sum / gate_struct_abs_count
        else:
            results['mplp_gate_abs_struct_mean'] = None
        if feat_abs_count > 0:
            results['mplp_feat_abs_mean'] = feat_abs_sum / feat_abs_count
        else:
            results['mplp_feat_abs_mean'] = None
        if results.get('mplp_gate_abs_struct_mean') is not None and results.get('mplp_feat_abs_mean') is not None:
            results['mplp_gate_struct_abs_ratio'] = results['mplp_gate_abs_struct_mean'] / (results['mplp_feat_abs_mean'] + 1e-8)
        else:
            results['mplp_gate_struct_abs_ratio'] = None

        if hybrid_w_count > 0:
            results['hybrid3_w_std'] = hybrid_w_std_sum / hybrid_w_count
            results['hybrid3_w_mplp'] = hybrid_w_mplp_sum / hybrid_w_count
            results['hybrid3_w_ncn'] = hybrid_w_ncn_sum / hybrid_w_count
        else:
            results['hybrid3_w_std'] = None
            results['hybrid3_w_mplp'] = None
            results['hybrid3_w_ncn'] = None

        def _select_default_metric(branch_results):
            if not branch_results:
                return None, None
            if evaluation_metric in branch_results:
                return branch_results[evaluation_metric], evaluation_metric
            if default_metric in branch_results:
                return branch_results[default_metric], default_metric
            if 'hits@100' in branch_results:
                return branch_results['hits@100'], 'hits@100'
            return None, None

        hybrid_std_metric, hybrid_std_metric_name = _select_default_metric(hybrid_std_results)
        hybrid_mplp_metric, hybrid_mplp_metric_name = _select_default_metric(hybrid_mplp_results)
        hybrid_ncn_metric, hybrid_ncn_metric_name = _select_default_metric(hybrid_ncn_results)
        results['hybrid3_std_only_metric'] = hybrid_std_metric
        results['hybrid3_std_only_metric_name'] = hybrid_std_metric_name
        results['hybrid3_mplp_only_metric'] = hybrid_mplp_metric
        results['hybrid3_mplp_only_metric_name'] = hybrid_mplp_metric_name
        results['hybrid3_ncn_only_metric'] = hybrid_ncn_metric
        results['hybrid3_ncn_only_metric_name'] = hybrid_ncn_metric_name
        results['hybrid3_ncn_neg_nonzero_overlap_count'] = ncn_neg_overlap_nonzero
        results['hybrid3_ncn_neg_overlap_total_count'] = ncn_neg_overlap_total
        if return_scores:
            results['pos_scores'] = pos_scores.detach().cpu()
            results['neg_scores'] = neg_scores.detach().cpu()
            if lp_edge_probe_debug or return_context_dot_scores:
                def _dot_scores_for_edges(edges, batch_size=65536):
                    scores = []
                    for start in range(0, edges.size(0), batch_size):
                        batch_edges = edges[start:start + batch_size]
                        hu = node_embeddings[batch_edges[:, 0]].float()
                        hv = node_embeddings[batch_edges[:, 1]].float()
                        scores.append((hu * hv).sum(dim=1).detach().cpu())
                    return torch.cat(scores, dim=0)

            if lp_edge_probe_debug:
                results['dot_pos_scores'] = _dot_scores_for_edges(pos_edges)
                results['dot_neg_scores'] = _dot_scores_for_edges(neg_edges_to_use)
            if return_context_dot_scores:
                context_dot_scores = _dot_scores_for_edges(context_edge_pairs)
                context_labels_cpu = context_labels.detach().cpu().float()
                results['lp_tta_context_dot_sep'] = (
                    context_dot_scores[context_labels_cpu == 1].mean()
                    - context_dot_scores[context_labels_cpu == 0].mean()
                ).item() if (context_labels_cpu == 1).any() and (context_labels_cpu == 0).any() else None
                try:
                    from sklearn.metrics import roc_auc_score
                    if (context_labels_cpu == 1).any() and (context_labels_cpu == 0).any():
                        results['lp_tta_context_dot_auc'] = roc_auc_score(
                            context_labels_cpu.numpy(),
                            context_dot_scores.numpy()
                        )
                    else:
                        results['lp_tta_context_dot_auc'] = None
                except Exception:
                    results['lp_tta_context_dot_auc'] = None

        # Move persistent tensors back to CPU to free GPU memory
        test_edges['edge_pairs'] = test_edges['edge_pairs'].cpu()
        test_edges['labels'] = test_edges['labels'].cpu()
        context_edges['edge_pairs'] = context_edges['edge_pairs'].cpu()
        context_edges['labels'] = context_edges['labels'].cpu()
        if neg_edges is not None:
            neg_edges = neg_edges.cpu()
        
        return results
        
    except Exception as e:
        if rank == 0:
            print(f"Error during evaluation: {e}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
        raise


def _compute_lp_metrics_from_scores(pos_scores, neg_scores, dataset_name, k_values, lp_metric, rank=0):
    """Compute the same core LP metrics from already-computed edge scores."""
    evaluation_metric = get_evaluation_metric(dataset_name, lp_metric)
    default_metric = get_dataset_default_metric(dataset_name)
    evaluator_name = dataset_name if isinstance(dataset_name, str) and dataset_name.startswith('ogbl-') else 'ogbl-ppa'
    evaluator = Evaluator(name=evaluator_name)

    results = {}
    for k in k_values:
        evaluator.K = k
        results[f'hits@{k}'] = evaluator.eval({
            'y_pred_pos': pos_scores.cpu(),
            'y_pred_neg': neg_scores.cpu(),
        })[f'hits@{k}']

    try:
        from sklearn.metrics import roc_auc_score, accuracy_score
        pos_labels = torch.ones(pos_scores.size(0))
        neg_labels = torch.zeros(neg_scores.size(0))
        all_labels = torch.cat([pos_labels, neg_labels]).cpu().numpy()
        all_probs = torch.sigmoid(torch.cat([pos_scores, neg_scores])).cpu().numpy()
        results['auc'] = roc_auc_score(all_labels, all_probs)
        results['acc'] = accuracy_score(all_labels, (all_probs > 0.5).astype(int))
    except Exception as e:
        if rank == 0:
            print(f"Warning: could not compute LP AUC/ACC from TTA scores: {e}")

    if dataset_name == 'ogbl-citation2':
        results['mrr'] = compute_mrr_citation2(pos_scores, neg_scores)

    if evaluation_metric in results:
        metric_name = evaluation_metric
    elif default_metric in results:
        metric_name = default_metric
    elif 'hits@100' in results:
        metric_name = 'hits@100'
    else:
        metric_name = None

    results['default_metric'] = results[metric_name] if metric_name is not None else 0.0
    results['default_metric_name'] = metric_name or 'unavailable'
    return results


def _mask_edges_from_sparse_adj(adj_t, edge_pairs, num_nodes):
    """Return adj_t with both directions of edge_pairs removed."""
    if adj_t is None or edge_pairs is None or edge_pairs.numel() == 0:
        return adj_t, 0

    row, col, edge_attr = adj_t.coo()
    edge_pairs = edge_pairs.to(row.device)
    remove_src = edge_pairs[:, 0].long()
    remove_dst = edge_pairs[:, 1].long()
    remove_keys = torch.cat([
        remove_src * num_nodes + remove_dst,
        remove_dst * num_nodes + remove_src,
    ], dim=0)
    remove_keys = torch.unique(remove_keys)

    edge_keys = row.long() * num_nodes + col.long()
    keep_mask = ~torch.isin(edge_keys, remove_keys)
    removed = int((~keep_mask).sum().item())
    if removed == 0:
        return adj_t, 0

    masked_attr = edge_attr[keep_mask] if edge_attr is not None else None
    masked_adj = SparseTensor(
        row=row[keep_mask],
        col=col[keep_mask],
        value=masked_attr,
        sparse_sizes=adj_t.sparse_sizes()
    ).coalesce()
    return masked_adj, removed


@torch.no_grad()
def evaluate_link_prediction_with_tta(model, predictor, data, test_edges, context_edges, batch_size,
                                      att, mlp, projector=None, identity_projection=None, rank=0,
                                      normalize_class_h=False, degree=False, evaluator=None,
                                      neg_edges=None, k_values=[20, 50, 100], use_full_adj_for_test=True,
                                      lp_metric='auto', lp_concat_common_neighbors=False,
                                      num_augmentations=5, normalize_views=False,
                                      use_batchnorm=False, linear_projection=False,
                                      lp_edge_probe_debug=False,
                                      lp_edge_probe_samples=512,
                                      context_gate=False,
                                      context_gate_tolerance=0.0,
                                      train_gate=False,
                                      train_gate_edges=None,
                                      train_gate_pos_samples=256,
                                      train_gate_neg_ratio=1.0,
                                      train_gate_hits_k=20,
                                      train_gate_tolerance=0.0):
    """
    Minimal LP TTA: evaluate the original view plus K random-projection feature
    views, average positive/negative edge logits, then compute LP metrics once.
    """
    dataset_name = getattr(data, 'name', 'unknown')
    device = data.x.device
    target_dim = data.x.size(1)
    num_augmentations = 5 if num_augmentations is None else int(num_augmentations)
    train_gate_data = None
    train_gate_key = f'hits@{int(train_gate_hits_k)}'

    if train_gate and train_gate_edges is not None and train_gate_edges.get('edge_pairs', None) is not None:
        gate_edge_pairs = train_gate_edges['edge_pairs'].detach().cpu()
        gate_labels = train_gate_edges['labels'].detach().cpu()
        pos_idx = torch.nonzero(gate_labels == 1, as_tuple=False).view(-1)
        neg_idx = torch.nonzero(gate_labels == 0, as_tuple=False).view(-1)
        if pos_idx.numel() > 0 and neg_idx.numel() > 0:
            generator = torch.Generator(device='cpu')
            generator.manual_seed(20260502)
            pos_count = min(int(train_gate_pos_samples), pos_idx.numel())
            neg_count = min(max(int(round(pos_count * float(train_gate_neg_ratio))), int(train_gate_hits_k)), neg_idx.numel())
            pos_perm = torch.randperm(pos_idx.numel(), generator=generator)[:pos_count]
            neg_perm = torch.randperm(neg_idx.numel(), generator=generator)[:neg_count]
            sampled_idx = torch.cat([pos_idx[pos_perm], neg_idx[neg_perm]], dim=0)
            train_gate_data = {
                'edge_pairs': gate_edge_pairs[sampled_idx].detach().cpu(),
                'labels': gate_labels[sampled_idx].detach().cpu(),
            }
            train_gate_pos_edges_to_mask = gate_edge_pairs[pos_idx[pos_perm]].detach().cpu()
        else:
            train_gate_pos_edges_to_mask = None
    else:
        train_gate_pos_edges_to_mask = None

    def _eval_view(view_data, view_label):
        view_result = evaluate_link_prediction(
            model, predictor, view_data, test_edges, context_edges, batch_size,
            att, mlp, projector, identity_projection, rank, normalize_class_h,
            degree, evaluator, neg_edges, k_values, use_full_adj_for_test,
            lp_metric, lp_concat_common_neighbors, return_scores=True,
            lp_edge_probe_debug=lp_edge_probe_debug,
            lp_edge_probe_samples=lp_edge_probe_samples,
            lp_edge_probe_label=view_label,
            return_context_dot_scores=context_gate
        )
        if train_gate_data is not None:
            gate_view_data = view_data.clone()
            masked_adj, removed_edges = _mask_edges_from_sparse_adj(
                view_data.adj_t, train_gate_pos_edges_to_mask, view_data.num_nodes
            )
            gate_view_data.adj_t = masked_adj
            if hasattr(gate_view_data, 'full_adj_t'):
                gate_view_data.full_adj_t = None
            gate_result = evaluate_link_prediction(
                model, predictor, gate_view_data, train_gate_data, context_edges, batch_size,
                att, mlp, projector, identity_projection, rank, normalize_class_h,
                degree, evaluator=None, neg_edges=None, k_values=[int(train_gate_hits_k)],
                use_full_adj_for_test=False, lp_metric=f'hits@{int(train_gate_hits_k)}',
                lp_concat_common_neighbors=lp_concat_common_neighbors, return_scores=True,
                lp_edge_probe_debug=False
            )
            view_result['lp_tta_train_gate_metric'] = gate_result.get(train_gate_key, None)
            view_result['lp_tta_train_gate_pos_scores'] = gate_result.get('pos_scores', None)
            view_result['lp_tta_train_gate_neg_scores'] = gate_result.get('neg_scores', None)
            view_result['lp_tta_train_gate_removed_adj_edges'] = removed_edges
        return view_result

    view_results = []
    original_results = _eval_view(data, "original")
    view_results.append(original_results)

    for aug_idx in range(num_augmentations):
        seed = 999000 + aug_idx
        data_for_aug = data.clone()
        if hasattr(data, 'x_original') and data.x_original is not None:
            data_for_aug.x = data.x_original.to(device).clone()
        else:
            data_for_aug.x = data.x.detach().clone()

        data_aug = apply_random_projection_augmentation(
            data_for_aug,
            activation_pool=['identity'] if linear_projection else None,
            seed=seed,
            verbose=False,
            rank=rank,
        )

        q = min(target_dim, data_aug.x.size(0), data_aug.x.size(1))
        U, S, _ = torch.pca_lowrank(data_aug.x.to(device), q=q)
        x_aug = torch.mm(U, torch.diag(S))
        if x_aug.size(1) < target_dim:
            pad = torch.zeros(x_aug.size(0), target_dim - x_aug.size(1), device=device, dtype=x_aug.dtype)
            x_aug = torch.cat([x_aug, pad], dim=1)
        if normalize_views:
            if use_batchnorm:
                x_aug = (x_aug - x_aug.mean(dim=0, keepdim=True)) / (x_aug.std(dim=0, keepdim=True, unbiased=False) + 1e-5)
            else:
                x_aug = F.normalize(x_aug, p=2, dim=1)
        data_aug.x = x_aug

        view_results.append(_eval_view(data_aug, f"aug{aug_idx + 1}"))

    keep_mask = [True] * len(view_results)
    train_gate_tta_metric = None
    train_gate_candidate_indices = list(range(len(view_results)))
    if context_gate:
        original_gate = view_results[0].get('lp_tta_context_dot_auc', None)
        if original_gate is not None:
            for idx, view_result in enumerate(view_results[1:], start=1):
                gate_value = view_result.get('lp_tta_context_dot_auc', None)
                if gate_value is None or gate_value + float(context_gate_tolerance) < original_gate:
                    keep_mask[idx] = False

    if train_gate_data is not None:
        original_train_gate = view_results[0].get('lp_tta_train_gate_metric', None)
        train_gate_candidate_indices = [idx for idx, keep in enumerate(keep_mask) if keep or idx == 0]
        candidate_view_results = [view_results[idx] for idx in train_gate_candidate_indices]
        if (
            original_train_gate is not None and
            all(
                r.get('lp_tta_train_gate_pos_scores', None) is not None
                and r.get('lp_tta_train_gate_neg_scores', None) is not None
                for r in candidate_view_results
            )
        ):
            candidate_gate_pos_scores = torch.stack(
                [r['lp_tta_train_gate_pos_scores'] for r in candidate_view_results], dim=0
            ).mean(dim=0)
            candidate_gate_neg_scores = torch.stack(
                [r['lp_tta_train_gate_neg_scores'] for r in candidate_view_results], dim=0
            ).mean(dim=0)
            train_gate_tta_metric = _compute_lp_metrics_from_scores(
                candidate_gate_pos_scores, candidate_gate_neg_scores, dataset_name,
                [int(train_gate_hits_k)], f'hits@{int(train_gate_hits_k)}', rank
            ).get(train_gate_key, None)
            if train_gate_tta_metric is None or train_gate_tta_metric + float(train_gate_tolerance) < original_train_gate:
                keep_mask = [idx == 0 for idx in range(len(view_results))]
            else:
                keep_mask = [idx in train_gate_candidate_indices for idx in range(len(view_results))]
        else:
            keep_mask = [idx == 0 for idx in range(len(view_results))]

    kept_indices = [idx for idx, keep in enumerate(keep_mask) if keep or idx == 0]
    selected_view_results = [view_results[idx] for idx in kept_indices]

    pos_scores = torch.stack([r['pos_scores'] for r in selected_view_results], dim=0).mean(dim=0)
    neg_scores = torch.stack([r['neg_scores'] for r in selected_view_results], dim=0).mean(dim=0)
    results = _compute_lp_metrics_from_scores(pos_scores, neg_scores, dataset_name, k_values, lp_metric, rank)
    results['lp_tta_original_metric'] = original_results.get('default_metric', 0.0)
    results['lp_tta_num_views'] = len(view_results)
    results['lp_tta_num_kept_views'] = len(selected_view_results)

    if rank == 0:
        view_metrics = [r.get('default_metric', 0.0) for r in view_results]
        print(
            f"[LP_TTA] {dataset_name}: original={view_metrics[0]:.4f} "
            f"mean_view={sum(view_metrics) / len(view_metrics):.4f} "
            f"agg={results['default_metric']:.4f} views={len(view_results)} kept={len(selected_view_results)} "
            f"norm_views={normalize_views} linear_proj={linear_projection}"
        )
        if context_gate:
            gate_values = [r.get('lp_tta_context_dot_auc', None) for r in view_results]
            gate_text = ",".join("nan" if v is None else f"{v:.3f}" for v in gate_values)
            print(
                f"[LP_TTA_GATE] {dataset_name}: metric=context_dot_auc "
                f"original={gate_values[0] if gate_values[0] is not None else float('nan'):.4f} "
                f"kept={kept_indices}/{len(view_results)} tol={float(context_gate_tolerance):.4f} "
                f"values=[{gate_text}]"
        )
        if train_gate_data is not None:
            train_gate_values = [r.get('lp_tta_train_gate_metric', None) for r in view_results]
            train_gate_text = ",".join("nan" if v is None else f"{v:.3f}" for v in train_gate_values)
            removed_text = ",".join(str(r.get('lp_tta_train_gate_removed_adj_edges', 0)) for r in view_results)
            candidate_gate_pos_scores = None
            candidate_gate_neg_scores = None
            candidate_view_results = [view_results[idx] for idx in train_gate_candidate_indices]
            if all(
                r.get('lp_tta_train_gate_pos_scores', None) is not None
                and r.get('lp_tta_train_gate_neg_scores', None) is not None
                for r in candidate_view_results
            ):
                candidate_gate_pos_scores = torch.stack(
                    [r['lp_tta_train_gate_pos_scores'] for r in candidate_view_results], dim=0
                ).mean(dim=0)
                candidate_gate_neg_scores = torch.stack(
                    [r['lp_tta_train_gate_neg_scores'] for r in candidate_view_results], dim=0
                ).mean(dim=0)
            train_gate_decision = "accept" if set(kept_indices) == set(train_gate_candidate_indices) else "reject"
            print(
                f"[LP_TTA_TRAIN_GATE] {dataset_name}: metric={train_gate_key} "
                f"original={train_gate_values[0] if train_gate_values[0] is not None else float('nan'):.4f} "
                f"tta={train_gate_tta_metric if train_gate_tta_metric is not None else float('nan'):.4f} "
                f"decision={train_gate_decision} candidates={train_gate_candidate_indices}/{len(view_results)} "
                f"kept={kept_indices}/{len(view_results)} tol={float(train_gate_tolerance):.4f} "
                f"gate_edges={train_gate_data['edge_pairs'].size(0)} values=[{train_gate_text}] "
                f"masked_adj_edges=[{removed_text}]"
            )
            original_gate_pos = view_results[0].get('lp_tta_train_gate_pos_scores', None)
            original_gate_neg = view_results[0].get('lp_tta_train_gate_neg_scores', None)
            if original_gate_pos is not None and original_gate_neg is not None:
                def _gate_quantile_text(label, pos_scores_q, neg_scores_q):
                    pos_scores_q = pos_scores_q.float()
                    neg_scores_q = neg_scores_q.float()
                    pos_q = torch.quantile(pos_scores_q, torch.tensor([0.10, 0.50, 0.90], device=pos_scores_q.device))
                    neg_q = torch.quantile(neg_scores_q, torch.tensor([0.90, 0.95, 0.99], device=neg_scores_q.device))
                    k_eff = min(int(train_gate_hits_k), neg_scores_q.numel())
                    topk_thr = torch.topk(neg_scores_q, k_eff).values[-1].item() if k_eff > 0 else float('nan')
                    pos_above = (pos_scores_q > topk_thr).float().mean().item() if k_eff > 0 else float('nan')
                    return (
                        f"{label}: pos_p10/50/90={pos_q[0].item():.3f}/{pos_q[1].item():.3f}/{pos_q[2].item():.3f} "
                        f"neg_p90/95/99={neg_q[0].item():.3f}/{neg_q[1].item():.3f}/{neg_q[2].item():.3f} "
                        f"neg_top{int(train_gate_hits_k)}_thr={topk_thr:.3f} pos_above_thr={pos_above:.3f}"
                    )

                diag_parts = [_gate_quantile_text("orig", original_gate_pos, original_gate_neg)]
                if candidate_gate_pos_scores is not None and candidate_gate_neg_scores is not None:
                    diag_parts.append(_gate_quantile_text("tta", candidate_gate_pos_scores, candidate_gate_neg_scores))
                print(f"[LP_TTA_TRAIN_GATE_DIST] {dataset_name}: " + " | ".join(diag_parts))
        if lp_edge_probe_debug and all('dot_pos_scores' in r and 'dot_neg_scores' in r for r in view_results):
            dot_pos_scores = torch.stack([r['dot_pos_scores'] for r in selected_view_results], dim=0).mean(dim=0)
            dot_neg_scores = torch.stack([r['dot_neg_scores'] for r in selected_view_results], dim=0).mean(dim=0)
            dot_tta_results = _compute_lp_metrics_from_scores(
                dot_pos_scores, dot_neg_scores, dataset_name, k_values, lp_metric, rank
            )
            dot_original_results = _compute_lp_metrics_from_scores(
                view_results[0]['dot_pos_scores'], view_results[0]['dot_neg_scores'],
                dataset_name, k_values, lp_metric, rank
            )
            dot_parts = []
            for k in k_values:
                key = f'hits@{int(k)}'
                if key in dot_tta_results:
                    dot_parts.append(
                        f"{key} orig={dot_original_results.get(key, 0.0):.4f} "
                        f"tta={dot_tta_results[key]:.4f}"
                    )
            print(
                f"[LP_DOT_TTA] {dataset_name}: "
                + " | ".join(dot_parts)
                + f" views={len(view_results)} kept={len(selected_view_results)} "
                + f"norm_views={normalize_views} linear_proj={linear_projection}"
            )

    return results
