import torch
import math
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from ogb.linkproppred import Evaluator
from torch_sparse import SparseTensor
import copy
import time
import psutil
import os
from .data_utils import edge_dropout_sparse_tensor, feature_dropout


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

                # --- Optional: Masking Target Edges ---
                adj_for_gnn = data.adj_t
                if mask_target_edges:
                    # Get batch labels and find positive edges in current batch
                    batch_labels_check = labels[batch_indices]
                    batch_pos_mask = batch_labels_check == 1
                    
                    if batch_pos_mask.any():
                        # Get the actual batch indices that correspond to positive edges
                        batch_pos_indices = batch_indices[batch_pos_mask]
                        
                        # Map these batch indices to positions in the pos_train_edges tensor
                        indices_to_mask_in_pos_list = []
                        for batch_pos_idx in batch_pos_indices:
                            if batch_pos_idx.item() in pos_indices_map:
                                indices_to_mask_in_pos_list.append(pos_indices_map[batch_pos_idx.item()])
                        
                        if indices_to_mask_in_pos_list:
                            pos_adjmask[indices_to_mask_in_pos_list] = False
                            
                            # Build graph from all positive edges *not* in the current batch
                            edge = pos_train_edges[pos_adjmask].t()
                            adj_for_gnn = SparseTensor.from_edge_index(edge, sparse_sizes=(data.num_nodes, data.num_nodes)).to(device)
                            adj_for_gnn = adj_for_gnn.to_symmetric().coalesce()

                            # Reset the mask for the next iteration
                            pos_adjmask[indices_to_mask_in_pos_list] = True

                # Recompute embeddings and prototypes for each batch to maintain the computation graph
                data_for_gnn = copy.copy(data)
                data_for_gnn.adj_t = adj_for_gnn
                node_embeddings = get_node_embeddings(model, data_for_gnn, projector, identity_projection, use_full_adj=False, args=args, rank=rank)

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
    if dataset_name == 'ogbl-collab':
        return 'hits@50'
    elif dataset_name == 'ogbl-citation2':
        return 'mrr'
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

@torch.no_grad()
def evaluate_link_prediction(model, predictor, data, test_edges, context_edges, batch_size,
                             att, mlp, projector=None, identity_projection=None, rank=0,
                             normalize_class_h=False, degree=False, evaluator=None,
                             neg_edges=None, k_values=[20, 50, 100], use_full_adj_for_test=True,
                             lp_metric='auto', lp_concat_common_neighbors=False):
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
        finally:
            pass

        # Compute Hits@K using OGB evaluator
        dataset_name = getattr(data, 'name', 'unknown')
        # Use user-specified metric or dataset default
        evaluation_metric = get_evaluation_metric(dataset_name, lp_metric)
        default_metric = get_dataset_default_metric(dataset_name)
        
        if evaluator is None:
            # Choose evaluator based on dataset
            if dataset_name == 'ogbl-citation2':
                evaluator = Evaluator(name='ogbl-citation2')
            elif dataset_name == 'ogbl-collab':
                evaluator = Evaluator(name='ogbl-collab')
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
