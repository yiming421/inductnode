import torch
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
            
            from scripts.joint_training import resolve_context_shots
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
        
        for batch_indices in dataloader:
            # Batch-level context refresh for this LP dataset
            if args is not None:
                context_edges, train_mask = refresh_lp_context_if_needed(data, batch_count, epoch, args, context_edges, train_mask, train_edges)
            
            st = time.time()
            try:
                # batch_indices from DataLoader should already be on CPU
                # No need to move them around
                
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

                batch_labels = labels[batch_indices]
                batch_edges = edge_pairs[batch_indices]

                # Get embeddings for target edges
                src_embeds = node_embeddings[batch_edges[:, 0]]
                dst_embeds = node_embeddings[batch_edges[:, 1]]
                target_edge_embeds = src_embeds * dst_embeds

                # Get link prototypes (binary class embeddings)
                link_prototypes = get_link_prototypes(node_embeddings, context_edges, att, mlp, normalize_class_h)
                if link_prototypes is None:
                    if rank == 0:
                        print("Warning: Could not form link prototypes. Skipping batch.")
                    continue

                # Use unified PFNPredictorNodeCls for link prediction
                pred_output = predictor(data_for_gnn, context_edge_embeds, target_edge_embeds, context_labels.long(),
                                      link_prototypes, "link_prediction")
                if len(pred_output) == 3:  # MoE case with auxiliary loss
                    scores, link_prototypes, auxiliary_loss = pred_output
                else:  # Standard case
                    scores, link_prototypes = pred_output
                    auxiliary_loss = 0.0
                
                # Use the train_mask to ensure loss is only calculated on non-context edges
                # Make sure the mask is properly aligned with batch indices
                if train_mask.size(0) != edge_pairs.size(0):
                    if rank == 0:
                        print(f"Warning: train_mask size {train_mask.size(0)} doesn't match edge_pairs size {edge_pairs.size(0)}")
                    # Create a default mask that includes all edges in the batch
                    mask_for_loss = torch.ones(batch_indices.size(0), dtype=torch.bool, device=device)
                else:
                    # Index with CPU batch_indices, then move result to GPU
                    mask_for_loss = train_mask[batch_indices].to(device)
                
                # Use CrossEntropyLoss for multi-class classification (link vs no-link)
                nll_loss = F.cross_entropy(scores[mask_for_loss], batch_labels[mask_for_loss].long())
                
                # Compute optional orthogonal loss on prototypes
                if orthogonal_push > 0:
                    proto_norm = F.normalize(link_prototypes, p=2, dim=1)
                    proto_matrix = proto_norm @ proto_norm.T
                    mask = ~torch.eye(proto_matrix.size(0), device=device, dtype=torch.bool)
                    orthogonal_loss = torch.sum(proto_matrix[mask]**2)
                else:
                    orthogonal_loss = torch.tensor(0.0, device=device)
                    
                loss = nll_loss + orthogonal_push * orthogonal_loss + auxiliary_loss
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
                             lp_metric='auto'):
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
        
        # Get node embeddings - use full_adj_t for test evaluation if available
        node_embeddings = get_node_embeddings(model, data, projector, identity_projection, use_full_adj_for_test, args=None, rank=rank)
        
        # Get context edge embeddings for PFN predictor
        context_edge_pairs = context_edges['edge_pairs'].to(device)
        context_labels = context_edges['labels'].to(device)
        context_src_embeds = node_embeddings[context_edge_pairs[:, 0]]
        context_dst_embeds = node_embeddings[context_edge_pairs[:, 1]]
        context_edge_embeds = context_src_embeds * context_dst_embeds
        
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

        # Compute predictions for positive edges
        pos_scores = []
        pos_dataloader = DataLoader(range(pos_edges.size(0)), batch_size, shuffle=False)
        for batch_idx in pos_dataloader:
            batch_edges = pos_edges[batch_idx]
            
            src_embeds = node_embeddings[batch_edges[:, 0]]
            dst_embeds = node_embeddings[batch_edges[:, 1]]
            target_edge_embeds = src_embeds * dst_embeds
            
            # Use the unified predictor for link prediction
            pred_output = predictor(data, context_edge_embeds, target_edge_embeds, context_labels.long(),
                                   link_prototypes, "link_prediction")
            if len(pred_output) == 3:  # MoE case with auxiliary loss
                batch_scores, _, _ = pred_output  # Discard auxiliary loss during evaluation
            else:  # Standard case
                batch_scores, _ = pred_output
            # Use the positive class score (class 1) and move to CPU immediately
            pos_scores.append(batch_scores[:, 1].cpu())
        
        pos_scores = torch.cat(pos_scores, dim=0)
        
        # Compute predictions for negative edges
        neg_scores = []
        neg_dataloader = DataLoader(range(neg_edges_to_use.size(0)), batch_size, shuffle=False)
        for batch_idx in neg_dataloader:
            batch_edges = neg_edges_to_use[batch_idx]
            
            src_embeds = node_embeddings[batch_edges[:, 0]]
            dst_embeds = node_embeddings[batch_edges[:, 1]]
            target_edge_embeds = src_embeds * dst_embeds
            
            # Use the unified predictor for link prediction
            pred_output = predictor(data, context_edge_embeds, target_edge_embeds, context_labels.long(),
                                   link_prototypes, "link_prediction")
            if len(pred_output) == 3:  # MoE case with auxiliary loss
                batch_scores, _, _ = pred_output  # Discard auxiliary loss during evaluation
            else:  # Standard case
                batch_scores, _ = pred_output
            # Use the positive class score (class 1) and move to CPU immediately  
            neg_scores.append(batch_scores[:, 1].cpu())
        
        neg_scores = torch.cat(neg_scores, dim=0)
        
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
            all_scores = torch.cat([pos_scores, neg_scores]).cpu().numpy()

            # Compute AUC
            auc_score = roc_auc_score(all_labels, all_scores)
            results['auc'] = auc_score

            # Compute accuracy (using 0.5 as threshold)
            predictions = (all_scores > 0.5).astype(int)
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
        return {f'hits@{k}': 0.0 for k in k_values}