import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from ogb.linkproppred import Evaluator
from torch_sparse import SparseTensor
import copy

from .model import PFNPredictorNodeCls
from torch.nn.parallel import DistributedDataParallel as DDP

def get_node_embeddings(model, data, projector=None, identity_projection=None):
    """
    Get node embeddings using the same model and preprocessing as node classification.
    
    Args:
        model: Trained GNN model
        data: Graph data
        projector: Optional projector module
        identity_projection: Optional identity projection module
    
    Returns:
        Node embeddings [num_nodes, hidden_dim]
    """
    # Apply same projection strategies as node classification
    if hasattr(data, 'needs_identity_projection') and data.needs_identity_projection and identity_projection is not None:
        x_input = identity_projection(data.x)
    elif hasattr(data, 'needs_projection') and data.needs_projection and projector is not None:
        projected_features = projector(data.x)
        x_input = projected_features
    else:
        x_input = data.x
    
    # Get node embeddings
    node_embeddings = model(x_input, data.adj_t)
    return node_embeddings

def get_link_prototypes(node_embeddings, context_data, att_pool, mlp_pool, normalize=False):
    """Generates 'link' and 'no-link' prototypes using the context set."""
    context_edges = context_data['edge_pairs']
    context_labels = context_data['labels']

    src_embeds = node_embeddings[context_edges[:, 0]]
    dst_embeds = node_embeddings[context_edges[:, 1]]

    # Combine node embeddings to get edge embeddings (Hadamard product is a good choice)
    edge_embeddings = src_embeds * dst_embeds

    pos_mask = context_labels == 1
    neg_mask = context_labels == 0

    pos_edge_embeddings = edge_embeddings[pos_mask]
    neg_edge_embeddings = edge_embeddings[neg_mask]

    if pos_edge_embeddings.numel() == 0 or neg_edge_embeddings.numel() == 0:
        return None # Not enough context to form prototypes

    # Use AttentionPool and MLP to get prototypes, similar to node classification
    if att_pool:
        # For link prediction, pool each set (pos/neg) into a single prototype.
        # We treat each set as having a single class (label 0).
        pos_labels = torch.zeros(pos_edge_embeddings.size(0), dtype=torch.long, device=pos_edge_embeddings.device)
        neg_labels = torch.zeros(neg_edge_embeddings.size(0), dtype=torch.long, device=neg_edge_embeddings.device)
        
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
                          batch_size, att, mlp, projector=None, identity_projection=None, 
                          clip_grad=1.0, rank=0, orthogonal_push=0.0, 
                          normalize_class_h=False, epoch=0, mask_target_edges=False, degree=False):
    """
    Train link prediction using the PFN methodology.
    """
    model.train()
    predictor.train()
    if att: att.train()
    if mlp: mlp.train()
    if projector: projector.train()
    if identity_projection: identity_projection.train()
    
    edge_pairs = train_edges['edge_pairs']
    labels = train_edges['labels']
    
    # The dataloader iterates over indices of the FULL training set
    indices = torch.arange(edge_pairs.size(0))
    
    # Use DistributedSampler if in DDP mode
    sampler = None
    if dist.is_initialized() and dist.get_world_size() > 1:
        sampler = DistributedSampler(indices, shuffle=True)
        sampler.set_epoch(epoch)

    dataloader = DataLoader(indices, batch_size=batch_size, sampler=sampler, shuffle=(sampler is None))
    
    # Correctly pre-filter for positive edges, as in the reference
    pos_train_mask = (labels == 1)
    pos_train_edges = edge_pairs[pos_train_mask]
    # The mask should be on the positive edges only
    pos_adjmask = torch.ones(pos_train_edges.size(0), dtype=torch.bool, device=edge_pairs.device)
    
    # Map original indices to their position in the positive-only list
    pos_indices_map = {orig_idx.item(): pos_idx for pos_idx, orig_idx in enumerate(torch.where(pos_train_mask)[0])}
    
    total_loss = 0
    for batch_idx in dataloader:
        optimizer.zero_grad()

        # --- Optional: Masking Target Edges ---
        adj_for_gnn = data.adj_t
        if mask_target_edges:
            # Identify which positive edges from the full list are in this batch
            batch_pos_indices = [idx.item() for idx in batch_idx if labels[idx] == 1]
            
            # Convert these to indices within the pos_train_edges tensor
            indices_to_mask_in_pos_list = [pos_indices_map[idx] for idx in batch_pos_indices if idx in pos_indices_map]
            
            if indices_to_mask_in_pos_list:
                pos_adjmask[indices_to_mask_in_pos_list] = False
                
                # Build graph from all positive edges *not* in the current batch
                edge = pos_train_edges[pos_adjmask].t()
                adj_for_gnn = SparseTensor.from_edge_index(edge, sparse_sizes=(data.num_nodes, data.num_nodes)).to(edge.device)
                adj_for_gnn = adj_for_gnn.to_symmetric().coalesce()

                # Reset the mask for the next iteration
                pos_adjmask[indices_to_mask_in_pos_list] = True

        # Recompute embeddings and prototypes for each batch to maintain the computation graph
        data_for_gnn = copy.copy(data)
        data_for_gnn.adj_t = adj_for_gnn
        node_embeddings = get_node_embeddings(model, data_for_gnn, projector, identity_projection)
        # -----------------------------------------

        # Get context edge embeddings for PFN predictor
        context_edge_pairs = context_edges['edge_pairs']
        context_labels = context_edges['labels']
        context_src_embeds = node_embeddings[context_edge_pairs[:, 0]]
        context_dst_embeds = node_embeddings[context_edge_pairs[:, 1]]
        context_edge_embeds = context_src_embeds * context_dst_embeds

        batch_labels = labels[batch_idx]
        batch_edges = edge_pairs[batch_idx]

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

        # Use PFNPredictorNodeCls for binary classification
        # context_x: context edge embeddings, target_x: target edge embeddings
        # context_y: context labels, class_x: link prototypes (neg=0, pos=1)
        
        # Create data adapter for PFN predictor (designed for node classification)
        # - Treats edge labels as "node" labels for pooling
        # - Treats all context edges as "context samples"
        predictor_data = copy.copy(data_for_gnn)
        predictor_data.y = context_labels.long()  # Convert to long for indexing
        predictor_data.context_sample = torch.arange(len(context_labels))
        
        scores = predictor(predictor_data, context_edge_embeds, target_edge_embeds, context_labels.long(), link_prototypes)
        
        # Use the train_mask to ensure loss is only calculated on non-context edges
        mask_for_loss = train_mask[batch_idx]
        
        # Use CrossEntropyLoss for multi-class classification (link vs no-link)
        nll_loss = F.cross_entropy(scores[mask_for_loss], batch_labels[mask_for_loss].long())
        
        # Compute optional orthogonal loss on prototypes
        if orthogonal_push > 0:
            proto_norm = F.normalize(link_prototypes, p=2, dim=1)
            proto_matrix = proto_norm @ proto_norm.T
            mask = ~torch.eye(proto_matrix.size(0), device=proto_matrix.device, dtype=torch.bool)
            orthogonal_loss = torch.sum(proto_matrix[mask]**2)
        else:
            orthogonal_loss = torch.tensor(0.0, device=nll_loss.device)
            
        loss = nll_loss + orthogonal_push * orthogonal_loss
        
        loss.backward()
        
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
            if att: torch.nn.utils.clip_grad_norm_(att.parameters(), clip_grad)
            if mlp: torch.nn.utils.clip_grad_norm_(mlp.parameters(), clip_grad)
            if projector: torch.nn.utils.clip_grad_norm_(projector.parameters(), clip_grad)
            if identity_projection: torch.nn.utils.clip_grad_norm_(identity_projection.parameters(), clip_grad)
        
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

@torch.no_grad()
def evaluate_link_prediction(model, predictor, data, test_edges, context_edges, batch_size,
                             att, mlp, projector=None, identity_projection=None, rank=0, 
                             normalize_class_h=False, degree=False, evaluator=None, 
                             neg_edges=None, k_values=[20, 50, 100]):
    """
    Evaluate link prediction using the PFN methodology with Hits@K metric.
    """
    model.eval()
    predictor.eval()
    if att: att.eval()
    if mlp: mlp.eval()
    if projector: projector.eval()
    if identity_projection: identity_projection.eval()
    
    # Get node embeddings
    node_embeddings = get_node_embeddings(model, data, projector, identity_projection)
    
    # Get context edge embeddings for PFN predictor
    context_edge_pairs = context_edges['edge_pairs']
    context_labels = context_edges['labels']
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
    edge_pairs = test_edges['edge_pairs']
    labels = test_edges['labels']
    
    # Split into positive and negative edges
    pos_mask = labels == 1
    neg_mask = labels == 0
    pos_edges = edge_pairs[pos_mask]
    neg_edges_from_test = edge_pairs[neg_mask]
    
    # Use provided negative edges if available, otherwise use negatives from test set
    if neg_edges is not None:
        neg_edges_to_use = neg_edges
    else:
        neg_edges_to_use = neg_edges_from_test
    
    # Compute predictions for positive edges
    pos_scores = []
    pos_dataloader = DataLoader(range(pos_edges.size(0)), batch_size, shuffle=False)
    for batch_idx in pos_dataloader:
        batch_edges = pos_edges[batch_idx]
        
        src_embeds = node_embeddings[batch_edges[:, 0]]
        dst_embeds = node_embeddings[batch_edges[:, 1]]
        target_edge_embeds = src_embeds * dst_embeds
        
        # Create data adapter for PFN predictor
        predictor_data = copy.copy(data)
        predictor_data.y = context_labels.long()  # Convert to long for indexing
        predictor_data.context_sample = torch.arange(len(context_labels))
        
        batch_scores = predictor(predictor_data, context_edge_embeds, target_edge_embeds, context_labels.long(), link_prototypes)
        # Use the positive class score (class 1)
        pos_scores.append(batch_scores[:, 1])
    
    pos_scores = torch.cat(pos_scores, dim=0)
    
    # Compute predictions for negative edges
    neg_scores = []
    neg_dataloader = DataLoader(range(neg_edges_to_use.size(0)), batch_size, shuffle=False)
    for batch_idx in neg_dataloader:
        batch_edges = neg_edges_to_use[batch_idx]
        
        src_embeds = node_embeddings[batch_edges[:, 0]]
        dst_embeds = node_embeddings[batch_edges[:, 1]]
        target_edge_embeds = src_embeds * dst_embeds
        
        # Create data adapter for PFN predictor
        predictor_data = copy.copy(data)
        predictor_data.y = context_labels.long()  # Convert to long for indexing
        predictor_data.context_sample = torch.arange(len(context_labels))
        
        batch_scores = predictor(predictor_data, context_edge_embeds, target_edge_embeds, context_labels.long(), link_prototypes)
        # Use the positive class score (class 1)
        neg_scores.append(batch_scores[:, 1])
    
    neg_scores = torch.cat(neg_scores, dim=0)
    
    # Compute Hits@K using OGB evaluator
    if evaluator is None:
        evaluator = Evaluator(name='ogbl-ppa')  # Use a standard link prediction evaluator
    
    results = {}
    for k in k_values:
        evaluator.K = k
        hits_k = evaluator.eval({
            'y_pred_pos': pos_scores.cpu(),
            'y_pred_neg': neg_scores.cpu(),
        })[f'hits@{k}']
        results[f'hits@{k}'] = hits_k
    
    return results 