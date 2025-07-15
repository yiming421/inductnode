import torch
import torch.nn as nn

def process_node_features(
    context_h: torch.Tensor,
    data: object, # PyG Data object, expected to have .y, .context_sample, and optionally .adj_t
    degree_normalize: bool = False,
    attention_pool_module: nn.Module = None, # e.g., instance of AttentionPool
    mlp_module: nn.Module = None,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Processes node features by selecting context nodes, optionally applying degree normalization,
    performing pooling (attention-based or mean), and optionally applying an MLP.

    Args:
        context_h (torch.Tensor): Context node features.
        data (object): A PyTorch Geometric Data object or similar, expected to have:
                       - data.y: Tensor of labels for all nodes [num_all_nodes].
                       - data.context_sample: Tensor of indices for context nodes [num_context_nodes].
                       - data.adj_t: Sparse adjacency tensor (if degree_normalize is True).
        degree_normalize (bool): If True, apply degree normalization to features.
        attention_pool_module (nn.Module, optional): An attention pooling module.
                                                     If None, mean pooling is used.
        mlp_module (nn.Module, optional): An MLP module to apply to the pooled features.

    Returns:
        torch.Tensor: Processed class representations [num_classes, output_feature_dim].
                      output_feature_dim depends on h.size(1) and attention_pool_module's output.
    """
    dev = context_h.device
    num_classes = int(data.y.max().item() + 1)

    context_y = data.y[data.context_sample]

    if degree_normalize:
        deg = data.adj_t.sum(dim=1).to(dtype=context_h.dtype, device=dev)
        degree_inv = (deg + 1e-9).pow(-1) 
        degree_inv[deg == 0] = 0 
        context_h = context_h * degree_inv[data.context_sample].view(-1, 1)

    pooled_class_h = torch.zeros(num_classes, context_h.size(1), device=dev, dtype=context_h.dtype)

    if attention_pool_module is not None:
        pooled_class_h = attention_pool_module(context_h, context_y, num_classes)
    else:
        pooled_class_h = torch.scatter_reduce(
            pooled_class_h, 0, context_y.view(-1, 1).expand(-1, context_h.size(1)), context_h,
            reduce='mean', include_self=False
        ) 
                
    if mlp_module is not None:
        final_class_h = mlp_module(pooled_class_h)
    else:
        final_class_h = pooled_class_h
    
    if normalize:
        final_class_h = nn.functional.normalize(final_class_h, p=2, dim=-1)

    return final_class_h

def acc(y_true, y_pred):
    import numpy as np
    y_true = y_true.cpu().numpy().flatten()
    y_pred = y_pred.cpu().numpy().flatten()
    correct = y_true == y_pred
    return float(np.sum(correct)) / len(correct)

def apply_final_pca(projected_features, target_dim, use_full_pca=False):
    """Apply PCA to projected features to get them in proper PCA form"""
    if use_full_pca:
        U, S, V = torch.svd(projected_features)
        U = U[:, :target_dim]
        S = S[:target_dim]
    else:
        U, S, V = torch.pca_lowrank(projected_features, q=target_dim)
    
    return torch.mm(U, torch.diag(S))

def pfn_link_score(target_edge_embedding, link_prototypes):
    """
    Computes PFN-style link prediction scores by comparing edge embeddings to prototypes.
    
    Args:
        target_edge_embedding: Embeddings of target edges [num_edges, hidden_dim]
        link_prototypes: Prototypes for 'no-link' (0) and 'link' (1) [2, hidden_dim]
    
    Returns:
        Scores (logits) for each class (no-link, link) [num_edges, 2]
    """
    # Normalize for cosine similarity, which is standard in PFN
    target_norm = torch.nn.functional.normalize(target_edge_embedding, p=2, dim=-1)
    prototypes_norm = torch.nn.functional.normalize(link_prototypes, p=2, dim=-1)
    
    # Compute dot product (cosine similarity) with both prototypes
    scores = torch.matmul(target_norm, prototypes_norm.t())
    return scores