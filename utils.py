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