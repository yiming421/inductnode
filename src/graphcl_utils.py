"""
Graph Contrastive Learning (GraphCL) utilities.

This module provides augmentation functions and contrastive loss for GraphCL.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.utils import dropout_edge, subgraph, k_hop_subgraph
import random


def augment_graph(data, aug_type='edge_drop', aug_ratio=0.2):
    """
    Apply graph augmentation to create a view for contrastive learning.

    Note: For feat_mask, this assumes data.x has already been projected via identity projection.

    Args:
        data: PyG Data or Batch object
        aug_type: Type of augmentation ('edge_drop', 'feat_mask', 'subgraph', 'none')
        aug_ratio: Augmentation strength (e.g., drop ratio, mask ratio)

    Returns:
        Augmented data object
    """
    if aug_type == 'none':
        return data

    # Handle batch vs single graph
    is_batch = isinstance(data, Batch)

    if aug_type == 'edge_drop':
        return edge_drop_augmentation(data, aug_ratio, is_batch)
    elif aug_type == 'feat_mask':
        return feature_mask_augmentation(data, aug_ratio, is_batch)
    elif aug_type == 'subgraph':
        return subgraph_augmentation(data, aug_ratio, is_batch)
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")


def edge_drop_augmentation(data, drop_ratio=0.2, is_batch=False):
    """
    Randomly drop edges from the graph.

    Args:
        data: PyG Data or Batch object
        drop_ratio: Probability of dropping each edge
        is_batch: Whether data is a batch of graphs

    Returns:
        Data with dropped edges (creates a copy)
    """
    # Create a copy to avoid modifying original
    aug_data = data.clone()

    # dropout_edge handles both single and batched graphs correctly
    edge_index, edge_attr = dropout_edge(
        aug_data.edge_index,
        p=drop_ratio,
        force_undirected=True,
        training=True
    )
    aug_data.edge_index = edge_index
    if hasattr(aug_data, 'edge_attr') and aug_data.edge_attr is not None:
        aug_data.edge_attr = edge_attr

    return aug_data


def feature_mask_augmentation(data, mask_ratio=0.2, is_batch=False):
    """
    Randomly mask node features.

    Note: Assumes data.x has already been projected via identity projection if needed.
    Simply masks the features that are passed in.

    Args:
        data: PyG Data or Batch object
        mask_ratio: Probability of masking each feature dimension
        is_batch: Whether data is a batch of graphs

    Returns:
        Data with masked features (creates a copy)
    """
    aug_data = data.clone()

    # Get node features (assumed to be already projected)
    x = aug_data.x

    # Create random mask: 1 = keep, 0 = mask
    mask = torch.bernoulli(torch.ones_like(x) * (1 - mask_ratio))

    # Apply mask (masked values become 0)
    x_masked = x * mask

    # Store masked features
    aug_data.x = x_masked

    return aug_data


def subgraph_augmentation(data, sample_ratio=0.2, is_batch=False):
    """
    Sample a subgraph by randomly dropping nodes.

    Args:
        data: PyG Data or Batch object
        sample_ratio: Ratio of nodes to drop (will keep 1-sample_ratio nodes)
        is_batch: Whether data is a batch of graphs

    Returns:
        Subgraph data (creates a copy)
    """
    if is_batch:
        # For batched graphs, apply subgraph sampling per graph
        return batch_subgraph_augmentation(data, sample_ratio)
    else:
        return single_graph_subgraph_augmentation(data, sample_ratio)


def single_graph_subgraph_augmentation(data, drop_ratio=0.2):
    """
    Sample a subgraph from a single graph by randomly keeping nodes.

    Args:
        data: PyG Data object
        drop_ratio: Ratio of nodes to drop

    Returns:
        Subgraph data
    """
    num_nodes = data.num_nodes
    num_keep = max(1, int(num_nodes * (1 - drop_ratio)))

    # Randomly select nodes to keep
    perm = torch.randperm(num_nodes)
    subset = perm[:num_keep]

    # Extract subgraph
    edge_index, edge_attr = subgraph(
        subset,
        data.edge_index,
        edge_attr=data.edge_attr if hasattr(data, 'edge_attr') else None,
        relabel_nodes=True,
        num_nodes=num_nodes
    )

    # Create new data object
    aug_data = Data(
        x=data.x[subset],
        edge_index=edge_index,
        edge_attr=edge_attr
    )

    # Copy additional attributes
    for key in data.keys():
        if key not in ['x', 'edge_index', 'edge_attr', 'num_nodes']:
            aug_data[key] = data[key]

    return aug_data


def batch_subgraph_augmentation(batch_data, drop_ratio=0.2):
    """
    Apply subgraph sampling to a batch of graphs.

    Args:
        batch_data: PyG Batch object
        drop_ratio: Ratio of nodes to drop per graph

    Returns:
        Batched subgraphs
    """
    # Separate batch into individual graphs
    data_list = batch_data.to_data_list()

    # Apply subgraph augmentation to each
    aug_list = []
    for data in data_list:
        aug_data = single_graph_subgraph_augmentation(data, drop_ratio)
        aug_list.append(aug_data)

    # Re-batch
    return Batch.from_data_list(aug_list)


def nt_xent_loss(z1, z2, temperature=0.5):
    """
    Normalized Temperature-scaled Cross-Entropy Loss (NT-Xent) for contrastive learning.
    Also known as InfoNCE loss.

    Args:
        z1: Embeddings from view 1, shape (batch_size, embedding_dim)
        z2: Embeddings from view 2, shape (batch_size, embedding_dim)
        temperature: Temperature parameter for scaling

    Returns:
        Scalar loss value
    """
    batch_size = z1.shape[0]

    # Normalize embeddings
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)

    # Concatenate z1 and z2 to create 2N samples
    z = torch.cat([z1, z2], dim=0)  # (2*batch_size, embedding_dim)

    # Compute similarity matrix
    sim_matrix = torch.mm(z, z.T) / temperature  # (2N, 2N)

    # Create mask to exclude self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim_matrix = sim_matrix.masked_fill(mask, -9e15)

    # Positive pairs: (i, i+N) and (i+N, i)
    # For i in [0, N), the positive pair is at index i+N
    # For i in [N, 2N), the positive pair is at index i-N

    # Create labels: for sample i, its positive is at i+N (mod 2N)
    pos_indices = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),  # For z1, positives are z2
        torch.arange(0, batch_size, device=z.device)  # For z2, positives are z1
    ])

    # Compute log-softmax
    log_prob = F.log_softmax(sim_matrix, dim=1)

    # Select log-probabilities of positive pairs
    loss = -log_prob[torch.arange(2 * batch_size, device=z.device), pos_indices].mean()

    return loss


def simclr_loss(z1, z2, temperature=0.5):
    """
    SimCLR contrastive loss (alias for NT-Xent loss).

    Args:
        z1: Embeddings from view 1, shape (batch_size, embedding_dim)
        z2: Embeddings from view 2, shape (batch_size, embedding_dim)
        temperature: Temperature parameter for scaling

    Returns:
        Scalar loss value
    """
    return nt_xent_loss(z1, z2, temperature)


