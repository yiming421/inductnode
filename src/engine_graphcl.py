"""
Training and evaluation engine for GraphCL (Graph Contrastive Learning).

GraphCL is a self-supervised task that learns graph representations by maximizing
agreement between differently augmented views of the same graph.
"""

import time
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.loader import DataLoader
from torch_sparse import SparseTensor
import numpy as np

from .graphcl_utils import augment_graph, nt_xent_loss


def train_graphcl(model, projection_head, data_loader, optimizer, args,
                  device='cuda', identity_projection=None, rank=0, lambda_=1.0):
    """
    Train GraphCL for one epoch.

    Args:
        model: GNN encoder (e.g., PureGCN_v1)
        projection_head: MLP projection head for contrastive learning
        data_loader: DataLoader for graphs
        optimizer: Optimizer
        args: Arguments containing GraphCL configuration
        device: Device for computation
        identity_projection: Identity projection layer (if using identity projection)
        rank: Process rank for distributed training
        lambda_: Loss scaling factor (applied before backward pass)

    Returns:
        dict: Training statistics (loss, etc.)
    """
    model.train()
    projection_head.train()
    if identity_projection is not None:
        identity_projection.train()

    total_loss = 0
    num_batches = 0
    start_time = time.time()

    for batch_idx, batch_data in enumerate(data_loader):
        batch_data = batch_data.to(device)

        # Get node features
        x = batch_data.x

        # Apply identity projection ONCE if needed
        if identity_projection is not None:
            x = identity_projection(x)  # e.g., 128-dim -> 512-dim

        # Create a temporary data object with projected features
        # This will be used for augmentation
        batch_data_proj = batch_data.clone()
        batch_data_proj.x = x

        # Generate two augmented views from the PROJECTED features
        aug_data_1 = augment_graph(
            batch_data_proj,
            aug_type=args.graphcl_aug1_type,
            aug_ratio=args.graphcl_aug1_ratio
        )
        aug_data_2 = augment_graph(
            batch_data_proj,
            aug_type=args.graphcl_aug2_type,
            aug_ratio=args.graphcl_aug2_ratio
        )

        # Extract features and edge indices for both views
        x1 = aug_data_1.x
        x2 = aug_data_2.x
        edge_index_1 = aug_data_1.edge_index
        edge_index_2 = aug_data_2.edge_index

        # Get batch indices for pooling
        batch_1 = aug_data_1.batch if hasattr(aug_data_1, 'batch') else torch.zeros(x1.size(0), dtype=torch.long, device=device)
        batch_2 = aug_data_2.batch if hasattr(aug_data_2, 'batch') else torch.zeros(x2.size(0), dtype=torch.long, device=device)

        # Convert edge_index to SparseTensor (adj_t) format required by model
        num_nodes_1 = x1.size(0)
        num_nodes_2 = x2.size(0)

        adj_t_1 = SparseTensor.from_edge_index(
            edge_index_1,
            sparse_sizes=(num_nodes_1, num_nodes_1)
        ).to_symmetric().coalesce()

        adj_t_2 = SparseTensor.from_edge_index(
            edge_index_2,
            sparse_sizes=(num_nodes_2, num_nodes_2)
        ).to_symmetric().coalesce()

        # Forward pass through GNN encoder for both views
        output_1 = model(x1, adj_t_1, batch_1)
        output_2 = model(x2, adj_t_2, batch_2)

        # Handle virtual node case (model returns tuple if virtual node is used)
        if isinstance(output_1, tuple):
            node_emb_1, _ = output_1  # Discard virtual node output
        else:
            node_emb_1 = output_1

        if isinstance(output_2, tuple):
            node_emb_2, _ = output_2  # Discard virtual node output
        else:
            node_emb_2 = output_2

        # Pool to graph-level embeddings
        graph_emb_1 = pool_graph_embeddings(node_emb_1, batch_1, args.graph_pooling)  # (batch_size, hidden_dim)
        graph_emb_2 = pool_graph_embeddings(node_emb_2, batch_2, args.graph_pooling)  # (batch_size, hidden_dim)

        # Project to contrastive space using MLP projection head
        z1 = projection_head(graph_emb_1)  # (batch_size, projection_dim)
        z2 = projection_head(graph_emb_2)  # (batch_size, projection_dim)

        # Compute contrastive loss
        loss = nt_xent_loss(z1, z2, temperature=args.graphcl_temperature)

        # Scale loss by lambda before backward pass
        scaled_loss = loss * lambda_

        # Backward pass
        optimizer.zero_grad()
        scaled_loss.backward()

        # Gradient clipping if specified
        if hasattr(args, 'clip_grad') and args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            torch.nn.utils.clip_grad_norm_(projection_head.parameters(), args.clip_grad)
            if identity_projection is not None:
                torch.nn.utils.clip_grad_norm_(identity_projection.parameters(), args.clip_grad)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Logging
        if rank == 0 and (batch_idx + 1) % 10 == 0:
            print(f"  GraphCL Batch {batch_idx + 1}/{len(data_loader)}, Loss: {loss.item():.4f}")

    avg_loss = total_loss / max(num_batches, 1)
    elapsed_time = time.time() - start_time

    if rank == 0:
        print(f"GraphCL Training - Avg Loss: {avg_loss:.4f}, Time: {elapsed_time:.2f}s")

    return {
        'loss': avg_loss,
        'time': elapsed_time
    }


def pool_graph_embeddings(node_embeddings, batch, pooling_method='mean'):
    """
    Pool node embeddings to graph-level embeddings.

    Args:
        node_embeddings: Node embeddings (num_nodes, hidden_dim)
        batch: Batch assignment for each node
        pooling_method: Pooling method ('mean', 'max', 'sum')

    Returns:
        Graph embeddings (batch_size, hidden_dim)
    """
    if pooling_method == 'mean':
        return global_mean_pool(node_embeddings, batch)
    elif pooling_method == 'max':
        return global_max_pool(node_embeddings, batch)
    elif pooling_method == 'sum' or pooling_method == 'add':
        return global_add_pool(node_embeddings, batch)
    else:
        raise ValueError(f"Unknown pooling method: {pooling_method}")


def prepare_graphcl_data(datasets, args, device='cuda', rank=0):
    """
    Prepare data loaders for GraphCL task.

    Args:
        datasets: List of dataset objects
        args: Arguments containing GraphCL configuration
        device: Device for computation
        rank: Process rank for distributed training

    Returns:
        DataLoader for GraphCL training
    """
    # Combine all datasets for GraphCL
    all_graphs = []
    for dataset in datasets:
        # For graph classification datasets, use all graphs (train + val + test)
        # The goal is self-supervised learning, so we can use all data
        if hasattr(dataset, '__len__'):
            all_graphs.extend([dataset[i] for i in range(len(dataset))])
        else:
            raise ValueError(f"Dataset {dataset.name if hasattr(dataset, 'name') else 'unknown'} does not support indexing")

    if rank == 0:
        print(f"GraphCL: Collected {len(all_graphs)} graphs from {len(datasets)} datasets")

    # Create data loader
    batch_size = getattr(args, 'graphcl_batch_size', 256)
    data_loader = DataLoader(
        all_graphs,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for CUDA compatibility
        pin_memory=True
    )

    return data_loader


def create_graphcl_projection_head(args, device='cuda'):
    """
    Create projection head for GraphCL using the MLP class from model.py.

    Args:
        args: Arguments containing model configuration
        device: Device for computation

    Returns:
        MLP projection head
    """
    from .model import MLP

    hidden_dim = args.hidden
    projection_dim = args.graphcl_projection_dim
    dropout = args.dp if hasattr(args, 'dp') else 0.0
    norm = args.norm if hasattr(args, 'norm') else False

    # Create 2-layer MLP: hidden_dim -> hidden_dim -> projection_dim
    projection_head = MLP(
        in_channels=hidden_dim,
        hidden_channels=hidden_dim,
        out_channels=projection_dim,
        num_layers=2,
        dropout=dropout,
        norm=norm,
        tailact=False,  # No activation after final layer
        norm_affine=args.mlp_norm_affine if hasattr(args, 'mlp_norm_affine') else True
    ).to(device)

    return projection_head


def load_graphcl_datasets(dataset_names, args, device='cuda', rank=0):
    """
    Load datasets for GraphCL task.

    Args:
        dataset_names: List of dataset names
        args: Arguments
        device: Device
        rank: Process rank

    Returns:
        List of dataset objects
    """
    from .data_gc import load_dataset

    datasets = []
    for dataset_name in dataset_names:
        if rank == 0:
            print(f"Loading GraphCL dataset: {dataset_name}")

        # Load dataset using the unified load_dataset function
        result = load_dataset(
            dataset_name,
            root='./dataset',
            embedding_family=getattr(args, 'embedding_family', 'ST')
        )

        if result is None:
            if rank == 0:
                print(f"Warning: Failed to load dataset {dataset_name}, skipping")
            continue

        # Handle both single dataset and (dataset, mapping) returns
        if isinstance(result, tuple) and len(result) == 2:
            dataset, _ = result  # Ignore FUG mapping for GraphCL
        else:
            dataset = result

        datasets.append(dataset)

    if rank == 0:
        print(f"Loaded {len(datasets)} datasets for GraphCL")

    return datasets


def evaluate_graphcl(model, projection_head, data_loader, args, device='cuda',
                    identity_projection=None, rank=0):
    """
    Evaluate GraphCL (compute contrastive loss on validation/test set).

    Note: For self-supervised learning, evaluation is mainly for monitoring.
    The real evaluation is on downstream tasks (transfer learning).

    Args:
        model: GNN encoder
        projection_head: Projection head
        data_loader: DataLoader
        args: Arguments
        device: Device
        identity_projection: Identity projection layer
        rank: Process rank

    Returns:
        dict: Evaluation statistics
    """
    model.eval()
    projection_head.eval()
    if identity_projection is not None:
        identity_projection.eval()

    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch_data in enumerate(data_loader):
            batch_data = batch_data.to(device)

            # Get node features
            x = batch_data.x

            # Apply identity projection if needed
            if identity_projection is not None:
                x = identity_projection(x)

            # Create temporary data object with projected features
            batch_data_proj = batch_data.clone()
            batch_data_proj.x = x

            # Generate two augmented views
            aug_data_1 = augment_graph(
                batch_data_proj,
                aug_type=args.graphcl_aug1_type,
                aug_ratio=args.graphcl_aug1_ratio
            )
            aug_data_2 = augment_graph(
                batch_data_proj,
                aug_type=args.graphcl_aug2_type,
                aug_ratio=args.graphcl_aug2_ratio
            )

            # Extract features and edge indices
            x1 = aug_data_1.x
            x2 = aug_data_2.x
            edge_index_1 = aug_data_1.edge_index
            edge_index_2 = aug_data_2.edge_index

            batch_1 = aug_data_1.batch if hasattr(aug_data_1, 'batch') else torch.zeros(x1.size(0), dtype=torch.long, device=device)
            batch_2 = aug_data_2.batch if hasattr(aug_data_2, 'batch') else torch.zeros(x2.size(0), dtype=torch.long, device=device)

            # Convert edge_index to SparseTensor
            num_nodes_1 = x1.size(0)
            num_nodes_2 = x2.size(0)

            adj_t_1 = SparseTensor.from_edge_index(
                edge_index_1,
                sparse_sizes=(num_nodes_1, num_nodes_1)
            ).to_symmetric().coalesce()

            adj_t_2 = SparseTensor.from_edge_index(
                edge_index_2,
                sparse_sizes=(num_nodes_2, num_nodes_2)
            ).to_symmetric().coalesce()

            # Forward pass
            output_1 = model(x1, adj_t_1, batch_1)
            output_2 = model(x2, adj_t_2, batch_2)

            # Handle virtual node case
            if isinstance(output_1, tuple):
                node_emb_1, _ = output_1
            else:
                node_emb_1 = output_1

            if isinstance(output_2, tuple):
                node_emb_2, _ = output_2
            else:
                node_emb_2 = output_2

            graph_emb_1 = pool_graph_embeddings(node_emb_1, batch_1, args.graph_pooling)
            graph_emb_2 = pool_graph_embeddings(node_emb_2, batch_2, args.graph_pooling)

            z1 = projection_head(graph_emb_1)
            z2 = projection_head(graph_emb_2)

            loss = nt_xent_loss(z1, z2, temperature=args.graphcl_temperature)

            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)

    if rank == 0:
        print(f"GraphCL Evaluation - Avg Loss: {avg_loss:.4f}")

    return {
        'loss': avg_loss
    }
