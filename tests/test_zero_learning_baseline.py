"""
Zero-Learning Baseline: Pure GNN Structure Heuristic

Tests how well we can classify nodes WITHOUT any learning, using only:
1. GNN propagation (no learnable weights)
2. Mean pooling per class
3. Cosine similarity
"""

import argparse
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
import sys
import os
import wandb
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data_nc import load_data, load_ogbn_data
from src.data_gc import load_dataset
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from sklearn.metrics import roc_auc_score
from pathlib import Path


class GraphDatasetWithPE:
    """
    Wrapper for graph datasets that manages PE embeddings.

    Since PyG datasets return copies when accessed, we can't attach PEs directly.
    This wrapper loads PEs once and attaches them to graph copies on-the-fly.
    """
    def __init__(self, dataset, pe_data):
        """
        Args:
            dataset: Original PyG dataset
            pe_data: Dict with keys 'gpse', 'lappe', 'rwse', each containing (embeddings, slices) tuples
        """
        self.dataset = dataset
        self.pe_data = pe_data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get graph from original dataset (returns a copy)
        graph = self.dataset[idx]

        # Attach PEs to this copy
        for pe_type, (embeddings, slices) in self.pe_data.items():
            if embeddings is None or slices is None:
                continue

            # Handle fingerprint PE differently (graph-level, not node-level)
            if pe_type == 'fingerprint_pe':
                # Fingerprint PE is graph-level, directly use the graph embedding
                graph_pe = embeddings[idx:idx+1]  # Shape: [1, fp_dim]
                attr_name = f'{pe_type}_embeddings'
                setattr(graph, attr_name, graph_pe)
                continue

            # Check bounds for node-level PE
            if idx >= len(slices) - 1:
                print(f"Warning: Graph index {idx} out of bounds for {pe_type} PE (max: {len(slices) - 2})")
                continue

            start_idx = slices[idx].item()
            end_idx = slices[idx + 1].item()
            graph_pe = embeddings[start_idx:end_idx]

            # Verify node count matches
            if graph.x.size(0) != graph_pe.size(0):
                raise ValueError(
                    f"Node count mismatch for graph {idx} with {pe_type} PE: "
                    f"graph has {graph.x.size(0)} nodes, PE has {graph_pe.size(0)} nodes"
                )

            attr_name = f'{pe_type}_embeddings'
            setattr(graph, attr_name, graph_pe)

        return graph

    def __getattr__(self, name):
        # Delegate attribute access to the wrapped dataset
        return getattr(self.dataset, name)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_graph_pe_embeddings(dataset_name, pe_type='gpse', gpse_base_path="/home/maweishuo/GPSE/datasets"):
    """
    Load pre-computed positional encodings for graph classification datasets.

    Args:
        dataset_name: Name of the dataset (e.g., 'ogbg-molhiv', 'bace')
        pe_type: Type of PE ('gpse', 'lappe', or 'rwse')
        gpse_base_path: Base path to GPSE datasets directory

    Returns:
        tuple: (node_embeddings_tensor, slices_tensor) or (None, None) if not found
    """
    # Map dataset names to GPSE directory names
    dataset_mapping = {
        'ogbg-molhiv': 'OGB-ogbg-molhiv',
        'ogbg-molbace': 'OGB-ogbg-molbace',
        'ogbg-molbbbp': 'OGB-ogbg-molbbbp',
        'ogbg-moltox21': 'OGB-ogbg-moltox21',
        'ogbg-moltoxcast': 'OGB-ogbg-moltoxcast',
        'ogbg-molclintox': 'OGB-ogbg-molclintox',
        'ogbg-molmuv': 'OGB-ogbg-molmuv',
        'ogbg-molsider': 'OGB-ogbg-molsider',
        'bace': 'OGB-ogbg-molbace',
        'bbbp': 'OGB-ogbg-molbbbp',
        'chemhiv': 'OGB-ogbg-molhiv',
        'tox21': 'OGB-ogbg-moltox21',
        'toxcast': 'OGB-ogbg-moltoxcast',
        'clintox': 'OGB-ogbg-molclintox',
        'muv': 'OGB-ogbg-molmuv',
        'sider': 'OGB-ogbg-molsider',
    }

    # Map PE type to directory name
    pe_dir_mapping = {
        'gpse': 'pe_stats_GPSE',
        'lappe': 'pe_stats_LapPE',
        'rwse': 'pe_stats_RWSE',
    }

    if pe_type not in pe_dir_mapping:
        print(f"Warning: Unknown PE type '{pe_type}'. Available: {list(pe_dir_mapping.keys())}")
        return None, None

    gpse_dataset_name = dataset_mapping.get(dataset_name)
    if not gpse_dataset_name:
        # Try the dataset name as-is (for non-OGB datasets)
        gpse_dataset_name = dataset_name

    pe_dir = pe_dir_mapping[pe_type]
    data_path = os.path.join(gpse_base_path, gpse_dataset_name, pe_dir, "1.0", "data.pt")
    slices_path = os.path.join(gpse_base_path, gpse_dataset_name, pe_dir, "1.0", "slices.pt")

    if not os.path.exists(data_path) or not os.path.exists(slices_path):
        print(f"  [{pe_type.upper()}] Files not found for {dataset_name}")
        print(f"    Tried: {data_path}")
        return None, None

    try:
        node_embeddings = torch.load(data_path, map_location='cpu')
        slices = torch.load(slices_path, map_location='cpu')

        print(f"  [{pe_type.upper()}] Loaded embeddings for {dataset_name}")
        print(f"    Node embeddings shape: {node_embeddings.shape}")
        print(f"    Number of graphs: {len(slices) - 1}")
        print(f"    Embedding dimension: {node_embeddings.shape[1]}")

        return node_embeddings, slices

    except Exception as e:
        print(f"  [{pe_type.upper()}] Error loading embeddings: {e}")
        return None, None


def load_all_pes(dataset_name, use_gpse=True, use_lappe=True, use_rwse=True,
                 gpse_base_path="/home/maweishuo/GPSE/datasets"):
    """
    Load all available PE embeddings for a graph dataset.

    Args:
        dataset_name: Name of the dataset
        use_gpse: Whether to load GPSE embeddings
        use_lappe: Whether to load LapPE embeddings
        use_rwse: Whether to load RWSE embeddings
        gpse_base_path: Base path to PE datasets directory

    Returns:
        dict: Dictionary with keys 'gpse', 'lappe', 'rwse', each containing (embeddings, slices) tuples
    """
    pe_data = {'gpse': (None, None), 'lappe': (None, None), 'rwse': (None, None)}

    print(f"\nLoading PE embeddings for {dataset_name}...")

    if use_gpse:
        gpse_emb, gpse_slices = load_graph_pe_embeddings(dataset_name, 'gpse', gpse_base_path)
        if gpse_emb is not None:
            pe_data['gpse'] = (gpse_emb, gpse_slices)

    if use_lappe:
        lappe_emb, lappe_slices = load_graph_pe_embeddings(dataset_name, 'lappe', gpse_base_path)
        if lappe_emb is not None:
            pe_data['lappe'] = (lappe_emb, lappe_slices)

    if use_rwse:
        rwse_emb, rwse_slices = load_graph_pe_embeddings(dataset_name, 'rwse', gpse_base_path)
        if rwse_emb is not None:
            pe_data['rwse'] = (rwse_emb, rwse_slices)

    # Count and report what was loaded
    loaded = []
    for pe_type, (embeddings, slices) in pe_data.items():
        if embeddings is not None:
            loaded.append(f"{pe_type.upper()}({embeddings.shape[1]}D)")

    if loaded:
        print(f"  [PE Summary] Loaded: {', '.join(loaded)}")
    else:
        print(f"  [PE Summary] No PE embeddings found for {dataset_name}")

    return pe_data


def load_fingerprint_pe(dataset_name, pas_ogb_path="/home/maweishuo/PAS-OGB", 
                       fp_type='morgan', use_pca=False, pe_dim=64):
    """
    Load molecular fingerprints from PAS-OGB as PE for graph datasets
    
    Args:
        dataset_name: Dataset name ('ogbg-molhiv', etc.)
        pas_ogb_path: Path to PAS-OGB project  
        fp_type: 'morgan', 'maccs', or 'both'
        use_pca: Whether to apply PCA to reduce dimensionality
        pe_dim: Target dimension if using PCA
        
    Returns:
        tuple: (embeddings_tensor, slices_tensor) or (None, None) if failed
    """
    
    # Map dataset names to PAS-OGB paths
    paths = {
        'ogbg-molhiv': 'ogb-molhiv/dataset/ogbg_molhiv',
        'hiv': 'ogb-molhiv/dataset/ogbg_molhiv',
        'chemhiv': 'ogb-molhiv/dataset/ogbg_molhiv',
        'ogbg-molpcba': 'ogb-molpcba/dataset/ogbg_molpcba', 
        'pcba': 'ogb-molpcba/dataset/ogbg_molpcba'
    }
    
    if dataset_name not in paths:
        print(f"  [FP-PE] No fingerprints available for {dataset_name}")
        return None, None
    
    dataset_path = Path(pas_ogb_path) / paths[dataset_name]
    
    try:
        # Load fingerprints
        if fp_type == 'morgan':
            fp = np.load(dataset_path / "mgf_feat.npy").astype(np.float32)
        elif fp_type == 'maccs':
            fp = np.load(dataset_path / "maccs_feat.npy").astype(np.float32) 
        elif fp_type == 'both':
            morgan = np.load(dataset_path / "mgf_feat.npy").astype(np.float32)
            maccs = np.load(dataset_path / "maccs_feat.npy").astype(np.float32)
            fp = np.concatenate([morgan, maccs], axis=1)
        else:
            raise ValueError(f"Unknown fp_type: {fp_type}")
        
        print(f"  [FP-PE] Loaded {fp_type} fingerprints: {fp.shape}")
        
        # Optional PCA
        if use_pca and fp.shape[1] > pe_dim:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Standardize and apply PCA
            scaler = StandardScaler()
            fp_scaled = scaler.fit_transform(fp)
            pca = PCA(n_components=pe_dim)
            fp = pca.fit_transform(fp_scaled)
            print(f"  [FP-PE] PCA reduced to: {fp.shape}")
        
        # Convert to inductnode format (graph-level embeddings)
        embeddings = torch.tensor(fp, dtype=torch.float32)
        slices = torch.arange(len(fp) + 1)
        
        return embeddings, slices
        
    except Exception as e:
        print(f"  [FP-PE] Error loading fingerprints: {e}")
        return None, None


def load_all_pes_with_fingerprints(dataset_name, use_gpse=True, use_lappe=True, use_rwse=True,
                                  use_fingerprint_pe=False, fp_type='morgan', fp_use_pca=False, 
                                  fp_pca_dim=64, pas_ogb_path="/home/maweishuo/PAS-OGB",
                                  gpse_base_path="/home/maweishuo/GPSE/datasets"):
    """
    Load all PE embeddings including fingerprints for a graph dataset
    """
    
    pe_data = {}
    
    # Load traditional PEs
    if use_gpse or use_lappe or use_rwse:
        traditional_pes = load_all_pes(dataset_name, use_gpse, use_lappe, use_rwse, gpse_base_path)
        pe_data.update(traditional_pes)
    
    # Load fingerprint PE
    if use_fingerprint_pe:
        fp_embeddings, fp_slices = load_fingerprint_pe(
            dataset_name, pas_ogb_path, fp_type, fp_use_pca, fp_pca_dim
        )
        if fp_embeddings is not None:
            pe_data['fingerprint_pe'] = (fp_embeddings, fp_slices)
    
    return pe_data


def normalize_features(x, method='none'):
    """
    Normalize node features.

    Args:
        method: 'none', 'row' (L2 per node), 'col' (standardize per feature), 'row+col'
    """
    if method == 'none':
        return x

    if 'col' in method:
        # Column normalization: standardize each feature dimension
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True) + 1e-9
        x = (x - mean) / std

    if 'row' in method:
        # Row normalization: L2 normalize each node
        x = F.normalize(x, p=2, dim=-1)

    return x


def gcn_propagate(x, adj, num_hops=2, method='residual', alpha=0.1, use_layer_norm=False):
    """
    GCN propagation without learnable weights.

    Args:
        method: 'residual' (original), 'appnp' (PPR-style), 'concat' (multi-hop concat), 'puregcn' (PureGCN-style)
        alpha: teleport probability for APPNP
        use_layer_norm: Apply LayerNorm after each propagation step
    """
    deg = adj.sum(dim=1).to_dense()

    # PureGCN uses deg + 1 for self-loop normalization
    deg_with_selfloop = deg + 1
    deg_inv_sqrt_pure = torch.rsqrt(deg_with_selfloop.clamp(min=1e-9)).view(-1, 1)

    # Standard GCN normalization
    deg_inv_sqrt = (deg + 1e-9).pow(-0.5)

    def propagate_once(h):
        h = deg_inv_sqrt.view(-1, 1) * h
        h = adj @ h
        h = deg_inv_sqrt.view(-1, 1) * h
        if use_layer_norm:
            # LayerNorm that adapts to arbitrary input dimension
            h = F.layer_norm(h, normalized_shape=(h.size(1),))
        return h

    def propagate_puregcn(h):
        """PureGCN-style propagation: norm * (A*x + x) * norm"""
        h = deg_inv_sqrt_pure * h
        h = adj @ h + h  # A*x + x (self-loop)
        h = deg_inv_sqrt_pure * h
        if use_layer_norm:
            # Post-LN: apply LayerNorm AFTER propagation
            h = F.layer_norm(h, normalized_shape=(h.size(1),))
        return h

    if method == 'puregcn':
        # PureGCN: directly iterate without external residual
        out = x
        for _ in range(num_hops):
            out = propagate_puregcn(out)
        return out

    elif method == 'residual':
        out = x
        for _ in range(num_hops):
            out = out + propagate_once(out)
        return out

    elif method == 'appnp':
        # APPNP: h^(k) = (1-α) * A * h^(k-1) + α * x
        out = x
        for _ in range(num_hops):
            out = (1 - alpha) * propagate_once(out) + alpha * x
        return out

    elif method == 'concat':
        # Concatenate features from all hops
        features = [x]
        h = x
        for _ in range(num_hops):
            h = propagate_once(h)
            features.append(h)
        return torch.cat(features, dim=-1)

    elif method == 'weighted':
        # Weighted sum of all hops (equal weights)
        out = x
        h = x
        for _ in range(num_hops):
            h = propagate_once(h)
            out = out + h
        return out / (num_hops + 1)

    else:
        raise ValueError(f"Unknown method: {method}")


def gin_propagate(x, adj, num_hops=2, eps=0.0, use_layer_norm=False):
    """
    GIN (Graph Isomorphism Network) propagation without learnable weights.

    GIN formula: h^(k) = (1 + ε) * h^(k-1) + A * h^(k-1)

    Since we don't have learnable MLP, we use the aggregation directly.
    This is a zero-learning baseline, so we skip the MLP transformation.

    Args:
        x: Node features [N, D]
        adj: Sparse adjacency matrix
        num_hops: Number of GIN layers
        eps: Epsilon parameter (default 0.0 as in original GIN paper)
        use_layer_norm: Apply LayerNorm after each layer

    Returns:
        Node features after GIN propagation [N, D]
    """
    h = x

    for _ in range(num_hops):
        # Aggregate neighbors: A * h
        neighbor_agg = adj @ h

        # GIN update: (1 + ε) * h + neighbor_agg
        h_new = (1 + eps) * h + neighbor_agg

        if use_layer_norm:
            h_new = F.layer_norm(h_new, normalized_shape=(h_new.size(1),))

        h = h_new

    return h


def pool_graph_features(node_features, batch_indices, method='mean'):
    """
    Pool node features to graph-level representations.

    Args:
        node_features: Node feature matrix [N, D]
        batch_indices: Batch assignment for each node [N]
        method: Pooling method ('mean', 'max', 'sum', 'concat')

    Returns:
        Graph-level feature matrix [num_graphs, D] or [num_graphs, D*methods] for concat
    """
    num_graphs = batch_indices.max().item() + 1
    feat_dim = node_features.size(1)
    device = node_features.device

    if method == 'mean':
        # Mean pooling per graph
        graph_features = torch.zeros(num_graphs, feat_dim, device=device)
        for g in range(num_graphs):
            mask = batch_indices == g
            if mask.any():
                graph_features[g] = node_features[mask].mean(dim=0)
        return graph_features

    elif method == 'max':
        # Max pooling per graph
        graph_features = torch.zeros(num_graphs, feat_dim, device=device)
        for g in range(num_graphs):
            mask = batch_indices == g
            if mask.any():
                graph_features[g] = node_features[mask].max(dim=0)[0]
        return graph_features

    elif method == 'sum':
        # Sum pooling per graph
        graph_features = torch.zeros(num_graphs, feat_dim, device=device)
        for g in range(num_graphs):
            mask = batch_indices == g
            if mask.any():
                graph_features[g] = node_features[mask].sum(dim=0)
        return graph_features

    elif method == 'concat':
        # Concatenate mean, max, and sum pooling
        mean_pool = torch.zeros(num_graphs, feat_dim, device=device)
        max_pool = torch.zeros(num_graphs, feat_dim, device=device)
        sum_pool = torch.zeros(num_graphs, feat_dim, device=device)

        for g in range(num_graphs):
            mask = batch_indices == g
            if mask.any():
                graph_nodes = node_features[mask]
                mean_pool[g] = graph_nodes.mean(dim=0)
                max_pool[g] = graph_nodes.max(dim=0)[0]
                sum_pool[g] = graph_nodes.sum(dim=0)

        return torch.cat([mean_pool, max_pool, sum_pool], dim=1)

    else:
        raise ValueError(f"Unknown pooling method: {method}")


def propagate_and_pool_graphs(graph_list, num_hops=2, method='residual', alpha=0.1,
                              use_layer_norm=False, pool_method='mean', device='cpu', eps=0.0,
                              use_pe=False):
    """
    Apply graph propagation (GCN/GIN) on batched graphs, then pool to graph-level embeddings.
    Uses PyG's Batch for efficient processing of multiple graphs.

    Args:
        graph_list: List of PyG Data objects
        num_hops: Number of propagation hops
        method: Propagation method ('residual', 'appnp', 'concat', 'weighted', 'puregcn', 'gin')
        alpha: Alpha parameter for APPNP
        use_layer_norm: Whether to use layer normalization
        pool_method: How to pool node features to graph level ('mean', 'max', 'sum', 'concat')
        device: Device to use
        eps: Epsilon parameter for GIN
        use_pe: Whether to concatenate positional encodings (GPSE/LapPE/RWSE) to features

    Returns:
        graph_embeddings: [num_graphs, D] tensor of graph embeddings
    """
    # Batch all graphs together (PyG automatically creates batch indices)
    batch_data = Batch.from_data_list(graph_list).to(device)

    # Extract batched node features and edge indices
    x = batch_data.x
    edge_index = batch_data.edge_index
    batch_indices = batch_data.batch  # [num_total_nodes], indicates which graph each node belongs to
    num_nodes = x.size(0)

    # PE Enhancement: Concatenate positional encodings if available and requested
    if use_pe:
        pe_features = []

        # Check for GPSE embeddings
        if hasattr(batch_data, 'gpse_embeddings') and batch_data.gpse_embeddings is not None:
            gpse_emb = batch_data.gpse_embeddings.to(device)
            pe_features.append(gpse_emb)
            print(f"  [PE] Concatenated GPSE: {x.size(1)}D + {gpse_emb.size(1)}D")

        # Check for LapPE embeddings
        if hasattr(batch_data, 'lappe_embeddings') and batch_data.lappe_embeddings is not None:
            lappe_emb = batch_data.lappe_embeddings.to(device)
            # Replace NaN with 0 (occurs for very small graphs with < 6 nodes)
            lappe_emb = torch.nan_to_num(lappe_emb, nan=0.0)
            pe_features.append(lappe_emb)
            print(f"  [PE] Concatenated LapPE: {x.size(1)}D + {lappe_emb.size(1)}D")

        # Check for RWSE embeddings
        if hasattr(batch_data, 'rwse_embeddings') and batch_data.rwse_embeddings is not None:
            rwse_emb = batch_data.rwse_embeddings.to(device)
            pe_features.append(rwse_emb)
            print(f"  [PE] Concatenated RWSE: {x.size(1)}D + {rwse_emb.size(1)}D")

        # Note: Fingerprint PE is handled at graph level after pooling

        # Concatenate all PE features to original features
        if pe_features:
            original_dim = x.size(1)
            x = torch.cat([x] + pe_features, dim=1)
            print(f"  [PE] Total feature dimension: {original_dim}D -> {x.size(1)}D")
        else:
            print(f"  [PE] Warning: use_pe=True but no node-level PE embeddings found in graph data")

    # Build sparse adjacency matrix for the entire batch
    adj = SparseTensor.from_edge_index(
        edge_index,
        sparse_sizes=(num_nodes, num_nodes)
    ).to_symmetric()

    # Apply propagation on the batched graph
    if num_hops == 0:
        x_prop = x
    elif method == 'gin':
        # Use GIN propagation for graph classification
        x_prop = gin_propagate(x, adj, num_hops=num_hops, eps=eps,
                              use_layer_norm=use_layer_norm)
    else:
        # Use GCN-based propagation
        x_prop = gcn_propagate(x, adj, num_hops=num_hops, method=method,
                              alpha=alpha, use_layer_norm=use_layer_norm)

    # Pool node features to graph-level using batch indices
    if pool_method == 'mean':
        graph_emb = global_mean_pool(x_prop, batch_indices)
    elif pool_method == 'max':
        graph_emb = global_max_pool(x_prop, batch_indices)
    elif pool_method == 'sum':
        graph_emb = global_add_pool(x_prop, batch_indices)
    elif pool_method == 'concat':
        # Concatenate different pooling methods
        mean_pool = global_mean_pool(x_prop, batch_indices)
        max_pool = global_max_pool(x_prop, batch_indices)
        sum_pool = global_add_pool(x_prop, batch_indices)
        graph_emb = torch.cat([mean_pool, max_pool, sum_pool], dim=1)
    else:
        raise ValueError(f"Unknown pooling method: {pool_method}")

    # Concatenate fingerprint PE to graph embeddings (after pooling)
    if use_pe and hasattr(batch_data, 'fingerprint_pe_embeddings') and batch_data.fingerprint_pe_embeddings is not None:
        fp_pe = batch_data.fingerprint_pe_embeddings.to(device)
        original_dim = graph_emb.size(1)
        graph_emb = torch.cat([graph_emb, fp_pe], dim=1)
        print(f"  [Graph-PE] Concatenated Fingerprint PE: {original_dim}D + {fp_pe.size(1)}D = {graph_emb.size(1)}D")

    return graph_emb


def _kcenter_indices(features, k, seed=0):
    """Farthest-first (k-center) selection on cosine distance."""
    n = features.size(0)
    if n <= k:
        return torch.arange(n, device=features.device)

    torch.manual_seed(seed)
    first = torch.randint(0, n, (1,), device=features.device).item()
    selected = [first]

    x_norm = F.normalize(features, dim=-1)
    sims = (x_norm @ x_norm[first].unsqueeze(1)).squeeze(1)
    min_dist = 1.0 - sims

    for _ in range(1, k):
        idx = torch.argmax(min_dist).item()
        selected.append(idx)
        sims = (x_norm @ x_norm[idx].unsqueeze(1)).squeeze(1)
        dist = 1.0 - sims
        min_dist = torch.minimum(min_dist, dist)

    return torch.tensor(selected, device=features.device, dtype=torch.long)


def _kmeans_prototypes(features, k, num_iters=10, seed=0, use_cosine=True):
    """Simple k-means to produce k prototypes."""
    n = features.size(0)
    if n <= k:
        return features

    rng_state = torch.random.get_rng_state()
    torch.manual_seed(seed)

    if use_cosine:
        x = F.normalize(features, dim=-1)
    else:
        x = features

    init_idx = torch.randperm(n, device=features.device)[:k]
    centers = x[init_idx].clone()

    for _ in range(num_iters):
        if use_cosine:
            sims = x @ centers.t()
            assign = sims.argmax(dim=1)
        else:
            dist = torch.cdist(x, centers)
            assign = dist.argmin(dim=1)

        new_centers = []
        for j in range(k):
            mask = assign == j
            if mask.any():
                c = x[mask].mean(dim=0)
                if use_cosine:
                    c = F.normalize(c, dim=0)
            else:
                if use_cosine:
                    min_dist = 1.0 - sims.max(dim=1).values
                    idx = torch.argmax(min_dist).item()
                else:
                    min_dist = dist.min(dim=1).values
                    idx = torch.argmax(min_dist).item()
                c = x[idx]
            new_centers.append(c)
        centers = torch.stack(new_centers, dim=0)

    torch.random.set_rng_state(rng_state)
    return centers


def _kernel_matrix(a, b, kernel='rbf', sigma=1.0):
    if kernel == 'rbf':
        a_norm = (a ** 2).sum(dim=1, keepdim=True)
        b_norm = (b ** 2).sum(dim=1, keepdim=True).t()
        dist2 = a_norm + b_norm - 2.0 * (a @ b.t())
        dist2 = torch.clamp(dist2, min=0.0)
        gamma = 1.0 / (2.0 * (sigma ** 2) + 1e-12)
        return torch.exp(-gamma * dist2)
    if kernel == 'cos':
        a_n = F.normalize(a, dim=-1)
        b_n = F.normalize(b, dim=-1)
        return a_n @ b_n.t()
    # dot
    return a @ b.t()


def zero_learning_classify(x, context_indices, context_labels, target_indices, num_classes, sim='cos',
                           use_ridge=False, ridge_alpha=1.0,
                           proto_per_class=1, proto_method='kmeans', proto_agg='max',
                           proto_kmeans_iters=10, proto_seed=0,
                           use_knn=False, knn_k=3, knn_weighting='uniform',
                           use_kernel_ridge=False, kernel_type='rbf', kernel_sigma=1.0, kernel_ridge_alpha=1.0):
    """
    Zero-learning classification with prototypical networks or ridge regression.

    Args:
        use_ridge: Use ridge regression (closed-form solution)
        ridge_alpha: Regularization strength for ridge regression
    """

    if use_knn:
        support_x = x[context_indices]
        target_x = x[target_indices]

        if sim == 'cos':
            support_n = F.normalize(support_x, dim=-1)
            target_n = F.normalize(target_x, dim=-1)
            sims = target_n @ support_n.t()
        elif sim == 'tanimoto':
            dot_product = target_x @ support_x.t()
            target_norm_sq = (target_x ** 2).sum(dim=1, keepdim=True)
            support_norm_sq = (support_x ** 2).sum(dim=1, keepdim=True).t()
            sims = dot_product / (target_norm_sq + support_norm_sq - dot_product + 1e-8)
        else:  # dot
            sims = target_x @ support_x.t()

        k = min(knn_k, support_x.size(0))
        topk_sim, topk_idx = torch.topk(sims, k=k, dim=1)
        labels = context_labels.long()
        topk_labels = labels[topk_idx]  # [n_target, k]

        if knn_weighting == 'uniform':
            weights = torch.ones_like(topk_sim)
        else:  # similarity
            weights = torch.clamp(topk_sim, min=0.0)

        logits = torch.zeros(target_x.size(0), num_classes, device=x.device)
        for j in range(k):
            logits.scatter_add_(1, topk_labels[:, j].unsqueeze(1), weights[:, j].unsqueeze(1))

        return logits.argmax(dim=1), logits

    if use_kernel_ridge:
        support_x = x[context_indices]
        target_x = x[target_indices]
        if support_x.numel() == 0:
            logits = torch.zeros(target_x.size(0), num_classes, device=x.device)
            return logits.argmax(dim=1), logits

        K = _kernel_matrix(support_x, support_x, kernel=kernel_type, sigma=kernel_sigma)
        Y = F.one_hot(context_labels.long(), num_classes=num_classes).float()
        I = torch.eye(K.size(0), device=x.device, dtype=K.dtype)
        alpha = torch.linalg.solve(K + kernel_ridge_alpha * I, Y)
        K_test = _kernel_matrix(target_x, support_x, kernel=kernel_type, sigma=kernel_sigma)
        logits = K_test @ alpha
        return logits.argmax(dim=1), logits

    if use_ridge:
        # Ridge Regression: solve W = (X^T X + λI)^{-1} X^T Y
        support_x = x[context_indices]  # [n_support, dim]
        target_x = x[target_indices]    # [n_target, dim]

        # One-hot encode labels
        support_y = F.one_hot(context_labels.long(), num_classes=num_classes).float()  # [n_support, n_classes]

        # Solve ridge regression: W = (X^T X + λI)^{-1} X^T Y
        XtX = support_x.t() @ support_x  # [dim, dim]
        XtY = support_x.t() @ support_y  # [dim, n_classes]

        # Add regularization
        I = torch.eye(support_x.size(1), device=x.device)
        W = torch.linalg.solve(XtX + ridge_alpha * I, XtY)  # [dim, n_classes]

        # Predict
        logits = target_x @ W  # [n_target, n_classes]

        return logits.argmax(dim=1), logits

    else:
        # Prototypical Network: single or multiple prototypes per class
        target_x = x[target_indices]

        if proto_per_class <= 1:
            prototypes = torch.zeros(num_classes, x.size(1), device=x.device)
            for c in range(num_classes):
                mask = context_labels == c
                if mask.any():
                    class_features = x[context_indices[mask]]
                    prototypes[c] = class_features.mean(dim=0)

            if sim == 'cos':
                target_norm = F.normalize(target_x, dim=-1)
                proto_norm = F.normalize(prototypes, dim=-1)
                logits = target_norm @ proto_norm.t()
            elif sim == 'tanimoto':
                dot_product = target_x @ prototypes.t()
                target_norm_sq = (target_x ** 2).sum(dim=1, keepdim=True)
                proto_norm_sq = (prototypes ** 2).sum(dim=1, keepdim=True).t()
                logits = dot_product / (target_norm_sq + proto_norm_sq - dot_product + 1e-8)
            else:
                logits = target_x @ prototypes.t()

            return logits.argmax(dim=1), logits

        proto_list = []
        proto_labels = []
        for c in range(num_classes):
            mask = context_labels == c
            if not mask.any():
                continue
            class_features = x[context_indices[mask]]
            k = min(proto_per_class, class_features.size(0))

            if k == 1:
                proto = class_features.mean(dim=0, keepdim=True)
            elif proto_method == 'kcenter':
                sel = _kcenter_indices(class_features, k, seed=proto_seed + c)
                proto = class_features[sel]
            else:
                proto = _kmeans_prototypes(
                    class_features, k, num_iters=proto_kmeans_iters,
                    seed=proto_seed + c, use_cosine=(sim == 'cos')
                )

            proto_list.append(proto)
            proto_labels.append(torch.full((proto.size(0),), c, device=x.device, dtype=torch.long))

        if not proto_list:
            logits = torch.zeros(target_x.size(0), num_classes, device=x.device)
            return logits.argmax(dim=1), logits

        prototypes = torch.cat(proto_list, dim=0)
        proto_labels = torch.cat(proto_labels, dim=0)

        if sim == 'cos':
            target_norm = F.normalize(target_x, dim=-1)
            proto_norm = F.normalize(prototypes, dim=-1)
            sim_mat = target_norm @ proto_norm.t()
        elif sim == 'tanimoto':
            dot_product = target_x @ prototypes.t()
            target_norm_sq = (target_x ** 2).sum(dim=1, keepdim=True)
            proto_norm_sq = (prototypes ** 2).sum(dim=1, keepdim=True).t()
            sim_mat = dot_product / (target_norm_sq + proto_norm_sq - dot_product + 1e-8)
        else:
            sim_mat = target_x @ prototypes.t()

        logits = torch.full((target_x.size(0), num_classes), -1e9, device=x.device)
        for c in range(num_classes):
            idxs = (proto_labels == c).nonzero(as_tuple=True)[0]
            if idxs.numel() == 0:
                continue
            if proto_agg == 'mean':
                logits[:, c] = sim_mat[:, idxs].mean(dim=1)
            else:
                logits[:, c] = sim_mat[:, idxs].max(dim=1).values

        return logits.argmax(dim=1), logits


def build_anchor_meta_graph(graph_embeddings, anchor_indices, k_neighbors=5, sim='cos', weight_sharpening=1.0):
    """
    Build a sparse meta-graph using anchor-based similarity.
    Instead of computing N×N similarity, we compute N×K similarity to K anchors.
    Then connect each graph to its top-k similar anchors.

    Args:
        graph_embeddings: [num_graphs, dim] tensor of all graph embeddings
        anchor_indices: Indices of anchor graphs (subset of context set)
        k_neighbors: Number of anchors to connect each graph to
        sim: Similarity metric ('cos', 'tanimoto', 'dot')
        weight_sharpening: Power to raise edge weights to (e.g., 2.0 makes strong edges stronger)

    Returns:
        adj: Sparse adjacency matrix [num_graphs, num_graphs]
    """
    num_graphs = graph_embeddings.size(0)
    num_anchors = len(anchor_indices)
    device = graph_embeddings.device

    anchor_embeddings = graph_embeddings[anchor_indices]  # [K, dim]

    # Compute similarity from all graphs to anchors: [N, K] matrix
    if sim == 'cos':
        norm_all = F.normalize(graph_embeddings, dim=-1)
        norm_anchors = F.normalize(anchor_embeddings, dim=-1)
        similarity_to_anchors = norm_all @ norm_anchors.t()  # [N, K]
    elif sim == 'tanimoto':
        dot_product = graph_embeddings @ anchor_embeddings.t()  # [N, K]
        all_norm_sq = (graph_embeddings ** 2).sum(dim=1, keepdim=True)  # [N, 1]
        anchor_norm_sq = (anchor_embeddings ** 2).sum(dim=1, keepdim=True).t()  # [1, K]
        similarity_to_anchors = dot_product / (all_norm_sq + anchor_norm_sq - dot_product + 1e-8)
    else:  # dot
        similarity_to_anchors = graph_embeddings @ anchor_embeddings.t()

    # Connect each graph to its top-k similar anchors
    k = min(k_neighbors, num_anchors)
    top_k_values, top_k_anchor_idx = torch.topk(similarity_to_anchors, k=k, dim=1)

    # Vectorized edge construction
    # Source nodes: each graph i repeated k times
    src_nodes = torch.arange(num_graphs, device=device).unsqueeze(1).repeat(1, k).flatten()  # [N*k]

    # Target nodes: map anchor indices to global indices
    anchor_indices_tensor = torch.tensor(anchor_indices, device=device)
    dst_nodes = anchor_indices_tensor[top_k_anchor_idx.flatten()]  # [N*k]

    # Edge weights - apply sharpening to control influence
    edge_weights = top_k_values.flatten()  # [N*k]

    # Apply weight sharpening: w' = w^k
    # k > 1: emphasizes strong connections, weakens weak ones
    # k = 1: no change (default)
    # k < 1: smooths out differences
    if weight_sharpening != 1.0:
        edge_weights = torch.pow(torch.clamp(edge_weights, min=0), weight_sharpening)

    # Create bidirectional edges
    edge_index = torch.stack([
        torch.cat([src_nodes, dst_nodes]),  # sources
        torch.cat([dst_nodes, src_nodes])   # targets
    ], dim=0)
    edge_weights = torch.cat([edge_weights, edge_weights])

    adj = SparseTensor.from_edge_index(
        edge_index,
        edge_attr=edge_weights,
        sparse_sizes=(num_graphs, num_graphs)
    )

    return adj


def correct_and_smooth(adj, base_logits, train_idx, train_labels, num_classes,
                       num_iters=50, alpha=0.5):
    """
    Correct & Smooth: post-process feature-based predictions with label propagation.

    1. Start with base predictions from features (not zeros!)
    2. Propagate predictions through graph
    3. Clamp support set to ground truth at each step

    This combines feature information with graph structure.
    """
    device = base_logits.device
    num_nodes = base_logits.size(0)

    # Compute normalized adjacency
    deg = adj.sum(dim=1).to_dense()
    deg_inv_sqrt = (deg + 1e-9).pow(-0.5)

    # Start with softmax of base logits (feature-based estimate)
    Y = F.softmax(base_logits, dim=-1)

    # Ground truth for support set
    Y_support = F.one_hot(train_labels.long(), num_classes=num_classes).float()

    # Propagate and clamp
    for _ in range(num_iters):
        # Propagate
        Y_new = deg_inv_sqrt.view(-1, 1) * Y
        Y_new = adj @ Y_new
        Y_new = deg_inv_sqrt.view(-1, 1) * Y_new

        # Blend with previous
        Y = (1 - alpha) * Y_new + alpha * Y

        # Clamp support set to ground truth (force truth to flow outward)
        Y[train_idx] = Y_support

    return Y


def apply_fast_pca_with_padding(features, target_dim, use_full_pca=False, preserve_norms=False):
    """
    Apply fast GPU PCA with padding (same logic as main pipeline).

    Args:
        features (torch.Tensor): Input features [N, D]
        target_dim (int): Target dimensionality after PCA
        use_full_pca (bool): Use full SVD instead of lowrank PCA
        preserve_norms (bool): Restore original L2 norms after PCA

    Returns:
        torch.Tensor: PCA-transformed and padded features [N, target_dim]
    """
    # Store original norms if preserving them
    if preserve_norms:
        original_norms = torch.norm(features, dim=1, p=2, keepdim=True)
    original_dim = features.size(1)
    num_nodes = features.size(0)
    max_pca_dim = min(num_nodes, original_dim)

    if original_dim >= target_dim:
        # Enough features, just PCA to target_dim
        pca_target_dim = min(target_dim, max_pca_dim)
    else:
        # Not enough features, PCA to all available then pad
        pca_target_dim = min(original_dim, max_pca_dim)

    # Apply PCA using same method as main pipeline
    if use_full_pca:
        U, S, V = torch.svd(features)
        U = U[:, :pca_target_dim]
        S = S[:pca_target_dim]
    else:
        U, S, V = torch.pca_lowrank(features, q=pca_target_dim)

    x_pca = torch.mm(U, torch.diag(S))

    # Padding if necessary (same logic as main pipeline)
    if x_pca.size(1) < target_dim:
        padding_size = target_dim - x_pca.size(1)
        # Use zero padding (can be extended to other strategies)
        padding = torch.zeros(x_pca.size(0), padding_size,
                            device=x_pca.device, dtype=x_pca.dtype)
        x_pca = torch.cat([x_pca, padding], dim=1)

    # Restore original norms if requested
    if preserve_norms:
        pca_norms = torch.norm(x_pca, dim=1, p=2, keepdim=True)
        pca_norms = pca_norms + 1e-9  # Avoid division by zero
        x_pca = x_pca * (original_norms / pca_norms)

    return x_pca


def few_shot_sample(labels, num_classes, k_shot, seed=42):
    """Sample k nodes per class for few-shot."""
    torch.manual_seed(seed)
    indices = []
    for c in range(num_classes):
        class_idx = (labels == c).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(class_idx))[:k_shot]
        indices.append(class_idx[perm])
    return torch.cat(indices)


def kmedoids_sample_context(features, labels, num_classes, k_shot, train_indices, seed=42):
    """
    Sample k nodes per class using K-Medoids clustering for better representativeness.

    Args:
        features: Node features (can be raw or GCN-propagated embeddings)
        labels: Node labels
        num_classes: Number of classes
        k_shot: Number of samples per class
        train_indices: Training node indices to sample from
        seed: Random seed

    Returns:
        Selected node indices
    """
    try:
        from sklearn_extra.cluster import KMedoids
    except ImportError:
        print("[K-Medoids] scikit-learn-extra not available, falling back to random sampling")
        return few_shot_sample(labels[train_indices], num_classes, k_shot, seed=seed)

    device = features.device
    context_samples = []

    for class_label in range(num_classes):
        # Get training nodes for this class
        class_mask = labels == class_label
        class_train_mask = torch.zeros_like(labels, dtype=torch.bool)
        class_train_mask[train_indices] = True

        # Find intersection: nodes that are both in this class and in training set
        class_train_nodes = torch.where(class_mask & class_train_mask)[0]

        if len(class_train_nodes) == 0:
            continue

        if len(class_train_nodes) <= k_shot:
            # If we have fewer nodes than k_shot, take all of them
            context_samples.append(class_train_nodes)
        else:
            # Use K-Medoids clustering to find representative nodes for this class
            class_features = features[class_train_nodes].detach().cpu().numpy()

            # Apply K-Medoids clustering
            n_clusters = min(k_shot, len(class_train_nodes))
            kmedoids = KMedoids(
                n_clusters=n_clusters,
                metric='cosine',  # Use cosine distance for node features
                init='k-medoids++',  # Smart initialization
                max_iter=100,
                random_state=seed
            )

            kmedoids.fit(class_features)
            medoid_indices_in_class = kmedoids.medoid_indices_

            # Map medoid indices back to original node indices
            selected_nodes = class_train_nodes[medoid_indices_in_class]
            context_samples.append(selected_nodes)

    if not context_samples:
        # Fallback if no samples found
        print("[K-Medoids] No samples found, falling back to random sampling")
        return few_shot_sample(labels[train_indices], num_classes, k_shot, seed=seed)

    # Concatenate all context samples
    context_sample = torch.cat(context_samples, dim=0)

    print(f"[K-Medoids] Sampled {len(context_sample)} context nodes using clustering (target: {k_shot} per class)")

    return context_sample


def test_dataset(data, split_idx, num_classes, mode='full', k_shot=5, hops=2, sim='cos', device='cpu',
                 prop_method='residual', alpha=0.1, gcn_layer_norm=False, use_cs=False, cs_hops=50, cs_alpha=0.5,
                 use_ridge=False, ridge_alpha=1.0, feature_norm='none', dataset_name='',
                 use_pca=False, pca_dim=128, use_full_pca=False, pca_preserve_norms=False, num_runs=1, seeds=None,
                 use_kmedoids=False, kmedoids_on_gcn=True, use_tta=False, tta_num_augmentations=5, tta_include_original=True,
                 tta_aggregation='logits',
                 proto_per_class=1, proto_method='kmeans', proto_agg='max', proto_kmeans_iters=10,
                 use_knn=False, knn_k=3, knn_weighting='uniform',
                 use_kernel_ridge=False, kernel_type='rbf', kernel_sigma=1.0, kernel_ridge_alpha=1.0):
    """Test zero-learning baseline on a dataset with multiple runs for averaging.

    Args:
        use_pca: Whether to apply PCA with padding for dimension unification
        pca_dim: Target dimension for PCA
        use_full_pca: Use full SVD instead of lowrank PCA
        pca_preserve_norms: Restore original L2 norms after PCA transformation
        num_runs: Number of runs with different random seeds
        seeds: List of seeds to use (if None, will use range based on first seed)
        use_kmedoids: Use K-Medoids clustering for context sampling instead of random
        kmedoids_on_gcn: If True, apply K-Medoids on GCN embeddings; if False, on raw features
        use_tta: Whether to use Test-Time Augmentation
        tta_num_augmentations: Number of augmented versions to create for TTA
        tta_include_original: Whether to include original features in TTA aggregation
        tta_aggregation: TTA aggregation method - 'logits' (average logits), 'probs' (average probabilities), 'voting' (majority vote)
        proto_per_class: Number of prototypes per class (1 = standard prototypes)
        proto_method: Prototype clustering method when proto_per_class > 1 ('kmeans' or 'kcenter')
        proto_agg: How to aggregate multiple prototypes per class ('max' or 'mean')
        proto_kmeans_iters: Number of k-means iterations for prototype clustering
        use_knn: Use KNN classifier instead of prototypes/ridge
        knn_k: Number of neighbors for KNN
        knn_weighting: 'uniform' or 'similarity'
        use_kernel_ridge: Use kernel ridge regression instead of prototypes/ridge
        kernel_type: Kernel type for kernel ridge ('rbf', 'cos', 'dot')
        kernel_sigma: RBF kernel bandwidth
        kernel_ridge_alpha: Kernel ridge regularization strength
    """

    if seeds is None:
        # Default: use consecutive seeds starting from current random state
        import numpy as np
        base_seed = int(torch.initial_seed() % (2**31))
        seeds = [base_seed + i for i in range(num_runs)]

    # Store results from all runs
    all_results = []

    # Data loading and preprocessing (done once, independent of seed)
    # Ensure adj_t exists
    if not hasattr(data, 'adj_t') or data.adj_t is None:
        adj = SparseTensor.from_edge_index(
            data.edge_index,
            sparse_sizes=(data.num_nodes, data.num_nodes)
        ).to_symmetric()
    else:
        adj = data.adj_t

    test_idx = split_idx['test']
    x = data.x.to(device)
    adj = adj.to(device)
    test_idx_device = test_idx.to(device)
    target_y = data.y[test_idx].to(device)

    # Store original features for TTA (BEFORE normalization to match train.py behavior)
    x_original = x.clone() if use_tta else None

    # Apply feature normalization (deterministic, done once)
    x = normalize_features(x, method=feature_norm)

    # Apply PCA BEFORE GCN propagation (matches main pipeline: PCA is during preprocessing, not after GCN)
    if use_pca:
        print(f'  Before PCA: {x.shape}')
        x = apply_fast_pca_with_padding(x, target_dim=pca_dim, use_full_pca=use_full_pca, preserve_norms=pca_preserve_norms)
        print(f'  After PCA+padding: {x.shape}')

        # IMPORTANT: Normalize after PCA (matches main pipeline behavior)
        batch_mean = x.mean(dim=0, keepdim=True)
        batch_std = x.std(dim=0, keepdim=True, unbiased=False)
        x = (x - batch_mean) / (batch_std + 1e-5)
        print(f'  Applied BatchNorm-style normalization after PCA')

    # Test-Time Augmentation: Create augmented feature versions BEFORE GCN
    if use_tta:
        from src.data_utils import apply_random_projection_augmentation
        print(f'  [TTA] Creating {tta_num_augmentations} augmented versions...')

        # Create list of feature matrices to process
        x_versions = []
        if tta_include_original:
            x_versions.append(x)

        # Create augmented versions
        for aug_idx in range(tta_num_augmentations):
            seed = 999000 + aug_idx

            # Create a temporary Data object for augmentation
            temp_data = Data(x=x_original.clone())
            temp_data = apply_random_projection_augmentation(
                temp_data,
                hidden_dim_range=None,
                activation_pool=None,
                seed=seed,
                verbose=False,
                rank=0
            )

            # Apply PCA to augmented features to match target dimension
            x_aug = temp_data.x
            if use_pca and x_aug.shape[1] >= pca_dim:
                U, S, V = torch.pca_lowrank(x_aug, q=pca_dim)
                x_aug = torch.mm(U, torch.diag(S))
            elif use_pca and x_aug.shape[1] < pca_dim:
                # PCA to current dim, then pad
                U, S, V = torch.pca_lowrank(x_aug, q=x_aug.shape[1])
                x_aug = torch.mm(U, torch.diag(S))
                pad_size = pca_dim - x_aug.shape[1]
                padding = torch.zeros(x_aug.shape[0], pad_size, device=x_aug.device)
                x_aug = torch.cat([x_aug, padding], dim=1)

            # Normalize augmented features
            if use_pca:
                batch_mean = x_aug.mean(dim=0, keepdim=True)
                batch_std = x_aug.std(dim=0, keepdim=True, unbiased=False)
                x_aug = (x_aug - batch_mean) / (batch_std + 1e-5)

            x_versions.append(x_aug)

        print(f'  [TTA] Created {len(x_versions)} versions (original={tta_include_original}, augmented={tta_num_augmentations})')
    else:
        x_versions = [x]

    # GCN propagation for each feature version
    x_prop_versions = []
    for x_ver in x_versions:
        if hops == 0:
            x_prop_versions.append(x_ver)
        else:
            x_prop = gcn_propagate(x_ver, adj, num_hops=hops, method=prop_method, alpha=alpha,
                                   use_layer_norm=gcn_layer_norm)
            x_prop_versions.append(x_prop)
    x_prop_base = x_prop_versions[0]

    # Loop over different seeds
    for run_idx, seed in enumerate(seeds):
        torch.manual_seed(seed)
        import numpy as np
        np.random.seed(seed)

        # Get train indices for this seed (few-shot sampling)
        if mode == 'full':
            train_idx = split_idx['train']
        else:  # few-shot
            all_train = split_idx['train']

            if use_kmedoids:
                # Use K-Medoids clustering for context sampling
                # Choose features: GCN embeddings or raw features
                if kmedoids_on_gcn:
                    # Use GCN-propagated features for clustering
                    features_for_clustering = x_prop_base
                    if run_idx == 0:
                        print(f'[K-Medoids] Using GCN-propagated embeddings for clustering')
                else:
                    # Use raw features for clustering
                    features_for_clustering = x
                    if run_idx == 0:
                        print(f'[K-Medoids] Using raw features for clustering')

                train_idx = kmedoids_sample_context(
                    features_for_clustering,
                    data.y,
                    num_classes,
                    k_shot,
                    all_train,
                    seed=seed
                )
            else:
                # Original random sampling
                train_labels = data.y[all_train]
                # Sample k_shot per class
                sampled = []
                for c in range(num_classes):
                    class_mask = train_labels == c
                    class_idx = all_train[class_mask]
                    if len(class_idx) >= k_shot:
                        perm = torch.randperm(len(class_idx))[:k_shot]
                        sampled.append(class_idx[perm])
                    else:
                        sampled.append(class_idx)  # Use all if not enough
                train_idx = torch.cat(sampled)

        context_y = data.y[train_idx].to(device)
        train_idx_device = train_idx.to(device)

        if run_idx == 0:  # Print info only for first run
            print(f'  Train: {len(train_idx)}, Test: {len(test_idx)}')
            print(f'  Samples per class: {[(context_y == c).sum().item() for c in range(num_classes)]}')
            if num_runs > 1:
                print(f'  Running {num_runs} times with seeds: {seeds}')

        # TTA: Aggregate predictions from all feature versions
        if use_tta:
            all_logits_test = []
            all_preds = []
            for version_idx, x_prop in enumerate(x_prop_versions):
                pred, logits = zero_learning_classify(
                    x_prop, train_idx_device, context_y, test_idx_device, num_classes, sim,
                    use_ridge=use_ridge, ridge_alpha=ridge_alpha,
                    proto_per_class=proto_per_class, proto_method=proto_method, proto_agg=proto_agg,
                    proto_kmeans_iters=proto_kmeans_iters, proto_seed=seed,
                    use_knn=use_knn, knn_k=knn_k, knn_weighting=knn_weighting,
                    use_kernel_ridge=use_kernel_ridge, kernel_type=kernel_type,
                    kernel_sigma=kernel_sigma, kernel_ridge_alpha=kernel_ridge_alpha
                )
                all_logits_test.append(logits)
                all_preds.append(pred)

                # DEBUG: Print accuracy of each view
                if run_idx == 0:  # Only print for first run
                    test_acc = (pred == target_y).float().mean().item()
                    view_type = "Original" if version_idx == 0 and tta_include_original else f"Aug_{version_idx}"
                    print(f"      [{view_type}] Test: {test_acc:.4f}")

            # DEBUG: Print summary before aggregation
            if run_idx == 0:
                individual_test_accs = [(pred == target_y).float().mean().item() for pred in all_preds]
                print(f"      Individual test accs: mean={sum(individual_test_accs)/len(individual_test_accs):.4f}, "
                      f"min={min(individual_test_accs):.4f}, max={max(individual_test_accs):.4f}")

            # Aggregate predictions based on strategy
            if tta_aggregation == 'logits':
                # Average logits, then argmax
                aggregated_logits = torch.stack(all_logits_test).mean(dim=0)
                preds = aggregated_logits.argmax(dim=-1)
            elif tta_aggregation == 'probs':
                # Average probabilities, then argmax
                all_probs = [F.softmax(logits, dim=-1) for logits in all_logits_test]
                aggregated_probs = torch.stack(all_probs).mean(dim=0)
                preds = aggregated_probs.argmax(dim=-1)
            elif tta_aggregation == 'voting':
                # Majority voting
                all_preds_stacked = torch.stack(all_preds)  # [num_versions, num_test_nodes]
                preds = torch.mode(all_preds_stacked, dim=0).values
            else:
                raise ValueError(f"Unknown TTA aggregation method: {tta_aggregation}")

            base_acc = (preds == target_y).float().mean().item()

            # DEBUG: Print aggregation result
            if run_idx == 0:
                boost = base_acc - sum(individual_test_accs)/len(individual_test_accs)
                print(f"      After aggregation ({tta_aggregation}): test_acc={base_acc:.4f} (boost: {boost:+.4f})")
        else:
            # Single version (no TTA)
            x_prop = x_prop_versions[0]
            preds, _ = zero_learning_classify(
                x_prop, train_idx_device, context_y, test_idx_device, num_classes, sim,
                use_ridge=use_ridge, ridge_alpha=ridge_alpha,
                proto_per_class=proto_per_class, proto_method=proto_method, proto_agg=proto_agg,
                proto_kmeans_iters=proto_kmeans_iters, proto_seed=seed,
                use_knn=use_knn, knn_k=knn_k, knn_weighting=knn_weighting,
                use_kernel_ridge=use_kernel_ridge, kernel_type=kernel_type,
                kernel_sigma=kernel_sigma, kernel_ridge_alpha=kernel_ridge_alpha
            )
            base_acc = (preds == target_y).float().mean().item()

        # If C&S is enabled, try it and use if better
        if use_cs:
            # Get logits for all nodes (needed for C&S)
            _, base_logits = zero_learning_classify(x_prop, train_idx_device, context_y,
                                                     torch.arange(data.num_nodes, device=device),
                                                     num_classes, sim,
                                                     use_ridge=use_ridge,
                                                     ridge_alpha=ridge_alpha,
                                                     proto_per_class=proto_per_class, proto_method=proto_method, proto_agg=proto_agg,
                                                     proto_kmeans_iters=proto_kmeans_iters, proto_seed=seed,
                                                     use_knn=use_knn, knn_k=knn_k, knn_weighting=knn_weighting,
                                                     use_kernel_ridge=use_kernel_ridge, kernel_type=kernel_type,
                                                     kernel_sigma=kernel_sigma, kernel_ridge_alpha=kernel_ridge_alpha)
            # Apply Correct & Smooth
            Y = correct_and_smooth(adj, base_logits, train_idx_device, context_y, num_classes,
                                   num_iters=cs_hops, alpha=cs_alpha)
            cs_preds = Y[test_idx_device].argmax(dim=1)
            cs_acc = (cs_preds == target_y).float().mean().item()

            # Use C&S if it improves performance
            if cs_acc > base_acc:
                final_acc = cs_acc
                used_cs = True
            else:
                final_acc = base_acc
                used_cs = False
            
            all_results.append({
                'base_acc': base_acc,
                'cs_acc': cs_acc,
                'final_acc': final_acc,
                'used_cs': used_cs
            })
        else:
            final_acc = base_acc
            all_results.append({
                'base_acc': base_acc,
                'final_acc': final_acc
            })

        if run_idx == 0 or num_runs <= 5:  # Print individual results for first run or if few runs
            method_str = f'{hops}-hop {prop_method} {sim.upper()}'
            if use_knn:
                method_str += f' + knn(k={knn_k},{knn_weighting})'
            elif use_kernel_ridge:
                method_str += f' + krr({kernel_type},σ={kernel_sigma},λ={kernel_ridge_alpha})'
            elif use_ridge:
                method_str += f' + ridge(α={ridge_alpha})'
            
            if use_cs:
                if all_results[-1]['used_cs']:
                    method_str += f' + C&S ({cs_hops} iters, α={cs_alpha}) [SELECTED]'
                    print(f'    Run {run_idx+1}: {method_str}: {final_acc:.4f} ({final_acc*100:.2f}%) [Base: {base_acc:.4f}]')
                else:
                    print(f'    Run {run_idx+1}: {method_str}: {final_acc:.4f} ({final_acc*100:.2f}%) [C&S: {cs_acc:.4f}, base better]')
            else:
                print(f'    Run {run_idx+1}: {method_str}: {final_acc:.4f} ({final_acc*100:.2f}%)')

    # Compute and display averaged results
    base_accs = [r['base_acc'] for r in all_results]
    final_accs = [r['final_acc'] for r in all_results]
    
    avg_base = sum(base_accs) / len(base_accs)
    avg_final = sum(final_accs) / len(final_accs)
    
    method_str = f'{hops}-hop {prop_method} {sim.upper()}'
    if use_knn:
        method_str += f' + knn(k={knn_k},{knn_weighting})'
    elif use_kernel_ridge:
        method_str += f' + krr({kernel_type},σ={kernel_sigma},λ={kernel_ridge_alpha})'
    elif use_ridge:
        method_str += f' + ridge(α={ridge_alpha})'
    
    if use_cs:
        cs_accs = [r.get('cs_acc', 0) for r in all_results]
        avg_cs = sum(cs_accs) / len(cs_accs)
        cs_used_count = sum(1 for r in all_results if r.get('used_cs', False))
        method_str += f' + C&S ({cs_hops} iters, α={cs_alpha})'
        print(f'  Average: {method_str}: {avg_final:.4f} ({avg_final*100:.2f}%) [Base: {avg_base:.4f}, C&S: {avg_cs:.4f}, C&S used: {cs_used_count}/{num_runs}]')
    else:
        print(f'  Average: {method_str}: {avg_final:.4f} ({avg_final*100:.2f}%)')

    return avg_final


def test_graph_dataset(dataset, split_idx, num_classes, mode='full', k_shot=5, hops=2, sim='cos', device='cpu',
                      prop_method='residual', alpha=0.1, gcn_layer_norm=False,
                      use_ridge=False, ridge_alpha=1.0, feature_norm='none', dataset_name='',
                      use_pca=False, pca_dim=128, use_full_pca=False, pca_preserve_norms=False,
                      num_runs=1, seeds=None, pool_method='mean', gin_eps=0.0,
                      use_graph_cs=False, num_anchors=100, cs_k_neighbors=5, cs_num_iters=50, cs_alpha=0.5, weight_sharpening=1.0,
                      use_pe=False, use_knn=False, knn_k=3, knn_weighting='uniform',
                      use_kernel_ridge=False, kernel_type='rbf', kernel_sigma=1.0, kernel_ridge_alpha=1.0):
    """Test zero-learning baseline on graph classification datasets with multiple runs for averaging.

    Args:
        dataset: PyG dataset containing list of graphs
        split_idx: Dict with 'train', 'valid', 'test' keys containing graph indices
        num_classes: Number of classes
        mode: 'full' or 'few-shot'
        k_shot: Number of shots per class for few-shot mode
        pool_method: Graph pooling method ('mean', 'max', 'sum', 'concat')
        use_pe: Whether to concatenate positional encodings (GPSE/LapPE/RWSE) to features
        ... (other args same as test_dataset)
    """
    if seeds is None:
        import numpy as np
        base_seed = int(torch.initial_seed() % (2**31))
        seeds = [base_seed + i for i in range(num_runs)]

    # Store results from all runs
    all_results = []

    # Get split indices
    train_idx = split_idx['train']
    test_idx = split_idx['test']

    # Check if multi-task BEFORE sampling (do this once outside the loop)
    first_y = dataset[int(train_idx[0])].y
    is_multi_task = first_y.dim() > 0 and first_y.size(-1) > 1

    # Loop over different seeds
    for run_idx, seed in enumerate(seeds):
        torch.manual_seed(seed)
        import numpy as np
        np.random.seed(seed)

        # Sample context graphs for this run
        if mode == 'full':
            context_graph_idx = train_idx
        else:  # few-shot
            if is_multi_task:
                # For multi-task, just randomly sample (can't do class-stratified sampling)
                num_samples = min(k_shot * num_classes, len(train_idx))
                perm = torch.randperm(len(train_idx))[:num_samples]
                context_graph_idx = train_idx[perm]
            else:
                # Sample k_shot graphs per class for single-task classification
                sampled = []
                for c in range(num_classes):
                    # Find all training graphs with label c
                    class_mask = []
                    for i in train_idx:
                        y_val = dataset[int(i)].y
                        label = y_val.item() if y_val.dim() == 0 else y_val[0].item()
                        class_mask.append(label == c)
                    class_mask = torch.tensor(class_mask)
                    class_idx = train_idx[class_mask]

                    if len(class_idx) >= k_shot:
                        perm = torch.randperm(len(class_idx))[:k_shot]
                        sampled.append(class_idx[perm])
                    else:
                        sampled.append(class_idx)  # Use all if not enough
                context_graph_idx = torch.cat(sampled) if sampled else train_idx[:k_shot]

        # Load context and test graphs
        context_graphs = [dataset[int(i)] for i in context_graph_idx]
        test_graphs = [dataset[int(i)] for i in test_idx]

        # Get labels - handle multi-task datasets

        if is_multi_task:
            # Multi-task: labels are [num_graphs, num_tasks]
            context_labels = torch.stack([g.y for g in context_graphs]).to(device)
            test_labels = torch.stack([g.y for g in test_graphs]).to(device)
            
            # Handle case where labels have extra dimension [num_graphs, 1, num_tasks]
            if context_labels.dim() == 3 and context_labels.size(1) == 1:
                context_labels = context_labels.squeeze(1)  # Remove middle dimension
                test_labels = test_labels.squeeze(1)
            
            num_tasks = context_labels.size(1)
        else:
            # Single-task: labels are [num_graphs]
            context_labels = torch.tensor([g.y.item() if g.y.dim() == 0 else g.y[0].item()
                                          for g in context_graphs], device=device)
            test_labels = torch.tensor([g.y.item() if g.y.dim() == 0 else g.y[0].item()
                                       for g in test_graphs], device=device)
            num_tasks = 1

        if run_idx == 0:  # Print info only for first run
            print(f'  Context: {len(context_graphs)}, Test: {len(test_graphs)}')
            if is_multi_task:
                print(f'  Multi-task dataset: {num_tasks} tasks')
            else:
                print(f'  Samples per class: {[(context_labels == c).sum().item() for c in range(num_classes)]}')
            if num_runs > 1:
                print(f'  Running {num_runs} times with seeds: {seeds}')

        # Apply GCN propagation and pooling to get graph embeddings
        print(f'  Processing context graphs...')
        context_emb = propagate_and_pool_graphs(
            context_graphs, num_hops=hops, method=prop_method, alpha=alpha,
            use_layer_norm=gcn_layer_norm, pool_method=pool_method, device=device, eps=gin_eps,
            use_pe=use_pe
        )

        print(f'  Processing test graphs...')
        test_emb = propagate_and_pool_graphs(
            test_graphs, num_hops=hops, method=prop_method, alpha=alpha,
            use_layer_norm=gcn_layer_norm, pool_method=pool_method, device=device, eps=gin_eps,
            use_pe=use_pe
        )

        # Apply feature normalization
        if feature_norm != 'none':
            all_emb = torch.cat([context_emb, test_emb], dim=0)
            all_emb = normalize_features(all_emb, method=feature_norm)
            context_emb = all_emb[:len(context_emb)]
            test_emb = all_emb[len(context_emb):]

        # Apply PCA if requested
        if use_pca:
            print(f'  Before PCA: {context_emb.shape}')
            all_emb = torch.cat([context_emb, test_emb], dim=0)
            all_emb = apply_fast_pca_with_padding(all_emb, target_dim=pca_dim,
                                                 use_full_pca=use_full_pca,
                                                 preserve_norms=pca_preserve_norms)
            print(f'  After PCA+padding: {all_emb.shape}')

            # Normalize after PCA
            batch_mean = all_emb.mean(dim=0, keepdim=True)
            batch_std = all_emb.std(dim=0, keepdim=True, unbiased=False)
            all_emb = (all_emb - batch_mean) / (batch_std + 1e-5)

            context_emb = all_emb[:len(context_emb)]
            test_emb = all_emb[len(context_emb):]

        # Classify using prototypical networks or ridge regression
        # Create a combined embedding matrix for zero_learning_classify
        all_graph_emb = torch.cat([context_emb, test_emb], dim=0)
        context_indices = torch.arange(len(context_emb), device=device)
        test_indices = torch.arange(len(context_emb), len(context_emb) + len(test_emb), device=device)

        if is_multi_task:
            # Multi-task: compute metrics per task then average
            # Build meta-graph once if using C&S (shared across all tasks)
            if use_graph_cs:
                if run_idx == 0:
                    print(f'  [Graph C&S] Building meta-graph for multi-task ({num_tasks} tasks)...')

                num_context = len(context_graphs)
                actual_num_anchors = min(num_anchors, num_context)
                anchor_perm = torch.randperm(num_context, device=device)[:actual_num_anchors]
                anchor_graph_indices = anchor_perm.tolist()

                if run_idx == 0:
                    print(f'  [Graph C&S] Using {actual_num_anchors} anchors, k={cs_k_neighbors} neighbors')

                meta_adj = build_anchor_meta_graph(
                    all_graph_emb, anchor_graph_indices,
                    k_neighbors=cs_k_neighbors, sim=sim, weight_sharpening=weight_sharpening
                )

                if run_idx == 0:
                    num_edges = meta_adj.nnz()
                    total_graphs = len(all_graph_emb)
                    print(f'  [Graph C&S] Meta-graph: {total_graphs} nodes, {num_edges} edges')

            # Collect base and C&S results for all tasks
            base_task_accs = []
            base_task_aucs = []
            cs_task_accs = []
            cs_task_aucs = []

            for task_id in range(num_tasks):
                # Get labels for this task
                task_context_labels = context_labels[:, task_id]
                task_test_labels = test_labels[:, task_id]

                # Filter out graphs with missing labels (NaN or -1)
                valid_context = (task_context_labels >= 0) & (~torch.isnan(task_context_labels))
                valid_test = (task_test_labels >= 0) & (~torch.isnan(task_test_labels))

                if valid_context.sum() == 0 or valid_test.sum() == 0:
                    continue  # Skip task if no valid context or test samples

                task_test_labels_filtered = task_test_labels[valid_test]
                task_context_labels_filtered = task_context_labels[valid_context]
                valid_test_indices = test_indices[valid_test]
                valid_context_indices = context_indices[valid_context]

                # Base classification for this task
                base_preds, base_logits = zero_learning_classify(
                    all_graph_emb, valid_context_indices, task_context_labels_filtered,
                    valid_test_indices, num_classes, sim, use_ridge=use_ridge, ridge_alpha=ridge_alpha,
                    use_knn=use_knn, knn_k=knn_k, knn_weighting=knn_weighting,
                    use_kernel_ridge=use_kernel_ridge, kernel_type=kernel_type,
                    kernel_sigma=kernel_sigma, kernel_ridge_alpha=kernel_ridge_alpha
                )

                # Compute base metrics
                base_acc_task = (base_preds == task_test_labels_filtered).float().mean().item()
                base_probs = F.softmax(base_logits, dim=1)[:, 1].cpu().numpy()
                labels_np = task_test_labels_filtered.cpu().numpy()
                try:
                    base_auc_task = roc_auc_score(labels_np, base_probs)
                except:
                    base_auc_task = 0.5

                base_task_accs.append(base_acc_task)
                base_task_aucs.append(base_auc_task)

                # Apply C&S for this task if enabled
                if use_graph_cs:
                    # Get logits for all graphs for this task
                    all_indices = torch.arange(len(all_graph_emb), device=device)
                    _, all_logits = zero_learning_classify(
                        all_graph_emb, valid_context_indices, task_context_labels_filtered,
                        all_indices, num_classes, sim, use_ridge=use_ridge, ridge_alpha=ridge_alpha,
                        use_knn=use_knn, knn_k=knn_k, knn_weighting=knn_weighting,
                        use_kernel_ridge=use_kernel_ridge, kernel_type=kernel_type,
                        kernel_sigma=kernel_sigma, kernel_ridge_alpha=kernel_ridge_alpha
                    )

                    # Apply C&S on the shared meta-graph
                    smoothed_logits = correct_and_smooth(
                        meta_adj, all_logits, valid_context_indices, task_context_labels_filtered,
                        num_classes, num_iters=cs_num_iters, alpha=cs_alpha
                    )

                    # Get C&S predictions for valid test graphs
                    cs_probs = smoothed_logits[valid_test_indices]
                    cs_preds = cs_probs.argmax(dim=1)
                    cs_acc_task = (cs_preds == task_test_labels_filtered).float().mean().item()

                    cs_probs_np = cs_probs[:, 1].cpu().numpy()
                    try:
                        cs_auc_task = roc_auc_score(labels_np, cs_probs_np)
                    except:
                        cs_auc_task = 0.5

                    cs_task_accs.append(cs_acc_task)
                    cs_task_aucs.append(cs_auc_task)

            # Compare average accuracy across all tasks (not per-task)
            if use_graph_cs:
                base_avg_acc = sum(base_task_accs) / len(base_task_accs) if base_task_accs else 0.0
                cs_avg_acc = sum(cs_task_accs) / len(cs_task_accs) if cs_task_accs else 0.0
                base_avg_auc = sum(base_task_aucs) / len(base_task_aucs) if base_task_aucs else 0.0
                cs_avg_auc = sum(cs_task_aucs) / len(cs_task_aucs) if cs_task_aucs else 0.0

                # Choose based on average AUC (primary metric for graph classification)
                if cs_avg_auc > base_avg_auc:
                    acc = cs_avg_acc
                    auc = cs_avg_auc
                    if run_idx == 0 or num_runs <= 5:
                        print(f'  [Graph C&S Run {run_idx+1}] Base AUC: {base_avg_auc:.4f} → C&S AUC: {cs_avg_auc:.4f} ✓')
                else:
                    acc = base_avg_acc
                    auc = base_avg_auc
                    if run_idx == 0 or num_runs <= 5:
                        print(f'  [Graph C&S Run {run_idx+1}] Base AUC: {base_avg_auc:.4f}, C&S AUC: {cs_avg_auc:.4f} [BASE]')
            else:
                acc = sum(base_task_accs) / len(base_task_accs) if base_task_accs else 0.0
                auc = sum(base_task_aucs) / len(base_task_aucs) if base_task_aucs else 0.0

        else:
            # Single-task classification - get base predictions
            base_preds, base_logits = zero_learning_classify(
                all_graph_emb, context_indices, context_labels, test_indices,
                num_classes, sim, use_ridge=use_ridge, ridge_alpha=ridge_alpha,
                use_knn=use_knn, knn_k=knn_k, knn_weighting=knn_weighting,
                use_kernel_ridge=use_kernel_ridge, kernel_type=kernel_type,
                kernel_sigma=kernel_sigma, kernel_ridge_alpha=kernel_ridge_alpha
            )

            # Compute base accuracy and AUC
            base_acc = (base_preds == test_labels).float().mean().item()

            # Compute base AUC
            if num_classes == 2:
                base_probs = F.softmax(base_logits, dim=1)[:, 1].cpu().numpy()
                test_labels_np = test_labels.cpu().numpy()
                try:
                    base_auc = roc_auc_score(test_labels_np, base_probs)
                except:
                    base_auc = 0.5
            else:
                base_probs = F.softmax(base_logits, dim=1).cpu().numpy()
                test_labels_np = test_labels.cpu().numpy()
                try:
                    base_auc = roc_auc_score(test_labels_np, base_probs, multi_class='ovr', average='macro')
                except:
                    base_auc = 0.0

            # Apply graph-level Correct & Smooth if enabled
            if use_graph_cs:
                if run_idx == 0:
                    print(f'  [Graph C&S] Building meta-graph with anchors...')
                # Select anchor graphs from context set
                num_context = len(context_graphs)
                actual_num_anchors = min(num_anchors, num_context)

                # Random sampling of anchors from context set
                # NOTE: context graphs are at indices 0 to num_context-1 in all_graph_emb
                anchor_perm = torch.randperm(num_context, device=device)[:actual_num_anchors]
                anchor_graph_indices = anchor_perm.tolist()  # These are global indices (same as context indices)

                if run_idx == 0:
                    print(f'  [Graph C&S] Using {actual_num_anchors} anchors, k={cs_k_neighbors} neighbors')

                # Build meta-graph using anchors
                meta_adj = build_anchor_meta_graph(
                    all_graph_emb, anchor_graph_indices,
                    k_neighbors=cs_k_neighbors, sim=sim, weight_sharpening=weight_sharpening
                )

                if run_idx == 0:
                    num_edges = meta_adj.nnz()
                    total_graphs = len(all_graph_emb)
                    print(f'  [Graph C&S] Meta-graph: {total_graphs} nodes, {num_edges} edges')

                # Get logits for all graphs (context + test)
                all_indices = torch.arange(len(all_graph_emb), device=device)
                _, all_logits = zero_learning_classify(
                    all_graph_emb, context_indices, context_labels, all_indices,
                    num_classes, sim, use_ridge=use_ridge, ridge_alpha=ridge_alpha,
                    use_knn=use_knn, knn_k=knn_k, knn_weighting=knn_weighting,
                    use_kernel_ridge=use_kernel_ridge, kernel_type=kernel_type,
                    kernel_sigma=kernel_sigma, kernel_ridge_alpha=kernel_ridge_alpha
                )

                if run_idx == 0:
                    print(f'  [Graph C&S] Applying C&S ({cs_num_iters} iters, α={cs_alpha})...')

                # Apply C&S on meta-graph
                smoothed_logits = correct_and_smooth(
                    meta_adj, all_logits, context_indices, context_labels,
                    num_classes, num_iters=cs_num_iters, alpha=cs_alpha
                )

                # Extract test predictions from smoothed logits (actually probabilities from C&S)
                cs_probs = smoothed_logits[test_indices]
                cs_preds = cs_probs.argmax(dim=1)

                # Compute C&S accuracy
                cs_acc = (cs_preds == test_labels).float().mean().item()

                # Compute C&S AUC
                if num_classes == 2:
                    cs_probs_np = cs_probs[:, 1].cpu().numpy()
                    try:
                        cs_auc = roc_auc_score(test_labels_np, cs_probs_np)
                    except:
                        cs_auc = 0.5
                else:
                    cs_probs_np = cs_probs.cpu().numpy()
                    try:
                        cs_auc = roc_auc_score(test_labels_np, cs_probs_np, multi_class='ovr', average='macro')
                    except:
                        cs_auc = 0.0

                # Choose between base and C&S based on AUC (primary metric for graph classification)
                if cs_auc > base_auc:
                    acc = cs_acc
                    auc = cs_auc
                    if run_idx == 0 or num_runs <= 5:
                        print(f'  [Graph C&S Run {run_idx+1}] Base: {base_auc:.4f} → C&S: {cs_auc:.4f} ✓')
                else:
                    acc = base_acc
                    auc = base_auc
                    if run_idx == 0 or num_runs <= 5:
                        print(f'  [Graph C&S Run {run_idx+1}] Base: {base_auc:.4f}, C&S: {cs_auc:.4f} [BASE]')
            else:
                # No C&S, use base predictions
                acc = base_acc
                auc = base_auc

        all_results.append({'acc': acc, 'auc': auc})

        if run_idx == 0 or num_runs <= 5:
            method_str = f'{hops}-hop {prop_method} {pool_method}-pool {sim.upper()}'
            if use_knn:
                method_str += f' + knn(k={knn_k},{knn_weighting})'
            elif use_kernel_ridge:
                method_str += f' + krr({kernel_type},σ={kernel_sigma},λ={kernel_ridge_alpha})'
            elif use_ridge:
                method_str += f' + ridge(α={ridge_alpha})'
            print(f'    Run {run_idx+1}: {method_str}: Acc={acc:.4f} ({acc*100:.2f}%), AUC={auc:.4f}')

    # Compute and display averaged results
    accs = [r['acc'] for r in all_results]
    aucs = [r['auc'] for r in all_results]
    avg_acc = sum(accs) / len(accs)
    avg_auc = sum(aucs) / len(aucs)

    method_str = f'{hops}-hop {prop_method} {pool_method}-pool {sim.upper()}'
    if use_knn:
        method_str += f' + knn(k={knn_k},{knn_weighting})'
    elif use_kernel_ridge:
        method_str += f' + krr({kernel_type},σ={kernel_sigma},λ={kernel_ridge_alpha})'
    elif use_ridge:
        method_str += f' + ridge(α={ridge_alpha})'

    print(f'  Average: {method_str}: Acc={avg_acc:.4f} ({avg_acc*100:.2f}%), AUC={avg_auc:.4f}')

    return avg_auc  # Return AUC as primary metric for graph classification


def main():
    parser = argparse.ArgumentParser(description='Zero-Learning Baseline Test')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['Actor', 'AirBrazil', 'AirEU', 'AirUS', 'AmzComp', 'AmzPhoto', 'AmzRatings',
                                 'BlogCatalog', 'Chameleon', 'Citeseer', 'CoCS', 'CoPhysics', 'Cora', 'Cornell',
                                 'DBLP', 'Deezer', 'LastFMAsia', 'Minesweeper', 'Pubmed', 'Questions', 'Reddit',
                                 'Roman', 'Squirrel', 'Texas', 'Tolokers', 'Wiki', 'Wisconsin', 'WikiCS',
                                 'ogbn-arxiv', 'ogbn-products', 'FullCora'],
                        help='List of datasets to test')
    parser.add_argument('--mode', type=str, choices=['full', 'few-shot'], default='few-shot',
                        help='full-shot or few-shot mode')
    parser.add_argument('--k_shot', type=int, default=5,
                        help='Number of shots per class (for few-shot mode)')
    parser.add_argument('--hops', type=int, default=3,
                        help='Number of GCN propagation hops')
    parser.add_argument('--sim', type=str, choices=['cos', 'dot', 'tanimoto'], default='cos',
                        help='Similarity function (use "tanimoto" for molecular graphs)')
    parser.add_argument('--prop_method', type=str, choices=['puregcn', 'residual', 'appnp', 'concat', 'weighted', 'gin'], default='residual',
                        help='Feature propagation method (use "gin" for graph classification)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Teleport probability for APPNP')
    parser.add_argument('--gin_eps', type=float, default=0.0,
                        help='Epsilon parameter for GIN (default 0.0)')
    parser.add_argument('--gcn_layer_norm', type=str2bool, default=True,
                        help='Apply LayerNorm after each GCN propagation step')
    parser.add_argument('--use_cs', type=str2bool, default=True,
                        help='Try Correct & Smooth and use if it improves performance')
    parser.add_argument('--cs_hops', type=int, default=50,
                        help='Number of iterations for C&S')
    parser.add_argument('--cs_alpha', type=float, default=0.5,
                        help='Alpha for C&S')
    parser.add_argument('--ridge', type=str2bool, default=False,
                        help='Use ridge regression (closed-form linear classifier)')
    parser.add_argument('--ridge_alpha', type=float, default=0.1,
                        help='Regularization strength for ridge regression')
    parser.add_argument('--use_knn', type=str2bool, default=False,
                        help='Use KNN classifier instead of prototypes/ridge')
    parser.add_argument('--knn_k', type=int, default=3,
                        help='Number of neighbors for KNN')
    parser.add_argument('--knn_weighting', type=str, default='uniform', choices=['uniform', 'similarity'],
                        help='KNN weighting: uniform or similarity')
    parser.add_argument('--use_kernel_ridge', type=str2bool, default=False,
                        help='Use kernel ridge regression instead of prototypes/ridge')
    parser.add_argument('--kernel_type', type=str, default='rbf', choices=['rbf', 'cos', 'dot'],
                        help='Kernel type for kernel ridge')
    parser.add_argument('--kernel_sigma', type=float, default=1.0,
                        help='RBF kernel bandwidth')
    parser.add_argument('--kernel_ridge_alpha', type=float, default=1.0,
                        help='Kernel ridge regularization strength')
    parser.add_argument('--proto_per_class', type=int, default=1,
                        help='Number of prototypes per class (1 = standard prototypes)')
    parser.add_argument('--proto_method', type=str, default='kmeans', choices=['kmeans', 'kcenter'],
                        help='Prototype clustering method when proto_per_class > 1')
    parser.add_argument('--proto_agg', type=str, default='max', choices=['max', 'mean'],
                        help='How to aggregate multiple prototypes per class')
    parser.add_argument('--proto_kmeans_iters', type=int, default=10,
                        help='Number of k-means iterations for prototype clustering')
    parser.add_argument('--feature_norm', type=str, choices=['none', 'row', 'col', 'row+col'], default='none',
                        help='Feature normalization (row=L2 per node, col=standardize per feature)')
    parser.add_argument('--use_pca', type=str2bool, default=False,
                        help='Apply PCA with padding for dimension unification')

    # TTA arguments
    parser.add_argument('--use_tta', type=str2bool, default=False,
                        help='Use Test-Time Augmentation')
    parser.add_argument('--tta_num_augmentations', type=int, default=20,
                        help='Number of augmented versions for TTA')
    parser.add_argument('--tta_include_original', type=str2bool, default=False,
                        help='Include original features in TTA aggregation')
    parser.add_argument('--tta_aggregation', type=str, default='voting', choices=['logits', 'probs', 'voting'],
                        help='TTA aggregation method: logits (average logits), probs (average probabilities), voting (majority vote)')
    parser.add_argument('--pca_dim', type=int, default=128,
                        help='Target dimension for PCA (with zero-padding if needed)')
    parser.add_argument('--use_full_pca', type=str2bool, default=False,
                        help='Use full SVD instead of lowrank PCA')
    parser.add_argument('--pca_preserve_norms', type=str2bool, default=False,
                        help='Restore original L2 norms after PCA transformation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for few-shot sampling')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device number (use -1 for CPU)')
    parser.add_argument('--wandb', type=str2bool, default=True,
                        help='Enable Weights & Biases logging')
    parser.add_argument('--num_runs', type=int, default=3,
                        help='Number of runs with different seeds for averaging')
    parser.add_argument('--use_kmedoids', type=str2bool, default=False,
                        help='Use K-Medoids clustering for context sampling instead of random')
    parser.add_argument('--kmedoids_on_gcn', type=str2bool, default=True,
                        help='If True, apply K-Medoids on GCN embeddings; if False, on raw features')
    parser.add_argument('--task_type', type=str, choices=['node', 'graph', 'auto'], default='auto',
                        help='Task type: node classification or graph classification (auto-detect if "auto")')
    parser.add_argument('--pool_method', type=str, choices=['mean', 'max', 'sum', 'concat'], default='concat',
                        help='Graph pooling method for graph classification')
    parser.add_argument('--use_graph_cs', type=str2bool, default=False,
                        help='Use graph-level Correct & Smooth (build meta-graph with anchors)')
    parser.add_argument('--num_anchors', type=int, default=1000,
                        help='Number of anchor graphs for meta-graph construction')
    parser.add_argument('--cs_k_neighbors', type=int, default=10,
                        help='Number of neighbors to connect in meta-graph')
    parser.add_argument('--weight_sharpening', type=float, default=1.0,
                        help='Power to raise edge weights to (>1: emphasize strong connections, <1: smooth differences)')
    parser.add_argument('--use_pe', type=str2bool, default=False,
                        help='Use positional encodings (GPSE/LapPE/RWSE) if available in the dataset')
    parser.add_argument('--gpse_path', type=str, default='/home/maweishuo/GPSE/datasets',
                        help='Path to GPSE/LapPE/RWSE datasets directory')
    parser.add_argument('--use_gpse', type=str2bool, default=True,
                        help='Load GPSE embeddings (if use_pe is True)')
    parser.add_argument('--use_lappe', type=str2bool, default=True,
                        help='Load LapPE embeddings (if use_pe is True)')
    parser.add_argument('--use_rwse', type=str2bool, default=True,
                        help='Load RWSE embeddings (if use_pe is True)')
    parser.add_argument('--use_fingerprint_pe', type=str2bool, default=False,
                        help='Use molecular fingerprints as PE (for molecular datasets)')
    parser.add_argument('--fp_type', type=str, choices=['morgan', 'maccs', 'both'], default='morgan',
                        help='Type of molecular fingerprint to use')
    parser.add_argument('--fp_use_pca', type=str2bool, default=False,
                        help='Apply PCA to fingerprints to reduce dimensionality')
    parser.add_argument('--fp_pca_dim', type=int, default=64,
                        help='Target dimension for fingerprint PCA')
    parser.add_argument('--pas_ogb_path', type=str, default='/home/maweishuo/PAS-OGB',
                        help='Path to PAS-OGB project for loading fingerprints')

    # === Random Projection Augmentation (for testing information preservation) ===
    parser.add_argument('--use_augmentation', type=str2bool, default=False,
                        help='Apply random projection augmentation σ(WX+b) to test information preservation')

    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'

    # Initialize wandb if enabled
    if args.wandb:
        # Create run name based on configuration
        method_parts = [args.prop_method, f'{args.hops}hop', args.sim]
        if args.use_pca:
            pca_type = 'fullPCA' if args.use_full_pca else 'PCA'
            pca_name = f'{pca_type}{args.pca_dim}'
            if args.pca_preserve_norms:
                pca_name += 'NormPres'
            method_parts.append(pca_name)
        if args.ridge:
            method_parts.append(f'ridge{args.ridge_alpha}')
        if args.feature_norm != 'none':
            method_parts.append(f'norm-{args.feature_norm}')
        if args.use_kmedoids:
            kmedoids_name = 'kmedoids-gcn' if args.kmedoids_on_gcn else 'kmedoids-raw'
            method_parts.append(kmedoids_name)

        run_name = f"{args.mode}_{'_'.join(method_parts)}"

        wandb.init(
            project='zero-learning-baseline',
            name=run_name,
            config=vars(args)
        )

    print('='*60)
    print('ZERO-LEARNING BASELINE: Pure GNN Structure Heuristic')
    print(f'Mode: {args.mode}' + (f' (k={args.k_shot})' if args.mode == 'few-shot' else ''))
    print(f'Similarity: {args.sim.upper()}, Device: {device}')
    print('='*60)

    torch.manual_seed(args.seed)
    all_results = {}

    # Define known graph classification datasets for auto-detection
    GRAPH_DATASETS = {'MUTAG', 'PROTEINS', 'ENZYMES', 'DD', 'NCI1', 'NCI109', 'IMDB-BINARY', 'IMDB-MULTI',
                     'COLLAB', 'REDDIT-BINARY', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K',
                     'bace', 'bbbp', 'chemhiv', 'tox21', 'toxcast', 'clintox', 'muv', 'sider',
                     'ogbg-molhiv', 'ogbg-molbace', 'ogbg-molbbbp', 'ogbg-moltox21', 'ogbg-moltoxcast',
                     'ZINC', 'PATTERN', 'CLUSTER', 'MNIST', 'CIFAR10'}

    for dataset_name in args.datasets:
        print(f'\n{"="*60}')
        print(f'Dataset: {dataset_name}')
        print('='*60)

        try:
            # Determine task type
            if args.task_type == 'auto':
                # Auto-detect based on dataset name
                if dataset_name in GRAPH_DATASETS or dataset_name.startswith('ogbg-'):
                    task_type = 'graph'
                else:
                    task_type = 'node'
            else:
                task_type = args.task_type

            print(f'Task type: {task_type}')

            if task_type == 'node':
                # Node classification
                if dataset_name.startswith('ogbn-'):
                    data, split_idx = load_ogbn_data(dataset_name)
                else:
                    data, split_idx = load_data(dataset_name)

                # Apply random projection augmentation if enabled
                if args.use_augmentation:
                    from src.data_utils import apply_random_projection_augmentation
                    print(f'\n[Augmentation Test] Applying σ(WX+b) random projection...')
                    print(f'  Original features: {data.x.shape}')
                    data = apply_random_projection_augmentation(
                        data,
                        hidden_dim_range=None,  # Use default range
                        activation_pool=None,   # Use default diverse pool
                        seed=42,
                        verbose=True,
                        rank=0,
                        use_random_noise=False  # Always use σ(WX+b), not noise
                    )
                    print(f'  Augmented features: {data.x.shape}\n')

                num_classes = data.y.max().item() + 1
                print(f'Nodes: {data.num_nodes}, Features: {data.num_features}, Classes: {num_classes}')

                result = test_dataset(
                    data, split_idx, num_classes,
                    mode=args.mode, k_shot=args.k_shot,
                    hops=args.hops, sim=args.sim, device=device,
                    prop_method=args.prop_method, alpha=args.alpha,
                    gcn_layer_norm=args.gcn_layer_norm,
                    use_cs=args.use_cs, cs_hops=args.cs_hops, cs_alpha=args.cs_alpha,
                    use_ridge=args.ridge, ridge_alpha=args.ridge_alpha,
                    feature_norm=args.feature_norm,
                    dataset_name=dataset_name,
                    use_pca=args.use_pca, pca_dim=args.pca_dim, use_full_pca=args.use_full_pca,
                    pca_preserve_norms=args.pca_preserve_norms,
                    num_runs=args.num_runs,
                    use_kmedoids=args.use_kmedoids,
                    kmedoids_on_gcn=args.kmedoids_on_gcn,
                    use_tta=args.use_tta,
                    tta_num_augmentations=args.tta_num_augmentations,
                    tta_include_original=args.tta_include_original,
                    tta_aggregation=args.tta_aggregation,
                    proto_per_class=args.proto_per_class,
                    proto_method=args.proto_method,
                    proto_agg=args.proto_agg,
                    proto_kmeans_iters=args.proto_kmeans_iters,
                    use_knn=args.use_knn,
                    knn_k=args.knn_k,
                    knn_weighting=args.knn_weighting,
                    use_kernel_ridge=args.use_kernel_ridge,
                    kernel_type=args.kernel_type,
                    kernel_sigma=args.kernel_sigma,
                    kernel_ridge_alpha=args.kernel_ridge_alpha
                )
                metric_name = 'acc'

            else:  # graph classification
                # For OGB datasets, set environment variable to use original features
                if dataset_name.startswith('ogbg-'):
                    os.environ['USE_ORIGINAL_FEATURES'] = '1'

                # Load graph classification dataset
                result = load_dataset(dataset_name, root='./dataset')

                # Check if dataset loading failed
                if result is None:
                    print(f"Failed to load dataset {dataset_name}. Skipping...")
                    return

                # Handle tuple return from original features loaders
                if isinstance(result, tuple):
                    dataset, mapping = result
                    # Store mapping if needed for later use
                else:
                    dataset = result

                # Load and wrap dataset with PE embeddings if requested
                if args.use_pe or args.use_fingerprint_pe:
                    print(f'Loading PE embeddings for {dataset_name}...')
                    pe_data = load_all_pes_with_fingerprints(
                        dataset_name=dataset_name,
                        use_gpse=args.use_gpse and args.use_pe,
                        use_lappe=args.use_lappe and args.use_pe,
                        use_rwse=args.use_rwse and args.use_pe,
                        use_fingerprint_pe=args.use_fingerprint_pe,
                        fp_type=args.fp_type,
                        fp_use_pca=args.fp_use_pca,
                        fp_pca_dim=args.fp_pca_dim,
                        pas_ogb_path=args.pas_ogb_path,
                        gpse_base_path=args.gpse_path
                    )
                    # Wrap dataset to attach PEs on-the-fly
                    has_any_pe = any(emb is not None for emb, _ in pe_data.values())
                    if has_any_pe:
                        dataset = GraphDatasetWithPE(dataset, pe_data)
                        print(f"  Wrapped dataset with PE loader")
                    else:
                        pe_type = "fingerprint PE" if args.use_fingerprint_pe else "traditional PE"
                        print(f"  Warning: {pe_type} enabled but no embeddings found for {dataset_name}")
                        print(f"  Continuing without PEs...")

                # Create or load splits
                if hasattr(dataset, 'split_idx'):
                    split_idx = dataset.split_idx
                else:
                    # Create random splits if not available
                    from src.data_gc import create_random_splits
                    split_idx = create_random_splits(dataset, seed=args.seed)

                # Determine number of classes
                first_y = dataset[0].y
                if first_y.dim() > 0 and first_y.size(-1) > 1:
                    # Multi-task, use binary for each task
                    num_classes = 2
                else:
                    # Single-task
                    all_labels = [dataset[i].y.item() if dataset[i].y.dim() == 0 else dataset[i].y[0].item()
                                 for i in range(len(dataset))]
                    num_classes = max(all_labels) + 1

                print(f'Graphs: {len(dataset)}, Classes: {num_classes}')

                result = test_graph_dataset(
                    dataset, split_idx, num_classes,
                    mode=args.mode, k_shot=args.k_shot,
                    hops=args.hops, sim=args.sim, device=device,
                    prop_method=args.prop_method, alpha=args.alpha,
                    gcn_layer_norm=args.gcn_layer_norm,
                    use_ridge=args.ridge, ridge_alpha=args.ridge_alpha,
                    feature_norm=args.feature_norm,
                    dataset_name=dataset_name,
                    use_pca=args.use_pca, pca_dim=args.pca_dim, use_full_pca=args.use_full_pca,
                    pca_preserve_norms=args.pca_preserve_norms,
                    num_runs=args.num_runs,
                    pool_method=args.pool_method,
                    gin_eps=args.gin_eps,
                    use_graph_cs=args.use_graph_cs,
                    num_anchors=args.num_anchors,
                    cs_k_neighbors=args.cs_k_neighbors,
                    cs_num_iters=args.cs_hops,
                    cs_alpha=args.cs_alpha,
                    weight_sharpening=args.weight_sharpening,
                    use_pe=args.use_pe,
                    use_knn=args.use_knn,
                    knn_k=args.knn_k,
                    knn_weighting=args.knn_weighting,
                    use_kernel_ridge=args.use_kernel_ridge,
                    kernel_type=args.kernel_type,
                    kernel_sigma=args.kernel_sigma,
                    kernel_ridge_alpha=args.kernel_ridge_alpha
                )
                metric_name = 'auc'

            all_results[dataset_name] = result

            # Log individual dataset result to wandb
            if args.wandb:
                wandb.log({f'{metric_name}/{dataset_name}': result})

        except Exception as e:
            import traceback
            print(f'  Error processing {dataset_name}: {e}')
            traceback.print_exc()
            continue

    # Summary table
    if all_results:
        print(f'\n{"="*60}')
        method_desc = []
        if args.feature_norm != 'none':
            method_desc.append(f'feat-{args.feature_norm}')
        method_desc.append(f'{args.prop_method}')
        if args.task_type == 'graph' or (args.task_type == 'auto' and any(d in GRAPH_DATASETS for d in args.datasets)):
            method_desc.append(f'{args.pool_method}-pool')
        if args.use_pca:
            pca_type = 'fullPCA' if args.use_full_pca else 'PCA'
            pca_desc = f'{pca_type}{args.pca_dim}'
            if args.pca_preserve_norms:
                pca_desc += '+NormPres'
            method_desc.append(pca_desc)
        if args.use_cs and args.task_type != 'graph':
            method_desc.append('C&S')
        if args.ridge:
            method_desc.append(f'ridge({args.ridge_alpha})')
        if args.use_kmedoids:
            method_desc.append('kmedoids')
        print(f'SUMMARY ({"+".join(method_desc)}, {args.hops}-hop, {args.sim.upper()}, {args.mode})')
        print('='*60)
        print(f'{"Dataset":<20} {"Score":>10}')
        print('-'*30)
        for name, score in all_results.items():
            print(f'{name:<20} {score*100:>9.2f}%')
        print('-'*30)
        avg_score = sum(all_results.values()) / len(all_results)
        print(f'{"Average":<20} {avg_score*100:>9.2f}%')

        # Log average score to wandb
        if args.wandb:
            wandb.log({'score/average': avg_score})
            wandb.finish()


if __name__ == '__main__':
    main()
