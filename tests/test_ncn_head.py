import os
import sys

import torch
from torch_sparse import SparseTensor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import PFNPredictorNodeCls


def _set_mlp_identity(mlp):
    for layer in mlp.lins:
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.zeros_(layer.bias)
            layer.weight.data.zero_()
            dim = min(layer.in_features, layer.out_features)
            layer.weight.data[:dim, :dim] = torch.eye(dim)


def test_ncn_head_common_neighbor_pooling_order():
    torch.manual_seed(0)

    hidden_dim = 4
    num_nodes = 4

    # Undirected edges for a small graph
    undirected_edges = [
        (0, 1), (0, 2), (0, 3), (1, 2), (2, 3)
    ]
    edge_index = torch.tensor(
        [(u, v) for u, v in undirected_edges] +
        [(v, u) for u, v in undirected_edges],
        dtype=torch.long
    ).t()

    adj_t = SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(num_nodes, num_nodes)
    ).coalesce()

    # Node embeddings: all ones => pooled CN features proportional to CN count
    node_emb = torch.ones((num_nodes, hidden_dim))

    # Target edges with different CN counts
    target_edges = torch.tensor([[1, 3], [0, 1]], dtype=torch.long)  # [E, 2]
    target_edge_embeds = torch.zeros((target_edges.size(0), hidden_dim))

    # Minimal context (not used by NCN head, but required by interface)
    context_x = torch.zeros((1, hidden_dim))
    context_y = torch.tensor([0], dtype=torch.long)
    class_x = torch.zeros((2, hidden_dim))

    predictor = PFNPredictorNodeCls(
        hidden_dim=hidden_dim,
        nhead=1,
        num_layers=0,
        mlp_layers=2,
        dropout=0.0,
        norm=False,
        separate_att=False,
        degree=False,
        att=None,
        mlp=None,
        sim='dot',
        padding='zero',
        norm_affine=False,
        normalize=False,
        use_first_half_embedding=False,
        use_full_embedding=False,
        norm_type='post',
        ffn_expansion_ratio=4,
        use_matching_network=False,
        matching_network_projection='linear',
        matching_network_temperature=1.0,
        matching_network_learnable_temp=True,
        nc_sim='dot',
        nc_ridge_alpha=1.0,
        lp_sim='dot',
        lp_ridge_alpha=1.0,
        gc_sim='dot',
        gc_ridge_alpha=1.0,
        head_num_layers=1,
        skip_token_formulation=True,
        lp_use_linear_predictor=False,
        lp_head_type='ncn',
        mplp_signature_dim=64,
        mplp_num_hops=2,
        mplp_feature_combine='hadamard',
        mplp_prop_type='combine',
        mplp_signature_sampling='torchhd',
        mplp_use_subgraph=True,
        mplp_use_degree='none',
        ncn_beta=1.0,
        ncn_cndeg=-1,
        nc_head_num_layers=None,
        lp_head_num_layers=None,
        lp_concat_common_neighbors=False,
    )
    predictor.eval()

    # Make NCN head deterministic: identity xcn/xij, sum features for final logit
    head = predictor.lp_head
    _set_mlp_identity(head.xcnlin)
    _set_mlp_identity(head.xijlin)
    head.beta.data.fill_(1.0)
    head.lin.weight.data.fill_(1.0)
    head.lin.bias.data.zero_()

    logits, _ = predictor(
        data=None,
        context_x=context_x,
        target_x=target_edge_embeds,
        context_y=context_y,
        class_x=class_x,
        task_type='link_prediction',
        adj_t=adj_t,
        lp_edges=target_edges.t(),
        node_emb=node_emb
    )

    # Compute expected CN counts via naive set intersection
    neighbors = {i: set() for i in range(num_nodes)}
    for u, v in edge_index.t().tolist():
        neighbors[u].add(v)
    counts = []
    for u, v in target_edges.tolist():
        counts.append(len(neighbors[u] & neighbors[v]))

    expected = torch.tensor([c * hidden_dim for c in counts], dtype=logits.dtype)
    assert logits.shape == expected.shape
    assert torch.allclose(logits.cpu(), expected, atol=1e-6)
    assert logits[0].item() > logits[1].item()
