import os
import sys

import torch
from torch_sparse import SparseTensor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model import UnifiedGNN


def _make_small_adj(num_nodes=4):
    # Simple undirected chain 0-1-2-3
    edges = [(0, 1), (1, 2), (2, 3)]
    edge_index = torch.tensor(edges + [(v, u) for u, v in edges], dtype=torch.long).t()
    adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes)).coalesce()
    return adj_t


def test_unifiedgnn_gat_supports_sparse_tensor():
    torch.manual_seed(0)
    num_nodes = 4
    in_feats = 4
    x = torch.randn(num_nodes, in_feats)
    adj_t = _make_small_adj(num_nodes)

    model = UnifiedGNN(
        model_type='gcn',
        in_feats=in_feats,
        h_feats=in_feats,
        prop_step=1,
        conv='GAT',
        multilayer=False,
        norm=False,
        relu=False,
        dropout=0.0,
    )

    out = model(x, adj_t)
    assert out.shape == (num_nodes, in_feats)
    assert torch.isfinite(out).all()


def test_unifiedgnn_gin_aggr_is_wired():
    model_sum = UnifiedGNN(
        model_type='gcn',
        in_feats=4,
        h_feats=4,
        prop_step=1,
        conv='GIN',
        gin_aggr='sum',
        multilayer=False,
        norm=False,
        relu=False,
        dropout=0.0,
    )
    assert model_sum.conv1.aggr == 'add'

    model_mean = UnifiedGNN(
        model_type='gcn',
        in_feats=4,
        h_feats=4,
        prop_step=1,
        conv='GIN',
        gin_aggr='mean',
        multilayer=False,
        norm=False,
        relu=False,
        dropout=0.0,
    )
    assert model_mean.conv1.aggr == 'mean'


def test_unifiedgnn_input_norm_is_norm_only():
    model = UnifiedGNN(
        model_type='gcn',
        in_feats=4,
        h_feats=4,
        prop_step=1,
        conv='GCN',
        multilayer=False,
        norm=True,
        relu=True,
        dropout=0.5,
        input_norm=True,
    )

    x = torch.tensor([[0.0, 1.0, 2.0, 3.0],
                      [1.0, 2.0, 3.0, 4.0]])

    norm_only = model._apply_input_norm(x.clone(), 0)
    expected = model.norms[0](x)
    assert torch.allclose(norm_only, expected)

    activated = model._apply_norm_and_activation(x.clone(), 0)
    assert activated.min().item() >= 0.0
    assert norm_only.min().item() < 0.0
