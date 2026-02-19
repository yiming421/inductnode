import os
import sys

import torch
from torch_sparse import SparseTensor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import FAGCN


def _toy_graph():
    # Small directed graph without duplicate edges for deterministic checks.
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 1, 4, 5, 2, 6, 0],
            [1, 2, 3, 4, 0, 5, 6, 6, 0, 3],
        ],
        dtype=torch.long,
    )
    num_nodes = 7
    # Canonical PyG adj_t convention is transposed sparse adjacency.
    adj = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        sparse_sizes=(num_nodes, num_nodes),
    ).coalesce()
    adj_t = adj.t().coalesce()
    x = torch.randn(num_nodes, 11, dtype=torch.float32)
    return x, edge_index, adj_t


def test_fagcn_matches_manual_faconv_stack_on_sparse_tensor():
    torch.manual_seed(1234)
    x, _, adj_t = _toy_graph()

    model = FAGCN(
        in_feats=11,
        h_feats=16,
        prop_step=3,
        dropout=0.0,
        norm=False,
        relu=False,
        res=False,
        eps=0.1,
        attn_dropout=0.0,
    )
    model.eval()

    with torch.no_grad():
        out_model = model(x, adj_t)

        # Manual forward with the same submodules validates wrapper logic.
        h = model.lin(x)
        h0 = h
        for conv in model.convs:
            h = conv(h, h0, adj_t)
        out_manual = h

    torch.testing.assert_close(out_model, out_manual, rtol=1e-6, atol=1e-7)


def test_fagcn_sparse_tensor_and_edge_index_paths_are_equivalent():
    torch.manual_seed(1234)
    x, edge_index, adj_t = _toy_graph()

    model = FAGCN(
        in_feats=11,
        h_feats=16,
        prop_step=2,
        dropout=0.0,
        norm=False,
        relu=False,
        res=False,
        eps=0.2,
        attn_dropout=0.0,
    )
    model.eval()

    with torch.no_grad():
        out_from_adj = model(x, adj_t)
        out_from_edge = model(x, edge_index)

    torch.testing.assert_close(out_from_adj, out_from_edge, rtol=1e-6, atol=1e-7)


def test_fagcn_backward_populates_gradients():
    torch.manual_seed(1234)
    x, _, adj_t = _toy_graph()

    model = FAGCN(
        in_feats=11,
        h_feats=16,
        prop_step=3,
        dropout=0.0,
        norm=True,
        relu=True,
        res=True,
        eps=0.1,
        attn_dropout=0.0,
    )
    model.train()

    out = model(x, adj_t)
    loss = out.pow(2).mean()
    loss.backward()

    assert model.lin.weight.grad is not None
    conv_has_grad = any(
        p.grad is not None for conv in model.convs for p in conv.parameters()
    )
    assert conv_has_grad
