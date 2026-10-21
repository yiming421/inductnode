"""Small, data-free checks of the three GILT prediction paths."""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_sparse import SparseTensor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import PureGCN_v1, PFNPredictorNodeCls
from src.utils import process_node_features

def components():
    torch.manual_seed(42)
    n, d = (18, 16)
    row = torch.arange(n)
    adj = SparseTensor(row=row, col=(row + 1) % n, sparse_sizes=(n, n)).to_symmetric()
    data = Data(x=torch.randn(n, d), y=torch.arange(n) % 3, context_sample=torch.arange(6), adj_t=adj)
    model = PureGCN_v1(d, num_layers=6, hidden=d, norm=True)
    predictor = PFNPredictorNodeCls(d, nhead=4, num_layers=2, norm=True, normalize=True, norm_type='pre', head_num_layers=0, nc_head_num_layers=0, lp_head_num_layers=2, gc_head_num_layers=1, mplp_signature_dim=64, mplp_use_degree='mlp')
    return (model, predictor, data)

def test_queries_cannot_influence_other_queries():
    model, predictor, data = components()
    model.eval()
    predictor.eval()
    h = model(data.x, data.adj_t)
    support = h[:6]
    prototypes = process_node_features(support, data, normalize=True)
    query = h[6:]
    for task in ('node_classification', 'graph_classification'):
        logits, _ = predictor(data, support, query, data.y[:6], prototypes, task_type=task)
        changed = query.clone()
        changed[1:] += 100
        changed_logits, _ = predictor(data, support, changed, data.y[:6], prototypes, task_type=task)
        torch.testing.assert_close(logits[0], changed_logits[0], atol=1e-06, rtol=1e-06)
        single, _ = predictor(data, support, query[:1], data.y[:6], prototypes, task_type=task)
        torch.testing.assert_close(logits[:1], single, atol=1e-06, rtol=1e-06)

def test_all_task_paths_backpropagate():
    for task in ('node_classification', 'link_prediction', 'graph_classification'):
        model, predictor, data = components()
        h = model(data.x, data.adj_t)
        support = h[:6]
        prototypes = process_node_features(support, data, normalize=True)
        kwargs = {}
        if task == 'link_prediction':
            edges = torch.stack((torch.arange(6, 12), torch.arange(7, 13)))
            target = h[edges[0]] * h[edges[1]]
            kwargs = dict(adj_t=data.adj_t, lp_edges=edges, node_emb=h)
        else:
            target = h[6:12]
        logits, _ = predictor(data, support, target, data.y[:6], prototypes, task_type=task, **kwargs)
        assert torch.isfinite(logits).all()
        loss = F.binary_cross_entropy_with_logits(logits, torch.arange(6).float() % 2) if task == 'link_prediction' else F.cross_entropy(logits, data.y[6:12])
        loss.backward()
        gradients = [p.grad for p in predictor.transformer_row.parameters() if p.grad is not None]
        assert gradients and all((torch.isfinite(g).all() for g in gradients))
        assert sum((g.abs().sum() for g in gradients)) > 0
