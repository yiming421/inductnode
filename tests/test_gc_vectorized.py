import os
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from torch_geometric.data import Data
    from src.data_gc import prepare_graph_data_for_pfn, create_data_loaders
    from src.engine_gc import (
        train_graph_classification_multitask_vectorized,
        evaluate_graph_classification_multitask_vectorized,
    )
    _GEOM_AVAILABLE = True
except Exception as exc:  # pragma: no cover - optional dependency
    _GEOM_AVAILABLE = False
    _IMPORT_ERROR = exc


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, num_classes, num_node_features, name="pcba"):
        self.graphs = graphs
        self.num_classes = num_classes
        self.num_node_features = num_node_features
        self.name = name
        self.node_embs = None

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]


class DummyModel(torch.nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.lin = torch.nn.Linear(in_dim, in_dim, bias=False)

    def forward(self, x, adj_t, batch):
        return self.lin(x)


@unittest.skipUnless(_GEOM_AVAILABLE, "torch_geometric is required for GC vectorized tests")
class TestGCVectorized(unittest.TestCase):
    def _build_dataset(self):
        torch.manual_seed(0)
        feat_dim = 8
        num_tasks = 3
        graphs = []
        node_embs = []
        offset = 0

        labels = [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, float('nan'), 1.0],  # include NaN to exercise task_mask
        ]

        for i in range(4):
            num_nodes = 3
            x_idx = torch.arange(offset, offset + num_nodes, dtype=torch.long)
            edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
            y = torch.tensor(labels[i], dtype=torch.float)
            graphs.append(Data(x=x_idx, edge_index=edge_index, y=y))
            node_embs.append(torch.randn(num_nodes, feat_dim))
            offset += num_nodes

        dataset = SimpleDataset(graphs, num_classes=2, num_node_features=feat_dim, name="pcba")
        dataset.node_embs = torch.cat(node_embs, dim=0)
        return dataset, feat_dim, num_tasks

    def test_vectorized_train_eval(self):
        dataset, feat_dim, num_tasks = self._build_dataset()

        split_idx = {
            'train': torch.tensor([0, 1, 2], dtype=torch.long),
            'valid': torch.tensor([3], dtype=torch.long),
            'test': torch.tensor([3], dtype=torch.long),
        }

        dataset_info = prepare_graph_data_for_pfn(dataset, split_idx, context_k=1, device='cpu')
        dataset_info['needs_identity_projection'] = False

        data_loaders = create_data_loaders(
            dataset, split_idx, batch_size=2, shuffle=False, task_idx=None, use_index_tracking=False
        )

        model = DummyModel(feat_dim)
        predictor = torch.nn.Identity()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        args = SimpleNamespace(
            edge_dropout_enabled=False,
            feature_dropout_enabled=False,
            gc_sim='dot',
        )

        loss = train_graph_classification_multitask_vectorized(
            model, predictor, dataset_info, data_loaders, optimizer,
            pooling_method='mean', device='cpu', clip_grad=1.0,
            identity_projection=None, args=args, lambda_=1.0
        )

        self.assertTrue(torch.isfinite(torch.tensor(loss)))

        results = evaluate_graph_classification_multitask_vectorized(
            model, predictor, dataset_info, data_loaders,
            pooling_method='mean', device='cpu',
            dataset_name='pcba', identity_projection=None, args=args
        )

        self.assertIn('test', results)


if __name__ == "__main__":
    unittest.main()
