import math
import os
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import engine_gc


class _DummyAdj:
    def to_symmetric(self):
        return self

    def coalesce(self):
        return self


class _DummySparseTensor:
    @staticmethod
    def from_edge_index(edge_index, sparse_sizes=None):
        return _DummyAdj()


class _DummyBatch:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to(self, device):
        for key, value in list(self.__dict__.items()):
            if torch.is_tensor(value):
                setattr(self, key, value.to(device))
        return self


class _TinyModel(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim, bias=False)

    def forward(self, x, adj_t, batch):
        return self.linear(x)


class _TinyPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.7))
        self.bias = nn.Parameter(torch.tensor(-0.1))


def _fake_pool_graph_embeddings(node_embeddings, batch, pooling_method="mean", virtualnode_embeddings=None):
    if virtualnode_embeddings is not None:
        return virtualnode_embeddings
    num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
    pooled = []
    for graph_idx in range(num_graphs):
        pooled.append(node_embeddings[batch == graph_idx].mean(dim=0))
    return torch.stack(pooled, dim=0) if pooled else node_embeddings.new_zeros((0, node_embeddings.size(1)))


def _fake_context_embeddings(
    model,
    context_structure,
    dataset,
    pooling_method,
    device,
    identity_projection,
    dataset_info,
    return_timing=False,
    sync_cuda=False,
    batch_size=None,
):
    num_tasks = max(context_structure.keys()) + 1 if context_structure else dataset_info["num_tasks"]
    hidden_dim = model.linear.weight.shape[0]
    base = model.linear.weight.mean()
    direction = torch.linspace(0.15, 0.45, steps=hidden_dim, device=base.device).view(1, hidden_dim)

    pos = []
    neg = []
    for task_idx in range(num_tasks):
        delta = (task_idx + 1) * 0.03
        pos.append((base + delta) * direction)
        neg.append((base - delta) * direction)

    if return_timing:
        return pos, neg, {
            "encode_time": 0.0,
            "overhead_time": 0.0,
            "concat_time": 0.0,
            "total_time": 0.0,
            "num_context_batches": 1,
        }
    return pos, neg


def _fake_multitask_pfn_logits(
    predictor,
    target_embeddings,
    pos_embeds_by_task,
    neg_embeds_by_task,
    normalize_class_h=True,
    device=None,
    task_indices=None,
):
    if task_indices is None:
        task_indices = list(range(len(pos_embeds_by_task)))

    logits = []
    valid_mask = []
    for task_idx in task_indices:
        pos_e = pos_embeds_by_task[task_idx]
        neg_e = neg_embeds_by_task[task_idx]
        if pos_e is None or neg_e is None or pos_e.numel() == 0 or neg_e.numel() == 0:
            logits.append(torch.zeros(target_embeddings.size(0), device=target_embeddings.device))
            valid_mask.append(False)
            continue

        proto = pos_e.mean(dim=0) - neg_e.mean(dim=0)
        logits_task = predictor.scale * (target_embeddings * proto).sum(dim=1) + predictor.bias
        logits.append(logits_task)
        valid_mask.append(True)

    if logits:
        logits = torch.stack(logits, dim=1)
    else:
        logits = torch.empty((target_embeddings.size(0), 0), device=target_embeddings.device)
    return logits, torch.tensor(valid_mask, dtype=torch.bool, device=target_embeddings.device)


def _make_batch(num_graphs=3, nodes_per_graph=2, in_dim=5, num_tasks=4):
    total_nodes = num_graphs * nodes_per_graph
    x = torch.arange(total_nodes * in_dim, dtype=torch.float32).view(total_nodes, in_dim) / 50.0
    batch = torch.arange(num_graphs).repeat_interleave(nodes_per_graph)

    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    y = torch.tensor(
        [
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    task_mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, True],
            [True, False, True, True],
        ],
        dtype=torch.bool,
    )

    return _DummyBatch(
        x=x,
        batch=batch,
        edge_index=edge_index,
        num_nodes=total_nodes,
        num_graphs=num_graphs,
        y=y,
        task_mask=task_mask,
    )


def _run_one_step(monkeypatch, chunk_size, seed=7):
    torch.manual_seed(seed)

    monkeypatch.setattr(engine_gc, "SparseTensor", _DummySparseTensor)
    monkeypatch.setattr(engine_gc, "pool_graph_embeddings", _fake_pool_graph_embeddings)
    monkeypatch.setattr(engine_gc, "_create_all_task_context_embeddings", _fake_context_embeddings)
    monkeypatch.setattr(engine_gc, "_vectorized_multitask_pfn_logits", _fake_multitask_pfn_logits)
    monkeypatch.setattr(engine_gc, "_get_node_embedding_table", lambda *args, **kwargs: torch.empty(1))
    monkeypatch.setattr(
        engine_gc,
        "_safe_lookup_node_embeddings",
        lambda node_emb_table, x, context="", batch_data=None, dataset_info=None: x.float(),
    )
    monkeypatch.setattr(engine_gc, "batch_edge_dropout", lambda batch_data, rate, training=True: batch_data)

    in_dim = 5
    hidden_dim = 3
    num_tasks = 4

    model = _TinyModel(in_dim=in_dim, hidden_dim=hidden_dim)
    predictor = _TinyPredictor()
    optimizer = torch.optim.SGD(list(model.parameters()) + list(predictor.parameters()), lr=0.1)

    step_counter = {"count": 0}
    original_step = optimizer.step

    def counted_step(*args, **kwargs):
        step_counter["count"] += 1
        return original_step(*args, **kwargs)

    optimizer.step = counted_step

    dataset = [SimpleNamespace(y=torch.zeros(num_tasks, dtype=torch.float32))]
    dataset_info = {
        "dataset": dataset,
        "context_graphs": {t: {} for t in range(num_tasks)},
        "needs_identity_projection": False,
        "num_tasks": num_tasks,
    }
    data_loaders = {"train": [_make_batch(num_tasks=num_tasks)]}

    args = SimpleNamespace(
        gc_supervised_mlp=False,
        gc_profile_context=False,
        gc_sim="dot",
        gc_ridge_alpha=1.0,
        gc_batch_size=16,
        gc_vec_task_chunk_size=chunk_size,
        edge_dropout_enabled=False,
    )

    loss = engine_gc.train_graph_classification_multitask_vectorized(
        model=model,
        predictor=predictor,
        dataset_info=dataset_info,
        data_loaders=data_loaders,
        optimizer=optimizer,
        pooling_method="mean",
        device="cpu",
        clip_grad=0.0,
        orthogonal_push=0.0,
        normalize_class_h=True,
        identity_projection=None,
        args=args,
        lambda_=1.0,
    )

    return {
        "loss": loss,
        "step_count": step_counter["count"],
        "model_state": {k: v.detach().clone() for k, v in model.state_dict().items()},
        "predictor_state": {k: v.detach().clone() for k, v in predictor.state_dict().items()},
    }


def _assert_states_close(state_a, state_b, rtol=1e-6, atol=1e-8):
    assert set(state_a.keys()) == set(state_b.keys())
    for key in state_a:
        assert torch.allclose(state_a[key], state_b[key], rtol=rtol, atol=atol), f"Mismatch at parameter: {key}"


def test_gc_vectorized_chunked_backward_matches_full_batch(monkeypatch):
    baseline = _run_one_step(monkeypatch, chunk_size=0, seed=123)
    chunk_1 = _run_one_step(monkeypatch, chunk_size=1, seed=123)
    chunk_2 = _run_one_step(monkeypatch, chunk_size=2, seed=123)
    chunk_big = _run_one_step(monkeypatch, chunk_size=999, seed=123)

    # Still full-batch semantics: exactly one optimizer step per batch.
    assert baseline["step_count"] == 1
    assert chunk_1["step_count"] == 1
    assert chunk_2["step_count"] == 1
    assert chunk_big["step_count"] == 1

    # Loss values and parameter updates must match baseline.
    assert math.isclose(chunk_1["loss"], baseline["loss"], rel_tol=1e-6, abs_tol=1e-8)
    assert math.isclose(chunk_2["loss"], baseline["loss"], rel_tol=1e-6, abs_tol=1e-8)
    assert math.isclose(chunk_big["loss"], baseline["loss"], rel_tol=1e-6, abs_tol=1e-8)

    _assert_states_close(chunk_1["model_state"], baseline["model_state"])
    _assert_states_close(chunk_2["model_state"], baseline["model_state"])
    _assert_states_close(chunk_big["model_state"], baseline["model_state"])

    _assert_states_close(chunk_1["predictor_state"], baseline["predictor_state"])
    _assert_states_close(chunk_2["predictor_state"], baseline["predictor_state"])
    _assert_states_close(chunk_big["predictor_state"], baseline["predictor_state"])
