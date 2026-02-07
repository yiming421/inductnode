import math
import os
import sys
import types
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch_sparse import SparseTensor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.engine_nc import train as train_nc
from src.model import PureGCN_v1


class ToyPredictor(nn.Module):
    """Small trainable NC head used to verify cache correctness."""

    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, data, context_h, target_h, context_y, class_h):
        logits = self.linear(target_h)
        return logits, class_h


def _build_toy_data(num_nodes=12, feat_dim=8):
    # Deterministic, non-random features so both runs are directly comparable.
    x = torch.linspace(-1.0, 1.0, steps=num_nodes * feat_dim).view(num_nodes, feat_dim)
    y = torch.tensor([i % 3 for i in range(num_nodes)], dtype=torch.long)

    # Undirected ring graph.
    row = []
    col = []
    for i in range(num_nodes):
        j = (i + 1) % num_nodes
        row.extend([i, j])
        col.extend([j, i])
    row = torch.tensor(row, dtype=torch.long)
    col = torch.tensor(col, dtype=torch.long)
    adj_t = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes)).coalesce()

    # One context node per class.
    context_sample = torch.tensor([0, 1, 2], dtype=torch.long)

    data = SimpleNamespace()
    data.x = x
    data.y = y
    data.adj_t = adj_t
    data.context_sample = context_sample
    data.name = "toy_nc_cache"
    return data


def _make_args(cache_mode, hidden_dim):
    return SimpleNamespace(
        hidden=hidden_dim,
        seed=42,
        context_batch_refresh_interval=0,
        use_random_projection_augmentation=False,
        augmentation_mode="preprocessing",
        augmentation_regenerate_interval=1,
        use_train_time_augmentation=False,
        train_tta_num_augmentations=0,
        train_tta_include_original=False,
        edge_dropout_enabled=True,
        edge_dropout_rate=0.0,
        feature_dropout_enabled=True,
        feature_dropout_rate=0.0,
        use_contrastive_augmentation_loss=False,
        nc_static_embedding_cache=cache_mode,
    )


def _attach_forward_counter(model):
    counter = {"num_calls": 0}
    original_forward = model.forward

    def counted_forward(self, *args, **kwargs):
        counter["num_calls"] += 1
        return original_forward(*args, **kwargs)

    model.forward = types.MethodType(counted_forward, model)
    return counter


def _run_train_once(cache_mode, trainable_model, seed):
    torch.manual_seed(seed)

    feat_dim = 8
    hidden_dim = feat_dim + 2 if trainable_model else feat_dim
    num_classes = 3
    batch_size = 4

    data = _build_toy_data(num_nodes=12, feat_dim=feat_dim)
    train_idx = torch.arange(0, 12, dtype=torch.long)
    num_batches = math.ceil(train_idx.numel() / batch_size)

    model = PureGCN_v1(
        input_dim=feat_dim,
        num_layers=2,
        hidden=hidden_dim,
        dp=0.0,
        norm=False,
        res=False,
        relu=False,
        norm_affine=False,
        use_virtual_node=False,
    )
    predictor = ToyPredictor(hidden_dim=hidden_dim, num_classes=num_classes)
    optimizer = torch.optim.SGD(predictor.parameters(), lr=0.05)
    args = _make_args(cache_mode=cache_mode, hidden_dim=hidden_dim)

    call_counter = _attach_forward_counter(model)

    loss_dict = train_nc(
        model=model,
        data=data,
        train_idx=train_idx,
        optimizer=optimizer,
        pred=predictor,
        batch_size=batch_size,
        degree=False,
        att=None,
        mlp=None,
        orthogonal_push=0.0,
        normalize_class_h=False,
        clip_grad=0.0,
        projector=None,
        rank=0,
        epoch=0,
        identity_projection=None,
        lambda_=1.0,
        args=args,
        external_embeddings=None,
    )

    predictor_state = {
        name: tensor.detach().clone()
        for name, tensor in predictor.state_dict().items()
    }

    return {
        "loss": loss_dict,
        "predictor_state": predictor_state,
        "forward_calls": call_counter["num_calls"],
        "num_batches": num_batches,
    }


def test_nc_static_embedding_cache_auto_matches_off_and_reduces_forwards():
    # Parameter-free PureGCN_v1 path should allow cache in auto mode.
    baseline = _run_train_once(cache_mode="off", trainable_model=False, seed=123)
    cached = _run_train_once(cache_mode="auto", trainable_model=False, seed=123)

    assert baseline["forward_calls"] == baseline["num_batches"]
    assert cached["forward_calls"] == 1

    for key in ("total", "nll", "de", "contrastive"):
        assert math.isclose(
            baseline["loss"][key],
            cached["loss"][key],
            rel_tol=1e-7,
            abs_tol=1e-9,
        ), f"Loss mismatch for key={key}: {baseline['loss'][key]} vs {cached['loss'][key]}"

    for name in baseline["predictor_state"]:
        assert torch.allclose(
            baseline["predictor_state"][name],
            cached["predictor_state"][name],
            rtol=1e-6,
            atol=1e-8,
        ), f"Predictor weight mismatch: {name}"


def test_nc_static_embedding_cache_auto_disabled_for_trainable_model():
    # input_dim != hidden introduces trainable input projection; auto must disable cache.
    result = _run_train_once(cache_mode="auto", trainable_model=True, seed=321)

    assert result["num_batches"] > 1
    assert result["forward_calls"] == result["num_batches"]


def test_nc_static_embedding_cache_force_rejects_trainable_model():
    # force bypasses model-type checks, but must still reject trainable NC encoders.
    result = _run_train_once(cache_mode="force", trainable_model=True, seed=456)

    assert result["num_batches"] > 1
    assert result["forward_calls"] == result["num_batches"]
