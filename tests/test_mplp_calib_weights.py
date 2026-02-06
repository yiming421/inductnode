import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import _fit_mplp_calib_sgd


def test_mplp_calib_weights_improve_loss_and_signs():
    torch.manual_seed(0)
    n = 2000
    feat = torch.randn(n)
    struct = torch.randn(n)
    true_w = torch.tensor([2.0, 0.5, -0.1])
    logits = true_w[0] * feat + true_w[1] * struct + true_w[2]
    probs = torch.sigmoid(logits)
    labels = torch.bernoulli(probs)

    w, b = _fit_mplp_calib_sgd(
        feat, struct, labels, w_max=10.0, reg=1e-3, steps=10, lr=0.1
    )
    pred_logits = w[0] * feat + w[1] * struct + b

    loss0 = F.binary_cross_entropy_with_logits(torch.zeros_like(labels), labels)
    loss1 = F.binary_cross_entropy_with_logits(pred_logits, labels)

    assert loss1 < loss0 * 0.9
    assert w[0].item() > 0.0
    assert w[1].item() > 0.0


def test_mplp_calib_weights_struct_clamp():
    torch.manual_seed(1)
    n = 2000
    feat = torch.randn(n)
    struct = torch.randn(n)
    logits = 5.0 * struct
    labels = torch.bernoulli(torch.sigmoid(logits))

    w, b = _fit_mplp_calib_sgd(
        feat, struct, labels, w_max=0.2, reg=1e-3, steps=10, lr=0.1
    )
    assert w[1].item() <= 0.2001
    assert w[1].item() >= 0.01
    assert b.item() <= 0.2001
    assert b.item() >= -0.2001


def test_mplp_calib_sgd_fits_separable_data():
    torch.manual_seed(2)
    n = 5000
    feat = torch.randn(n)
    struct = torch.randn(n)
    true_logits = 2.0 * feat + 1.0 * struct
    labels = (true_logits > 0).float()

    w, b = _fit_mplp_calib_sgd(
        feat, struct, labels, w_max=10.0, reg=1e-3, steps=50, lr=0.2
    )
    pred_logits = w[0] * feat + w[1] * struct + b
    preds = (pred_logits > 0).float()
    acc = (preds == labels).float().mean().item()

    assert acc > 0.9
    assert w[0].item() > 0.0
    assert w[1].item() > 0.0
