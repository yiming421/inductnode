import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import PFNTransformerLayer, PFNPredictorNodeCls


def test_pfn_transformer_layer_ffn_only_skips_attention_modules():
    layer = PFNTransformerLayer(
        hidden_dim=8,
        n_head=2,
        mlp_layers=2,
        dropout=0.0,
        norm=False,
        separate_att=True,
        norm_type='post',
        ffn_only=True,
    )

    assert layer.ffn_only is True
    assert layer.self_att is None
    assert layer.cross_att is None

    x_context = torch.randn(5, 3, 8)
    x_target = torch.randn(7, 3, 8)
    out_context, out_target = layer(x_context, x_target)

    assert out_context.shape == x_context.shape
    assert out_target.shape == x_target.shape
    assert torch.isfinite(out_context).all()
    assert torch.isfinite(out_target).all()


def test_pfn_predictor_propagates_ffn_only_flag_to_all_layers():
    predictor = PFNPredictorNodeCls(
        hidden_dim=8,
        nhead=2,
        num_layers=3,
        mlp_layers=2,
        dropout=0.0,
        norm=False,
        separate_att=True,
        transformer_ffn_only=True,
    )

    assert len(predictor.transformer_row) == 3
    assert all(layer.ffn_only for layer in predictor.transformer_row)
    assert all(layer.self_att is None for layer in predictor.transformer_row)
    assert all(layer.cross_att is None for layer in predictor.transformer_row)
