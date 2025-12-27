"""
GraphPFN transformer layer components.

Adapted from TabPFN's architecture:
https://github.com/automl/TabPFN
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from .attention import MultiHeadAttention
from .mlp import MLP

if TYPE_CHECKING:
    from .config import GraphPFNConfig

HIDDEN_SIZE_LIMIT = 512


class LayerNorm(torch.nn.LayerNorm):
    """Custom LayerNorm module adapted from TabPFN.

    This module extends the PyTorch LayerNorm implementation to handle FP16 inputs
    efficiently. When input is FP16 and normalized shape is small (<512),
    computation is forced to FP16 for better performance.

    Args:
        *args: Positional arguments passed to the base LayerNorm class.
        **kwargs: Keyword arguments passed to the base LayerNorm class.
    """

    @functools.wraps(torch.nn.LayerNorm.__init__)
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Perform layer normalization on the input tensor.

        Args:
            input: The input tensor.

        Returns:
            The layer normalized tensor.
        """
        x = input
        input_shape = x.shape

        # Reshape to apply layernorm
        x = x.reshape(-1, *self.normalized_shape)

        # FP16 optimization: force fp16 computation for small hidden sizes
        # torch.amp.autocast wants to run layer_norm in fp32, but that's 2x slower
        # Only safe for hidden sizes < 512 to avoid numerical instabilities
        if x.dtype == torch.float16 and sum(self.normalized_shape) < HIDDEN_SIZE_LIMIT:
            with torch.amp.autocast("cuda" if x.is_cuda else "cpu", enabled=False):
                x = super().forward(x)
        else:
            x = super().forward(x)

        return x.reshape(input_shape)


class PerFeatureEncoderLayer(nn.Module):
    """Transformer encoder layer with dual attention (features + items).
    
    Simplified from TabPFN's PerFeatureEncoderLayer, keeping essential functionality:
    - Attention between feature groups
    - Attention between items (nodes) with KV caching
    - MLP with residual connections
    - Post-norm (LayerNorm after each sublayer)
    
    Removed optimizations:
    - multiquery_item_attention_for_test_set
    - save_peak_mem_factor memory chunking
    - pre_norm support
    - second_mlp
    - att_src decoder support
    
    Args:
        config: GraphPFN configuration
        dim_feedforward: Hidden dimension for MLP
        activation: Activation function ('gelu' or 'relu')
        attention_between_features: Whether to use dual attention
        device: Device for parameters
        dtype: Data type for parameters
    """
    
    def __init__(
        self,
        *,
        config: 'GraphPFNConfig',
        dim_feedforward: int,
        activation: str = "gelu",
        attention_between_features: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        
        # Validate
        assert config.emsize % config.nhead == 0, \
            f"emsize ({config.emsize}) must be divisible by nhead ({config.nhead})"
        
        d_k = config.emsize // config.nhead
        d_v = config.emsize // config.nhead
        
        # Attention between features (dual attention)
        self.self_attn_between_features: MultiHeadAttention | None = None
        if attention_between_features:
            self.self_attn_between_features = MultiHeadAttention(
                d_model=config.emsize,
                nhead=config.nhead,
                d_k=d_k,
                d_v=d_v,
                dropout=config.dropout,
                device=device,
                dtype=dtype,
            )
        
        # Attention between items (nodes)
        self.self_attn_between_items = MultiHeadAttention(
            d_model=config.emsize,
            nhead=config.nhead,
            d_k=d_k,
            d_v=d_v,
            dropout=config.dropout,
            device=device,
            dtype=dtype,
        )
        
        # MLP
        self.mlp = MLP(
            size=config.emsize,
            hidden_size=dim_feedforward,
            activation=activation,
            device=device,
            dtype=dtype,
        )
        
        # Layer norms (post-norm: apply after each sublayer)
        self.layer_norms = nn.ModuleList()
        num_sublayers = 2 if attention_between_features else 1  # attn_items + mlp (+ attn_features)
        num_sublayers += 1  # Always have mlp
        
        for _ in range(num_sublayers):
            self.layer_norms.append(
                LayerNorm(config.emsize, elementwise_affine=True)
            )
    
    def forward(
        self,
        state: torch.Tensor,
        single_eval_pos: int,
        *,
        cache_trainset_representation: bool = False,
    ) -> torch.Tensor:
        """Forward pass through the encoder layer.
        
        Args:
            state: Input tensor [batch, num_nodes, num_feature_groups, emsize]
            single_eval_pos: Split point between train (context) and test (target) nodes
            cache_trainset_representation: If True, cache K,V for context nodes
            
        Returns:
            Transformed tensor [batch, num_nodes, num_feature_groups, emsize]
        """
        assert len(state.shape) == 4, \
            f"state must be [batch, num_nodes, num_feature_groups, emsize], got {state.shape}"
        
        # If using cached representations, don't cache again
        if cache_trainset_representation and not single_eval_pos:
            # Inference mode with cached K,V
            pass
        
        # Build list of sublayers to apply
        sublayers = []
        
        # 1. Optional: Attention between features
        if self.self_attn_between_features is not None:
            def attn_between_features(x: torch.Tensor) -> torch.Tensor:
                # Apply attention across feature groups (dim=2)
                # No need to transpose - attention works on dim -2
                result = self.self_attn_between_features(
                    x,
                    add_input=False,  # No residual - pre-norm handles it
                )
                return result
            sublayers.append(attn_between_features)
        else:
            assert state.shape[2] == 1, \
                f"Without feature attention, expect 1 feature group, got {state.shape[2]}"

        # 2. Attention between items (nodes)
        def attn_between_items(x: torch.Tensor) -> torch.Tensor:
            # Transpose to make items (nodes) the sequence dimension
            # x: [batch, num_nodes, num_feature_groups, emsize]
            # Need: [batch, num_feature_groups, num_nodes, emsize] for attention
            x_transposed = x.transpose(1, 2)

            # Prepare key/value source (context nodes)
            if single_eval_pos:
                # Cross-attention: target attends to context
                x_kv = x_transposed[:, :, :single_eval_pos, :]
            else:
                # Self-attention (or using cached K,V)
                x_kv = None

            # Apply attention
            x_out = self.self_attn_between_items(
                x_transposed,
                x_kv=x_kv,
                cache_kv=cache_trainset_representation and single_eval_pos > 0,
                use_cached_kv=cache_trainset_representation and single_eval_pos == 0,
                add_input=False,  # No residual - pre-norm handles it
            )

            # Transpose back
            return x_out.transpose(1, 2)

        sublayers.append(attn_between_items)

        # 3. MLP
        def mlp_forward(x: torch.Tensor) -> torch.Tensor:
            result = self.mlp(x, add_input=False)  # No residual - pre-norm handles it
            return result

        sublayers.append(mlp_forward)

        # Apply sublayers with pre-norm
        # Pre-norm: LayerNorm -> Sublayer -> Add residual
        for idx, (sublayer, layer_norm) in enumerate(zip(sublayers, self.layer_norms)):
            normalized = layer_norm(state)
            if idx == 0:  # Only print for first sublayer to avoid spam
                print(f"[PRE-NORM] Before LayerNorm: norm={state.norm().item():.2f}, After LayerNorm: norm={normalized.norm().item():.2f}")
            sublayer_output = sublayer(normalized)
            state = state + sublayer_output  # Add residual

        return state
    
    def empty_trainset_representation_cache(self) -> None:
        """Clear cached K,V representations."""
        self.self_attn_between_items.clear_cache()
        if self.self_attn_between_features is not None:
            self.self_attn_between_features.clear_cache()
