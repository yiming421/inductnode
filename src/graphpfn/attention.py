"""
Multi-head attention module for GraphPFN.

Simplified version of TabPFN's attention with KV caching support.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from .config import GraphPFNConfig

TRACE_ATTN = False


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional KV caching.

    Simplified from TabPFN's full_attention.py, keeping core functionality:
    - Standard multi-head attention
    - KV caching for context nodes
    - Cross-attention support

    Args:
        d_model: Model dimension (input/output size)
        nhead: Number of attention heads
        d_k: Dimension of keys (default: d_model // nhead)
        d_v: Dimension of values (default: d_model // nhead)
        dropout: Dropout probability
        device: Device for parameters
        dtype: Data type for parameters
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 8,
        d_k: int | None = None,
        d_v: int | None = None,
        dropout: float = 0.0,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_k if d_k is not None else d_model // nhead
        self.d_v = d_v if d_v is not None else d_model // nhead
        self.dropout_p = dropout

        # Query, Key, Value projections
        self.w_q = nn.Parameter(
            torch.empty(nhead, d_model, self.d_k, device=device, dtype=dtype)
        )
        self.w_k = nn.Parameter(
            torch.empty(nhead, d_model, self.d_k, device=device, dtype=dtype)
        )
        self.w_v = nn.Parameter(
            torch.empty(nhead, d_model, self.d_v, device=device, dtype=dtype)
        )

        # Output projection
        self.w_out = nn.Parameter(
            torch.empty(nhead, self.d_v, d_model, device=device, dtype=dtype)
        )

        # KV cache
        self._k_cache: torch.Tensor | None = None
        self._v_cache: torch.Tensor | None = None

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform."""
        # Initialize each head like an independent Linear(d_model -> d_k/d_v)
        for h in range(self.nhead):
            nn.init.xavier_uniform_(self.w_q[h])
            nn.init.xavier_uniform_(self.w_k[h])
            nn.init.xavier_uniform_(self.w_v[h])
            # Output projection per head (d_v -> d_model)
            nn.init.xavier_uniform_(self.w_out[h])

    @property
    def has_cached_kv(self) -> bool:
        """Check if KV cache exists."""
        return self._k_cache is not None and self._v_cache is not None

    def clear_cache(self):
        """Clear KV cache."""
        self._k_cache = None
        self._v_cache = None

    def forward(
        self,
        x: torch.Tensor,
        x_kv: torch.Tensor | None = None,
        *,
        cache_kv: bool = False,
        use_cached_kv: bool = False,
        add_input: bool = False,
    ) -> torch.Tensor:
        """Forward pass of multi-head attention.

        Args:
            x: Query tensor [batch, ..., seq_len, d_model]
            x_kv: Key/Value tensor for cross-attention. If None, uses x (self-attention)
            cache_kv: If True, cache the K,V for future use
            use_cached_kv: If True, use cached K,V instead of computing
            add_input: If True, add residual connection

        Returns:
            Output tensor [batch, ..., seq_len, d_model]
        """
        assert not (cache_kv and use_cached_kv), \
            "Cannot cache and use cached KV at the same time"

        input_tensor = x
        input_shape = x.shape

        # Flatten batch dimensions: [batch, ..., seq_len, d_model] -> [B, L, d_model]
        # where B = product of all batch dims
        x = x.reshape(-1, input_shape[-2], input_shape[-1])
        batch_size, seq_len_q, _ = x.shape

        if x_kv is not None:
            x_kv = x_kv.reshape(-1, x_kv.shape[-2], x_kv.shape[-1])
            seq_len_kv = x_kv.shape[1]
        else:
            x_kv = x
            seq_len_kv = seq_len_q

        # Compute Q, K, V
        # Q: [B, L_q, d_model] @ [nhead, d_model, d_k] -> [B, L_q, nhead, d_k]
        q = torch.einsum('bld,hdk->blhk', x, self.w_q)
        if TRACE_ATTN:
            print(f"[ATTN TRACE] Step 1 - Input x: norm={x.norm().item():.2f}")
            print(f"[ATTN TRACE] Step 2 - After Q projection: norm={q.norm().item():.2f}")

        if use_cached_kv and self.has_cached_kv:
            # Use cached K, V
            k = self._k_cache
            v = self._v_cache
        else:
            # Compute K, V
            k = torch.einsum('bld,hdk->blhk', x_kv, self.w_k)  # [B, L_kv, nhead, d_k]
            v = torch.einsum('bld,hdv->blhv', x_kv, self.w_v)  # [B, L_kv, nhead, d_v]
            if TRACE_ATTN:
                print(f"[ATTN TRACE] Step 3 - After K projection: norm={k.norm().item():.2f}")
                print(f"[ATTN TRACE] Step 4 - After V projection: norm={v.norm().item():.2f}")

            if cache_kv:
                # Cache K, V
                self._k_cache = k.detach()
                self._v_cache = v.detach()

        # Reshape for attention: [B, nhead, L, d]
        q = q.transpose(1, 2)  # [B, nhead, L_q, d_k]
        k = k.transpose(1, 2)  # [B, nhead, L_kv, d_k]
        v = v.transpose(1, 2)  # [B, nhead, L_kv, d_v]

        # Scaled dot-product attention
        # Use PyTorch 2.0+ optimized function if available
        if hasattr(F, 'scaled_dot_product_attention'):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout_p if self.training else 0.0,
            )  # [B, nhead, L_q, d_v]
            if TRACE_ATTN:
                print(f"[ATTN TRACE] Step 5 - After attention: norm={attn_output.norm().item():.2f}")
        else:
            # Manual implementation
            scale = 1.0 / math.sqrt(self.d_k)
            attn_scores = torch.einsum('bhqd,bhkd->bhqk', q, k) * scale  # [B, nhead, L_q, L_kv]
            attn_weights = F.softmax(attn_scores, dim=-1)
            if self.training and self.dropout_p > 0:
                attn_weights = F.dropout(attn_weights, p=self.dropout_p)
            attn_output = torch.einsum('bhqk,bhkv->bhqv', attn_weights, v)  # [B, nhead, L_q, d_v]
            if TRACE_ATTN:
                print(f"[ATTN TRACE] Step 5 - After attention: norm={attn_output.norm().item():.2f}")

        # Transpose back: [B, nhead, L_q, d_v] -> [B, L_q, nhead, d_v]
        attn_output = attn_output.transpose(1, 2)

        # Output projection: [B, L_q, nhead, d_v] @ [nhead, d_v, d_model] -> [B, L_q, d_model]
        output = torch.einsum('blhv,hvo->blo', attn_output, self.w_out)
        if TRACE_ATTN:
            print(f"[ATTN TRACE] Step 6 - After output projection: norm={output.norm().item():.2f}")

        # Reshape back to original shape
        output = output.reshape(input_shape)

        # Optional residual connection
        if add_input:
            # DEBUG: Check magnitude ratio
            input_norm = input_tensor.norm().item()
            attn_norm = output.norm().item()
            ratio = input_norm / (attn_norm + 1e-9)
            if ratio > 5.0 or ratio < 0.2:  # Log if imbalanced
                print(f"[ATTN MAGNITUDE] Input norm: {input_norm:.4f}, Attn output norm: {attn_norm:.4f}, Ratio: {ratio:.2f}x")

            output = output + input_tensor

        return output
