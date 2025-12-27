"""
Parameter-free GNN model for GraphPFN foundational model.

This GNN accepts arbitrary input dimensions since it has no learnable projection layers.
Only uses:
- PureGCNConv (parameter-free message passing)
- LayerNorm without affine (dimension-agnostic normalization)
- Dropout (no parameters)
- Residual connections (no parameters)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse.matmul import spmm_add


class PureGCNConv(nn.Module):
    """Parameter-free GCN convolution layer."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, adj_t):
        norm = torch.rsqrt_((1+adj_t.sum(dim=-1))).reshape(-1, 1)
        x = norm * x
        x = spmm_add(adj_t, x) + x
        x = norm * x
        return x


class ParameterFreeGCN(nn.Module):
    """Parameter-free GCN for GraphPFN foundational model.

    Accepts arbitrary input dimensions. Uses:
    - PureGCNConv: parameter-free message passing
    - LayerNorm without affine: handles any dimension
    - Dropout: regularization
    - Residual connections: gradient flow

    Args:
        num_layers: Number of GCN layers
        dropout: Dropout rate
        use_norm: Whether to use LayerNorm
        use_residual: Whether to use residual connections
    """

    def __init__(
        self,
        num_layers: int = 2,
        dropout: float = 0.0,
        use_norm: bool = True,
        use_residual: bool = True,
    ):
        super().__init__()

        self.conv = PureGCNConv()
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_norm = use_norm
        self.use_residual = use_residual

        # LayerNorm WITHOUT affine parameters (elementwise_affine=False)
        # This makes it dimension-agnostic!
        if self.use_norm:
            # We can't pre-create LayerNorm because we don't know the dimension
            # Will create on first forward pass
            self.norm_dim = None
            self.norms = None

    def forward(self, x, adj_t):
        """
        Forward pass through parameter-free GCN.

        Args:
            x: Node features [num_nodes, feature_dim] (arbitrary dimension!)
            adj_t: Sparse adjacency tensor

        Returns:
            Node embeddings [num_nodes, feature_dim] (same dimension as input)
        """
        # Create LayerNorms on first forward pass (now we know the dimension)
        if self.use_norm and (self.norms is None or self.norm_dim != x.shape[1]):
            self.norm_dim = x.shape[1]
            self.norms = nn.ModuleList([
                nn.LayerNorm(self.norm_dim, elementwise_affine=False).to(x.device)
                for _ in range(self.num_layers)
            ])
        elif self.use_norm and self.norms is not None:
            # Check if device changed - use first parameter of first norm
            first_norm_device = next(self.norms[0].parameters(), None)
            if first_norm_device is None:  # No parameters (elementwise_affine=False)
                # Check using buffers instead
                first_norm_device = next(self.norms[0].buffers(), None)
            if first_norm_device is not None and first_norm_device.device != x.device:
                # Move norms to correct device if input device changed
                self.norms = self.norms.to(x.device)

        ori = x
        for i in range(self.num_layers):
            if i != 0:
                if self.use_residual:
                    x = x + ori
                if self.use_norm:
                    x = self.norms[i](x)
                if self.dropout > 0:
                    x = F.dropout(x, p=self.dropout, training=self.training)

            x = self.conv(x, adj_t)

        # Final residual connection
        if self.use_residual:
            x = x + ori

        return x
