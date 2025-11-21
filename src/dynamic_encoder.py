"""
Dynamic Encoder Module for Feature Projection

This module implements the Dynamic Encoder (DE) from the FUG paper, which learns
to project features of varying dimensions to a unified space in an end-to-end manner.

Key Concepts:
1. Column Sampling: Sample n_s nodes from the graph to create a fixed-size input
2. Transposed Processing: Process features column-wise (d, n_s) instead of row-wise
3. Learnable Projection: Generate projection matrix T (d, k) via MLP
4. Uniformity Loss: Prevent collapse by pushing mean of T toward origin

Architecture:
    Input: (n_s, d) sampled node features
    → Transpose to (d, n_s)
    → MLP: (d, n_s) → (d, hidden) → (d, k)
    → L2 Normalize: Each row to unit norm
    → Output: Projection matrix T (d, k)

Usage:
    Universal Projection: X @ T where X is (N, d), T is (d, k) → (N, k)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicEncoder(nn.Module):
    """
    Dynamic Encoder: Learns projection matrix via column sampling and MLP.

    Based on FUG paper's DimensionNN_V2 architecture.
    """

    def __init__(self,
                 sample_size,
                 hidden_dim,
                 output_dim,
                 activation='prelu',
                 use_layernorm=True,
                 dropout=0.0,
                 norm_affine=True):
        """
        Initialize Dynamic Encoder.

        Default architecture follows FUG's DimensionNN_V2 with enhancements:
        - 3-layer MLP: Linear(n_s, h) → Act → Linear(h, h) → Act → Linear(h, k)
        - LayerNorm enabled by default for better stability
        - Dropout matches main model dropout for consistency
        - L2 normalization on output

        Args:
            sample_size (int): Number of nodes to sample (n_s)
            hidden_dim (int): Hidden dimension for MLP (h)
            output_dim (int): Output projection dimension (k)
            activation (str): Activation function ('prelu', 'relu', 'gelu', 'silu')
            use_layernorm (bool): Add LayerNorm after each linear layer (default: True)
            dropout (float): Dropout rate, should match main model dropout (default: 0.0)
            norm_affine (bool): Use learnable affine params in LayerNorm (default: True)
        """
        super(DynamicEncoder, self).__init__()

        self.sample_size = sample_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_layernorm = use_layernorm
        self.dropout_rate = dropout

        # 3-layer MLP: (n_s) → (hidden) → (hidden) → (k)
        # Note: Input to first linear is n_s (after transpose)
        self.lin_in = nn.Linear(sample_size, hidden_dim)
        self.lin_h1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin_out = nn.Linear(hidden_dim, output_dim)

        # Activation function
        if activation == 'prelu':
            self.act = nn.PReLU()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Optional LayerNorm (applied AFTER linear, BEFORE activation)
        if use_layernorm:
            # Note: Normalize over the feature dimension (dim=1)
            # Input shape to norm: (d, hidden)
            self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=norm_affine)
            self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=norm_affine)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        # Optional Dropout
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        # Cache for uniformity loss
        self.cached_projection = None

    def encode(self, x):
        """
        MLP forward pass with optional LayerNorm and Dropout.

        Architecture:
        - Layer 1: Linear → [LayerNorm] → Act → [Dropout]
        - Layer 2: Linear → [LayerNorm] → Act → [Dropout]
        - Layer 3: Linear (no activation/norm)

        Args:
            x: (d, n_s) transposed features

        Returns:
            z: (d, k) projection matrix (before normalization)
        """
        # Layer 1: Linear → [Norm] → Act → [Dropout]
        z = self.lin_in(x)
        z = self.norm1(z)
        z = self.act(z)
        z = self.dropout(z)

        # Layer 2: Linear → [Norm] → Act → [Dropout]
        z = self.lin_h1(z)
        z = self.norm2(z)
        z = self.act(z)
        z = self.dropout(z)

        # Layer 3: Linear (no activation, will be normalized in forward())
        z = self.lin_out(z)

        return z

    def forward(self, sampled_features):
        """
        Generate projection matrix from sampled features.

        Args:
            sampled_features: (n_s, d) or (batch_size, n_s, d) sampled node features

        Returns:
            projection_matrix: (d, k) normalized projection matrix
        """
        # Handle batch dimension (for future extension)
        if sampled_features.dim() == 3:
            batch_size = sampled_features.size(0)
            raise NotImplementedError("Batched DE not yet supported")

        # Transpose: (n_s, d) → (d, n_s)
        x_transposed = sampled_features.T

        # MLP encode: (d, n_s) → (d, k)
        projection_matrix = self.encode(x_transposed)

        # NOTE: Do NOT normalize the projection matrix - it kills gradients
        # and constrains what can be learned. Let it learn freely.

        # Cache for loss computation
        self.cached_projection = projection_matrix

        return projection_matrix

    def uniformity_loss(self):
        """
        Compute uniformity loss to prevent collapse.

        Loss: ||mean(T, dim=0)||^2

        This pushes the mean of all basis vectors toward the origin,
        forcing them to spread uniformly on the unit hypersphere.

        Returns:
            loss: Scalar tensor
        """
        if self.cached_projection is None:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        # Mean across dimension 0 (average all d basis vectors)
        mean_vector = self.cached_projection.mean(dim=0)

        # L2 norm squared
        loss = mean_vector.pow(2).mean()

        return loss


def sample_feature_columns(x, sample_size, training=True, deterministic_eval=True):
    """
    Sample n_s nodes for DE column sampling.

    Strategy:
    - Training: Random sampling (acts as augmentation)
    - Evaluation: Fixed sampling (deterministic, uses first n_s nodes)
    - Small graphs (N < sample_size): Pad with zeros

    Args:
        x (torch.Tensor): Node features (N, d)
        sample_size (int): Number of nodes to sample (n_s)
        training (bool): Whether in training mode
        deterministic_eval (bool): Use deterministic sampling during eval

    Returns:
        sampled (torch.Tensor): Sampled features (sample_size, d)
    """
    N, d = x.size()
    device = x.device
    dtype = x.dtype

    # Case 1: Graph smaller than sample size → pad with zeros
    if N < sample_size:
        padding = torch.zeros(sample_size - N, d, device=device, dtype=dtype)
        return torch.cat([x, padding], dim=0)

    # Case 2: Graph exactly sample size → return as is
    if N == sample_size:
        return x

    # Case 3: Graph larger than sample size → sample
    if training:
        # Random sampling for augmentation
        indices = torch.randperm(N, device=device)[:sample_size]
    else:
        if deterministic_eval:
            # Deterministic: always use first n_s nodes
            indices = torch.arange(sample_size, device=device)
        else:
            # Random even in eval (for ablation studies)
            indices = torch.randperm(N, device=device)[:sample_size]

    return x[indices]


def apply_dynamic_projection(x, projection_matrix, normalize=True, layer_norm=None):
    """
    Apply learned projection matrix to features.

    Args:
        x (torch.Tensor): Node features (N, d)
        projection_matrix (torch.Tensor): Projection matrix (d, k)
        normalize (bool): Whether to apply LayerNorm to output
        layer_norm (nn.LayerNorm, optional): Learnable LayerNorm module

    Returns:
        x_proj (torch.Tensor): Projected features (N, k)
    """
    # Matrix multiplication: (N, d) @ (d, k) → (N, k)
    x_proj = torch.mm(x, projection_matrix)

    # Apply LayerNorm instead of L2 norm to preserve std≈1.0
    # L2 norm causes std=1/sqrt(k)≈0.06 which is too small for GNN
    if normalize:
        if layer_norm is not None:
            x_proj = layer_norm(x_proj)
        else:
            # Fallback to non-learnable LayerNorm
            x_proj = F.layer_norm(x_proj, [x_proj.size(-1)])

    return x_proj


# Utility function for debugging
def compute_projection_statistics(projection_matrix):
    """
    Compute statistics about the projection matrix for debugging.

    Args:
        projection_matrix (torch.Tensor): (d, k) projection matrix

    Returns:
        stats (dict): Dictionary of statistics
    """
    with torch.no_grad():
        stats = {}

        # Mean norm of basis vectors (should be ~1 after normalization)
        stats['mean_norm'] = torch.norm(projection_matrix, p=2, dim=1).mean().item()

        # Mean of projection (should be close to 0 for good uniformity)
        mean_vector = projection_matrix.mean(dim=0)
        stats['mean_vector_norm'] = torch.norm(mean_vector, p=2).item()

        # Correlation between columns (should be low for diversity)
        normalized_cols = F.normalize(projection_matrix, p=2, dim=0)
        correlation_matrix = torch.mm(normalized_cols.T, normalized_cols)
        # Off-diagonal elements
        mask = ~torch.eye(correlation_matrix.size(0), dtype=torch.bool, device=correlation_matrix.device)
        stats['mean_correlation'] = correlation_matrix[mask].abs().mean().item()

        # Variance explained (proxy for expressiveness)
        stats['variance_explained'] = projection_matrix.pow(2).sum().item()

        return stats
