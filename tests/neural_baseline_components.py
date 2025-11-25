"""
Neural Components for Zero-Learning Baseline Extension

This module contains neural components to extend the zero-learning baseline:
- Fast GPU PCA dimension unification using torch.pca_lowrank (same as main pipeline)
- MLP feature transformer as a pluggable component
- Proper padding handling like the main pipeline
- Works with existing prototypical/ridge regression classification heads
"""

import torch
import torch.nn.functional as F
from torch import nn


def apply_fast_pca_with_padding(features, target_dim, use_full_pca=False):
    """
    Apply fast GPU PCA with padding (same logic as main pipeline).
    
    Args:
        features (torch.Tensor): Input features [N, D]
        target_dim (int): Target dimensionality after PCA
        use_full_pca (bool): Use full SVD instead of lowrank PCA
        
    Returns:
        torch.Tensor: PCA-transformed and padded features [N, target_dim]
    """
    original_dim = features.size(1)
    num_nodes = features.size(0)
    max_pca_dim = min(num_nodes, original_dim)
    
    if original_dim >= target_dim:
        # Enough features, just PCA to target_dim
        pca_target_dim = min(target_dim, max_pca_dim)
    else:
        # Not enough features, PCA to all available then pad
        pca_target_dim = min(original_dim, max_pca_dim)
    
    # Apply PCA using same method as main pipeline
    if use_full_pca:
        U, S, V = torch.svd(features)
        U = U[:, :pca_target_dim]
        S = S[:pca_target_dim]
    else:
        U, S, V = torch.pca_lowrank(features, q=pca_target_dim)
    
    x_pca = torch.mm(U, torch.diag(S))
    
    # Padding if necessary (same logic as main pipeline)
    if x_pca.size(1) < target_dim:
        padding_size = target_dim - x_pca.size(1)
        # Use zero padding (can be extended to other strategies)
        padding = torch.zeros(x_pca.size(0), padding_size, 
                            device=x_pca.device, dtype=x_pca.dtype)
        x_pca = torch.cat([x_pca, padding], dim=1)
    
    return x_pca


class MLPFeatureTransformer(nn.Module):
    """
    MLP for feature transformation - pluggable component.
    This transforms features that will be fed to existing classification heads.
    """
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=None, num_layers=2,
                 dropout=0.2, activation='relu', norm=True, final_norm=True):
        """
        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden dimension
            output_dim (int): Output dimension (default: same as hidden_dim)
            num_layers (int): Number of layers
            dropout (float): Dropout probability
            activation (str): Activation function ('relu', 'gelu', 'silu')
            norm (bool): Use layer normalization between hidden layers
            final_norm (bool): Use layer normalization after final output layer
        """
        super().__init__()
        
        if output_dim is None:
            output_dim = hidden_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.final_norm = final_norm

        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # Build layers
        layers = []
        current_dim = input_dim

        # Hidden layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self.activation)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        # Output layer (no activation)
        if num_layers > 0:
            layers.append(nn.Linear(current_dim, output_dim))

        self.mlp = nn.Sequential(*layers) if layers else nn.Identity()

        # Final layer normalization (applied after MLP output)
        if self.final_norm:
            self.output_norm = nn.LayerNorm(output_dim)
        else:
            self.output_norm = None
        
    def forward(self, x):
        """
        Transform features through MLP.

        Args:
            x (torch.Tensor): Input features [N, input_dim]

        Returns:
            torch.Tensor: Transformed features [N, output_dim]
        """
        x = self.mlp(x)
        if self.output_norm is not None:
            x = self.output_norm(x)
        return x


def create_neural_feature_processor(target_unified_dim=128, mlp_hidden=256, 
                                  mlp_output_dim=None, mlp_layers=2, mlp_dropout=0.2,
                                  use_pca=True, use_full_pca=False):
    """
    Create neural feature processor components.
    
    Args:
        target_unified_dim (int): Target dimension after PCA unification
        mlp_hidden (int): MLP hidden dimension
        mlp_output_dim (int): MLP output dimension (default: same as hidden)
        mlp_layers (int): Number of MLP layers
        mlp_dropout (float): MLP dropout
        use_pca (bool): Whether to use PCA unification
        use_full_pca (bool): Use full SVD instead of lowrank PCA
        
    Returns:
        tuple: (pca_processor_func, mlp_transformer)
    """
    # PCA processor (function that can be applied to any features)
    if use_pca:
        def pca_processor(features):
            return apply_fast_pca_with_padding(features, target_unified_dim, use_full_pca)
        mlp_input_dim = target_unified_dim
    else:
        pca_processor = None
        # Without PCA, need to know input_dim when creating MLP
        mlp_input_dim = None
    
    # MLP transformer (will be created when we know the actual input dim)
    def create_mlp_transformer(actual_input_dim):
        return MLPFeatureTransformer(
            input_dim=actual_input_dim,
            hidden_dim=mlp_hidden,
            output_dim=mlp_output_dim,
            num_layers=mlp_layers,
            dropout=mlp_dropout
        )
    
    return pca_processor, create_mlp_transformer, mlp_input_dim


def process_features_with_neural_pipeline(features, pca_processor=None, mlp_transformer=None):
    """
    Process features through the neural pipeline.
    
    Args:
        features (torch.Tensor): Input features [N, D]
        pca_processor (callable): PCA processing function
        mlp_transformer (MLPFeatureTransformer): MLP transformer
        
    Returns:
        torch.Tensor: Processed features [N, output_dim]
    """
    x = features
    
    # Step 1: PCA unification (if enabled)
    if pca_processor is not None:
        x = pca_processor(x)
    
    # Step 2: MLP transformation (if provided)
    if mlp_transformer is not None:
        x = mlp_transformer(x)
    
    return x