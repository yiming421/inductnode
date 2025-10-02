#!/usr/bin/env python3
"""
Test to replicate the exact transformer behavior and find the dimension issue
"""
import torch
import torch.nn as nn

# Import the actual classes
import sys
import os
sys.path.append('/home/maweishuo/inductnode/src')

# Simple MLP class (similar to what's in model.py)
class SimpleMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(nn.Linear(in_channels, hidden_channels))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout))

        # Final layer
        self.layers.append(nn.Linear(hidden_channels, out_channels))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class TestTransformerLayer(nn.Module):
    """Simplified version of PFNTransformerLayer to test dimensions"""
    def __init__(self, hidden_dim, n_head=1, dropout=0.2, unsqueeze=False):
        super().__init__()

        self.unsqueeze = unsqueeze

        # Attention layers (batch_first=False by default)
        self.self_att = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_head,
            dropout=dropout,
        )

        # FFN
        self.ffn = SimpleMLP(
            in_channels=hidden_dim,
            hidden_channels=4 * hidden_dim,
            out_channels=hidden_dim,
        )

        # Layer norms
        self.context_norm1 = nn.LayerNorm(hidden_dim)
        self.context_norm2 = nn.LayerNorm(hidden_dim)
        self.tar_norm1 = nn.LayerNorm(hidden_dim)
        self.tar_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x_context, x_target):
        print(f"\n=== Forward Pass Debug ===")
        print(f"Input x_context shape: {x_context.shape}")
        print(f"Input x_target shape: {x_target.shape}")

        # Check what format we need for attention
        # MultiheadAttention with batch_first=False expects (seq_len, batch_size, hidden_dim)

        # For self-attention, we need to figure out the sequence and batch dimensions
        # Current shape is likely (batch_size, seq_len, hidden_dim)
        # So we need to transpose to (seq_len, batch_size, hidden_dim)

        if len(x_context.shape) == 3:
            print(f"3D tensor detected, transposing for attention")
            # Transpose from (batch_size, seq_len, hidden_dim) to (seq_len, batch_size, hidden_dim)
            x_context_att_format = x_context.transpose(0, 1)
            x_target_att_format = x_target.transpose(0, 1)
        else:
            print(f"Unexpected tensor dimensions: {x_context.shape}")
            x_context_att_format = x_context
            x_target_att_format = x_target

        print(f"x_context_att_format shape: {x_context_att_format.shape}")
        print(f"x_target_att_format shape: {x_target_att_format.shape}")

        # Context self-attention
        print(f"\n--- Context Self-Attention ---")
        x_context_norm = self.context_norm1(x_context)
        x_context_norm_att_format = x_context_norm.transpose(0, 1) if len(x_context_norm.shape) == 3 else x_context_norm

        try:
            x_context_att, _ = self.self_att(x_context_norm_att_format, x_context_norm_att_format, x_context_norm_att_format)
            print(f"✓ Context attention output shape: {x_context_att.shape}")

            # Transpose back if needed
            if len(x_context_att.shape) == 3:
                x_context_att = x_context_att.transpose(0, 1)
                print(f"After transpose back: {x_context_att.shape}")

            x_context = x_context_att + x_context  # Residual connection
            print(f"After residual: {x_context.shape}")

        except Exception as e:
            print(f"✗ Context attention failed: {e}")
            return None, None

        # Context FFN
        print(f"\n--- Context FFN ---")
        x_context_norm = self.context_norm2(x_context)
        print(f"Before FFN: {x_context_norm.shape}")

        x_context_fnn = self.ffn(x_context_norm)
        print(f"After FFN: {x_context_fnn.shape}")

        if self.unsqueeze:
            print(f"Applying unsqueeze logic...")
            x_context_fnn = x_context_fnn.unsqueeze(1)
            print(f"After unsqueeze: {x_context_fnn.shape}")

            x_context_expanded = x_context.unsqueeze(1)
            print(f"Context expanded: {x_context_expanded.shape}")

            x_context_residual = x_context_fnn + x_context_expanded
            print(f"After residual: {x_context_residual.shape}")

            x_context = x_context_residual.squeeze(1)
            print(f"After squeeze back: {x_context.shape}")
        else:
            x_context = x_context_fnn + x_context
            print(f"Simple residual: {x_context.shape}")

        # Target cross-attention
        print(f"\n--- Target Cross-Attention ---")
        x_target_norm = self.tar_norm1(x_target)
        print(f"Target norm shape: {x_target_norm.shape}")
        print(f"Context for attention shape: {x_context.shape}")

        # Prepare for attention
        x_target_norm_att_format = x_target_norm.transpose(0, 1) if len(x_target_norm.shape) == 3 else x_target_norm
        x_context_att_format = x_context.transpose(0, 1) if len(x_context.shape) == 3 else x_context

        print(f"Target norm att format: {x_target_norm_att_format.shape}")
        print(f"Context att format: {x_context_att_format.shape}")

        try:
            x_target_att, _ = self.self_att(x_target_norm_att_format, x_context_att_format, x_context_att_format)
            print(f"✓ Target attention output shape: {x_target_att.shape}")

            # Transpose back
            if len(x_target_att.shape) == 3:
                x_target_att = x_target_att.transpose(0, 1)
                print(f"After transpose back: {x_target_att.shape}")

            x_target = x_target_att + x_target
            print(f"After residual: {x_target.shape}")

        except Exception as e:
            print(f"✗ Target attention failed: {e}")
            return None, None

        # Target FFN
        print(f"\n--- Target FFN ---")
        x_target_norm = self.tar_norm2(x_target)
        x_target_fnn = self.ffn(x_target_norm)

        if self.unsqueeze:
            x_target_fnn = x_target_fnn.unsqueeze(1)
            x_target_expanded = x_target.unsqueeze(1)
            x_target_residual = x_target_fnn + x_target_expanded
            x_target = x_target_residual.squeeze(1)
        else:
            x_target = x_target_fnn + x_target

        print(f"Final x_context shape: {x_context.shape}")
        print(f"Final x_target shape: {x_target.shape}")

        return x_context, x_target

def test_transformer_scenarios():
    """Test different scenarios that might cause the dimension issue"""
    print("=== Testing Transformer Scenarios ===")

    hidden_dim = 64
    batch_size = 8
    context_len = 10
    target_len = 5

    # Scenario 1: Normal case without unsqueeze
    print(f"\n--- Scenario 1: unsqueeze=False ---")
    transformer = TestTransformerLayer(hidden_dim, unsqueeze=False)

    x_context = torch.randn(batch_size, context_len, hidden_dim)
    x_target = torch.randn(batch_size, target_len, hidden_dim)

    result = transformer(x_context, x_target)
    if result[0] is not None:
        print(f"✓ SUCCESS: unsqueeze=False works")
    else:
        print(f"✗ FAILED: unsqueeze=False failed")

    # Scenario 2: With unsqueeze
    print(f"\n--- Scenario 2: unsqueeze=True ---")
    transformer_unsqueeze = TestTransformerLayer(hidden_dim, unsqueeze=True)

    x_context = torch.randn(batch_size, context_len, hidden_dim)
    x_target = torch.randn(batch_size, target_len, hidden_dim)

    result = transformer_unsqueeze(x_context, x_target)
    if result[0] is not None:
        print(f"✓ SUCCESS: unsqueeze=True works")
    else:
        print(f"✗ FAILED: unsqueeze=True failed")

if __name__ == "__main__":
    test_transformer_scenarios()