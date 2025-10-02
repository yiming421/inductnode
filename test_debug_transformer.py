#!/usr/bin/env python3
"""
Debug the exact tensor flow in transformer layer
"""
import torch
import torch.nn as nn
import sys

# Add base directory to path
sys.path.append('/home/maweishuo/inductnode')

# Import the actual classes
from src.model import PFNTransformerLayer

def test_transformer_layer_debug():
    """Debug the transformer layer tensor flow"""
    print("=== Debugging PFNTransformerLayer tensor flow ===")

    hidden_dim = 128  # This is the embed_dim (hidden_dim + d_label)
    transformer = PFNTransformerLayer(
        hidden_dim=hidden_dim,
        n_head=4,
        mlp_layers=2,
        dropout=0.2,
        norm=False,
        separate_att=False,
        unsqueeze=False,  # Our fix
        norm_affine=True,
        norm_type='post',
        use_moe=False
    )

    print(f"Created transformer layer with hidden_dim={hidden_dim}")

    # Simulate the exact tensors from the predictor
    num_context = 10
    num_target = 5
    seq_len = 1  # The predictor adds unsqueeze(1) making this the sequence dimension

    # These are the shapes after predictor's unsqueeze(1)
    context_tokens = torch.randn(num_context, seq_len, hidden_dim)  # [10, 1, 128]
    target_tokens = torch.randn(num_target, seq_len, hidden_dim)    # [5, 1, 128]

    print(f"\nInput shapes to transformer:")
    print(f"context_tokens: {context_tokens.shape}")
    print(f"target_tokens: {target_tokens.shape}")

    try:
        print(f"\n--- Calling transformer forward ---")
        result = transformer(context_tokens, target_tokens)

        if isinstance(result, tuple) and len(result) == 2:
            output_context, output_target = result
            print(f"✓ SUCCESS!")
            print(f"Output context shape: {output_context.shape}")
            print(f"Output target shape: {output_target.shape}")
        else:
            print(f"? Unexpected result format: {type(result)}")

        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()

        # Let's debug step by step
        print(f"\n--- Step by step debug ---")

        # Check attention input format expectations
        print(f"MultiheadAttention parameters:")
        print(f"  embed_dim: {transformer.self_att.embed_dim}")
        print(f"  num_heads: {transformer.self_att.num_heads}")
        print(f"  batch_first: {getattr(transformer.self_att, 'batch_first', False)}")

        # The issue might be tensor format for attention
        # MultiheadAttention with batch_first=False expects (seq_len, batch_size, embed_dim)
        # But we have (batch_size, seq_len, embed_dim)

        print(f"\nTrying manual tensor transpose for attention...")

        # Context self-attention debug
        print(f"Context self-attention:")
        x_context_norm = transformer.context_norm1(context_tokens)
        print(f"After norm: {x_context_norm.shape}")

        # Transpose for attention: (batch, seq, dim) -> (seq, batch, dim)
        x_context_att_input = x_context_norm.transpose(0, 1)
        print(f"After transpose for attention: {x_context_att_input.shape}")

        try:
            x_context_att, _ = transformer.self_att(x_context_att_input, x_context_att_input, x_context_att_input)
            print(f"✓ Attention worked! Output: {x_context_att.shape}")

            # Transpose back: (seq, batch, dim) -> (batch, seq, dim)
            x_context_att = x_context_att.transpose(0, 1)
            print(f"After transpose back: {x_context_att.shape}")

        except Exception as att_e:
            print(f"✗ Attention still failed: {att_e}")

        return False

if __name__ == "__main__":
    test_transformer_layer_debug()