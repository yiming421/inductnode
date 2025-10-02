#!/usr/bin/env python3
"""
Simple test to understand tensor dimensions in PFNTransformerLayer
"""
import torch
import torch.nn as nn

def test_multihead_attention_dimensions():
    """Test what dimensions MultiheadAttention expects and returns"""
    print("=== Testing MultiheadAttention Dimensions ===")

    # Create a simple attention layer
    hidden_dim = 64
    attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, dropout=0.0)

    # Test different input dimensions
    batch_size = 32
    seq_len_context = 10
    seq_len_target = 5

    print(f"Hidden dim: {hidden_dim}")
    print(f"Batch size: {batch_size}")
    print(f"Context sequence length: {seq_len_context}")
    print(f"Target sequence length: {seq_len_target}")

    # Test 1: Standard 3D input (seq_len, batch_size, hidden_dim)
    print("\n--- Test 1: 3D input (seq_len, batch_size, hidden_dim) ---")
    query_3d = torch.randn(seq_len_target, batch_size, hidden_dim)
    key_3d = torch.randn(seq_len_context, batch_size, hidden_dim)
    value_3d = torch.randn(seq_len_context, batch_size, hidden_dim)

    print(f"Query shape: {query_3d.shape}")
    print(f"Key shape: {key_3d.shape}")
    print(f"Value shape: {value_3d.shape}")

    try:
        output_3d, _ = attention(query_3d, key_3d, value_3d)
        print(f"✓ SUCCESS: Output shape: {output_3d.shape}")
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test 2: 4D input (should fail)
    print("\n--- Test 2: 4D input (should fail) ---")
    query_4d = query_3d.unsqueeze(0)  # Add extra dimension
    key_4d = key_3d.unsqueeze(0)
    value_4d = value_3d.unsqueeze(0)

    print(f"Query shape: {query_4d.shape}")
    print(f"Key shape: {key_4d.shape}")
    print(f"Value shape: {value_4d.shape}")

    try:
        output_4d, _ = attention(query_4d, key_4d, value_4d)
        print(f"✓ SUCCESS: Output shape: {output_4d.shape}")
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test 3: Different batch format (batch_size, seq_len, hidden_dim)
    print("\n--- Test 3: Batch-first format (batch_size, seq_len, hidden_dim) ---")
    query_batch_first = torch.randn(batch_size, seq_len_target, hidden_dim)
    key_batch_first = torch.randn(batch_size, seq_len_context, hidden_dim)
    value_batch_first = torch.randn(batch_size, seq_len_context, hidden_dim)

    print(f"Query shape: {query_batch_first.shape}")
    print(f"Key shape: {key_batch_first.shape}")
    print(f"Value shape: {value_batch_first.shape}")

    # Need to set batch_first=True
    attention_batch_first = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, dropout=0.0, batch_first=True)

    try:
        output_batch_first, _ = attention_batch_first(query_batch_first, key_batch_first, value_batch_first)
        print(f"✓ SUCCESS: Output shape: {output_batch_first.shape}")
    except Exception as e:
        print(f"✗ FAILED: {e}")

def test_mlp_dimensions():
    """Test what dimensions our MLP expects"""
    print("\n=== Testing MLP Dimensions ===")

    # Create a simple MLP (similar to what's used in the transformer)
    hidden_dim = 64
    mlp = nn.Sequential(
        nn.Linear(hidden_dim, 4 * hidden_dim),
        nn.ReLU(),
        nn.Linear(4 * hidden_dim, hidden_dim)
    )

    batch_size = 32
    seq_len = 10

    # Test different input shapes
    print(f"Hidden dim: {hidden_dim}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")

    # Test 1: 2D input (batch_size, hidden_dim)
    print("\n--- Test 1: 2D input (batch_size, hidden_dim) ---")
    input_2d = torch.randn(batch_size, hidden_dim)
    print(f"Input shape: {input_2d.shape}")

    try:
        output_2d = mlp(input_2d)
        print(f"✓ SUCCESS: Output shape: {output_2d.shape}")
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test 2: 3D input (batch_size, seq_len, hidden_dim)
    print("\n--- Test 2: 3D input (batch_size, seq_len, hidden_dim) ---")
    input_3d = torch.randn(batch_size, seq_len, hidden_dim)
    print(f"Input shape: {input_3d.shape}")

    try:
        output_3d = mlp(input_3d)
        print(f"✓ SUCCESS: Output shape: {output_3d.shape}")
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test 3: 3D input reshaped to 2D, then back
    print("\n--- Test 3: 3D -> 2D -> MLP -> 3D ---")
    input_3d = torch.randn(batch_size, seq_len, hidden_dim)
    print(f"Original input shape: {input_3d.shape}")

    # Reshape to 2D
    input_2d_reshaped = input_3d.view(-1, hidden_dim)
    print(f"Reshaped to 2D: {input_2d_reshaped.shape}")

    # Pass through MLP
    output_2d_reshaped = mlp(input_2d_reshaped)
    print(f"MLP output: {output_2d_reshaped.shape}")

    # Reshape back to 3D
    output_3d_reshaped = output_2d_reshaped.view(batch_size, seq_len, hidden_dim)
    print(f"✓ Final output shape: {output_3d_reshaped.shape}")

def test_actual_transformer_scenario():
    """Test the actual scenario from our transformer"""
    print("\n=== Testing Actual Transformer Scenario ===")

    # Simulate the actual dimensions we get from the PFN predictor
    batch_size = 32
    context_len = 10  # number of context samples
    target_len = 5    # number of target samples
    hidden_dim = 64

    print(f"Context length: {context_len}")
    print(f"Target length: {target_len}")
    print(f"Hidden dim: {hidden_dim}")
    print(f"Batch size: {batch_size}")

    # These are the shapes we typically get
    x_context = torch.randn(batch_size, context_len, hidden_dim)
    x_target = torch.randn(batch_size, target_len, hidden_dim)

    print(f"\nInitial shapes:")
    print(f"x_context: {x_context.shape}")
    print(f"x_target: {x_target.shape}")

    # Test with batch_first=False (default PyTorch MultiheadAttention)
    print("\n--- Testing with batch_first=False ---")
    attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, dropout=0.0, batch_first=False)

    # Need to transpose to (seq_len, batch_size, hidden_dim)
    x_context_transposed = x_context.transpose(0, 1)  # (context_len, batch_size, hidden_dim)
    x_target_transposed = x_target.transpose(0, 1)    # (target_len, batch_size, hidden_dim)

    print(f"After transpose:")
    print(f"x_context_transposed: {x_context_transposed.shape}")
    print(f"x_target_transposed: {x_target_transposed.shape}")

    try:
        # Self-attention on context
        context_att, _ = attention(x_context_transposed, x_context_transposed, x_context_transposed)
        print(f"✓ Context self-attention output: {context_att.shape}")

        # Cross-attention (target query, context key/value)
        target_att, _ = attention(x_target_transposed, x_context_transposed, x_context_transposed)
        print(f"✓ Target cross-attention output: {target_att.shape}")

    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test with batch_first=True
    print("\n--- Testing with batch_first=True ---")
    attention_batch_first = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, dropout=0.0, batch_first=True)

    try:
        # Self-attention on context
        context_att, _ = attention_batch_first(x_context, x_context, x_context)
        print(f"✓ Context self-attention output: {context_att.shape}")

        # Cross-attention (target query, context key/value)
        target_att, _ = attention_batch_first(x_target, x_context, x_context)
        print(f"✓ Target cross-attention output: {target_att.shape}")

    except Exception as e:
        print(f"✗ FAILED: {e}")

if __name__ == "__main__":
    print("Testing tensor dimensions for transformer components\n")

    test_multihead_attention_dimensions()
    test_mlp_dimensions()
    test_actual_transformer_scenario()

    print("\n=== Summary ===")
    print("• MultiheadAttention expects 3D tensors: (seq_len, batch_size, hidden_dim) or (batch_size, seq_len, hidden_dim) with batch_first=True")
    print("• MLPs work with both 2D and 3D tensors, applying to the last dimension")
    print("• The key is consistent dimension handling and proper transpose operations")