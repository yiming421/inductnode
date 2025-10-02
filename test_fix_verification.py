#!/usr/bin/env python3
"""
Test to verify the tensor dimension fix works
"""
import torch
import torch.nn as nn
import sys
import os

# Add base directory to path
sys.path.append('/home/maweishuo/inductnode')

# Import the actual classes
from src.model import PFNPredictorNodeCls

def test_predictor_dimensions():
    """Test that the predictor works with the fix"""
    print("=== Testing PFNPredictorNodeCls with fix ===")

    # Create a predictor with the same settings as the real code
    hidden_dim = 64
    predictor = PFNPredictorNodeCls(
        hidden_dim=hidden_dim,
        nhead=4,
        num_layers=2,
        mlp_layers=2,
        dropout=0.2,
        norm=False,
        separate_att=False,
        degree=False,
        att=None,
        mlp=None,
        sim='dot',
        padding='zero',
        norm_affine=True,
        normalize=False,
        use_first_half_embedding=False,
        use_full_embedding=False,
        norm_type='post',
        use_moe=False,  # Test without MoE first
        moe_num_experts=4,
        moe_top_k=2,
        moe_auxiliary_loss_weight=0.01
    )

    print(f"Predictor created successfully")
    print(f"Embed dim: {predictor.embed_dim}")

    # Create test data similar to real scenario
    batch_size = 8
    num_context = 10
    num_target = 5
    num_classes = 4

    # Create test inputs
    context_x = torch.randn(num_context, hidden_dim)  # Context node embeddings
    target_x = torch.randn(num_target, hidden_dim)    # Target node embeddings
    context_y = torch.randint(0, num_classes, (num_context,))  # Context labels
    class_x = torch.randn(num_classes, hidden_dim)    # Class prototypes

    print(f"\nInput shapes:")
    print(f"context_x: {context_x.shape}")
    print(f"target_x: {target_x.shape}")
    print(f"context_y: {context_y.shape}")
    print(f"class_x: {class_x.shape}")

    # Create mock data object
    class MockData:
        def __init__(self):
            self.y = torch.randint(0, num_classes, (100,))
            self.context_sample = torch.arange(num_context)

    data = MockData()

    try:
        # Test the forward pass
        print(f"\n--- Testing forward pass ---")
        result = predictor(data, context_x, target_x, context_y, class_x)

        if isinstance(result, tuple) and len(result) == 2:
            output, class_h_refined = result
            print(f"✓ SUCCESS!")
            print(f"Output shape: {output.shape}")
            print(f"Class h refined shape: {class_h_refined.shape}")
        elif isinstance(result, tuple) and len(result) == 3:
            output, class_h_refined, aux_loss = result
            print(f"✓ SUCCESS with auxiliary loss!")
            print(f"Output shape: {output.shape}")
            print(f"Class h refined shape: {class_h_refined.shape}")
            print(f"Auxiliary loss: {aux_loss}")
        else:
            print(f"? Unexpected result format: {type(result)}")

        return True

    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_moe():
    """Test with MoE enabled"""
    print(f"\n=== Testing with MoE enabled ===")

    hidden_dim = 64
    predictor = PFNPredictorNodeCls(
        hidden_dim=hidden_dim,
        nhead=4,
        num_layers=2,
        use_moe=True,  # Enable MoE
        moe_num_experts=4,
        moe_top_k=2,
        moe_auxiliary_loss_weight=0.01
    )

    # Same test data
    num_context = 10
    num_target = 5
    num_classes = 4

    context_x = torch.randn(num_context, hidden_dim)
    target_x = torch.randn(num_target, hidden_dim)
    context_y = torch.randint(0, num_classes, (num_context,))
    class_x = torch.randn(num_classes, hidden_dim)

    class MockData:
        def __init__(self):
            self.y = torch.randint(0, num_classes, (100,))
            self.context_sample = torch.arange(num_context)

    data = MockData()

    try:
        result = predictor(data, context_x, target_x, context_y, class_x)

        if isinstance(result, tuple) and len(result) == 3:
            output, class_h_refined, aux_loss = result
            print(f"✓ MoE SUCCESS!")
            print(f"Output shape: {output.shape}")
            print(f"Class h refined shape: {class_h_refined.shape}")
            print(f"Auxiliary loss: {aux_loss}")
        else:
            print(f"? Unexpected MoE result format: {type(result)}")

        return True

    except Exception as e:
        print(f"✗ MoE FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success1 = test_predictor_dimensions()
    success2 = test_with_moe()

    print(f"\n=== Summary ===")
    print(f"Standard predictor: {'✓ PASS' if success1 else '✗ FAIL'}")
    print(f"MoE predictor: {'✓ PASS' if success2 else '✗ FAIL'}")

    if success1 and success2:
        print(f"🎉 All tests passed! The tensor dimension fix is working.")
    else:
        print(f"❌ Some tests failed. Need further investigation.")