#!/usr/bin/env python3
"""Quick test script to verify random orthogonal projection works correctly."""

import torch
import sys
sys.path.insert(0, '/home/maweishuo/inductnode/src')

from data_utils import apply_random_orthogonal_projection

def test_random_orthogonal_projection():
    """Test the random orthogonal projection function."""
    print("Testing random orthogonal projection...")

    # Test case 1: Standard case (high to low dimension)
    print("\n=== Test 1: 1000D -> 128D projection ===")
    data = torch.randn(500, 1000)  # 500 samples, 1000 features
    result = apply_random_orthogonal_projection(data, input_dim=1000, target_dim=128, seed=42, rank=0)

    assert result.shape == (500, 128), f"Expected shape (500, 128), got {result.shape}"
    print(f"✓ Shape correct: {result.shape}")
    print(f"✓ Mean: {result.mean():.4f} (should be close to 0 after normalization)")
    print(f"✓ Std: {result.std():.4f}")

    # Test case 2: GPU test
    if torch.cuda.is_available():
        print("\n=== Test 2: GPU projection ===")
        data_gpu = torch.randn(500, 1000, device='cuda')
        result_gpu = apply_random_orthogonal_projection(data_gpu, input_dim=1000, target_dim=128, seed=42, rank=0)

        assert result_gpu.shape == (500, 128), f"Expected shape (500, 128), got {result_gpu.shape}"
        assert result_gpu.device.type == 'cuda', f"Expected CUDA device, got {result_gpu.device}"
        print(f"✓ GPU projection successful: {result_gpu.shape} on {result_gpu.device}")
    else:
        print("\n=== Test 2: GPU not available, skipping GPU test ===")

    # Test case 3: Verify determinism (same seed = same result)
    print("\n=== Test 3: Determinism check ===")
    data = torch.randn(100, 500)
    result1 = apply_random_orthogonal_projection(data, input_dim=500, target_dim=64, seed=123, rank=0)
    result2 = apply_random_orthogonal_projection(data, input_dim=500, target_dim=64, seed=123, rank=0)

    assert torch.allclose(result1, result2), "Results should be identical with same seed!"
    print(f"✓ Determinism verified: same seed produces same result")

    # Test case 4: Different seeds produce different results
    result3 = apply_random_orthogonal_projection(data, input_dim=500, target_dim=64, seed=456, rank=0)
    assert not torch.allclose(result1, result3), "Different seeds should produce different results!"
    print(f"✓ Different seeds produce different results")

    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)

if __name__ == '__main__':
    test_random_orthogonal_projection()
