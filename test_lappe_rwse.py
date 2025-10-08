#!/usr/bin/env python3
"""
Unit tests for LapPE and RWSE implementation
Tests loading, concatenation, and integration with the training pipeline
"""
import torch
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_gpse_loader():
    """Test GPSEEmbeddingLoader for LapPE and RWSE"""
    print("="*70)
    print("TEST 1: GPSEEmbeddingLoader - LapPE and RWSE loading")
    print("="*70)

    from src.gpse_loader import GPSEEmbeddingLoader

    loader = GPSEEmbeddingLoader(
        gpse_base_dir='../GPSE/datasets',
        verbose=True
    )

    # Test LapPE loading
    print("\n1.1 Testing LapPE loading for Cora...")
    try:
        lappe = loader.load_lappe('Cora')
        print(f"  ✓ LapPE loaded: shape={lappe.shape}, dtype={lappe.dtype}")
        assert lappe.shape[1] in [19, 20], f"Expected 19 or 20 dimensions, got {lappe.shape[1]}"
        assert not torch.isnan(lappe).any(), "LapPE contains NaN"
        assert not torch.isinf(lappe).any(), "LapPE contains Inf"
        print(f"  ✓ LapPE validation passed")
    except Exception as e:
        print(f"  ✗ LapPE loading failed: {e}")
        return False

    # Test RWSE loading
    print("\n1.2 Testing RWSE loading for Cora...")
    try:
        rwse = loader.load_rwse('Cora')
        print(f"  ✓ RWSE loaded: shape={rwse.shape}, dtype={rwse.dtype}")
        assert rwse.shape[1] == 20, f"Expected 20 dimensions, got {rwse.shape[1]}"
        assert not torch.isnan(rwse).any(), "RWSE contains NaN"
        assert not torch.isinf(rwse).any(), "RWSE contains Inf"
        print(f"  ✓ RWSE validation passed")
    except Exception as e:
        print(f"  ✗ RWSE loading failed: {e}")
        return False

    # Test node count consistency
    print("\n1.3 Testing node count consistency...")
    assert lappe.shape[0] == rwse.shape[0], f"Node count mismatch: LapPE={lappe.shape[0]}, RWSE={rwse.shape[0]}"
    print(f"  ✓ Node counts match: {lappe.shape[0]} nodes")

    # Test caching
    print("\n1.4 Testing caching...")
    lappe2 = loader.load_lappe('Cora')
    assert torch.equal(lappe, lappe2), "Cached LapPE differs from original"
    print(f"  ✓ Caching works correctly")

    # Test case-insensitive dataset name matching
    print("\n1.5 Testing case-insensitive loading (ogbn-arxiv)...")
    try:
        lappe_arxiv = loader.load_lappe('ogbn-arxiv')
        rwse_arxiv = loader.load_rwse('ogbn-arxiv')
        print(f"  ✓ ogbn-arxiv loaded: LapPE={lappe_arxiv.shape}, RWSE={rwse_arxiv.shape}")
        assert lappe_arxiv.shape[0] == 169343, f"Expected 169343 nodes, got {lappe_arxiv.shape[0]}"
    except Exception as e:
        print(f"  ✗ ogbn-arxiv loading failed: {e}")
        return False

    print("\n✅ TEST 1 PASSED: GPSEEmbeddingLoader works correctly\n")
    return True


def test_attach_to_data():
    """Test attaching LapPE and RWSE to PyG Data objects"""
    print("="*70)
    print("TEST 2: Attaching LapPE/RWSE to PyG Data objects")
    print("="*70)

    from src.gpse_loader import GPSEEmbeddingLoader
    from torch_geometric.data import Data

    loader = GPSEEmbeddingLoader(gpse_base_dir='../GPSE/datasets', verbose=True)

    # Create dummy data object
    print("\n2.1 Creating dummy data object...")
    num_nodes = 2708  # Cora size
    data = Data(
        x=torch.randn(num_nodes, 10),
        edge_index=torch.randint(0, num_nodes, (2, 1000)),
        name='Cora'
    )
    print(f"  ✓ Created data: {num_nodes} nodes, 10 features")

    # Attach LapPE
    print("\n2.2 Attaching LapPE...")
    try:
        loader.attach_lappe_to_data(data, 'Cora')
        assert hasattr(data, 'lappe_embeddings'), "data.lappe_embeddings not found"
        assert data.lappe_embeddings.shape[0] == num_nodes, f"Node count mismatch"
        print(f"  ✓ LapPE attached: {data.lappe_embeddings.shape}")
    except Exception as e:
        print(f"  ✗ LapPE attachment failed: {e}")
        return False

    # Attach RWSE
    print("\n2.3 Attaching RWSE...")
    try:
        loader.attach_rwse_to_data(data, 'Cora')
        assert hasattr(data, 'rwse_embeddings'), "data.rwse_embeddings not found"
        assert data.rwse_embeddings.shape[0] == num_nodes, f"Node count mismatch"
        print(f"  ✓ RWSE attached: {data.rwse_embeddings.shape}")
    except Exception as e:
        print(f"  ✗ RWSE attachment failed: {e}")
        return False

    print("\n✅ TEST 2 PASSED: Attachment to PyG Data works correctly\n")
    return True


def test_batch_attach():
    """Test batch attachment using attach_gpse_embeddings function"""
    print("="*70)
    print("TEST 3: Batch attachment via attach_gpse_embeddings()")
    print("="*70)

    from src.data import attach_gpse_embeddings
    from torch_geometric.data import Data

    # Create dummy data list
    print("\n3.1 Creating dummy dataset list...")
    datasets = ['Cora', 'Citeseer', 'Pubmed']
    data_list = []
    for ds_name in datasets:
        # Use actual node counts
        node_counts = {'Cora': 2708, 'Citeseer': 3327, 'Pubmed': 19717}
        num_nodes = node_counts[ds_name]
        data = Data(
            x=torch.randn(num_nodes, 10),
            edge_index=torch.randint(0, num_nodes, (2, 100)),
            name=ds_name
        )
        data_list.append(data)
    print(f"  ✓ Created {len(data_list)} dummy datasets")

    # Test LapPE only
    print("\n3.2 Testing LapPE-only loading...")
    try:
        count = attach_gpse_embeddings(
            data_list, datasets,
            gpse_dir='../GPSE/datasets',
            verbose=True,
            use_gpse=False,
            use_lappe=True,
            use_rwse=False
        )
        assert count == 3, f"Expected 3 datasets enhanced, got {count}"
        for data in data_list:
            assert hasattr(data, 'lappe_embeddings'), f"LapPE not attached to {data.name}"
            assert not hasattr(data, 'rwse_embeddings'), f"RWSE should not be attached"
        print(f"  ✓ LapPE-only loading successful: {count}/3 datasets")
    except Exception as e:
        print(f"  ✗ LapPE-only loading failed: {e}")
        return False

    # Reset data
    for data in data_list:
        if hasattr(data, 'lappe_embeddings'):
            delattr(data, 'lappe_embeddings')

    # Test RWSE only
    print("\n3.3 Testing RWSE-only loading...")
    try:
        count = attach_gpse_embeddings(
            data_list, datasets,
            gpse_dir='../GPSE/datasets',
            verbose=True,
            use_gpse=False,
            use_lappe=False,
            use_rwse=True
        )
        assert count == 3, f"Expected 3 datasets enhanced, got {count}"
        for data in data_list:
            assert hasattr(data, 'rwse_embeddings'), f"RWSE not attached to {data.name}"
            assert not hasattr(data, 'lappe_embeddings'), f"LapPE should not be attached"
        print(f"  ✓ RWSE-only loading successful: {count}/3 datasets")
    except Exception as e:
        print(f"  ✗ RWSE-only loading failed: {e}")
        return False

    # Reset data
    for data in data_list:
        if hasattr(data, 'rwse_embeddings'):
            delattr(data, 'rwse_embeddings')

    # Test both LapPE + RWSE
    print("\n3.4 Testing LapPE + RWSE loading...")
    try:
        count = attach_gpse_embeddings(
            data_list, datasets,
            gpse_dir='../GPSE/datasets',
            verbose=True,
            use_gpse=False,
            use_lappe=True,
            use_rwse=True
        )
        assert count == 3, f"Expected 3 datasets enhanced, got {count}"
        for data in data_list:
            assert hasattr(data, 'lappe_embeddings'), f"LapPE not attached to {data.name}"
            assert hasattr(data, 'rwse_embeddings'), f"RWSE not attached to {data.name}"
        print(f"  ✓ LapPE+RWSE loading successful: {count}/3 datasets")
    except Exception as e:
        print(f"  ✗ LapPE+RWSE loading failed: {e}")
        return False

    print("\n✅ TEST 3 PASSED: Batch attachment works correctly\n")
    return True


def test_feature_concatenation():
    """Test feature concatenation in data_utils.py"""
    print("="*70)
    print("TEST 4: Feature concatenation in process_data()")
    print("="*70)

    from src.data_utils import process_data
    from torch_geometric.data import Data

    # Create test data with LapPE and RWSE
    print("\n4.1 Creating test data with LapPE and RWSE...")
    num_nodes = 100
    feat_dim = 10
    lappe_dim = 19
    rwse_dim = 20

    data = Data(
        x=torch.randn(num_nodes, feat_dim),
        edge_index=torch.randint(0, num_nodes, (2, 200)),
        y=torch.randint(0, 5, (num_nodes,)),
        name='TestDataset'
    )

    # Add LapPE and RWSE
    data.lappe_embeddings = torch.randn(num_nodes, lappe_dim)
    data.rwse_embeddings = torch.randn(num_nodes, rwse_dim)

    # Add split indices
    split_idx = {
        'train': torch.arange(60),
        'valid': torch.arange(60, 80),
        'test': torch.arange(80, 100)
    }

    print(f"  ✓ Created test data:")
    print(f"    - Original features: {feat_dim}D")
    print(f"    - LapPE: {lappe_dim}D")
    print(f"    - RWSE: {rwse_dim}D")
    print(f"    - Expected concatenated: {feat_dim + lappe_dim + rwse_dim}D")

    # Process data
    print("\n4.2 Processing data with identity projection...")
    try:
        process_data(
            data=data,
            split_idx=split_idx,
            hidden=128,  # Add required hidden parameter
            context_num=5,
            use_identity_projection=True,
            projection_small_dim=32,
            projection_large_dim=64,
            rank=0
        )

        # Check that features were processed (data is modified in-place)
        print(f"  ✓ Data processed successfully")
        print(f"    - Processed features shape: {data.x.shape}")
        print(f"    - After PCA dimension: {data.x.shape[1]}D")

        # The concatenated features should be feat_dim + lappe_dim + rwse_dim before PCA
        # After PCA to small_dim (32), the output should be 32D
        assert data.x.shape[1] == 32, f"Expected 32D output after PCA, got {data.x.shape[1]}D"
        print(f"  ✓ Output dimension correct: {data.x.shape[1]}D")

        # Check that needs_identity_projection flag is set
        assert hasattr(data, 'needs_identity_projection') and data.needs_identity_projection, "Identity projection flag not set"
        assert data.projection_target_dim == 64, f"Expected projection target 64D, got {data.projection_target_dim}D"
        print(f"  ✓ Identity projection flag set: will project to {data.projection_target_dim}D during forward pass")

    except Exception as e:
        print(f"  ✗ Data processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✅ TEST 4 PASSED: Feature concatenation works correctly\n")
    return True


def test_all_datasets():
    """Test loading all 17 datasets"""
    print("="*70)
    print("TEST 5: Loading all 17 datasets")
    print("="*70)

    from src.gpse_loader import GPSEEmbeddingLoader

    datasets = [
        'Cora', 'Citeseer', 'Pubmed', 'WikiCS', 'ogbn-arxiv',
        'CS', 'Physics', 'Computers', 'Photo', 'Flickr',
        'usa', 'brazil', 'europe', 'wiki', 'blogcatalog', 'dblp', 'FacebookPagePage'
    ]

    loader = GPSEEmbeddingLoader(gpse_base_dir='../GPSE/datasets', verbose=False)

    print(f"\n5.1 Testing LapPE for all {len(datasets)} datasets...")
    lappe_success = 0
    for ds in datasets:
        try:
            lappe = loader.load_lappe(ds)
            assert lappe.shape[1] in [19, 20], f"{ds}: wrong LapPE dimension"
            assert not torch.isnan(lappe).any(), f"{ds}: LapPE has NaN"
            lappe_success += 1
            print(f"  ✓ {ds:20s} LapPE: {lappe.shape}")
        except Exception as e:
            print(f"  ✗ {ds:20s} LapPE failed: {e}")

    print(f"\n5.2 Testing RWSE for all {len(datasets)} datasets...")
    rwse_success = 0
    for ds in datasets:
        try:
            rwse = loader.load_rwse(ds)
            assert rwse.shape[1] == 20, f"{ds}: wrong RWSE dimension"
            assert not torch.isnan(rwse).any(), f"{ds}: RWSE has NaN"
            rwse_success += 1
            print(f"  ✓ {ds:20s} RWSE: {rwse.shape}")
        except Exception as e:
            print(f"  ✗ {ds:20s} RWSE failed: {e}")

    print(f"\n5.3 Results:")
    print(f"  - LapPE: {lappe_success}/{len(datasets)} datasets")
    print(f"  - RWSE: {rwse_success}/{len(datasets)} datasets")

    if lappe_success == len(datasets) and rwse_success == len(datasets):
        print("\n✅ TEST 5 PASSED: All 17 datasets loaded successfully\n")
        return True
    else:
        print("\n⚠ TEST 5 PARTIALLY PASSED: Some datasets failed\n")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" "*15 + "LapPE & RWSE UNIT TESTS")
    print("="*70 + "\n")

    results = []

    # Run tests
    results.append(("GPSEEmbeddingLoader", test_gpse_loader()))
    results.append(("Attach to Data", test_attach_to_data()))
    results.append(("Batch Attachment", test_batch_attach()))
    results.append(("Feature Concatenation", test_feature_concatenation()))
    results.append(("All 17 Datasets", test_all_datasets()))

    # Summary
    print("="*70)
    print(" "*20 + "TEST SUMMARY")
    print("="*70)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {test_name}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print("\n" + "="*70)
    print(f"  OVERALL: {passed}/{total} tests passed")
    print("="*70 + "\n")

    return all(p for _, p in results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
