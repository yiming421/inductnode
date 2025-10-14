#!/usr/bin/env python3
"""
Test script for MNISTSuperpixels dataset usage in the project.
Tests the original features loading functionality.

NOTE: MNISTSuperpixels only supports USE_ORIGINAL_FEATURES=1 mode,
similar to TU and GNN Benchmark datasets.
"""

import os
import sys
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def test_original_features_loading():
    """Test loading MNISTSuperpixels with original features"""
    print("\n" + "="*80)
    print("TEST 1: MNISTSuperpixels Original Features Loading")
    print("="*80)
    
    from data_fug import load_mnist_superpixels_original_features
    
    for train in [True, False]:
        split_name = 'train' if train else 'test'
        print(f"\n--- Testing MNISTSuperpixels {split_name} with original features ---")
        result = load_mnist_superpixels_original_features(
            root='./dataset/MNISTSuperpixels',
            train=train
        )
        
        if result is not None:
            dataset, original_features_mapping = result
            print(f"✓ Successfully loaded MNISTSuperpixels {split_name} with original features")
            print(f"  - Number of graphs: {len(dataset)}")
            
            # Check the mapping
            print(f"  - Node embeddings shape: {original_features_mapping['node_embs'].shape}")
            print(f"  - Uses FUG embeddings: {original_features_mapping['uses_fug_embeddings']}")
            print(f"  - Uses original features: {original_features_mapping['uses_original_features']}")
            print(f"  - Is multitask: {original_features_mapping['is_multitask']}")
            print(f"  - Number of graphs with mappings: {len(original_features_mapping['node_index_mapping'])}")
            
            # Check a sample mapping
            if len(dataset) > 0:
                sample_idx = 0
                sample_graph = dataset[sample_idx]
                sample_node_indices = original_features_mapping['node_index_mapping'][sample_idx]
                print(f"  - Sample graph {sample_idx} has {sample_graph.num_nodes} nodes")
                print(f"  - Mapped node indices: {sample_node_indices[:5]}... (showing first 5)")
                
                # Verify the mapping works
                sample_embeddings = original_features_mapping['node_embs'][sample_node_indices]
                print(f"  - Retrieved embeddings shape: {sample_embeddings.shape}")
                print(f"  - Expected: ({sample_graph.num_nodes}, {dataset.num_node_features})")
        else:
            print(f"✗ Failed to load MNISTSuperpixels {split_name} with original features")
    
    return True


def test_unified_loader():
    """Test loading MNISTSuperpixels through the unified loader"""
    print("\n" + "="*80)
    print("TEST 2: MNISTSuperpixels through Unified Loader")
    print("="*80)
    
    from data_gc import load_dataset
    
    # Test with original features mode (REQUIRED for MNISTSuperpixels)
    os.environ['USE_ORIGINAL_FEATURES'] = '1'
    os.environ['MNIST_SUPERPIXELS_ROOT'] = './dataset/MNISTSuperpixels'
    
    dataset_name = 'mnistsuperpixels'
    
    print(f"\n--- Testing {dataset_name} through unified loader ---")
    try:
        result = load_dataset(
            name=dataset_name,
            root='./dataset',
            embedding_family=None
        )
        
        if result is not None:
            if isinstance(result, tuple) and len(result) == 2:
                dataset, mapping = result
                print(f"✓ Successfully loaded {dataset_name} with mapping")
                print(f"  - Number of graphs: {len(dataset)}")
                if 'node_embs' in mapping:
                    print(f"  - External node embeddings: {mapping['node_embs'].shape}")
            else:
                dataset = result
                print(f"✓ Successfully loaded {dataset_name}")
                print(f"  - Number of graphs: {len(dataset)}")
                if hasattr(dataset, 'node_embs'):
                    print(f"  - Node embeddings: {dataset.node_embs.shape}")
        else:
            print(f"✗ Failed to load {dataset_name}")
            return False
    except Exception as e:
        print(f"✗ Error loading {dataset_name}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up environment
        if 'USE_ORIGINAL_FEATURES' in os.environ:
            del os.environ['USE_ORIGINAL_FEATURES']
    
    return True


def test_data_loader_compatibility():
    """Test that MNISTSuperpixels data works with PyTorch Geometric DataLoader"""
    print("\n" + "="*80)
    print("TEST 3: DataLoader Compatibility")
    print("="*80)
    
    from torch_geometric.loader import DataLoader
    from data_fug import load_mnist_superpixels_original_features
    
    dataset_name = 'MNISTSuperpixels'
    print(f"\n--- Testing {dataset_name} with DataLoader ---")
    
    result = load_mnist_superpixels_original_features(
        root='./dataset/MNISTSuperpixels',
        train=True
    )
    
    if result is None:
        print(f"✗ Could not load dataset for DataLoader test")
        return False
    
    dataset, mapping = result
    
    try:
        # Create a DataLoader
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Try to get one batch
        batch = next(iter(loader))
        
        print(f"✓ Successfully created DataLoader and retrieved batch")
        print(f"  - Batch size: {batch.num_graphs}")
        print(f"  - Total nodes in batch: {batch.num_nodes}")
        print(f"  - Total edges in batch: {batch.num_edges}")
        if hasattr(batch, 'x') and batch.x is not None:
            print(f"  - Batch node features shape: {batch.x.shape}")
        if hasattr(batch, 'y') and batch.y is not None:
            print(f"  - Batch labels shape: {batch.y.shape}")
            print(f"  - Sample labels: {batch.y[:10].flatten()}")
        
        return True
    except Exception as e:
        print(f"✗ Error with DataLoader: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graph_classification_integration():
    """Test integration with graph classification pipeline"""
    print("\n" + "="*80)
    print("TEST 4: Graph Classification Pipeline Integration")
    print("="*80)
    
    from data_gc import (
        load_dataset, 
        create_dataset_splits, 
        prepare_graph_data_for_pfn,
        process_graph_features
    )
    
    print(f"\n--- Testing full pipeline with MNISTSuperpixels ---")
    
    # Enable original features mode (REQUIRED for MNISTSuperpixels)
    os.environ['USE_ORIGINAL_FEATURES'] = '1'
    os.environ['MNIST_SUPERPIXELS_ROOT'] = './dataset/MNISTSuperpixels'
    
    try:
        # Load dataset
        dataset_name = 'mnistsuperpixels'
        print(f"1. Loading dataset...")
        result = load_dataset(dataset_name, root='./dataset')
        
        if isinstance(result, tuple):
            dataset, fug_mapping = result
            print(f"✓ Dataset loaded with external mapping: {len(dataset)} graphs")
        else:
            dataset = result
            fug_mapping = None
            print(f"✓ Dataset loaded: {len(dataset)} graphs")
        
        if dataset is None:
            print("✗ Failed to load dataset")
            return False
        
        # Create splits (use small subset for testing)
        print(f"\n2. Creating dataset splits...")
        split_idx = create_dataset_splits(
            dataset, 
            dataset_name,
            root='./dataset',
            train_ratio=0.01,  # Use only 1% for quick testing
            val_ratio=0.005,
            test_ratio=0.005,
            seed=42
        )
        print(f"✓ Splits created - Train: {len(split_idx['train'])}, "
              f"Val: {len(split_idx['val'])}, Test: {len(split_idx['test'])}")
        
        # Prepare data for PFN
        print(f"\n3. Preparing data for PFN (context sampling)...")
        processed_data = prepare_graph_data_for_pfn(
            dataset,
            split_idx,
            context_k=5,  # Small context for testing
            device='cpu'  # Use CPU for testing
        )
        
        # Add FUG mapping if available
        if fug_mapping is not None:
            processed_data['fug_mapping'] = fug_mapping
        
        print(f"✓ Data prepared - Multitask: {processed_data['is_multitask']}, "
              f"Classes: {processed_data['num_classes']}")
        
        # Process features with PCA
        print(f"\n4. Processing graph features with PCA...")
        processing_info = process_graph_features(
            dataset,
            hidden_dim=128,
            device='cpu',
            use_identity_projection=False,
            pca_device='cpu',
            dataset_name=dataset_name,
            processed_data=processed_data
        )
        print(f"✓ Features processed - Original dim: {processing_info['original_dim']}, "
              f"Processed dim: {processing_info['processed_dim']}")
        
        print(f"\n✓ Full pipeline test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up environment
        if 'USE_ORIGINAL_FEATURES' in os.environ:
            del os.environ['USE_ORIGINAL_FEATURES']


def main():
    """Run all tests"""
    print("\n" + "#"*80)
    print("# MNISTSuperpixels Dataset Testing Suite")
    print("# NOTE: MNISTSuperpixels only supports USE_ORIGINAL_FEATURES=1 mode")
    print("#"*80)
    
    results = []
    
    # Test 1: Original features loading
    try:
        results.append(("Original Features", test_original_features_loading()))
    except Exception as e:
        print(f"\n✗ Test 1 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Original Features", False))
    
    # Test 2: Unified loader
    try:
        results.append(("Unified Loader", test_unified_loader()))
    except Exception as e:
        print(f"\n✗ Test 2 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Unified Loader", False))
    
    # Test 3: DataLoader compatibility
    try:
        results.append(("DataLoader", test_data_loader_compatibility()))
    except Exception as e:
        print(f"\n✗ Test 3 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("DataLoader", False))
    
    # Test 4: Pipeline integration
    try:
        results.append(("Pipeline Integration", test_graph_classification_integration()))
    except Exception as e:
        print(f"\n✗ Test 4 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Pipeline Integration", False))
    
    # Summary
    print("\n" + "#"*80)
    print("# Test Summary")
    print("#"*80)
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    print("\n" + "="*80)
    if all_passed:
        print("All tests PASSED! ✓")
    else:
        print("Some tests FAILED! ✗")
    print("="*80 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
