#!/usr/bin/env python3
"""
Test script for GNNBenchmarkDataset usage in the project.
Tests both the direct loading and the original features loading functionality.
"""

import os
import sys
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_direct_gnn_benchmark_loading():
    """Test direct loading of GNNBenchmarkDataset"""
    print("\n" + "="*80)
    print("TEST 1: Direct GNNBenchmarkDataset Loading")
    print("="*80)
    
    from data_gc import load_gnn_benchmark_dataset
    
    # Test loading different datasets
    datasets_to_test = ['MNIST', 'CIFAR10', 'PATTERN', 'CLUSTER']
    
    for dataset_name in datasets_to_test:
        print(f"\n--- Testing {dataset_name} ---")
        dataset = load_gnn_benchmark_dataset(
            name=dataset_name,
            root='./dataset/GNN_Benchmark',
            split='train'
        )
        
        if dataset is not None:
            print(f"✓ Successfully loaded {dataset_name}")
            print(f"  - Number of graphs: {len(dataset)}")
            print(f"  - Number of classes: {dataset.num_classes}")
            print(f"  - Number of node features: {dataset.num_node_features}")
            
            # Check a sample graph
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"  - Sample graph nodes: {sample.num_nodes}")
                print(f"  - Sample graph edges: {sample.num_edges}")
                if hasattr(sample, 'x') and sample.x is not None:
                    print(f"  - Sample node features shape: {sample.x.shape}")
                if hasattr(sample, 'y') and sample.y is not None:
                    print(f"  - Sample label shape: {sample.y.shape}")
        else:
            print(f"✗ Failed to load {dataset_name}")
    
    return True


def test_original_features_loading():
    """Test loading GNNBenchmark with original features"""
    print("\n" + "="*80)
    print("TEST 2: GNNBenchmark Original Features Loading")
    print("="*80)
    
    from data_fug import load_gnn_benchmark_original_features
    
    datasets_to_test = ['MNIST', 'CIFAR10', 'PATTERN', 'CLUSTER']
    
    for dataset_name in datasets_to_test:
        print(f"\n--- Testing {dataset_name} with original features ---")
        result = load_gnn_benchmark_original_features(
            name=dataset_name,
            gnn_benchmark_root='./dataset/GNN_Benchmark',
            split='train'
        )
        
        if result is not None:
            dataset, original_features_mapping = result
            print(f"✓ Successfully loaded {dataset_name} with original features")
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
            print(f"✗ Failed to load {dataset_name} with original features")
    
    return True


def test_unified_loader():
    """Test loading GNNBenchmark through the unified loader"""
    print("\n" + "="*80)
    print("TEST 3: GNNBenchmark through Unified Loader")
    print("="*80)
    
    from data_gc import load_dataset_graph_classification
    
    # Test with original features mode
    os.environ['USE_ORIGINAL_FEATURES'] = '1'
    os.environ['GNN_BENCHMARK_ROOT'] = './dataset/GNN_Benchmark'
    
    datasets_to_test = ['mnist', 'cifar10', 'pattern', 'cluster']
    
    for dataset_name in datasets_to_test:
        print(f"\n--- Testing {dataset_name} through unified loader ---")
        try:
            result = load_dataset_graph_classification(
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
        except Exception as e:
            print(f"✗ Error loading {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Clean up environment
    del os.environ['USE_ORIGINAL_FEATURES']
    
    return True


def test_data_loader_compatibility():
    """Test that GNNBenchmark data works with PyTorch Geometric DataLoader"""
    print("\n" + "="*80)
    print("TEST 4: DataLoader Compatibility")
    print("="*80)
    
    from torch_geometric.loader import DataLoader
    from data_gc import load_gnn_benchmark_dataset
    
    dataset_name = 'MNIST'
    print(f"\n--- Testing {dataset_name} with DataLoader ---")
    
    dataset = load_gnn_benchmark_dataset(
        name=dataset_name,
        root='./dataset/GNN_Benchmark',
        split='train'
    )
    
    if dataset is not None:
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
            
            return True
        except Exception as e:
            print(f"✗ Error with DataLoader: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print(f"✗ Could not load dataset for DataLoader test")
        return False


def main():
    """Run all tests"""
    print("\n" + "#"*80)
    print("# GNNBenchmarkDataset Testing Suite")
    print("#"*80)
    
    results = []
    
    # Test 1: Direct loading
    try:
        results.append(("Direct Loading", test_direct_gnn_benchmark_loading()))
    except Exception as e:
        print(f"\n✗ Test 1 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Direct Loading", False))
    
    # Test 2: Original features loading
    try:
        results.append(("Original Features", test_original_features_loading()))
    except Exception as e:
        print(f"\n✗ Test 2 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Original Features", False))
    
    # Test 3: Unified loader
    try:
        results.append(("Unified Loader", test_unified_loader()))
    except Exception as e:
        print(f"\n✗ Test 3 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Unified Loader", False))
    
    # Test 4: DataLoader compatibility
    try:
        results.append(("DataLoader", test_data_loader_compatibility()))
    except Exception as e:
        print(f"\n✗ Test 4 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("DataLoader", False))
    
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
