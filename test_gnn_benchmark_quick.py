#!/usr/bin/env python3
"""
Quick test script for GNNBenchmarkDataset usage - tests MNIST only.
"""

import os
import sys
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    print("\n" + "#"*80)
    print("# Quick GNNBenchmarkDataset Test (MNIST only)")
    print("#"*80)
    
    # Test 1: Direct loading
    print("\n" + "="*80)
    print("TEST 1: Direct Loading")
    print("="*80)
    
    from data_gc import load_gnn_benchmark_dataset
    
    dataset = load_gnn_benchmark_dataset(
        name='MNIST',
        root='./dataset/GNN_Benchmark',
        split='train'
    )
    
    if dataset is not None:
        print(f"✓ Successfully loaded MNIST")
        print(f"  - Graphs: {len(dataset)}, Classes: {dataset.num_classes}, Features: {dataset.num_node_features}")
        sample = dataset[0]
        print(f"  - Sample: {sample.num_nodes} nodes, {sample.num_edges} edges")
        print(f"  - Features shape: {sample.x.shape}, Label: {sample.y.item()}")
    else:
        print("✗ Failed to load MNIST")
        return False
    
    # Test 2: Original features loading
    print("\n" + "="*80)
    print("TEST 2: Original Features Loading")
    print("="*80)
    
    from data_fug import load_gnn_benchmark_original_features
    
    result = load_gnn_benchmark_original_features(
        name='MNIST',
        gnn_benchmark_root='./dataset/GNN_Benchmark',
        split='train'
    )
    
    if result is not None:
        dataset, mapping = result
        print(f"✓ Successfully loaded MNIST with original features")
        print(f"  - Graphs: {len(dataset)}")
        print(f"  - Node embeddings: {mapping['node_embs'].shape}")
        print(f"  - Uses original features: {mapping['uses_original_features']}")
        
        # Test mapping
        sample_graph = dataset[0]
        sample_indices = mapping['node_index_mapping'][0]
        sample_embs = mapping['node_embs'][sample_indices]
        print(f"  - Sample mapping: {sample_graph.num_nodes} nodes -> embeddings {sample_embs.shape}")
        
        # Verify match
        if torch.allclose(sample_embs, sample_graph.x.float()):
            print(f"  ✓ Embeddings match original features!")
        else:
            print(f"  ✗ Warning: Embeddings don't match")
    else:
        print("✗ Failed to load with original features")
        return False
    
    # Test 3: DataLoader compatibility
    print("\n" + "="*80)
    print("TEST 3: DataLoader Compatibility")
    print("="*80)
    
    from torch_geometric.loader import DataLoader
    
    try:
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        batch = next(iter(loader))
        
        print(f"✓ Successfully created DataLoader")
        print(f"  - Batch: {batch.num_graphs} graphs, {batch.num_nodes} nodes")
        print(f"  - Features: {batch.x.shape}, Labels: {batch.y.shape}")
    except Exception as e:
        print(f"✗ DataLoader error: {e}")
        return False
    
    # Test 4: Unified loader with original features
    print("\n" + "="*80)
    print("TEST 4: Unified Loader (with USE_ORIGINAL_FEATURES=1)")
    print("="*80)
    
    os.environ['USE_ORIGINAL_FEATURES'] = '1'
    os.environ['GNN_BENCHMARK_ROOT'] = './dataset/GNN_Benchmark'
    
    from data_gc import load_dataset
    
    try:
        result = load_dataset(
            name='mnist',
            root='./dataset',
            embedding_family='ST'
        )
        
        if isinstance(result, tuple) and len(result) == 2:
            dataset, mapping = result
            print(f"✓ Unified loader returned dataset with mapping")
            print(f"  - Graphs: {len(dataset)}")
            print(f"  - External embeddings: {mapping['node_embs'].shape}")
        else:
            dataset = result
            print(f"✓ Unified loader returned dataset")
            print(f"  - Graphs: {len(dataset)}")
    except Exception as e:
        print(f"✗ Unified loader error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        del os.environ['USE_ORIGINAL_FEATURES']
    
    print("\n" + "="*80)
    print("✓ All tests PASSED!")
    print("="*80)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
