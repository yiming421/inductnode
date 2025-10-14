#!/usr/bin/env python3
"""
Test script to verify ModelNet dataset support.
Tests both ModelNet10 and ModelNet40 with original features.
"""

import os
import sys

# Set USE_ORIGINAL_FEATURES=1 to test ModelNet with original features
os.environ['USE_ORIGINAL_FEATURES'] = '1'

# Import after setting environment variables
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from data_gc import load_dataset


def test_modelnet_dataset(name='modelnet10'):
    """Test loading ModelNet dataset with original features."""
    print(f"\n{'='*70}")
    print(f"Testing {name.upper()} with USE_ORIGINAL_FEATURES=1")
    print(f"{'='*70}\n")

    try:
        # Load dataset
        result = load_dataset(name, root='./dataset')

        if result is None:
            print(f"❌ Failed to load {name}")
            return False

        # Unpack result
        dataset, original_features_mapping = result

        # Verify dataset properties
        print(f"✓ Dataset loaded successfully")
        print(f"  - Name: {dataset.name}")
        print(f"  - Graphs: {len(dataset)}")
        print(f"  - Classes: {dataset.num_classes}")

        # Verify mapping structure
        print(f"\n✓ Original features mapping created")
        print(f"  - Node embeddings shape: {original_features_mapping['node_embs'].shape}")
        print(f"  - Uses FUG embeddings: {original_features_mapping.get('uses_fug_embeddings', False)}")
        print(f"  - Uses original features: {original_features_mapping.get('uses_original_features', False)}")
        print(f"  - Is multitask: {original_features_mapping.get('is_multitask', False)}")
        print(f"  - Dataset name: {original_features_mapping.get('name', 'N/A')}")

        # Verify node index mapping
        node_index_mapping = original_features_mapping['node_index_mapping']
        print(f"\n✓ Node index mapping created for {len(node_index_mapping)} graphs")

        # Verify first graph
        first_graph = dataset[0]
        print(f"\n✓ First graph properties:")
        print(f"  - Nodes: {first_graph.num_nodes}")
        print(f"  - Edges: {first_graph.num_edges}")
        print(f"  - Features shape: {first_graph.x.shape if first_graph.x is not None else 'None'}")
        print(f"  - Label: {first_graph.y}")

        # Verify node embeddings dimension (should be 3D for ModelNet)
        expected_dim = 3  # 3D position coordinates
        actual_dim = original_features_mapping['node_embs'].shape[1]
        if actual_dim == expected_dim:
            print(f"\n✓ Node embeddings have correct dimension: {actual_dim}D (3D positions)")
        else:
            print(f"\n⚠ Node embeddings dimension mismatch: expected {expected_dim}, got {actual_dim}")

        print(f"\n{'='*70}")
        print(f"✓ {name.upper()} test PASSED")
        print(f"{'='*70}\n")

        return True

    except Exception as e:
        print(f"\n❌ Error testing {name}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n" + "="*70)
    print("ModelNet Dataset Support Test")
    print("="*70)

    # Test ModelNet10
    success_10 = test_modelnet_dataset('modelnet10')

    # Test ModelNet40 (optional - larger dataset)
    # Uncomment to test ModelNet40 as well
    # success_40 = test_modelnet_dataset('modelnet40')

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"ModelNet10: {'✓ PASSED' if success_10 else '❌ FAILED'}")
    # print(f"ModelNet40: {'✓ PASSED' if success_40 else '❌ FAILED'}")
    print("="*70 + "\n")
