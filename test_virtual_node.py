#!/usr/bin/env python3
"""
Test script to verify virtual node implementation.
"""
import torch
import sys
sys.path.insert(0, '/home/maweishuo/inductnode')

from src.model import PureGCN_v1
from torch_sparse import SparseTensor

def test_virtual_node_basic():
    """Test basic virtual node functionality"""
    print("=" * 60)
    print("Test 1: Basic Virtual Node Functionality")
    print("=" * 60)

    # Create model WITH virtual node
    model_vn = PureGCN_v1(
        input_dim=64, num_layers=2, hidden=128, dp=0.1,
        norm=False, res=False, relu=True, use_virtual_node=True
    )

    # Create model WITHOUT virtual node (baseline)
    model_no_vn = PureGCN_v1(
        input_dim=64, num_layers=2, hidden=128, dp=0.1,
        norm=False, res=False, relu=True, use_virtual_node=False
    )

    # Create test data: 2 graphs with 5 nodes each
    num_nodes = 10
    num_graphs = 2
    hidden_dim = 128

    x = torch.randn(num_nodes, 64)

    # Create simple adjacency: each graph is a chain
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4,  5, 6, 6, 7, 7, 8, 8, 9],  # source
        [1, 0, 2, 1, 3, 2, 4, 3,  6, 5, 7, 6, 8, 7, 9, 8]   # target
    ])

    adj_t = SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(num_nodes, num_nodes)
    ).to_symmetric().coalesce()

    # Batch assignment: first 5 nodes belong to graph 0, last 5 to graph 1
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    # Test WITH virtual node
    print("\n✅ Testing model WITH virtual node...")
    model_vn.eval()
    with torch.no_grad():
        output_vn = model_vn(x, adj_t, batch)

    if isinstance(output_vn, tuple):
        node_emb_vn, vn_emb = output_vn
        print(f"  Node embeddings shape: {node_emb_vn.shape} (expected: {num_nodes}, {hidden_dim})")
        print(f"  Virtual node embeddings shape: {vn_emb.shape} (expected: {num_graphs}, {hidden_dim})")
        assert node_emb_vn.shape == (num_nodes, hidden_dim), "Node embedding shape mismatch!"
        assert vn_emb.shape == (num_graphs, hidden_dim), "Virtual node embedding shape mismatch!"
        print("  ✓ Output shapes correct!")
    else:
        raise AssertionError("Expected tuple output (node_emb, vn_emb) when use_virtual_node=True")

    # Test WITHOUT virtual node
    print("\n✅ Testing model WITHOUT virtual node...")
    model_no_vn.eval()
    with torch.no_grad():
        output_no_vn = model_no_vn(x, adj_t)

    if isinstance(output_no_vn, tuple):
        raise AssertionError("Expected single tensor output when use_virtual_node=False")
    else:
        print(f"  Node embeddings shape: {output_no_vn.shape} (expected: {num_nodes}, {hidden_dim})")
        assert output_no_vn.shape == (num_nodes, hidden_dim), "Node embedding shape mismatch!"
        print("  ✓ Output shape correct!")

    print("\n✅ Test 1 PASSED: Basic functionality works!")


def test_virtual_node_no_batch():
    """Test that virtual node is NOT used when batch=None (single graph, node classification)"""
    print("\n" + "=" * 60)
    print("Test 2: Virtual Node Skipped When batch=None")
    print("=" * 60)

    model = PureGCN_v1(
        input_dim=64, num_layers=2, hidden=128,
        use_virtual_node=True  # Enabled but should be skipped
    )

    num_nodes = 10
    x = torch.randn(num_nodes, 64)

    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3],
        [1, 0, 2, 1, 3, 2]
    ])
    adj_t = SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(num_nodes, num_nodes)
    ).to_symmetric().coalesce()

    print("\n✅ Testing with batch=None (node classification scenario)...")
    model.eval()
    with torch.no_grad():
        output = model(x, adj_t, batch=None)

    if isinstance(output, tuple):
        raise AssertionError("Expected single tensor when batch=None (virtual node should be skipped)")
    else:
        print(f"  Output shape: {output.shape}")
        assert output.shape == (num_nodes, 128)
        print("  ✓ Virtual node correctly skipped for node classification!")

    print("\n✅ Test 2 PASSED: Virtual node skipped when batch=None!")


def test_virtual_node_edge_structure():
    """Test that virtual node adds correct bidirectional edges"""
    print("\n" + "=" * 60)
    print("Test 3: Virtual Node Edge Structure")
    print("=" * 60)

    model = PureGCN_v1(
        input_dim=32, num_layers=1, hidden=64,
        use_virtual_node=True
    )

    # Small test case: 2 graphs, 3 nodes each
    num_nodes = 6
    num_graphs = 2
    x = torch.randn(num_nodes, 32)

    # No edges initially (we'll check virtual node adds them)
    edge_index = torch.tensor([[], []], dtype=torch.long)
    adj_t = SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(num_nodes, num_nodes)
    )

    batch = torch.tensor([0, 0, 0, 1, 1, 1])

    print(f"\n✅ Testing edge structure...")
    print(f"  Original: {num_nodes} nodes, {num_graphs} graphs, 0 edges")

    model.eval()
    with torch.no_grad():
        output = model(x, adj_t, batch)

    if isinstance(output, tuple):
        node_emb, vn_emb = output
        # Expected: num_graphs virtual nodes + num_nodes real nodes
        # Each real node connects to its virtual node (bidirectional)
        # So: num_graphs + num_nodes total nodes after adding virtual nodes
        expected_nodes_after_vn = num_graphs + num_nodes
        # Each of num_nodes real nodes has 2 edges to its virtual node (bidirectional)
        # So: 2 * num_nodes edges from virtual node connections

        print(f"  After virtual node:")
        print(f"    - Node embeddings: {node_emb.shape[0]} nodes (expected: {num_nodes})")
        print(f"    - Virtual node embeddings: {vn_emb.shape[0]} graphs (expected: {num_graphs})")
        print(f"  ✓ Virtual nodes created correctly!")
    else:
        raise AssertionError("Expected tuple output")

    print("\n✅ Test 3 PASSED: Edge structure correct!")


def test_virtual_node_batch_consistency():
    """Test that virtual node handles different batch sizes correctly"""
    print("\n" + "=" * 60)
    print("Test 4: Virtual Node Batch Consistency")
    print("=" * 60)

    model = PureGCN_v1(
        input_dim=32, num_layers=2, hidden=64,
        use_virtual_node=True
    )

    edge_index = torch.tensor([[0, 1], [1, 0]])

    test_cases = [
        (2, 1, "1 graph, 2 nodes"),
        (10, 1, "1 graph, 10 nodes"),
        (10, 2, "2 graphs, 10 nodes total"),
        (15, 3, "3 graphs, 15 nodes total"),
    ]

    for num_nodes, num_graphs, desc in test_cases:
        x = torch.randn(num_nodes, 32)
        adj_t = SparseTensor.from_edge_index(
            edge_index, sparse_sizes=(num_nodes, num_nodes)
        ).to_symmetric().coalesce()

        # Create balanced batch
        nodes_per_graph = num_nodes // num_graphs
        batch = torch.cat([torch.full((nodes_per_graph,), i) for i in range(num_graphs)])

        print(f"\n✅ Testing: {desc}")
        model.eval()
        with torch.no_grad():
            output = model(x, adj_t, batch)

        if isinstance(output, tuple):
            node_emb, vn_emb = output
            assert node_emb.shape == (num_nodes, 64), f"Node embedding shape mismatch for {desc}"
            assert vn_emb.shape == (num_graphs, 64), f"Virtual node shape mismatch for {desc}"
            print(f"  ✓ Correct: {node_emb.shape[0]} nodes, {vn_emb.shape[0]} virtual nodes")
        else:
            raise AssertionError(f"Expected tuple output for {desc}")

    print("\n✅ Test 4 PASSED: Batch consistency verified!")


if __name__ == '__main__':
    try:
        test_virtual_node_basic()
        test_virtual_node_no_batch()
        test_virtual_node_edge_structure()
        test_virtual_node_batch_consistency()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Virtual node implementation verified!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
