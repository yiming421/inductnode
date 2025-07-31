import torch
from torch_sparse import SparseTensor
import numpy as np


class MockData:
    """Mock data object to simulate graph data"""
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes


def test_edge_masking_step_by_step():
    """
    Test the edge masking logic step by step to understand each line
    """
    # Setup test data
    device = 'cpu'
    num_nodes = 6
    data = MockData(num_nodes)
    
    # Create positive training edges (shape: [num_pos_edges, 2])
    pos_train_edges = torch.tensor([
        [0, 1],  # edge 0
        [1, 2],  # edge 1
        [2, 3],  # edge 2
        [3, 4],  # edge 3
        [4, 5],  # edge 4
    ])
    
    # Create mapping from edge index to position in pos_train_edges
    pos_indices_map = {i: i for i in range(len(pos_train_edges))}
    
    # Initialize adjacency mask (all edges visible initially)
    pos_adjmask = torch.ones(len(pos_train_edges), dtype=torch.bool)
    
    # Simulate a batch with indices and labels
    batch_idx = torch.tensor([0, 1, 2])  # batch contains first 3 samples
    labels = torch.tensor([1, 0, 1, 0, 1])  # labels for all 5 edges (0,2,4 are positive)
    
    print("=== INITIAL STATE ===")
    print(f"pos_train_edges:\n{pos_train_edges}")
    print(f"pos_indices_map: {pos_indices_map}")
    print(f"pos_adjmask: {pos_adjmask}")
    print(f"batch_idx: {batch_idx}")
    print(f"labels: {labels}")
    
    # Start of the masking logic
    mask_target_edges = True
    
    if mask_target_edges:
        print("\n=== STEP 1: GET BATCH LABELS ===")
        # Get batch labels and find positive edges in current batch
        batch_labels_check = labels[batch_idx]
        print(f"batch_labels_check = labels[batch_idx] = {batch_labels_check}")
        
        print("\n=== STEP 2: FIND POSITIVE EDGES ===")
        batch_pos_mask = batch_labels_check == 1
        print(f"batch_pos_mask = batch_labels_check == 1 = {batch_pos_mask}")
        print(f"batch_pos_mask.any() = {batch_pos_mask.any()}")
        
        if batch_pos_mask.any():
            print("\n=== STEP 3: GET POSITIVE BATCH INDICES ===")
            # Get the actual batch indices that correspond to positive edges
            batch_pos_indices = batch_idx[batch_pos_mask]
            print(f"batch_pos_indices = batch_idx[batch_pos_mask] = {batch_pos_indices}")
            
            print("\n=== STEP 4: MAP TO POSITIONS IN POS_TRAIN_EDGES ===")
            # Map these batch indices to positions in the pos_train_edges tensor
            indices_to_mask_in_pos_list = []
            for batch_pos_idx in batch_pos_indices:
                print(f"  Checking batch_pos_idx: {batch_pos_idx.item()}")
                if batch_pos_idx.item() in pos_indices_map:
                    pos_in_list = pos_indices_map[batch_pos_idx.item()]
                    indices_to_mask_in_pos_list.append(pos_in_list)
                    print(f"    Found in pos_indices_map, position: {pos_in_list}")
            
            print(f"indices_to_mask_in_pos_list: {indices_to_mask_in_pos_list}")
            
            if indices_to_mask_in_pos_list:
                print("\n=== STEP 5: MASK TARGET EDGES ===")
                print(f"pos_adjmask before masking: {pos_adjmask}")
                pos_adjmask[indices_to_mask_in_pos_list] = False
                print(f"pos_adjmask after masking: {pos_adjmask}")
                
                print("\n=== STEP 6: BUILD GRAPH WITHOUT TARGET EDGES ===")
                # Build graph from all positive edges *not* in the current batch
                edge = pos_train_edges[pos_adjmask].t()
                print(f"Remaining edges after masking:")
                print(f"pos_train_edges[pos_adjmask] = \n{pos_train_edges[pos_adjmask]}")
                print(f"edge = pos_train_edges[pos_adjmask].t() = \n{edge}")
                
                adj_for_gnn = SparseTensor.from_edge_index(edge, sparse_sizes=(data.num_nodes, data.num_nodes)).to(device)
                adj_for_gnn = adj_for_gnn.to_symmetric().coalesce()
                print(f"Created SparseTensor with shape: {adj_for_gnn.sizes()}")
                
                print("\n=== STEP 7: RESET MASK ===")
                # Reset the mask for the next iteration
                pos_adjmask[indices_to_mask_in_pos_list] = True
                print(f"pos_adjmask after reset: {pos_adjmask}")


def test_edge_masking_assertions():
    """
    Test with assertions to verify the logic works correctly
    """
    device = 'cpu'
    num_nodes = 4
    data = MockData(num_nodes)
    
    # Simple case: 3 edges, batch contains edge 1 which is positive
    pos_train_edges = torch.tensor([[0, 1], [1, 2], [2, 3]])
    pos_indices_map = {0: 0, 1: 1, 2: 2}
    pos_adjmask = torch.ones(3, dtype=torch.bool)
    
    batch_idx = torch.tensor([0, 1, 2])  # batch contains edges 0, 1, 2  
    labels = torch.tensor([0, 1, 0])  # only edge 1 is positive
    
    # Execute the logic
    mask_target_edges = True
    if mask_target_edges:
        batch_labels_check = labels[batch_idx]
        batch_pos_mask = batch_labels_check == 1
        
        if batch_pos_mask.any():
            batch_pos_indices = batch_idx[batch_pos_mask]
            
            indices_to_mask_in_pos_list = []
            for batch_pos_idx in batch_pos_indices:
                if batch_pos_idx.item() in pos_indices_map:
                    indices_to_mask_in_pos_list.append(pos_indices_map[batch_pos_idx.item()])
            
            if indices_to_mask_in_pos_list:
                # Assertions before masking
                assert pos_adjmask.sum() == 3, "Initially all edges should be visible"
                
                pos_adjmask[indices_to_mask_in_pos_list] = False
                
                # Assertions after masking
                assert pos_adjmask.sum() == 2, "One edge should be masked"
                assert not pos_adjmask[1], "Edge 1 should be masked"
                assert pos_adjmask[0] and pos_adjmask[2], "Edges 0 and 2 should remain visible"
                
                # Check remaining edges
                remaining_edges = pos_train_edges[pos_adjmask]
                expected_remaining = torch.tensor([[0, 1], [2, 3]])
                assert torch.equal(remaining_edges, expected_remaining), "Wrong edges remaining"
                
                # Reset and check
                pos_adjmask[indices_to_mask_in_pos_list] = True
                assert pos_adjmask.sum() == 3, "All edges should be visible after reset"


def test_complex_batch_scenario():
    """
    Test a more complex scenario with multiple positive edges in batch
    """
    device = 'cpu'
    num_nodes = 8
    data = MockData(num_nodes)
    
    # 6 edges total
    pos_train_edges = torch.tensor([
        [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]
    ])
    pos_indices_map = {i: i for i in range(6)}
    pos_adjmask = torch.ones(6, dtype=torch.bool)
    
    # Batch contains first 6 samples, where samples 1 and 5 correspond to positive edges
    batch_idx = torch.tensor([0, 1, 2, 3, 4, 5])
    labels = torch.tensor([0, 1, 0, 0, 0, 1])  # samples 1 and 5 are positive
    
    print("\n=== COMPLEX BATCH SCENARIO ===")
    print(f"Batch edges: {batch_idx}")
    print(f"Labels: {labels}")
    print(f"Positive edges in batch: {batch_idx[labels == 1]}")
    
    mask_target_edges = True
    if mask_target_edges:
        batch_labels_check = labels[batch_idx]
        batch_pos_mask = batch_labels_check == 1
        
        if batch_pos_mask.any():
            batch_pos_indices = batch_idx[batch_pos_mask]
            
            indices_to_mask_in_pos_list = []
            for batch_pos_idx in batch_pos_indices:
                if batch_pos_idx.item() in pos_indices_map:
                    indices_to_mask_in_pos_list.append(pos_indices_map[batch_pos_idx.item()])
            
            print(f"Edges to mask: {indices_to_mask_in_pos_list}")
            
            if indices_to_mask_in_pos_list:
                pos_adjmask[indices_to_mask_in_pos_list] = False
                remaining_edges = pos_train_edges[pos_adjmask]
                
                print(f"Remaining edges for GNN:\n{remaining_edges}")
                
                # Verify: should have edges 0, 2, 3, 4 (not 1, 5)
                expected_remaining = torch.tensor([[0, 1], [2, 3], [3, 4], [4, 5]])
                assert torch.equal(remaining_edges, expected_remaining)
                
                pos_adjmask[indices_to_mask_in_pos_list] = True


if __name__ == "__main__":
    print("Running edge masking tests...")
    test_edge_masking_step_by_step()
    print("\n" + "="*60)
    test_edge_masking_assertions()
    print("Assertions passed!")
    print("\n" + "="*60)
    test_complex_batch_scenario()
    print("Complex scenario test passed!")
    print("\nAll tests completed successfully!")