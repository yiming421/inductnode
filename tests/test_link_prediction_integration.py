import torch
import torch.nn as nn
import pytest
from torch_sparse import SparseTensor
from torch_geometric.data import Data

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Fix the relative import issue by importing utils first
import utils
from model import PFNPredictorNodeCls, PureGCN_v1, AttentionPool, MLP
from engine_link_pred import train_link_prediction, evaluate_link_prediction, get_node_embeddings, get_link_prototypes


class TestLinkPredictionIntegration:
    @pytest.fixture
    def setup_data(self):
        """Create test data for link prediction"""
        torch.manual_seed(42)
        
        # Create a small graph
        num_nodes = 20
        hidden_dim = 32
        
        # Node features
        x = torch.randn(num_nodes, hidden_dim)
        
        # Create edges (adjacency matrix)
        edge_index = torch.tensor([
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        ])
        
        # Convert to SparseTensor
        adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes))
        adj_t = adj_t.to_symmetric().coalesce()
        
        # Create graph data
        data = Data(x=x, adj_t=adj_t, num_nodes=num_nodes)
        
        # Add required attributes for PFN predictor (simulating node classification format)
        # For link prediction, we treat each context edge as a "node" with binary labels
        data.y = torch.tensor([0, 1])  # Binary classes: 0=neg, 1=pos
        data.context_sample = torch.arange(4)  # Indices for context edges (will be overridden in predictor)
        
        # Create training edges (positive and negative)
        train_pos_edges = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7]])
        train_neg_edges = torch.tensor([[0, 2], [1, 3], [4, 6], [5, 7]])
        
        train_edge_pairs = torch.cat([train_pos_edges, train_neg_edges], dim=0)
        train_labels = torch.cat([torch.ones(4, dtype=torch.long), torch.zeros(4, dtype=torch.long)], dim=0)
        
        train_edges = {
            'edge_pairs': train_edge_pairs,
            'labels': train_labels
        }
        
        # Create context edges (smaller subset)
        context_pos_edges = torch.tensor([[8, 9], [10, 11]])
        context_neg_edges = torch.tensor([[8, 10], [9, 11]])
        
        context_edge_pairs = torch.cat([context_pos_edges, context_neg_edges], dim=0)
        context_labels = torch.cat([torch.ones(2, dtype=torch.long), torch.zeros(2, dtype=torch.long)], dim=0)
        
        context_edges = {
            'edge_pairs': context_edge_pairs,
            'labels': context_labels
        }
        
        # Create test edges
        test_pos_edges = torch.tensor([[12, 13], [14, 15]])
        test_neg_edges = torch.tensor([[12, 14], [13, 15]])
        
        test_edge_pairs = torch.cat([test_pos_edges, test_neg_edges], dim=0)
        test_labels = torch.cat([torch.ones(2, dtype=torch.long), torch.zeros(2, dtype=torch.long)], dim=0)
        
        test_edges = {
            'edge_pairs': test_edge_pairs,
            'labels': test_labels
        }
        
        # Create train mask (all True for this test)
        train_mask = torch.ones(len(train_labels), dtype=torch.bool)
        
        return data, train_edges, context_edges, test_edges, train_mask, hidden_dim
    
    @pytest.fixture
    def setup_models(self, setup_data):
        """Create model components"""
        data, train_edges, context_edges, test_edges, train_mask, hidden_dim = setup_data
        
        # Create model components
        model = PureGCN_v1(input_dim=hidden_dim, hidden=hidden_dim, num_layers=2)
        predictor = PFNPredictorNodeCls(hidden_dim=hidden_dim, nhead=1, num_layers=1)
        att = AttentionPool(in_channels=hidden_dim, out_channels=hidden_dim, nhead=1)
        mlp = MLP(in_channels=hidden_dim, hidden_channels=hidden_dim, out_channels=hidden_dim, num_layers=2)
        
        return model, predictor, att, mlp
    
    def test_get_node_embeddings(self, setup_data, setup_models):
        """Test node embedding generation"""
        data, _, _, _, _, hidden_dim = setup_data
        model, _, _, _ = setup_models
        
        embeddings = get_node_embeddings(model, data)
        
        assert embeddings.shape == (data.num_nodes, hidden_dim)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()
    
    def test_get_link_prototypes(self, setup_data, setup_models):
        """Test link prototype generation"""
        data, _, context_edges, _, _, hidden_dim = setup_data
        model, _, att, mlp = setup_models
        
        node_embeddings = get_node_embeddings(model, data)
        prototypes = get_link_prototypes(node_embeddings, context_edges, att, mlp)
        
        assert prototypes is not None
        assert prototypes.shape == (2, hidden_dim)  # [neg_proto, pos_proto]
        assert not torch.isnan(prototypes).any()
        assert not torch.isinf(prototypes).any()
    
    def test_predictor_forward(self, setup_data, setup_models):
        """Test that PFNPredictorNodeCls works correctly for link prediction"""
        data, train_edges, context_edges, _, _, hidden_dim = setup_data
        model, predictor, att, mlp = setup_models
        
        # Get embeddings
        node_embeddings = get_node_embeddings(model, data)
        
        # Get context edge embeddings
        context_edge_pairs = context_edges['edge_pairs']
        context_labels = context_edges['labels']
        context_src_embeds = node_embeddings[context_edge_pairs[:, 0]]
        context_dst_embeds = node_embeddings[context_edge_pairs[:, 1]]
        context_edge_embeds = context_src_embeds * context_dst_embeds
        
        # Get target edge embeddings
        target_edge_pairs = train_edges['edge_pairs'][:2]  # Test with 2 edges
        target_src_embeds = node_embeddings[target_edge_pairs[:, 0]]
        target_dst_embeds = node_embeddings[target_edge_pairs[:, 1]]
        target_edge_embeds = target_src_embeds * target_dst_embeds
        
        # Get prototypes
        link_prototypes = get_link_prototypes(node_embeddings, context_edges, att, mlp)
        
        # Create data adapter for PFN predictor (designed for node classification)
        # - Treats edge labels as "node" labels for pooling
        # - Treats all context edges as "context samples"
        predictor_data = Data(x=data.x, adj_t=data.adj_t, num_nodes=data.num_nodes)
        predictor_data.y = context_labels  # Use context labels as the "node" labels
        predictor_data.context_sample = torch.arange(len(context_labels))  # All context edges are "context samples"
        
        # Test predictor forward pass
        scores = predictor(predictor_data, context_edge_embeds, target_edge_embeds, context_labels, link_prototypes)
        
        assert scores.shape == (2, 2)  # [num_targets, num_classes]
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()
        
        # Test that scores are reasonable (not all zeros)
        assert scores.abs().sum() > 0
    
    def test_train_link_prediction(self, setup_data, setup_models):
        """Test training function"""
        data, train_edges, context_edges, _, train_mask, _ = setup_data
        model, predictor, att, mlp = setup_models
        
        # Create optimizer
        optimizer = torch.optim.Adam([
            *model.parameters(),
            *predictor.parameters(),
            *att.parameters(),
            *mlp.parameters()
        ], lr=0.001)
        
        # Test training
        loss = train_link_prediction(
            model=model,
            predictor=predictor,
            data=data,
            train_edges=train_edges,
            context_edges=context_edges,
            train_mask=train_mask,
            optimizer=optimizer,
            batch_size=4,
            att=att,
            mlp=mlp,
            mask_target_edges=False
        )
        
        assert isinstance(loss, float)
        assert not torch.isnan(torch.tensor(loss))
        assert not torch.isinf(torch.tensor(loss))
        assert loss >= 0  # Loss should be non-negative
    
    def test_test_link_prediction(self, setup_data, setup_models):
        """Test testing function with Hits@K metric"""
        data, _, context_edges, test_edges, _, _ = setup_data
        model, predictor, att, mlp = setup_models
        
        # Test evaluation with Hits@K
        results = evaluate_link_prediction(
            model=model,
            predictor=predictor,
            data=data,
            test_edges=test_edges,
            context_edges=context_edges,
            batch_size=4,
            att=att,
            mlp=mlp,
            k_values=[20, 50]  # Use smaller K values for testing
        )
        
        assert isinstance(results, dict)
        assert 'hits@20' in results
        assert 'hits@50' in results
        
        for k, hits_k in results.items():
            assert isinstance(hits_k, float)
            assert 0 <= hits_k <= 1  # Hits@K should be between 0 and 1
            assert not torch.isnan(torch.tensor(hits_k))
            assert not torch.isinf(torch.tensor(hits_k))
    
    def test_edge_masking(self, setup_data, setup_models):
        """Test edge masking functionality"""
        data, train_edges, context_edges, _, train_mask, _ = setup_data
        model, predictor, att, mlp = setup_models
        
        # Create optimizer
        optimizer = torch.optim.Adam([
            *model.parameters(),
            *predictor.parameters(),
            *att.parameters(),
            *mlp.parameters()
        ], lr=0.001)
        
        # Test training with edge masking
        loss_with_masking = train_link_prediction(
            model=model,
            predictor=predictor,
            data=data,
            train_edges=train_edges,
            context_edges=context_edges,
            train_mask=train_mask,
            optimizer=optimizer,
            batch_size=4,
            att=att,
            mlp=mlp,
            mask_target_edges=True
        )
        
        assert isinstance(loss_with_masking, float)
        assert not torch.isnan(torch.tensor(loss_with_masking))
        assert not torch.isinf(torch.tensor(loss_with_masking))
        assert loss_with_masking >= 0
    
    def test_full_pipeline(self, setup_data, setup_models):
        """Test the complete training and testing pipeline"""
        data, train_edges, context_edges, test_edges, train_mask, _ = setup_data
        model, predictor, att, mlp = setup_models
        
        # Create optimizer
        optimizer = torch.optim.Adam([
            *model.parameters(),
            *predictor.parameters(),
            *att.parameters(),
            *mlp.parameters()
        ], lr=0.001)
        
        # Train for a few steps
        initial_results = evaluate_link_prediction(
            model=model,
            predictor=predictor,
            data=data,
            test_edges=test_edges,
            context_edges=context_edges,
            batch_size=4,
            att=att,
            mlp=mlp,
            k_values=[20, 50]  # Use smaller K values for testing
        )
        
        # Training step
        for _ in range(3):
            loss = train_link_prediction(
                model=model,
                predictor=predictor,
                data=data,
                train_edges=train_edges,
                context_edges=context_edges,
                train_mask=train_mask,
                optimizer=optimizer,
                batch_size=4,
                att=att,
                mlp=mlp
            )
            assert loss >= 0
        
        # Test after training
        final_results = evaluate_link_prediction(
            model=model,
            predictor=predictor,
            data=data,
            test_edges=test_edges,
            context_edges=context_edges,
            batch_size=4,
            att=att,
            mlp=mlp,
            k_values=[20, 50]  # Use smaller K values for testing
        )
        
        # Both results should be valid
        for k, hits_k in initial_results.items():
            assert 0 <= hits_k <= 1
        for k, hits_k in final_results.items():
            assert 0 <= hits_k <= 1
        
        print(f"Initial Hits@20: {initial_results['hits@20']:.4f}")
        print(f"Final Hits@20: {final_results['hits@20']:.4f}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 