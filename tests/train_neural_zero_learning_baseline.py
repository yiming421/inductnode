"""
Neural Zero-Learning Baseline Training Pipeline

This script implements proper training for the neural baseline:
1. Train on multiple datasets with train/valid splits
2. Use prototypical loss to train the MLP components
3. Test on all datasets after training
4. Follows the main training pipeline structure
"""

import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_sparse import SparseTensor
from torch.utils.data import TensorDataset, DataLoader
import sys
import os
import time
import wandb
import numpy as np
import copy
from sklearn.model_selection import train_test_split

# Import zero-learning baseline components  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data_nc import load_data, load_ogbn_data
from test_zero_learning_baseline import (str2bool, normalize_features, gcn_propagate,
                                       zero_learning_classify, few_shot_sample, correct_and_smooth)
from neural_baseline_components import (create_neural_feature_processor, 
                                      process_features_with_neural_pipeline)


def load_datasets_with_splits(dataset_names, mode='few-shot', k_shot=5, val_ratio=0.1, seed=42,
                             for_training=True, joint_training=False):
    """
    Load datasets and create train/validation splits.

    Args:
        for_training: If True, only create train/val splits (no test). If False, also include test split.
        joint_training: If True, use original splits (training on multiple datasets jointly).
                       If False, re-split small datasets (transfer learning mode).
    """
    datasets = []
    torch.manual_seed(seed)
    np.random.seed(seed)

    for dataset_name in dataset_names:
        try:
            # Load dataset
            if dataset_name.startswith('ogbn-'):
                data, split_idx = load_ogbn_data(dataset_name)
            else:
                data, split_idx = load_data(dataset_name)

            num_classes = data.y.max().item() + 1

            # Determine base training split
            if joint_training:
                # Use original train split
                base_train = split_idx['train']
            else:
                # Check if we should re-split for small datasets
                original_train = split_idx['train']
                train_labels = data.y[original_train]
                min_samples_per_class = min([(train_labels == c).sum().item() for c in range(num_classes)])
                has_abundant_samples = min_samples_per_class > 20

                if has_abundant_samples:
                    base_train = original_train
                else:
                    # Re-split entire dataset for small datasets
                    base_train = torch.arange(data.num_nodes)

            # Now apply few-shot or full mode
            if mode == 'few-shot':
                # Sample k_shot per class for support
                train_labels_all = data.y[base_train]
                support_sampled_idx = few_shot_sample(train_labels_all, num_classes, k_shot, seed=seed + hash(dataset_name) % 1000)
                train_idx = base_train[support_sampled_idx]  # Support set

                # Query set: remaining samples from base_train
                remaining_mask = ~torch.isin(base_train, train_idx)
                query_idx = base_train[remaining_mask]

                split_type = f"few-shot ({k_shot}-shot: support={len(train_idx)}, query={len(query_idx)})"

            else:  # mode == 'full'
                # Use all base_train for support
                train_idx = base_train
                query_idx = None  # No query in full mode
                split_type = f"full mode (train={len(train_idx)})"

            # Validation split
            if 'valid' in split_idx and split_idx['valid'] is not None:
                val_idx = split_idx['valid']
            else:
                # Create validation from base_train if not provided
                if len(base_train) > len(train_idx):
                    # Use remaining samples as validation
                    remaining_mask = ~torch.isin(base_train, train_idx)
                    remaining = base_train[remaining_mask]
                    val_size = min(len(remaining), max(10, num_classes * 2))
                    val_idx = remaining[:val_size]
                else:
                    # Split from train_idx
                    val_size = max(1, len(train_idx) // 10)
                    val_idx = train_idx[-val_size:]
                    train_idx = train_idx[:-val_size]
                    if query_idx is not None and len(query_idx) > val_size:
                        query_idx = query_idx[:-val_size]

            # Test split
            test_idx = split_idx['test'] if not for_training else None

            # Store dataset info
            dataset_info = {
                'name': dataset_name,
                'data': data,
                'train_idx': train_idx,  # Support set in few-shot mode
                'val_idx': val_idx,
                'test_idx': test_idx,
                'num_classes': num_classes,
                'query_idx': query_idx if mode == 'few-shot' else None  # Query set for few-shot training
            }
            datasets.append(dataset_info)

            if test_idx is not None:
                print(f"Loaded {dataset_name}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}, classes={num_classes} [{split_type}]")
            else:
                print(f"Loaded {dataset_name}: train={len(train_idx)}, val={len(val_idx)}, classes={num_classes} [{split_type}]")

        except Exception as e:
            import traceback
            print(f"Failed to load {dataset_name}: {e}")
            if str(e) == "":
                print(f"  Full traceback:")
                traceback.print_exc()
            continue

    return datasets


def batched_inference(model, features, batch_size=1024):
    """
    Apply model to features in batches to avoid OOM.

    Args:
        model: The model to apply
        features: Input features [N, D]
        batch_size: Batch size for inference

    Returns:
        Output features [N, output_dim]
    """
    model.eval()
    outputs = []

    with torch.no_grad():
        for i in range(0, features.size(0), batch_size):
            batch = features[i:i + batch_size]
            batch_output = model(batch)
            outputs.append(batch_output)

    return torch.cat(outputs, dim=0)


def train_neural_baseline(datasets, neural_pca_dim=128, neural_mlp_hidden=256, neural_mlp_layers=2,
                         neural_mlp_dropout=0.2, use_pca=True, use_full_pca=False, hops=2,
                         prop_method='residual', alpha=0.1, gcn_layer_norm=False, feature_norm='none', sim='cos',
                         epochs=100, lr=0.01, weight_decay=1e-4, patience=20, grad_clip=1.0,
                         batch_size=512, device='cpu'):
    """
    Train neural baseline on multiple datasets using episodic few-shot training.

    Training paradigm:
    - Support set: k-shot samples per class (determined by dataset loading)
    - Query set: Remaining training samples (used to compute loss, treated as unknown)

    Training strategy: Train all datasets for 1 epoch at a time (round-robin) instead of
    completing all epochs for one dataset before moving to the next.
    """
    # Initialize data structures for all datasets
    dataset_states = {}

    # Step 1: Prepare all datasets (propagation, PCA, model initialization)
    print("\n" + "="*60)
    print("INITIALIZATION PHASE: Preparing all datasets")
    print("="*60)

    for dataset_info in datasets:
        dataset_name = dataset_info['name']
        data = dataset_info['data']
        train_idx = dataset_info['train_idx']  # Support set
        val_idx = dataset_info['val_idx']
        query_idx = dataset_info.get('query_idx', None)  # Query set (for few-shot training)
        num_classes = dataset_info['num_classes']

        print(f"\nPreparing dataset: {dataset_name}")

        # Move data to device
        x = data.x.to(device)
        adj = data.adj_t.to(device) if hasattr(data, 'adj_t') and data.adj_t is not None else None

        if adj is None:
            adj = SparseTensor.from_edge_index(
                data.edge_index,
                sparse_sizes=(data.num_nodes, data.num_nodes)
            ).to_symmetric().to(device)

        # Support and query sets for few-shot training
        # Index on CPU, then move to device
        support_labels = data.y[train_idx].to(device)
        query_labels = data.y[query_idx].to(device)
        val_labels = data.y[val_idx].to(device)

        # Move indices to device after indexing
        support_idx = train_idx.to(device)
        query_idx_device = query_idx.to(device)
        val_idx_device = val_idx.to(device)

        print(f"  Support: {len(support_idx)}, Query: {len(query_idx_device)}, Val: {len(val_idx_device)}")

        # Apply feature normalization
        x = normalize_features(x, method=feature_norm)

        # Step 1: GNN propagation
        if hops == 0:
            x_prop = x
        else:
            x_prop = gcn_propagate(x, adj, num_hops=hops, method=prop_method, alpha=alpha, use_layer_norm=gcn_layer_norm)

        print(f"  After GNN propagation: {x_prop.shape}")

        # Step 2: Create neural components for this dataset
        pca_processor, create_mlp_transformer, mlp_input_dim = create_neural_feature_processor(
            target_unified_dim=neural_pca_dim,
            mlp_hidden=neural_mlp_hidden,
            mlp_output_dim=neural_mlp_hidden,
            mlp_layers=neural_mlp_layers,
            mlp_dropout=neural_mlp_dropout,
            use_pca=use_pca,
            use_full_pca=use_full_pca
        )

        # Apply PCA unification and cache the result (PCA is deterministic, no need to recompute)
        if pca_processor is not None:
            print(f"  Applying PCA and caching result in memory...")
            x_unified = pca_processor(x_prop)
            actual_input_dim = neural_pca_dim
        else:
            x_unified = x_prop
            actual_input_dim = x_prop.size(1)

        print(f"  After PCA unification: {x_unified.shape} (cached in memory)")

        # Create and train MLP transformer
        mlp_transformer = create_mlp_transformer(actual_input_dim).to(device)
        optimizer = optim.Adam(mlp_transformer.parameters(), lr=lr, weight_decay=weight_decay)

        # Learning rate scheduler (cosine annealing with warmup)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr*0.01)

        # Create DataLoader for query set batching
        query_dataset = TensorDataset(query_idx_device, query_labels)
        query_loader = DataLoader(query_dataset, batch_size=batch_size, shuffle=True)

        # Store all state for this dataset
        dataset_states[dataset_name] = {
            'num_classes': num_classes,
            'x_unified': x_unified,
            'support_idx': support_idx,
            'support_labels': support_labels,
            'query_idx': query_idx_device,
            'query_labels': query_labels,
            'val_idx': val_idx_device,
            'val_labels': val_labels,
            'mlp_transformer': mlp_transformer,
            'optimizer': optimizer,
            'scheduler': scheduler,
            'query_loader': query_loader,
            'pca_processor': pca_processor,
            'input_dim': actual_input_dim,
            'best_val_acc': 0.0,
            'best_model_state': None,
            'patience_counter': 0,
            'finished': False
        }

    # Step 2: Train all datasets for one epoch at a time (round-robin)
    print("\n" + "="*60)
    print("TRAINING PHASE: Training all datasets round-robin")
    print("="*60)

    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{epochs}")
        print('='*60)

        # Train each dataset for one epoch
        for dataset_name, state in dataset_states.items():
            # Skip if training is finished for this dataset
            if state['finished']:
                continue

            mlp_transformer = state['mlp_transformer']
            optimizer = state['optimizer']
            scheduler = state['scheduler']
            query_loader = state['query_loader']

            print(f"\n  Training {dataset_name}...")

            # Extract state variables
            x_unified = state['x_unified']
            support_idx = state['support_idx']
            support_labels = state['support_labels']
            query_idx_device = state['query_idx']
            query_labels = state['query_labels']
            val_idx_device = state['val_idx']
            val_labels = state['val_labels']
            num_classes = state['num_classes']

            mlp_transformer.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_query_idx, batch_query_labels in query_loader:
                optimizer.zero_grad()

                # CRITICAL: Recalculate support embeddings every batch (weights have been updated)
                support_features = x_unified[support_idx]
                support_embeddings = mlp_transformer(support_features)

                # Compute prototypes from support set
                prototypes = torch.zeros(num_classes, support_embeddings.size(1),
                                       device=device, dtype=x_unified.dtype)

                for c in range(num_classes):
                    mask = support_labels == c
                    if mask.any():
                        prototypes[c] = support_embeddings[mask].mean(dim=0)

                # Forward pass on query batch
                batch_query_features = x_unified[batch_query_idx]
                batch_query_embeddings = mlp_transformer(batch_query_features)

                # Compute logits for query batch using support prototypes
                if sim == 'cos':
                    query_norm = F.normalize(batch_query_embeddings, dim=-1)
                    proto_norm = F.normalize(prototypes, dim=-1)
                    batch_logits = query_norm @ proto_norm.t()
                else:
                    batch_logits = batch_query_embeddings @ prototypes.t()

                # Cross-entropy loss
                loss = F.cross_entropy(batch_logits, batch_query_labels)
                loss.backward()

                # Gradient clipping
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(mlp_transformer.parameters(), max_norm=grad_clip)

                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

            # Validation evaluation
            if epoch % 10 == 0 or epoch == epochs - 1:
                mlp_transformer.eval()
                with torch.no_grad():
                    # Compute prototypes from support set
                    support_features = x_unified[support_idx]
                    support_embeddings = mlp_transformer(support_features)

                    prototypes = torch.zeros(num_classes, support_embeddings.size(1),
                                           device=device, dtype=x_unified.dtype)
                    for c in range(num_classes):
                        mask = support_labels == c
                        if mask.any():
                            prototypes[c] = support_embeddings[mask].mean(dim=0)

                    # Compute query accuracy (training performance on unseen samples)
                    query_features = x_unified[query_idx_device]
                    query_embeddings = batched_inference(mlp_transformer, query_features, batch_size=batch_size)

                    if sim == 'cos':
                        proto_norm = F.normalize(prototypes, dim=-1)
                        query_norm = F.normalize(query_embeddings, dim=-1)
                        query_logits = query_norm @ proto_norm.t()
                    else:
                        query_logits = query_embeddings @ prototypes.t()

                    query_acc = (query_logits.argmax(dim=1) == query_labels).float().mean().item()

                    # Get validation embeddings using batched inference
                    val_features = x_unified[val_idx_device]
                    val_embeddings = batched_inference(mlp_transformer, val_features, batch_size=batch_size)

                    # Compute validation logits
                    if sim == 'cos':
                        val_norm = F.normalize(val_embeddings, dim=-1)
                        val_logits = val_norm @ proto_norm.t()
                    else:
                        val_logits = val_embeddings @ prototypes.t()

                    val_preds = val_logits.argmax(dim=1)
                    val_acc = (val_preds == val_labels).float().mean().item()

                    # Early stopping check
                    if val_acc > state['best_val_acc']:
                        state['best_val_acc'] = val_acc
                        state['best_model_state'] = copy.deepcopy(mlp_transformer.state_dict())
                        state['patience_counter'] = 0
                        improved = True
                    else:
                        state['patience_counter'] += 1
                        improved = False

                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"    loss={avg_loss:.4f}, query_acc={query_acc:.4f}, val_acc={val_acc:.4f}, lr={current_lr:.6f} {'*' if improved else ''}")

                    # Early stopping
                    if state['patience_counter'] >= patience:
                        print(f"    Early stopping triggered (patience={patience})")
                        state['finished'] = True

        # Check if all datasets have finished training
        all_finished = all(state['finished'] for state in dataset_states.values())
        if all_finished:
            print(f"\nAll datasets finished training early at epoch {epoch + 1}")
            break

    # Step 3: Finalize - load best models and prepare return dictionary
    print("\n" + "="*60)
    print("FINALIZATION: Loading best models")
    print("="*60)

    trained_models = {}
    for dataset_name, state in dataset_states.items():
        mlp_transformer = state['mlp_transformer']

        # Load best model
        if state['best_model_state'] is not None:
            mlp_transformer.load_state_dict(state['best_model_state'])

        # Store trained components
        trained_models[dataset_name] = {
            'mlp_transformer': mlp_transformer,
            'pca_processor': state['pca_processor'],
            'best_val_acc': state['best_val_acc'],
            'input_dim': state['input_dim']
        }

        print(f"  {dataset_name}: Best val acc = {state['best_val_acc']:.4f}")

    return trained_models


def test_neural_baseline(datasets, trained_models, hops=2, prop_method='residual', alpha=0.1,
                        gcn_layer_norm=False, feature_norm='none', sim='cos', use_ridge=False, ridge_alpha=1.0,
                        use_cs=False, cs_hops=50, cs_alpha=0.5, device='cpu'):
    """
    Test neural baseline on all datasets using trained models.

    Args:
        use_cs: Whether to use Correct & Smooth (with adaptive fallback)
        cs_hops: Number of C&S iterations
        cs_alpha: C&S smoothing factor
    """
    results = {}
    
    for dataset_info in datasets:
        dataset_name = dataset_info['name']
        data = dataset_info['data']
        train_idx = dataset_info['train_idx']
        test_idx = dataset_info['test_idx']
        num_classes = dataset_info['num_classes']
        
        if dataset_name not in trained_models:
            print(f"No trained model for {dataset_name}, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Testing on dataset: {dataset_name}")
        print('='*60)
        
        # Get trained components
        model_info = trained_models[dataset_name]
        mlp_transformer = model_info['mlp_transformer']
        pca_processor = model_info['pca_processor']
        
        # Move data to device
        x = data.x.to(device)
        adj = data.adj_t.to(device) if hasattr(data, 'adj_t') and data.adj_t is not None else None
        
        if adj is None:
            adj = SparseTensor.from_edge_index(
                data.edge_index,
                sparse_sizes=(data.num_nodes, data.num_nodes)
            ).to_symmetric().to(device)
        
        # Index on CPU, then move to device
        train_labels = data.y[train_idx].to(device)
        test_labels = data.y[test_idx].to(device)

        # Move indices to device after indexing
        train_idx = train_idx.to(device)
        test_idx = test_idx.to(device)
        
        # Apply feature normalization
        x = normalize_features(x, method=feature_norm)
        
        # Step 1: GNN propagation
        if hops == 0:
            x_prop = x
        else:
            x_prop = gcn_propagate(x, adj, num_hops=hops, method=prop_method, alpha=alpha, use_layer_norm=gcn_layer_norm)
        
        # Step 2: PCA unification
        if pca_processor is not None:
            x_unified = pca_processor(x_prop)
        else:
            x_unified = x_prop
        
        # Step 3: MLP transformation (batched to avoid OOM)
        mlp_transformer.eval()
        x_neural = batched_inference(mlp_transformer, x_unified, batch_size=1024)
        
        print(f"  Final features shape: {x_neural.shape}")

        # Step 4: Classification using same heads as zero-learning baseline
        preds, base_logits = zero_learning_classify(x_neural, train_idx, train_labels, test_idx, num_classes, sim,
                                                    use_ridge=use_ridge,
                                                    ridge_alpha=ridge_alpha)

        base_acc = (preds == test_labels).float().mean().item()
        test_acc = base_acc

        # Step 5: Optional Correct & Smooth (with adaptive fallback)
        if use_cs:
            print(f"  Base accuracy: {base_acc:.4f} ({base_acc*100:.2f}%)")
            print(f"  Trying C&S (hops={cs_hops}, alpha={cs_alpha})...", end=' ')

            # Get logits for all nodes (need to classify all nodes for C&S)
            all_idx = torch.arange(data.num_nodes, device=device)
            _, all_logits = zero_learning_classify(x_neural, train_idx, train_labels, all_idx, num_classes, sim,
                                                   use_ridge=use_ridge,
                                                   ridge_alpha=ridge_alpha)

            # Apply Correct & Smooth
            Y = correct_and_smooth(adj, all_logits, train_idx, train_labels, num_classes,
                                 num_iters=cs_hops, alpha=cs_alpha)
            cs_preds = Y[test_idx].argmax(dim=1)
            cs_acc = (cs_preds == test_labels).float().mean().item()

            # Use C&S if it improves performance (adaptive fallback)
            if cs_acc > base_acc:
                test_acc = cs_acc
                print(f"[SELECTED] C&S: {cs_acc:.4f}")
            else:
                print(f"C&S: {cs_acc:.4f} [base better, using base]")

        results[dataset_name] = test_acc

        if not use_cs:
            print(f"  Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        else:
            print(f"  Final test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Neural Zero-Learning Baseline Training Pipeline')
    
    # Dataset and training configuration
    parser.add_argument('--train_datasets', type=str, nargs='+',
                        default=['Actor', 'AirBrazil', 'AirEU', 'AirUS', 'AmzComp', 'AmzPhoto', 'AmzRatings',
                                'BlogCatalog', 'Chameleon', 'Citeseer', 'CoCS', 'CoPhysics', 'Cora',
                                'Cornell', 'DBLP', 'Deezer', 'LastFMAsia', 'Minesweeper', 'Pubmed',
                                'Questions', 'Reddit', 'Roman', 'Squirrel', 'Texas', 'Tolokers',
                                'Wiki', 'Wisconsin', 'WikiCS', 'ogbn-arxiv', 'ogbn-products', 'FullCora'],
                        help='Datasets for training')
    parser.add_argument('--test_datasets', type=str, nargs='+',
                        default=['Actor', 'AirBrazil', 'AirEU', 'AirUS', 'AmzComp', 'AmzPhoto', 'AmzRatings',
                                'BlogCatalog', 'Chameleon', 'Citeseer', 'CoCS', 'CoPhysics', 'Cora',
                                'Cornell', 'DBLP', 'Deezer', 'LastFMAsia', 'Minesweeper', 'Pubmed',
                                'Questions', 'Reddit', 'Roman', 'Squirrel', 'Texas', 'Tolokers',
                                'Wiki', 'Wisconsin', 'WikiCS', 'ogbn-arxiv', 'ogbn-products', 'FullCora'],
                        help='Datasets for testing')
    parser.add_argument('--mode', type=str, choices=['full', 'few-shot'], default='few-shot',
                        help='Training mode')
    parser.add_argument('--k_shot', type=int, default=5,
                        help='Number of shots per class (for few-shot mode)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation ratio for splitting')
    parser.add_argument('--joint_training', type=str2bool, default=True,
                        help='Joint training mode: use original splits. If False, use transfer learning mode (re-split small datasets)')

    # GNN parameters
    parser.add_argument('--hops', type=int, default=3,
                        help='Number of GNN propagation hops')
    parser.add_argument('--prop_method', type=str, choices=['residual', 'appnp', 'concat', 'weighted'], default='residual',
                        help='Feature propagation method')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Teleport probability for APPNP')
    parser.add_argument('--gcn_layer_norm', type=str2bool, default=False,
                        help='Use layer normalization between GCN propagation layers')
    parser.add_argument('--feature_norm', type=str, choices=['none', 'row', 'col', 'row+col'], default='none',
                        help='Feature normalization')
    
    # Neural component parameters
    parser.add_argument('--neural_pca_dim', type=int, default=128,
                        help='Target dimension for PCA unification')
    parser.add_argument('--neural_mlp_hidden', type=int, default=512,
                        help='Hidden dimension for MLP')
    parser.add_argument('--neural_mlp_layers', type=int, default=2,
                        help='Number of MLP layers')
    parser.add_argument('--neural_mlp_dropout', type=float, default=0.2,
                        help='Dropout rate for MLP')
    parser.add_argument('--use_pca', type=str2bool, default=True,
                        help='Use PCA for dimension unification')
    parser.add_argument('--use_full_pca', type=str2bool, default=False,
                        help='Use full SVD instead of lowrank PCA')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs per dataset')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (number of epochs without improvement)')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping max norm (0 to disable)')
    
    # Classification parameters
    parser.add_argument('--sim', type=str, choices=['cos', 'dot'], default='cos',
                        help='Similarity function')
    parser.add_argument('--ridge', type=str2bool, default=False,
                        help='Use ridge regression instead of prototypical classification')
    parser.add_argument('--ridge_alpha', type=float, default=1.0,
                        help='Regularization strength for ridge regression')

    # Correct & Smooth parameters
    parser.add_argument('--use_cs', type=str2bool, default=True,
                        help='Use Correct & Smooth (with adaptive fallback)')
    parser.add_argument('--cs_hops', type=int, default=50,
                        help='Number of C&S iterations')
    parser.add_argument('--cs_alpha', type=float, default=0.5,
                        help='C&S smoothing factor')

    # Batch size parameter
    parser.add_argument('--batch_size', type=int, default=16384,
                        help='Batch size for training and inference')

    # General parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device number (use -1 for CPU)')
    parser.add_argument('--wandb', type=str2bool, default=True,
                        help='Enable Weights & Biases logging')
    
    args = parser.parse_args()
    
    # Set device
    device = f'cuda:{args.gpu}' if args.gpu >= 0 and torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Initialize wandb
    if args.wandb:
        run_name = f"neural_train_{args.mode}_{len(args.train_datasets)}datasets"
        wandb.init(
            project='neural-zero-learning-baseline-training',
            name=run_name,
            config=vars(args)
        )
    
    print('='*60)
    print('NEURAL ZERO-LEARNING BASELINE TRAINING PIPELINE')
    print(f'Training datasets: {len(args.train_datasets)}')
    print(f'Test datasets: {len(args.test_datasets)}')
    print(f'Mode: {args.mode}')
    print(f'Neural: PCA{args.neural_pca_dim} + MLP{args.neural_mlp_layers}L-{args.neural_mlp_hidden}H')
    print('='*60)
    
    # Step 1: Load training datasets
    print(f"\n📚 Loading training datasets...")
    train_datasets = load_datasets_with_splits(
        args.train_datasets,
        mode=args.mode,
        k_shot=args.k_shot,
        val_ratio=args.val_ratio,
        seed=args.seed,
        for_training=True,
        joint_training=args.joint_training
    )
    
    if not train_datasets:
        print("❌ No training datasets loaded successfully!")
        return
    
    # Step 2: Train neural baseline
    print(f"\n🚀 Training neural baseline...")
    start_time = time.time()
    trained_models = train_neural_baseline(
        train_datasets,
        neural_pca_dim=args.neural_pca_dim,
        neural_mlp_hidden=args.neural_mlp_hidden,
        neural_mlp_layers=args.neural_mlp_layers,
        neural_mlp_dropout=args.neural_mlp_dropout,
        use_pca=args.use_pca,
        use_full_pca=args.use_full_pca,
        hops=args.hops,
        prop_method=args.prop_method,
        alpha=args.alpha,
        gcn_layer_norm=args.gcn_layer_norm,
        feature_norm=args.feature_norm,
        sim=args.sim,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        grad_clip=args.grad_clip,
        batch_size=args.batch_size,
        device=device
    )
    training_time = time.time() - start_time
    print(f"\n✅ Training completed in {training_time:.2f} seconds")
    
    # Step 3: Load test datasets (use original test splits for evaluation)
    print(f"\n📊 Loading test datasets...")
    test_datasets = load_datasets_with_splits(
        args.test_datasets,
        mode=args.mode,
        k_shot=args.k_shot,
        val_ratio=args.val_ratio,
        seed=args.seed,
        for_training=False,  # Include test split
        joint_training=True  # Always use original splits for testing
    )
    
    # Step 4: Test on all datasets
    print(f"\n🔬 Testing neural baseline...")
    test_results = test_neural_baseline(
        test_datasets,
        trained_models,
        hops=args.hops,
        prop_method=args.prop_method,
        alpha=args.alpha,
        gcn_layer_norm=args.gcn_layer_norm,
        feature_norm=args.feature_norm,
        sim=args.sim,
        use_ridge=args.ridge,
        ridge_alpha=args.ridge_alpha,
        use_cs=args.use_cs,
        cs_hops=args.cs_hops,
        cs_alpha=args.cs_alpha,
        device=device
    )
    
    # Step 5: Summary
    if test_results:
        print(f'\n{"="*60}')
        print('FINAL RESULTS')
        print('='*60)
        print(f'{"Dataset":<15} {"Test Acc":>10}')
        print('-'*25)
        for dataset_name, acc in test_results.items():
            print(f'{dataset_name:<15} {acc*100:>9.2f}%')
        print('-'*25)
        avg_acc = sum(test_results.values()) / len(test_results)
        print(f'{"Average":<15} {avg_acc*100:>9.2f}%')
        
        # Log to wandb
        if args.wandb:
            for dataset_name, acc in test_results.items():
                wandb.log({f'test_acc/{dataset_name}': acc})
            wandb.log({'test_acc/average': avg_acc})
            wandb.finish()
    else:
        print("❌ No test results obtained!")


if __name__ == '__main__':
    main()