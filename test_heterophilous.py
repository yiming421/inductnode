#!/usr/bin/env python3
"""
Test script for HeterophilousGraphDataset support.
Tests loading all 5 heterophilous datasets without upgrading PyG.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data import load_data_train

def test_heterophilous_datasets():
    """Test loading all heterophilous datasets"""
    datasets = ['Roman-empire', 'Amazon-ratings', 'Minesweeper', 'Tolokers', 'Questions']

    print("=" * 80)
    print("Testing Heterophilous Graph Datasets")
    print("=" * 80)

    for dataset_name in datasets:
        try:
            print(f"\n{'='*80}")
            print(f"Loading {dataset_name}...")
            print(f"{'='*80}")

            data, split_idx = load_data_train(dataset_name)

            print(f"\n✓ Successfully loaded {dataset_name}!")
            print(f"  Nodes: {data.num_nodes:,}")
            print(f"  Edges: {data.num_edges:,}")
            print(f"  Features: {data.x.shape[1] if data.x is not None else 'None'}")
            print(f"  Classes: {data.y.max().item() + 1}")
            print(f"  Train nodes: {len(split_idx['train']):,}")
            print(f"  Valid nodes: {len(split_idx['valid']):,}")
            print(f"  Test nodes: {len(split_idx['test']):,}")

            # Check if splits have multiple masks (heterophilous datasets have 10 splits)
            if hasattr(data, 'train_mask') and data.train_mask.dim() > 1:
                print(f"  Number of splits: {data.train_mask.shape[1]}")

        except Exception as e:
            print(f"\n✗ Failed to load {dataset_name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("Test completed!")
    print(f"{'='*80}")

if __name__ == '__main__':
    test_heterophilous_datasets()
