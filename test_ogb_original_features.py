"""
Test script to verify that all OGB molecular datasets are supported with original features.
"""
import os
import sys

# Set environment variable to use original features
os.environ['USE_ORIGINAL_FEATURES'] = '1'
os.environ['OGB_ROOT'] = './dataset/ogb'

from src.data_graph import load_dataset

# List of datasets to test
datasets_to_test = [
    'bace',
    'bbbp',
    'tox21',
    'clintox',
    'muv',
    'sider',
    'toxcast'
]

print("=" * 80)
print("Testing OGB Datasets with Original Features (9-dim)")
print("=" * 80)

results = {}

for dataset_name in datasets_to_test:
    print(f"\n{'='*80}")
    print(f"Testing: {dataset_name}")
    print(f"{'='*80}")

    try:
        result = load_dataset(dataset_name, root='./dataset')

        if result is None:
            print(f"❌ FAILED: {dataset_name} - returned None")
            results[dataset_name] = 'FAILED: returned None'
            continue

        # Check if it's a tuple (dataset, mapping)
        if isinstance(result, tuple) and len(result) == 2:
            dataset, mapping = result

            # Verify mapping structure
            if 'node_embs' in mapping and 'node_index_mapping' in mapping:
                node_embs = mapping['node_embs']
                print(f"✅ SUCCESS: {dataset_name}")
                print(f"   - Graphs: {len(dataset)}")
                print(f"   - Node embeddings shape: {node_embs.shape}")
                print(f"   - Feature dimension: {node_embs.shape[1]} (should be 9)")
                print(f"   - Uses original features: {mapping.get('uses_original_features', False)}")

                if node_embs.shape[1] == 9:
                    results[dataset_name] = 'SUCCESS (9-dim)'
                else:
                    results[dataset_name] = f'WARNING: dimension is {node_embs.shape[1]}, expected 9'
            else:
                print(f"❌ FAILED: {dataset_name} - invalid mapping structure")
                results[dataset_name] = 'FAILED: invalid mapping'
        else:
            print(f"❌ FAILED: {dataset_name} - unexpected return format")
            results[dataset_name] = 'FAILED: unexpected format'

    except Exception as e:
        print(f"❌ FAILED: {dataset_name} - Exception: {e}")
        results[dataset_name] = f'FAILED: {str(e)}'

# Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
for dataset_name, result in results.items():
    status_icon = "✅" if result.startswith("SUCCESS") else "❌"
    print(f"{status_icon} {dataset_name}: {result}")

# Count successes
successes = sum(1 for r in results.values() if r.startswith("SUCCESS"))
print(f"\n{successes}/{len(datasets_to_test)} datasets loaded successfully")
