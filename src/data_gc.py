from .data_features import load_ogb_original_features, load_tu_original_features
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
import numpy as np
import os
import random
import time
from pathlib import Path

def load_dataset(name):
    """Load numerical features for the molecular corpus or PROTEINS."""
    if name == 'PROTEINS':
        result = load_tu_original_features(name, tu_root=os.environ.get('TU_ROOT', './dataset/TU'))
    else:
        result = load_ogb_original_features(name, ogb_root=os.environ.get('OGB_ROOT', './dataset/ogb'))
    if result is None:
        raise RuntimeError(f'Could not load graph dataset {name}')
    result[0].name = name
    return result

def load_precomputed_splits(dataset_name, root='./dataset'):
    tag_dataset_root = str('')
    if tag_dataset_root and os.path.isdir(tag_dataset_root):
        root = tag_dataset_root
    split_path = os.path.join(root, dataset_name, 'e2e_graph_splits.pt')
    if not os.path.exists(split_path):
        print(f'No pre-computed splits found for {dataset_name} at {split_path}')
        return None
    try:
        splits = torch.load(split_path, map_location='cpu')
        split_key = f'{dataset_name}_e2e_graph'
        if split_key not in splits:
            print(f'Split key {split_key} not found in splits file')
            return None
        dataset_splits = splits[split_key]
        split_idx = {'train': np.array(dataset_splits['train']), 'val': np.array(dataset_splits['valid']), 'test': np.array(dataset_splits['test'])}
        print(f"Loaded pre-computed splits for {dataset_name} - Train: {len(split_idx['train'])}, Val: {len(split_idx['val'])}, Test: {len(split_idx['test'])}")
        return split_idx
    except Exception as e:
        print(f'Error loading pre-computed splits for {dataset_name}: {e}')
        return None

def create_scaffold_splits(dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42, pretraining_mode=False):
    """
    Create scaffold splits for molecular datasets (fallback when no pre-computed splits).
    TODO: Implement scaffold splitting using RDKit.
    
    For now, falls back to random splitting with warning.
    """
    print(f'WARNING: Scaffold splitting not yet implemented, using random splits for {dataset.name}')
    return create_random_splits(dataset, train_ratio, val_ratio, test_ratio, seed, pretraining_mode)

def create_random_splits(dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42, pretraining_mode=False):
    if pretraining_mode:
        assert abs(train_ratio + val_ratio - 1.0) < 1e-06, 'For pretraining mode, train_ratio + val_ratio must sum to 1.0'
    else:
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-06, 'Ratios must sum to 1.0'
    np.random.seed(seed)
    random.seed(seed)
    indices = np.random.permutation(len(dataset))
    train_end = int(len(dataset) * train_ratio)
    if pretraining_mode:
        val_end = len(dataset)
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        print(f'Pretraining split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}')
        return {'train': torch.tensor(train_idx), 'val': torch.tensor(val_idx)}
    else:
        val_end = int(len(dataset) * (train_ratio + val_ratio))
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        print(f'Random split sizes - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}')
        return {'train': torch.tensor(train_idx), 'val': torch.tensor(val_idx), 'test': torch.tensor(test_idx)}

def create_dataset_splits(dataset, dataset_name, root='./dataset', train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42, pretraining_mode=False):
    if pretraining_mode:
        train_ratio = 0.8
        val_ratio = 0.2
        test_ratio = 0.0
        print(f'Pretraining mode: using {train_ratio:.1%} train / {val_ratio:.1%} val split (no test)')
    if hasattr(dataset, 'get_idx_split'):
        try:
            ogb_splits = dataset.get_idx_split()
            split_idx = {'train': ogb_splits['train'].numpy() if hasattr(ogb_splits['train'], 'numpy') else ogb_splits['train'], 'val': ogb_splits['valid'].numpy() if hasattr(ogb_splits['valid'], 'numpy') else ogb_splits['valid'], 'test': ogb_splits['test'].numpy() if hasattr(ogb_splits['test'], 'numpy') else ogb_splits['test']}
            if pretraining_mode and 'test' in split_idx:
                print(f'[OGB] Merging test set into training for pretraining')
                train_tensor = torch.tensor(split_idx['train']) if isinstance(split_idx['train'], np.ndarray) else split_idx['train']
                test_tensor = torch.tensor(split_idx['test']) if isinstance(split_idx['test'], np.ndarray) else split_idx['test']
                val_tensor = torch.tensor(split_idx['val']) if isinstance(split_idx['val'], np.ndarray) else split_idx['val']
                combined_train = torch.cat([train_tensor, test_tensor])
                split_idx = {'train': combined_train, 'val': val_tensor}
            print(f"[OGB] Using official OGB splits for {dataset_name} - Train: {len(split_idx['train'])}, Val: {len(split_idx['val'])}" + (f", Test: {len(split_idx['test'])}" if 'test' in split_idx else ''))
            return split_idx
        except Exception as e:
            print(f'[OGB] Failed to get official OGB splits for {dataset_name}: {e}')
    split_idx = load_precomputed_splits(dataset_name, root)
    if split_idx is not None:
        if pretraining_mode and 'test' in split_idx:
            print(f'Merging test set into training for pretraining')
            train_tensor = torch.tensor(split_idx['train']) if isinstance(split_idx['train'], np.ndarray) else split_idx['train']
            test_tensor = torch.tensor(split_idx['test']) if isinstance(split_idx['test'], np.ndarray) else split_idx['test']
            val_tensor = torch.tensor(split_idx['val']) if isinstance(split_idx['val'], np.ndarray) else split_idx['val']
            combined_train = torch.cat([train_tensor, test_tensor])
            split_idx = {'train': combined_train, 'val': val_tensor}
        return split_idx
    molecular_datasets = {'bace', 'bbbp', 'hiv', 'chemhiv', 'ogbg-molhiv', 'pcba', 'chempcba', 'molpcba', 'ogbg-molpcba', 'muv', 'ogbg-molmuv', 'tox21', 'ogbg-moltox21', 'toxcast', 'ogbg-moltoxcast', 'clintox', 'ogbg-molclintox', 'sider', 'ogbg-molsider', 'chemblpre', 'ogbg-molbace', 'ogbg-molbbbp'}
    if dataset_name.lower() in molecular_datasets:
        return create_scaffold_splits(dataset, train_ratio, val_ratio, test_ratio, seed, pretraining_mode)
    print(f'Using random splits for {dataset_name}')
    return create_random_splits(dataset, train_ratio, val_ratio, test_ratio, seed, pretraining_mode)

def prepare_graph_data_for_pfn(dataset, split_idx, context_k=32, device='cuda'):
    sample_graph = dataset[0]
    is_multitask = sample_graph.y.numel() > 1
    if is_multitask:
        import time
        start_time = time.time()
        num_tasks = sample_graph.y.numel()
        print(f'Starting smart context sampling for {num_tasks} tasks...')
        sampling_start = time.time()
        task_contexts = {}
        reservoirs = {}
        counts = {}
        for task_idx in range(num_tasks):
            task_contexts[task_idx] = {0: [], 1: []}
            reservoirs[task_idx] = {0: [], 1: []}
            counts[task_idx] = {0: 0, 1: 0}
        print(f"  Vectorized sampling for {num_tasks} tasks across {len(split_idx['train'])} graphs...")
        train_indices = split_idx['train']
        train_indices_tensor = torch.as_tensor(train_indices, dtype=torch.long)
        num_train_samples = len(train_indices_tensor)
        train_labels = torch.empty((num_train_samples, num_tasks), dtype=torch.long)
        train_masks = torch.empty((num_train_samples, num_tasks), dtype=torch.bool)
        for i, idx in enumerate(train_indices_tensor):
            graph = dataset[idx.item()]
            if not hasattr(graph, 'task_mask'):
                if graph.y.dtype.is_floating_point:
                    mask = ~torch.isnan(graph.y)
                else:
                    mask = torch.ones_like(graph.y, dtype=torch.bool)
            else:
                mask = graph.task_mask
            train_masks[i] = mask.view(-1).bool()
            train_labels[i] = torch.where(mask.view(-1).bool(), graph.y.view(-1), -1)
        print('Pre-fetching complete.')
        reservoirs = [[[] for _ in range(2)] for _ in range(num_tasks)]
        for task_idx in range(num_tasks):
            for class_label in [0, 1]:
                mask_is_valid = train_masks[:, task_idx]
                label_is_correct = train_labels[:, task_idx] == class_label
                final_mask = mask_is_valid & label_is_correct
                relative_indices = torch.where(final_mask)[0]
                candidate_indices = train_indices_tensor[relative_indices].numpy()
                if len(candidate_indices) == 0:
                    continue
                if len(candidate_indices) <= context_k:
                    reservoirs[task_idx][class_label] = candidate_indices.tolist()
                else:
                    sampled_indices = np.random.choice(candidate_indices, size=context_k, replace=False)
                    reservoirs[task_idx][class_label] = sampled_indices.tolist()
        sampling_time = time.time() - sampling_start
        for task_idx in range(num_tasks):
            for class_label in [0, 1]:
                task_contexts[task_idx][class_label] = reservoirs[task_idx][class_label]
        dedup_start = time.time()
        unique_indices = set()
        for task_data in task_contexts.values():
            for class_data in task_data.values():
                unique_indices.update(class_data)
        print(f'Deduplication: {len(unique_indices)} unique graphs out of {num_tasks * 2 * context_k} total samples')
        gpu_graph_cache = {}
        for idx in unique_indices:
            idx_int = idx.item() if torch.is_tensor(idx) else idx
            gpu_graph_cache[idx] = dataset[idx_int]
        for task_idx in task_contexts:
            for class_label in task_contexts[task_idx]:
                indices = task_contexts[task_idx][class_label]
                graphs = [gpu_graph_cache[idx] for idx in indices]
                task_contexts[task_idx][class_label] = {'graphs': graphs, 'indices': indices}
        dedup_time = time.time() - dedup_start
        total_time = time.time() - start_time
        print(f'Context sampling completed in {total_time:.2f}s:')
        print(f'  - Lazy sampling: {sampling_time:.2f}s')
        print(f'  - Deduplication: {dedup_time:.2f}s')
        print(f'  - Memory efficiency: {(1 - len(unique_indices) / (num_tasks * 2 * context_k)) * 100:.1f}% reduction')
        final_context_graphs = task_contexts
        context_labels = {task_idx: {0: 0, 1: 1} for task_idx in range(num_tasks)}
        is_multitask_dataset = True
    else:
        import time
        start_time = time.time()
        print('Starting context sampling for single-task dataset...')

        def reservoir_sample_for_class(class_label, k):
            """Reservoir sampling for specific class in single-task dataset"""
            reservoir = []
            count = 0
            for idx in split_idx['train']:
                idx_int = idx.item() if torch.is_tensor(idx) else idx
                graph = dataset[idx_int]
                if graph.y.numel() == 1:
                    label = graph.y.item()
                else:
                    label = graph.y[0].item()
                if label == class_label:
                    count += 1
                    if len(reservoir) < k:
                        reservoir.append(idx)
                    else:
                        j = np.random.randint(0, count)
                        if j < k:
                            reservoir[j] = idx
            return reservoir
        all_labels_in_train = []
        for idx in split_idx['train']:
            idx_int = idx.item() if torch.is_tensor(idx) else idx
            graph = dataset[idx_int]
            label = graph.y.item() if graph.y.numel() == 1 else graph.y[0].item()
            all_labels_in_train.append(label)
        available_classes = sorted(set(all_labels_in_train))
        print(f'  Detected {len(available_classes)} classes in training set: {available_classes}')
        task_contexts = {0: {}}
        unique_indices = set()
        for class_label in available_classes:
            selected_indices = reservoir_sample_for_class(class_label, context_k)
            if len(selected_indices) < context_k and len(selected_indices) > 0:
                while len(selected_indices) < context_k:
                    selected_indices.append(np.random.choice(selected_indices))
            task_contexts[0][class_label] = selected_indices
            unique_indices.update(selected_indices)
        gpu_graph_cache = {}
        for idx in unique_indices:
            idx_int = idx.item() if torch.is_tensor(idx) else idx
            gpu_graph_cache[idx] = dataset[idx_int]
        for class_label in task_contexts[0]:
            indices = task_contexts[0][class_label]
            graphs = [gpu_graph_cache[idx] for idx in indices]
            task_contexts[0][class_label] = {'graphs': graphs, 'indices': indices}
        total_time = time.time() - start_time
        total_samples = sum((len(context_data['indices']) for context_data in task_contexts[0].values()))
        print(f'Single-task context sampling completed in {total_time:.2f}s')
        print(f'  - Classes found: {list(available_classes)}')
        print(f'  - Unique graphs: {len(unique_indices)} out of {total_samples} samples')
        print(f'  - Memory efficiency: {(1 - len(unique_indices) / max(1, total_samples)) * 100:.1f}% reduction')
        final_context_graphs = task_contexts
        context_labels = {0: {class_label: class_label for class_label in available_classes}}
        is_multitask_dataset = False
    ret = {'context_graphs': final_context_graphs, 'context_labels': context_labels, 'split_idx': split_idx, 'dataset': dataset, 'num_classes': dataset.num_classes, 'num_features': dataset.num_node_features, 'is_multitask': is_multitask_dataset, 'num_tasks': num_tasks if is_multitask_dataset else 1, 'name': dataset.name}
    return ret

def create_graph_batch(graphs, device='cuda'):
    batch = Batch.from_data_list(graphs).to(device)
    return batch

def load_all_graph_datasets(dataset_names, device='cuda', pretraining_mode=False, context_k=32, embedding_family='ST'):
    import time
    overall_start = time.time()
    datasets = []
    processed_data_list = []
    for name in dataset_names:
        print(f'\nLoading dataset: {name}')
        load_start = time.time()
        result = load_dataset(name)
        load_time = time.time() - load_start
        if result is None:
            print(f'Skipping {name} due to loading error')
            continue
        elif isinstance(result, tuple) and len(result) == 2:
            dataset, feature_mapping = result
            print(f'  Numerical dataset loaded with external mapping')
        else:
            dataset = result
            feature_mapping = None
        print(f'  Dataset loading: {load_time:.2f}s')
        split_start = time.time()
        split_idx = create_dataset_splits(dataset, name, pretraining_mode=pretraining_mode)
        split_time = time.time() - split_start
        print(f'  Split creation: {split_time:.2f}s')
        prep_start = time.time()
        processed_data = prepare_graph_data_for_pfn(dataset, split_idx, device=device, context_k=context_k)
        if feature_mapping is not None:
            processed_data['feature_mapping'] = feature_mapping
            print(f'  feature mapping attached to processed data')
        prep_time = time.time() - prep_start
        print(f'  Context preparation: {prep_time:.2f}s')
        datasets.append(dataset)
        processed_data_list.append(processed_data)
        total_dataset_time = time.time() - load_start
        print(f'Successfully processed {name} with {len(dataset)} graphs (total: {total_dataset_time:.2f}s)')
    overall_time = time.time() - overall_start
    print(f'\nAll datasets loaded in {overall_time:.2f}s')
    return (datasets, processed_data_list)

def create_task_filtered_datasets(dataset, split_idx, filter_split=None):
    sample_graph = dataset[0] if len(dataset) > 0 else None
    is_multitask = sample_graph is not None and sample_graph.y.numel() > 1
    if not is_multitask:
        print(f'Single-task dataset detected!')
        return {0: split_idx}
    if filter_split is not None and filter_split not in ['train', 'val', 'test']:
        raise ValueError(f"Invalid filter_split '{filter_split}'. Must be one of: 'train', 'val', 'test'")
    if filter_split is not None and filter_split not in split_idx:
        raise ValueError(f"Filter split '{filter_split}' not found in split_idx. Available splits: {list(split_idx.keys())}")
    num_tasks = sample_graph.y.numel()
    task_filtered_splits = {}
    if filter_split is not None:
        print(f"Prefiltering dataset for {num_tasks} tasks (filtering only '{filter_split}' split)...")
    else:
        print(f'Prefiltering dataset for {num_tasks} tasks...')
    print('Pre-computing task masks for the entire dataset...')
    all_task_masks = torch.empty((len(dataset), num_tasks), dtype=torch.bool)
    for i, graph in enumerate(dataset):
        if not hasattr(graph, 'task_mask'):
            if graph.y.dtype.is_floating_point:
                mask = ~torch.isnan(graph.y)
            else:
                mask = torch.ones_like(graph.y, dtype=torch.bool)
        else:
            mask = graph.task_mask
        all_task_masks[i] = mask.view(-1).bool()
    print('Pre-computation complete.')
    task_filtered_splits = {}
    for task_idx in range(num_tasks):
        task_filtered_splits[task_idx] = {}
        task_validity_mask = all_task_masks[:, task_idx]
        for split_name, indices in split_idx.items():
            if filter_split is None or split_name == filter_split:
                split_indices_tensor = torch.as_tensor(indices, dtype=torch.long)
                mask_for_this_split = task_validity_mask[split_indices_tensor]
                valid_indices = split_indices_tensor[mask_for_this_split]
                task_filtered_splits[task_idx][split_name] = valid_indices.tolist()
            else:
                task_filtered_splits[task_idx][split_name] = indices.tolist() if hasattr(indices, 'tolist') else indices
        if filter_split is None:
            total_samples = sum((len(indices) for indices in split_idx.values()))
            valid_samples = sum((len(indices) for indices in task_filtered_splits[task_idx].values()))
            print(f'  Task {task_idx}: {valid_samples}/{total_samples} samples ({valid_samples / total_samples * 100:.1f}%)')
        else:
            original_count = len(split_idx[filter_split])
            filtered_count = len(task_filtered_splits[task_idx][filter_split])
            print(f'  Task {task_idx} ({filter_split}): {filtered_count}/{original_count} samples ({filtered_count / original_count * 100:.1f}%)')
    return task_filtered_splits

class IndexTrackingDataLoader:

    def __init__(self, dataset, indices, batch_size=128, shuffle=True, num_workers=0, pin_memory=True):
        self.dataset = dataset
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.subset = Subset(dataset, indices)
        self.loader = DataLoader(self.subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

    def __iter__(self):
        """
        Iterator that yields batches with original dataset indices attached.
        """
        if self.shuffle:
            perm = torch.randperm(len(self.indices))
            sampled_indices = [self.indices[perm[i]] for i in range(len(self.indices))]
        else:
            sampled_indices = list(self.indices)
        for i in range(0, len(sampled_indices), self.batch_size):
            batch_subset_indices = sampled_indices[i:i + self.batch_size]
            graphs = [self.dataset[idx] for idx in batch_subset_indices]
            batch = Batch.from_data_list(graphs)
            batch.original_graph_indices = torch.tensor(batch_subset_indices, dtype=torch.long)
            yield batch

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

def create_data_loaders(dataset, split_idx, batch_size=128, shuffle=True, task_idx=None, use_index_tracking=False):
    loaders = {}
    num_workers = 0
    for split_name, indices in split_idx.items():
        subset = Subset(dataset, indices)
        shuffle_data = shuffle if split_name == 'train' else False
        use_pin_memory = torch.cuda.is_available()
        if use_index_tracking:
            loader = IndexTrackingDataLoader(dataset=dataset, indices=indices, batch_size=batch_size, shuffle=shuffle_data, num_workers=num_workers, pin_memory=use_pin_memory)
        else:
            loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle_data, num_workers=num_workers, pin_memory=use_pin_memory)
        loaders[split_name] = loader
    return loaders

def _save_pca_cache(dataset_name, target_dim, processed_features, cache_dir='./pca_cache'):
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / f'{dataset_name}_dim{target_dim}.pt'
    cache_data = {'processed_features': processed_features.cpu(), 'dataset_name': dataset_name, 'target_dim': target_dim, 'timestamp': time.time()}
    torch.save(cache_data, cache_file)
    print(f'PCA cache saved: {cache_file}')

def _load_pca_cache(dataset_name, target_dim, cache_dir='./pca_cache'):
    cache_path = Path(cache_dir)
    cache_file = cache_path / f'{dataset_name}_dim{target_dim}.pt'
    if not cache_file.exists():
        return None
    try:
        cache_data = torch.load(cache_file, map_location='cpu')
        print(f'PCA cache hit: {cache_file}')
        return cache_data['processed_features']
    except Exception as e:
        print(f'Error loading PCA cache {cache_file}: {e}')
        return None

def _apply_pca_unified(data_tensor, target_dim, use_full_pca, sign_normalize, pca_device, incremental_pca_batch_size, pca_sample_threshold=float('inf'), use_pca_cache=False, pca_cache_dir='./pca_cache', dataset_name=None, target_cuda_device=None, use_random_orthogonal=False):
    pca_start = time.time()
    n_samples = data_tensor.shape[0]
    print(f'Applying PCA on {n_samples:,} samples with target dimension {target_dim}, threshold {pca_sample_threshold:,}')
    if use_pca_cache and dataset_name:
        cached_result = _load_pca_cache(dataset_name, target_dim, pca_cache_dir)
        if cached_result is not None:
            return cached_result.to(data_tensor.device)
    cache_miss = use_pca_cache and dataset_name
    if cache_miss:
        pca_device = 'cpu'
    original_device = data_tensor.device
    target_device = None
    if pca_device == 'gpu':
        if torch.cuda.is_available():
            if target_cuda_device is not None:
                target_device = target_cuda_device
            else:
                target_device = torch.cuda.current_device()
        else:
            print(f'WARNING: GPU requested but CUDA not available, falling back to CPU')
            pca_device = 'cpu'
    if pca_device == 'cpu':
        from .data_utils import apply_incremental_pca_cpu
        processed_features = apply_incremental_pca_cpu(data_tensor, target_dim, incremental_pca_batch_size, sign_normalize, rank=0)
        processed_features = processed_features.to(data_tensor.device)
    else:
        use_sampling = n_samples > pca_sample_threshold
        if use_sampling:
            print(f'  Large dataset ({n_samples:,} nodes) - using sampled PCA with {pca_sample_threshold:,} samples')
            sample_indices = torch.randperm(n_samples, device=original_device)[:pca_sample_threshold]
            pca_data = data_tensor[sample_indices]
        else:
            print(f'  Small dataset ({n_samples:,} nodes) - using full PCA')
            pca_data = data_tensor
        if target_device is not None:
            pca_data = pca_data.to(target_device)
        if use_full_pca:
            U, S, V = torch.svd(pca_data)
            U = U[:, :target_dim]
            S = S[:target_dim]
        else:
            U, S, V = torch.pca_lowrank(pca_data, q=target_dim)
        batch_size = 500000
        num_batches = (n_samples + batch_size - 1) // batch_size
        processed_features = torch.zeros(n_samples, target_dim, device=original_device).pin_memory()
        data_tensor_pinned = data_tensor.pin_memory()
        V_gpu = V[:, :target_dim].to(target_device)
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_samples)
                data_batch_cpu = data_tensor_pinned[start_idx:end_idx]
                data_batch_gpu = data_batch_cpu.to(target_device, non_blocking=True)
                batch_transformed_gpu = data_batch_gpu @ V_gpu
                processed_features[start_idx:end_idx].copy_(batch_transformed_gpu, non_blocking=True)
            torch.cuda.synchronize()
        if sign_normalize:
            sign_sample_size = min(10000, n_samples)
            sign_indices = torch.randperm(n_samples, device=original_device)[:sign_sample_size]
            sample_transformed = processed_features[sign_indices]
            for i in range(target_dim):
                feature_vector = sample_transformed[:, i]
                if feature_vector.abs().argmax() < len(feature_vector) // 2:
                    processed_features[:, i] = -processed_features[:, i]
    pca_time = time.time() - pca_start
    method = 'cpu' if pca_device == 'cpu' else 'sampled-gpu' if n_samples > pca_sample_threshold else 'gpu'
    print(f'  PCA computation ({method}): {pca_time:.2f}s')
    if cache_miss:
        _save_pca_cache(dataset_name, target_dim, processed_features, pca_cache_dir)
    if target_device is not None and original_device != data_tensor.device:
        processed_features = processed_features.to(original_device)
    return processed_features

def process_graph_features(dataset, hidden_dim, device='cuda', projection_large_dim=256, use_full_pca=False, sign_normalize=False, normalize_data=False, use_batchnorm=False, dataset_name=None, pca_device='gpu', incremental_pca_batch_size=10000, pca_sample_threshold=500000, processed_data=None, pcba_context_only_pca=False, use_pca_cache=False, pca_cache_dir='./pca_cache'):
    import time
    sample = dataset[0] if len(dataset) > 0 else None
    if sample is None:
        raise ValueError('Empty dataset')
    feature_mapping = None
    node_embs = None
    if processed_data is not None and 'feature_mapping' in processed_data:
        feature_mapping = processed_data['feature_mapping']
        if feature_mapping.get('cache_mode', False) and use_pca_cache and dataset_name:
            print(f'🚀 Cache mode enabled for {dataset_name} - attempting to load cached PCA features')
            cached_result = _load_pca_cache(dataset_name, hidden_dim, pca_cache_dir)
            if cached_result is not None:
                print(f'✅ Successfully loaded cached PCA features for {dataset_name}')
                feature_mapping['node_embs'] = cached_result
                feature_mapping['cache_mode'] = False
                print(f'Set cached features as node_embs: {cached_result.shape}')
                node_embs = cached_result
                original_dim = cached_result.shape[1]
                print(f'Cache mode: Using cached embeddings {node_embs.shape} as normal flow')
            else:
                print(f"❌ Cache miss for {dataset_name} in cache mode - this shouldn't happen!")
                print(f'   Cache was detected during loading but file is missing/corrupted')
                if feature_mapping.get('embedding_file'):
                    print(f"🔄 Loading embeddings as fallback: {feature_mapping['embedding_file']}")
                    try:
                        node_embs = torch.load(feature_mapping['embedding_file'], map_location='cpu')
                        feature_mapping['node_embs'] = node_embs
                        feature_mapping['cache_mode'] = False
                        print(f'✅ Loaded fallback embeddings: {node_embs.shape}')
                    except Exception as e:
                        raise RuntimeError(f'Failed to load fallback embeddings: {e}')
        if feature_mapping.get('node_embs') is not None:
            if 'node_embs' not in locals() or node_embs is None:
                node_embs = feature_mapping['node_embs']
                print(f'Using numerical node features: {node_embs.shape}')
            if 'original_dim' not in locals():
                original_dim = node_embs.shape[1]
    elif hasattr(dataset, 'node_embs') and dataset.node_embs is not None:
        node_embs = dataset.node_embs
        original_dim = node_embs.shape[1]
        print(f'Using dataset node_embs: {node_embs.shape}')
    else:
        raise ValueError('Expected dataset.node_embs under unified setting or feature mapping in processed_data')
    print(f'Processing graph features: {original_dim}D -> {hidden_dim}D')
    st = time.time()
    if node_embs is None:
        raise ValueError('No node embeddings available - check feature mapping or dataset.node_embs')

    def get_graph_node_embeddings(graph_idx):
        if feature_mapping is not None:
            node_indices = feature_mapping['node_index_mapping'][graph_idx]
            return node_embs[node_indices]
        else:
            graph = dataset[graph_idx]
            return node_embs[graph.x]
    print('Unified setting: applying PCA to node_embs')
    processed_features = None
    target_dim = hidden_dim
    needs_identity_projection = False
    if dataset_name and dataset_name.lower() == 'pcba' and pcba_context_only_pca and (processed_data is not None):
        print('PCBA mode: Using context + test graphs for PCA fitting')
        context_node_embs = []
        context_graphs = processed_data['context_graphs']
        for task_idx in context_graphs:
            for class_label in context_graphs[task_idx]:
                context_data = context_graphs[task_idx][class_label]
                if isinstance(context_data, dict) and 'indices' in context_data:
                    indices = context_data['indices']
                    for graph_idx in indices:
                        context_node_embs.append(get_graph_node_embeddings(graph_idx))
                else:
                    print('[WARNING] Using slow fallback for context graph indexing!')
                    for graph in context_data:
                        graph_idx = None
                        for i, dataset_graph in enumerate(dataset):
                            if dataset_graph is graph:
                                graph_idx = i
                                break
                        if graph_idx is not None:
                            context_node_embs.append(get_graph_node_embeddings(graph_idx))
        test_node_embs = []
        test_split = processed_data['split_idx']['test']
        for test_idx in test_split:
            test_idx_int = test_idx.item() if torch.is_tensor(test_idx) else test_idx
            test_node_embs.append(get_graph_node_embeddings(test_idx_int))
        stacked_features = torch.cat(context_node_embs + test_node_embs, dim=0)
        print(f'PCBA PCA fitted on {stacked_features.size(0)} nodes from context + test graphs')
    else:
        stacked_features = node_embs
    if processed_features is None:
        original_dim = min(original_dim, stacked_features.size(0))
        if original_dim >= hidden_dim:
            processed_features = _apply_pca_unified(stacked_features, hidden_dim, use_full_pca, sign_normalize, pca_device, incremental_pca_batch_size, pca_sample_threshold, use_pca_cache, pca_cache_dir, dataset_name, device, False)
        else:
            pca_dim = min(original_dim, stacked_features.size(1))
            pca_features = _apply_pca_unified(stacked_features, pca_dim, use_full_pca, sign_normalize, pca_device, incremental_pca_batch_size, pca_sample_threshold, use_pca_cache, pca_cache_dir, dataset_name, device, False)
            if pca_features.size(1) < hidden_dim:
                padding_size = hidden_dim - pca_features.size(1)
                padding = torch.zeros(pca_features.size(0), padding_size, device=pca_features.device)
                processed_features = torch.cat([pca_features, padding], dim=1)
                print(f"Applied {'zero'} padding ({pca_features.size(1)} -> {hidden_dim})")
            else:
                processed_features = pca_features[:, :hidden_dim]
        target_dim = hidden_dim
        needs_identity_projection = False
    if normalize_data and True:
        if use_batchnorm:
            batch_mean = processed_features.mean(dim=0, keepdim=True)
            batch_std = processed_features.std(dim=0, keepdim=True, unbiased=False)
            processed_features = (processed_features - batch_mean) / (batch_std + 1e-05)
            print('Applied BatchNorm-style normalization')
        else:
            processed_features = F.normalize(processed_features, p=2, dim=1)
            print('Applied LayerNorm-style normalization')
    if feature_mapping is not None:
        feature_mapping['node_embs'] = processed_features
        print(f'  Updated feature mapping embeddings: {processed_features.shape}')
    else:
        dataset.node_embs = processed_features
        print(f'  Updated embedding table: {processed_features.shape} - all graphs now use reduced embeddings!')
    processing_info = {'original_dim': original_dim, 'processed_dim': processed_features.size(1), 'target_dim': target_dim, 'needs_identity_projection': needs_identity_projection, 'projection_target_dim': projection_large_dim if needs_identity_projection else None}
    print(f'Graph feature processing completed in {time.time() - st:.2f}s: {original_dim}D -> {processed_features.size(1)}D (optimized: processed embedding table directly!)')
    return processing_info
