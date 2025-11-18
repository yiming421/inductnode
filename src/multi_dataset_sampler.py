"""
Multi-dataset batch sampler with temperature-controlled sampling probabilities.

Implements temperature-based sampling for training on multiple graph classification datasets.
"""

import numpy as np


def compute_dataset_sampling_probs(dataset_sizes, temperature):
    """
    Compute sampling probabilities for datasets based on their sizes and temperature.

    The probability of sampling dataset i is:
        p_i = (N_i)^T / sum_j((N_j)^T)

    Where:
        - N_i is the number of training examples in dataset i (num_graphs * num_tasks)
        - T is the temperature parameter (0 <= T <= 1)
        - T=0: uniform sampling (all datasets equally likely)
        - T=1: size-proportional sampling (larger datasets more likely)
        - 0<T<1: interpolates between uniform and size-proportional

    Args:
        dataset_sizes (list): List of dataset sizes [N_1, N_2, ..., N_k]
        temperature (float): Temperature parameter controlling size influence (0 <= T <= 1)

    Returns:
        np.ndarray: Normalized probabilities for each dataset [p_1, p_2, ..., p_k]
    """
    dataset_sizes = np.array(dataset_sizes, dtype=np.float64)

    # Compute (N_i)^T for each dataset
    if temperature == 0:
        # Special case: uniform distribution
        weighted_sizes = np.ones_like(dataset_sizes)
    else:
        weighted_sizes = np.power(dataset_sizes, temperature)

    # Normalize to get probabilities
    probs = weighted_sizes / np.sum(weighted_sizes)

    return probs


class MultiDatasetBatchSampler:
    """
    Samples batches from multiple datasets with temperature-controlled probabilities.

    Uses two-stage sampling WITH REPLACEMENT:
    1. Sample a dataset based on temperature-weighted probabilities
    2. Uniformly sample a task within that dataset
    3. Get next batch from that (dataset, task) pair

    Epoch length is fixed to the total number of batches across all (dataset, task) pairs.
    """

    def __init__(self, train_processed_data_list, all_splits, temperature, batch_size,
                 device='cuda', verbose=True):
        """
        Args:
            train_processed_data_list (list): List of dataset info dicts
            all_splits (list): List of split indices for each dataset
            temperature (float): Temperature parameter for sampling (0 <= T <= 1)
            batch_size (int): Batch size for data loaders
            device (str): Device for computation
            verbose (bool): Whether to print sampling statistics
        """
        from .data_gc import create_data_loaders

        self.train_processed_data_list = train_processed_data_list
        self.all_splits = all_splits
        self.temperature = temperature
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

        # Compute dataset sizes (num_graphs * num_tasks)
        self.dataset_sizes = []
        self.num_tasks_per_dataset = []
        self.dataset_names = []

        for dataset_idx, (dataset_info, splits) in enumerate(zip(train_processed_data_list, all_splits)):
            num_graphs = len(splits['train'])
            num_tasks = dataset_info.get('num_tasks', 1)

            self.dataset_sizes.append(num_graphs * num_tasks)
            self.num_tasks_per_dataset.append(num_tasks)

            # Get dataset name for logging
            name = dataset_info['dataset'].name if hasattr(dataset_info['dataset'], 'name') else f'dataset_{dataset_idx}'
            self.dataset_names.append(name)

        # Compute dataset sampling probabilities
        self.dataset_probs = compute_dataset_sampling_probs(self.dataset_sizes, temperature)

        # Create data loaders for each (dataset, task) pair (lazily)
        self.data_loaders = {}
        self.data_iterators = {}

        # Calculate total batches per epoch
        self.total_batches = 0
        for dataset_idx, (dataset_info, splits) in enumerate(zip(train_processed_data_list, all_splits)):
            num_tasks = self.num_tasks_per_dataset[dataset_idx]
            num_graphs = len(splits['train'])
            batches_per_task = int(np.ceil(num_graphs / batch_size))
            self.total_batches += num_tasks * batches_per_task

        # Statistics tracking
        self.dataset_sample_counts = [0] * len(train_processed_data_list)
        self.task_sample_counts = {}  # (dataset_idx, task_idx) -> count
        self.batches_sampled = 0

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"MultiDatasetBatchSampler Initialized (WITH REPLACEMENT)")
            print(f"{'='*60}")
            print(f"Temperature: {temperature:.3f}")
            print(f"Number of datasets: {len(train_processed_data_list)}")
            print(f"\nDataset Information:")
            for idx, (name, size, num_tasks) in enumerate(zip(self.dataset_names, self.dataset_sizes, self.num_tasks_per_dataset)):
                print(f"  [{idx}] {name:20s}: {size:8d} examples ({num_tasks:3d} tasks)")
            print(f"\nSampling Probabilities:")
            for idx, (name, prob) in enumerate(zip(self.dataset_names, self.dataset_probs)):
                print(f"  [{idx}] {name:20s}: {prob:6.4f}")
            print(f"\nTotal batches per epoch: {self.total_batches}")
            print(f"{'='*60}\n")

    def _get_or_create_loader(self, dataset_idx, task_idx):
        """Get or create data loader for (dataset, task) pair."""
        from .data_gc import create_data_loaders

        key = (dataset_idx, task_idx)

        if key not in self.data_loaders:
            dataset_info = self.train_processed_data_list[dataset_idx]
            splits = self.all_splits[dataset_idx]

            # Check if we need task filtering
            use_full_batch = getattr(dataset_info, 'full_batch_training', False)

            if use_full_batch:
                # No task filtering - use all data
                task_splits = splits
            else:
                # Create task-filtered splits
                from .data_gc import create_task_filtered_datasets
                task_filtered_splits = create_task_filtered_datasets(
                    dataset_info['dataset'],
                    splits
                )
                task_splits = task_filtered_splits.get(task_idx, splits)

            # Check if FUG mapping is present
            use_fug_tracking = 'fug_mapping' in dataset_info

            # Create data loader
            loaders = create_data_loaders(
                dataset_info['dataset'],
                task_splits,
                batch_size=self.batch_size,
                shuffle=True,
                task_idx=task_idx,
                use_index_tracking=use_fug_tracking
            )

            self.data_loaders[key] = loaders['train']

        return self.data_loaders[key]

    def _get_infinite_iterator(self, dataset_idx, task_idx):
        """Get or create infinite iterator for (dataset, task) pair."""
        key = (dataset_idx, task_idx)

        if key not in self.data_iterators:
            loader = self._get_or_create_loader(dataset_idx, task_idx)
            # Create infinite iterator
            def infinite_loader():
                while True:
                    for batch in loader:
                        yield batch
            self.data_iterators[key] = infinite_loader()

        return self.data_iterators[key]

    def __iter__(self):
        """Reset statistics and return self."""
        self.batches_sampled = 0
        self.dataset_sample_counts = [0] * len(self.train_processed_data_list)
        self.task_sample_counts = {}
        return self

    def __len__(self):
        """Return total number of batches per epoch."""
        return self.total_batches

    def __next__(self):
        """Sample next batch from a randomly selected (dataset, task) pair."""
        if self.batches_sampled >= self.total_batches:
            if self.verbose:
                self._print_sampling_stats()
            raise StopIteration

        # Stage 1: Sample dataset with temperature-weighted probabilities
        dataset_idx = np.random.choice(len(self.dataset_probs), p=self.dataset_probs)

        # Stage 2: Uniformly sample task within selected dataset
        num_tasks = self.num_tasks_per_dataset[dataset_idx]
        task_idx = np.random.randint(0, num_tasks)

        # Get batch from infinite iterator
        iterator = self._get_infinite_iterator(dataset_idx, task_idx)
        batch = next(iterator)

        # Update statistics
        self.batches_sampled += 1
        self.dataset_sample_counts[dataset_idx] += 1
        key = (dataset_idx, task_idx)
        self.task_sample_counts[key] = self.task_sample_counts.get(key, 0) + 1

        return dataset_idx, task_idx, batch

    def _print_sampling_stats(self):
        """Print statistics about dataset sampling distribution."""
        print(f"\n{'='*60}")
        print(f"MultiDatasetBatchSampler Statistics")
        print(f"{'='*60}")
        print(f"Total batches sampled: {self.batches_sampled}")
        print(f"\nDataset Sample Counts:")
        for idx, (name, count) in enumerate(zip(self.dataset_names, self.dataset_sample_counts)):
            freq = count / self.batches_sampled if self.batches_sampled > 0 else 0
            expected = self.dataset_probs[idx]
            print(f"  [{idx}] {name:20s}: {count:5d} batches ({freq:6.4f} actual vs {expected:6.4f} expected)")

        print(f"\nTask Sample Distribution (Top 10 most sampled):")
        sorted_tasks = sorted(self.task_sample_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for (d_idx, t_idx), count in sorted_tasks:
            print(f"  Dataset {d_idx}, Task {t_idx}: {count} batches")
        print(f"{'='*60}\n")
