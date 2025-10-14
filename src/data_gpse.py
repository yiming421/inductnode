"""
GPSE Embedding Loader for Node Classification
Loads pre-computed GPSE, LapPE, and RWSE embeddings and enhances node features
"""
import torch
import os
from pathlib import Path


class GPSEEmbeddingLoader:
    """
    Loader for positional and structural encodings:
    - GPSE: 512-dimensional learned structural representations from pre-trained 20-layer GNN
    - LapPE: Laplacian Positional Encoding (top-k eigenvectors of graph Laplacian, 20-dim)
    - RWSE: Random Walk Structural Encoding (return probabilities, 20-dim)
    """

    def __init__(self, gpse_base_dir='../GPSE/datasets', verbose=False):
        """
        Args:
            gpse_base_dir: Base directory containing GPSE/LapPE/RWSE embeddings
            verbose: Print loading information
        """
        self.base_dir = Path(gpse_base_dir)
        self.cache = {}  # Cache loaded embeddings
        self.verbose = verbose

    def load_gpse(self, dataset_name):
        """
        Load GPSE embeddings for a dataset.

        Args:
            dataset_name: Name of the dataset (e.g., 'Cora', 'Citeseer')

        Returns:
            torch.Tensor: GPSE embeddings [num_nodes, 512]

        Raises:
            FileNotFoundError: If GPSE embeddings not found
        """
        # Check cache first
        if dataset_name in self.cache:
            if self.verbose:
                print(f"  [GPSE] Using cached embeddings for {dataset_name}")
            return self.cache[dataset_name]

        # Try multiple naming conventions
        # 1. Original name (e.g., 'Cora')
        # 2. Replace hyphen with underscore (e.g., 'ogbn-arxiv' -> 'ogbn_arxiv')
        # 3. Lowercase (e.g., 'USA' -> 'usa')
        # 4. Both (e.g., 'ogbn-arxiv' -> 'ogbn_arxiv')
        possible_names = [
            dataset_name,
            dataset_name.replace('-', '_'),
            dataset_name.lower(),
            dataset_name.replace('-', '_').lower()
        ]

        data_path = None
        for name in possible_names:
            candidate_path = self.base_dir / name / 'pe_stats_GPSE' / '1.0' / 'data.pt'
            if candidate_path.exists():
                data_path = candidate_path
                if self.verbose and name != dataset_name:
                    print(f"  [GPSE] Found {dataset_name} as '{name}'")
                break

        if data_path is None:
            raise FileNotFoundError(
                f"GPSE embeddings not found for {dataset_name}\n"
                f"Tried: {possible_names}\n"
                f"Please generate embeddings using generate_gpse_embeddings.py"
            )

        # Load embeddings
        embeddings = torch.load(data_path, map_location='cpu')

        if self.verbose:
            print(f"  [GPSE] Loaded embeddings for {dataset_name}: {embeddings.shape}")

        # Cache for future use
        self.cache[dataset_name] = embeddings

        return embeddings

    def load_lappe(self, dataset_name):
        """
        Load Laplacian PE embeddings for a dataset.

        Args:
            dataset_name: Name of the dataset (e.g., 'Cora', 'Citeseer')

        Returns:
            torch.Tensor: LapPE embeddings [num_nodes, 20] (or 19 with skip_zero_freq)

        Raises:
            FileNotFoundError: If LapPE embeddings not found
        """
        cache_key = f"lappe_{dataset_name}"
        if cache_key in self.cache:
            if self.verbose:
                print(f"  [LapPE] Using cached embeddings for {dataset_name}")
            return self.cache[cache_key]

        # Try multiple naming conventions
        possible_names = [
            dataset_name,
            dataset_name.replace('-', '_'),
            dataset_name.lower(),
            dataset_name.replace('-', '_').lower()
        ]

        data_path = None
        for name in possible_names:
            candidate_path = self.base_dir / name / 'pe_stats_LapPE' / '1.0' / 'data.pt'
            if candidate_path.exists():
                data_path = candidate_path
                if self.verbose and name != dataset_name:
                    print(f"  [LapPE] Found {dataset_name} as '{name}'")
                break

        if data_path is None:
            raise FileNotFoundError(
                f"LapPE embeddings not found for {dataset_name}\n"
                f"Tried: {possible_names}\n"
                f"Please generate embeddings using generate_lappe_rwse.py"
            )

        # Load embeddings
        embeddings = torch.load(data_path, map_location='cpu')

        if self.verbose:
            print(f"  [LapPE] Loaded embeddings for {dataset_name}: {embeddings.shape}")

        # Cache for future use
        self.cache[cache_key] = embeddings

        return embeddings

    def load_rwse(self, dataset_name):
        """
        Load Random Walk SE embeddings for a dataset.

        Args:
            dataset_name: Name of the dataset (e.g., 'Cora', 'Citeseer')

        Returns:
            torch.Tensor: RWSE embeddings [num_nodes, 20]

        Raises:
            FileNotFoundError: If RWSE embeddings not found
        """
        cache_key = f"rwse_{dataset_name}"
        if cache_key in self.cache:
            if self.verbose:
                print(f"  [RWSE] Using cached embeddings for {dataset_name}")
            return self.cache[cache_key]

        # Try multiple naming conventions
        possible_names = [
            dataset_name,
            dataset_name.replace('-', '_'),
            dataset_name.lower(),
            dataset_name.replace('-', '_').lower()
        ]

        data_path = None
        for name in possible_names:
            candidate_path = self.base_dir / name / 'pe_stats_RWSE' / '1.0' / 'data.pt'
            if candidate_path.exists():
                data_path = candidate_path
                if self.verbose and name != dataset_name:
                    print(f"  [RWSE] Found {dataset_name} as '{name}'")
                break

        if data_path is None:
            raise FileNotFoundError(
                f"RWSE embeddings not found for {dataset_name}\n"
                f"Tried: {possible_names}\n"
                f"Please generate embeddings using generate_lappe_rwse.py"
            )

        # Load embeddings
        embeddings = torch.load(data_path, map_location='cpu')

        if self.verbose:
            print(f"  [RWSE] Loaded embeddings for {dataset_name}: {embeddings.shape}")

        # Cache for future use
        self.cache[cache_key] = embeddings

        return embeddings

    def attach_gpse_to_data(self, data, dataset_name):
        """
        Attach GPSE embeddings to data object as an attribute.

        Args:
            data: PyG Data object
            dataset_name: Name of the dataset

        Modifies data in-place by adding .gpse_embeddings attribute
        """
        gpse_embeddings = self.load_gpse(dataset_name)

        # Verify dimensions
        if data.x.size(0) != gpse_embeddings.size(0):
            raise ValueError(
                f"Node count mismatch for {dataset_name}: "
                f"data.x has {data.x.size(0)} nodes, "
                f"GPSE has {gpse_embeddings.size(0)} nodes"
            )

        data.gpse_embeddings = gpse_embeddings

        if self.verbose:
            print(f"  [GPSE] Attached embeddings to {dataset_name}: {gpse_embeddings.shape}")

    def attach_lappe_to_data(self, data, dataset_name):
        """
        Attach LapPE embeddings to data object as an attribute.

        Args:
            data: PyG Data object
            dataset_name: Name of the dataset

        Modifies data in-place by adding .lappe_embeddings attribute
        """
        lappe_embeddings = self.load_lappe(dataset_name)

        # Verify dimensions
        if data.x.size(0) != lappe_embeddings.size(0):
            raise ValueError(
                f"Node count mismatch for {dataset_name}: "
                f"data.x has {data.x.size(0)} nodes, "
                f"LapPE has {lappe_embeddings.size(0)} nodes"
            )

        data.lappe_embeddings = lappe_embeddings

        if self.verbose:
            print(f"  [LapPE] Attached embeddings to {dataset_name}: {lappe_embeddings.shape}")

    def attach_rwse_to_data(self, data, dataset_name):
        """
        Attach RWSE embeddings to data object as an attribute.

        Args:
            data: PyG Data object
            dataset_name: Name of the dataset

        Modifies data in-place by adding .rwse_embeddings attribute
        """
        rwse_embeddings = self.load_rwse(dataset_name)

        # Verify dimensions
        if data.x.size(0) != rwse_embeddings.size(0):
            raise ValueError(
                f"Node count mismatch for {dataset_name}: "
                f"data.x has {data.x.size(0)} nodes, "
                f"RWSE has {rwse_embeddings.size(0)} nodes"
            )

        data.rwse_embeddings = rwse_embeddings

        if self.verbose:
            print(f"  [RWSE] Attached embeddings to {dataset_name}: {rwse_embeddings.shape}")

    def clear_cache(self):
        """Clear the embedding cache to free memory"""
        self.cache.clear()
