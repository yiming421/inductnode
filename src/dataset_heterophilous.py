"""
Custom HeterophilousGraphDataset for older PyG versions.

This module provides support for heterophilous graph datasets from:
"A Critical Look at the Evaluation of GNNs under Heterophily: Are We Really Making Progress?"
https://arxiv.org/abs/2302.11640

Supports: Roman-empire, Amazon-ratings, Minesweeper, Tolokers, Questions
"""

import os
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_undirected


class HeterophilousGraphDataset(InMemoryDataset):
    """
    The heterophilous graphs from the paper "A Critical Look at the Evaluation of GNNs
    under Heterophily: Are We Really Making Progress?" (https://arxiv.org/abs/2302.11640)

    Datasets:
    - Roman-empire: 22,662 nodes, 32,927 edges, 300 features, 18 classes
    - Amazon-ratings: 24,492 nodes, 93,050 edges, 300 features, 5 classes
    - Minesweeper: 10,000 nodes, 39,402 edges, 7 features, 2 classes
    - Tolokers: 11,758 nodes, 519,000 edges, 10 features, 2 classes
    - Questions: 48,921 nodes, 153,540 edges, 301 features, 2 classes

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (Roman-empire, Amazon-ratings,
                    Minesweeper, Tolokers, Questions).
        transform (callable, optional): A function/transform that takes in a Data object
                                       and returns a transformed version.
        pre_transform (callable, optional): A function/transform that takes in a Data object
                                           and returns a transformed version before saving.
        force_reload (bool, optional): Whether to re-process the dataset.
    """

    url = 'https://github.com/yandex-research/heterophilous-graphs/raw/main/data'

    def __init__(self, root, name, transform=None, pre_transform=None,
                 make_undirected=True, force_reload=False):
        self.name = name.lower().replace('-', '_')
        self.make_undirected = bool(make_undirected)
        assert self.name in [
            'roman_empire',
            'amazon_ratings',
            'minesweeper',
            'tolokers',
            'questions',
        ], f"Dataset {name} not supported. Choose from: Roman-empire, Amazon-ratings, Minesweeper, Tolokers, Questions"

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return os.path.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self):
        return os.path.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return f'{self.name}.npz'

    @property
    def processed_file_names(self):
        suffix = 'undirected' if self.make_undirected else 'directed'
        return f'data_{suffix}.pt'

    def download(self):
        """Download the dataset from GitHub"""
        import urllib.request
        url = f'{self.url}/{self.name}.npz'
        os.makedirs(self.raw_dir, exist_ok=True)
        file_path = os.path.join(self.raw_dir, self.raw_file_names)

        print(f"Downloading {self.name} from {url}...")
        urllib.request.urlretrieve(url, file_path)
        print(f"Downloaded to {file_path}")

    def process(self):
        """Process the raw data"""
        raw = np.load(self.raw_paths[0], 'r')
        x = torch.from_numpy(raw['node_features'])
        y = torch.from_numpy(raw['node_labels'])
        edge_index = torch.from_numpy(raw['edges']).t().contiguous()
        if self.make_undirected:
            edge_index = to_undirected(edge_index, num_nodes=x.size(0))
        train_mask = torch.from_numpy(raw['train_masks']).t().contiguous()
        val_mask = torch.from_numpy(raw['val_masks']).t().contiguous()
        test_mask = torch.from_numpy(raw['test_masks']).t().contiguous()

        data = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask,
                    val_mask=val_mask, test_mask=test_mask)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        data_list = [data]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def __repr__(self):
        direction = 'undirected' if self.make_undirected else 'directed'
        return f'{self.__class__.__name__}(name={self.name}, direction={direction})'
