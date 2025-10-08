import os.path as osp
from typing import Callable, List, Optional

import numpy as np
import torch
import json
import pandas as pd

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class TwitchFixed(InMemoryDataset):
    r"""The Twitch Gamer networks from the `"Multi-scale Attribution Methods
    for Network Anomaly Detection" <https://arxiv.org/abs/1901.08075>`_ paper.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (:obj:`"DE"`, :obj:`"EN"`,
            :obj:`"ES"`, :obj:`"FR"`, :obj:`"PT"`, :obj:`"RU"`).
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    """

    url = 'https://snap.stanford.edu/data/twitch.zip'

    # Map API names to actual directory names in the dataset
    name_mapping = {
        'DE': 'DE',
        'EN': 'ENGB',
        'ES': 'ES',
        'FR': 'FR',
        'PT': 'PTBR',
        'RU': 'RU'
    }

    def __init__(
        self,
        root: str,
        name: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ) -> None:
        self.name = name
        assert self.name in ['DE', 'EN', 'ES', 'FR', 'PT', 'RU']
        self.actual_name = self.name_mapping[self.name]
        super().__init__(root, transform, pre_transform)
        self.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        # DE uses musae_DE.json, others use musae_{REGION}_features.json
        if self.actual_name == 'DE':
            features_file = f'twitch/{self.actual_name}/musae_{self.actual_name}.json'
        else:
            features_file = f'twitch/{self.actual_name}/musae_{self.actual_name}_features.json'

        return [
            f'twitch/{self.actual_name}/musae_{self.actual_name}_edges.csv',
            features_file,
            f'twitch/{self.actual_name}/musae_{self.actual_name}_target.csv',
        ]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self) -> None:
        file_path = download_url(self.url, self.raw_dir)
        extract_zip(file_path, self.raw_dir)

    def process(self) -> None:
        edges = pd.read_csv(self.raw_paths[0])
        features = json.load(open(self.raw_paths[1]))
        target = pd.read_csv(self.raw_paths[2])

        xs = []
        n_feats = 128
        for i in target['id'].values:
            f = [0] * n_feats
            if str(i) in features:
                n_len = len(features[str(i)])
                f = features[str(
                    i)][:n_feats] if n_len >= n_feats else features[str(
                        i)] + [0] * (n_feats - n_len)
            xs.append(f)
        x = torch.from_numpy(np.array(xs)).to(torch.float)

        # Convert mature column (True/False) to binary labels (1/0)
        y = torch.from_numpy((target['mature'] == True).astype(int).values).to(torch.long)

        edge_index = torch.from_numpy(edges.values.astype(np.int64)).to(torch.long)
        edge_index = edge_index.t().contiguous()

        data = Data(x=x, y=y, edge_index=edge_index)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
