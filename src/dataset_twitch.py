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
        # Bump cache version because old processing could misalign x/y rows vs edge indices.
        return 'data_v2.pt'

    def download(self) -> None:
        file_path = download_url(self.url, self.raw_dir)
        extract_zip(file_path, self.raw_dir)

    def process(self) -> None:
        edges = pd.read_csv(self.raw_paths[0])
        with open(self.raw_paths[1], 'r') as f:
            features = json.load(f)
        target = pd.read_csv(self.raw_paths[2])

        if 'new_id' not in target.columns or 'mature' not in target.columns:
            raise ValueError(
                f"Twitch target file for {self.name} must contain 'new_id' and 'mature' columns."
            )
        if edges.shape[1] < 2:
            raise ValueError(f"Twitch edge file for {self.name} must contain at least two columns.")

        dup_count = int(target.duplicated(subset=['new_id']).sum())
        if dup_count > 0:
            print(f"[TwitchFixed] Warning: {self.name} has {dup_count} duplicate new_id rows. Keeping first.")
            target = target.drop_duplicates(subset=['new_id'], keep='first')

        # Crucial: index rows by sorted new_id so node index i matches edge endpoint id i
        # (or mapped i when ids are not perfectly contiguous).
        target = target.sort_values('new_id').reset_index(drop=True)
        new_ids = target['new_id'].astype(int).to_numpy()
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(new_ids.tolist())}

        n_feats = 128
        x_arr = np.zeros((len(new_ids), n_feats), dtype=np.float32)
        for idx, node_id in enumerate(new_ids):
            feats = features.get(str(int(node_id)), [])
            if feats:
                if len(feats) >= n_feats:
                    x_arr[idx] = np.asarray(feats[:n_feats], dtype=np.float32)
                else:
                    padded = feats + [0] * (n_feats - len(feats))
                    x_arr[idx] = np.asarray(padded, dtype=np.float32)
        x = torch.from_numpy(x_arr)

        # Convert mature column (True/False) to binary labels (1/0)
        y = torch.from_numpy((target['mature'] == True).astype(np.int64).values).to(torch.long)

        src = edges.iloc[:, 0].astype(np.int64).to_numpy()
        dst = edges.iloc[:, 1].astype(np.int64).to_numpy()
        mapped_src = []
        mapped_dst = []
        invalid_edges = 0
        for s, d in zip(src, dst):
            s_idx = node_id_to_idx.get(int(s))
            d_idx = node_id_to_idx.get(int(d))
            if s_idx is None or d_idx is None:
                invalid_edges += 1
                continue
            mapped_src.append(s_idx)
            mapped_dst.append(d_idx)

        if invalid_edges > 0:
            print(f"[TwitchFixed] Warning: Dropped {invalid_edges} edges with unknown node ids in {self.name}.")

        edge_index = torch.tensor([mapped_src, mapped_dst], dtype=torch.long)

        data = Data(x=x, y=y, edge_index=edge_index)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])
