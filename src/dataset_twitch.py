import os.path as osp
from pathlib import Path
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
        # v4: global shared feature-space dimension across regions + duplicate-label conflict guard.
        return 'data_v4.pt'

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
            # Safety check: conflicting labels for duplicated node ids would corrupt supervision.
            mature_conflicts = target.groupby('new_id')['mature'].nunique(dropna=False)
            if (mature_conflicts > 1).any():
                bad_ids = mature_conflicts[mature_conflicts > 1].index.tolist()
                raise ValueError(
                    f"Twitch target file for {self.name} has duplicate new_id rows with conflicting labels: {bad_ids[:10]}"
                )
            print(f"[TwitchFixed] Warning: {self.name} has {dup_count} duplicate new_id rows. Keeping first.")
            target = target.drop_duplicates(subset=['new_id'], keep='first')

        # Crucial: index rows by sorted new_id so node index i matches edge endpoint id i
        # (or mapped i when ids are not perfectly contiguous).
        target = target.sort_values('new_id').reset_index(drop=True)
        new_ids = target['new_id'].astype(int).to_numpy()
        node_id_to_idx = {node_id: idx for idx, node_id in enumerate(new_ids.tolist())}

        # Raw JSON stores feature ID lists (sparse categorical indicators), not dense values.
        # Decode into a multi-hot matrix over full feature vocabulary.
        max_feat_id = -1
        for vals in features.values():
            if vals:
                local_max = max(vals)
                if local_max > max_feat_id:
                    max_feat_id = int(local_max)
        if max_feat_id < 0:
            raise ValueError(f"Twitch feature file for {self.name} appears empty.")

        # Datasets share the same feature space across regions.
        # Infer a global feature dimension from all region feature files in this raw zip.
        global_max_feat_id = max_feat_id
        raw_twitch_root = Path(self.raw_dir) / 'twitch'
        if raw_twitch_root.exists():
            candidate_files = [
                raw_twitch_root / 'DE' / 'musae_DE.json',
                raw_twitch_root / 'ENGB' / 'musae_ENGB_features.json',
                raw_twitch_root / 'ES' / 'musae_ES_features.json',
                raw_twitch_root / 'FR' / 'musae_FR_features.json',
                raw_twitch_root / 'PTBR' / 'musae_PTBR_features.json',
                raw_twitch_root / 'RU' / 'musae_RU_features.json',
            ]
            for file_path in candidate_files:
                if not file_path.exists():
                    continue
                with open(file_path, 'r') as f:
                    region_features = json.load(f)
                for vals in region_features.values():
                    if vals:
                        region_max = max(vals)
                        if region_max > global_max_feat_id:
                            global_max_feat_id = int(region_max)

        n_feats = global_max_feat_id + 1
        x_arr = np.zeros((len(new_ids), n_feats), dtype=np.float32)
        for idx, node_id in enumerate(new_ids):
            feat_ids = features.get(str(int(node_id)), [])
            if not feat_ids:
                continue

            feat_ids = np.asarray(feat_ids, dtype=np.int64)
            valid = (feat_ids >= 0) & (feat_ids < n_feats)
            feat_ids = feat_ids[valid]
            if feat_ids.size > 0:
                x_arr[idx, np.unique(feat_ids)] = 1.0
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
