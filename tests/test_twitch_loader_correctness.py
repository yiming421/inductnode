import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset_twitch import TwitchFixed


REGION_TO_RAW_NAME = {
    "DE": "DE",
    "EN": "ENGB",
    "ES": "ES",
    "FR": "FR",
    "PT": "PTBR",
    "RU": "RU",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _all_feature_files(dataset_root: Path):
    files = []
    for region, raw_name in REGION_TO_RAW_NAME.items():
        raw_base = dataset_root / region / "raw" / "twitch" / raw_name
        feat_path = _feature_file(raw_base, raw_name)
        if feat_path.exists():
            files.append(feat_path)
    return files


def _global_feature_dim(dataset_root: Path) -> int:
    max_feat_id = -1
    for feat_path in _all_feature_files(dataset_root):
        with open(feat_path, "r") as f:
            features = json.load(f)
        for vals in features.values():
            if vals:
                local_max = max(vals)
                if local_max > max_feat_id:
                    max_feat_id = int(local_max)
    assert max_feat_id >= 0, "No Twitch feature ids found across regions"
    return max_feat_id + 1


def _feature_file(base: Path, raw_name: str) -> Path:
    de_style = base / f"musae_{raw_name}.json"
    other_style = base / f"musae_{raw_name}_features.json"
    if de_style.exists():
        return de_style
    return other_style


def _build_expected_from_raw(dataset_root: Path, region: str, n_feats: int):
    raw_name = REGION_TO_RAW_NAME[region]
    raw_base = dataset_root / region / "raw" / "twitch" / raw_name

    edges_path = raw_base / f"musae_{raw_name}_edges.csv"
    target_path = raw_base / f"musae_{raw_name}_target.csv"
    feature_path = _feature_file(raw_base, raw_name)

    assert edges_path.exists(), f"Missing edges file: {edges_path}"
    assert target_path.exists(), f"Missing target file: {target_path}"
    assert feature_path.exists(), f"Missing feature file: {feature_path}"

    edges_df = pd.read_csv(edges_path)
    target_df = pd.read_csv(target_path)
    with open(feature_path, "r") as f:
        features = json.load(f)

    # Loader behavior: keep first duplicate new_id row, then sort by new_id.
    target_df = target_df.drop_duplicates(subset=["new_id"], keep="first")
    target_df = target_df.sort_values("new_id").reset_index(drop=True)

    new_ids = target_df["new_id"].astype(int).to_numpy()
    node_id_to_idx = {nid: idx for idx, nid in enumerate(new_ids.tolist())}
    x_expected = np.zeros((len(new_ids), n_feats), dtype=np.float32)
    for i, node_id in enumerate(new_ids):
        feat_ids = np.asarray(features.get(str(int(node_id)), []), dtype=np.int64)
        if feat_ids.size == 0:
            continue
        feat_ids = feat_ids[(feat_ids >= 0) & (feat_ids < n_feats)]
        if feat_ids.size > 0:
            x_expected[i, np.unique(feat_ids)] = 1.0

    y_expected = (target_df["mature"] == True).astype(np.int64).to_numpy()  # noqa: E712

    src = edges_df.iloc[:, 0].astype(np.int64).to_numpy()
    dst = edges_df.iloc[:, 1].astype(np.int64).to_numpy()
    mapped_src = []
    mapped_dst = []
    dropped = 0
    for s, d in zip(src, dst):
        s_idx = node_id_to_idx.get(int(s))
        d_idx = node_id_to_idx.get(int(d))
        if s_idx is None or d_idx is None:
            dropped += 1
            continue
        mapped_src.append(s_idx)
        mapped_dst.append(d_idx)

    edge_expected = np.array([mapped_src, mapped_dst], dtype=np.int64)
    return x_expected, y_expected, edge_expected, dropped


@pytest.mark.parametrize("region", ["DE", "EN", "ES", "FR", "PT", "RU"])
def test_twitch_loader_matches_raw_semantics(region):
    dataset_root = _repo_root() / "dataset" / "Twitch"
    if not dataset_root.exists():
        pytest.skip(f"Twitch dataset root not found: {dataset_root}")

    n_feats = _global_feature_dim(dataset_root)
    dataset = TwitchFixed(root=str(dataset_root), name=region)
    data = dataset[0]

    # Cache version guard: this test is validating the v4 semantics.
    assert dataset.processed_paths[0].endswith("data_v4.pt")

    x_expected, y_expected, edge_expected, dropped = _build_expected_from_raw(dataset_root, region, n_feats)

    x_actual = data.x.cpu().numpy()
    y_actual = data.y.cpu().numpy()
    edge_actual = data.edge_index.cpu().numpy()

    # 1) Feature tensor must be exact multi-hot decode from raw JSON ID lists.
    assert x_actual.shape == x_expected.shape
    assert np.array_equal(x_actual, x_expected)

    # 2) Labels must align with sorted+dedup new_id order.
    assert y_actual.shape == y_expected.shape
    assert np.array_equal(y_actual, y_expected)

    # 3) Edge index must be exact remap through new_id -> row index, with invalid endpoints dropped.
    assert edge_actual.shape == edge_expected.shape
    assert np.array_equal(edge_actual, edge_expected)

    # 4) Additional invariants to catch common parsing regressions.
    assert np.all((x_actual == 0.0) | (x_actual == 1.0)), "Features must be binary multi-hot"
    assert set(np.unique(y_actual).tolist()).issubset({0, 1}), "Labels must be binary"
    assert edge_actual.size == 0 or (edge_actual.min() >= 0 and edge_actual.max() < data.num_nodes)

    # Current local data should not require dropping edges; keep this strict to catch future corruption.
    assert dropped == 0


def test_twitch_regions_share_feature_dimension():
    dataset_root = _repo_root() / "dataset" / "Twitch"
    if not dataset_root.exists():
        pytest.skip(f"Twitch dataset root not found: {dataset_root}")

    expected_dim = _global_feature_dim(dataset_root)
    for region in REGION_TO_RAW_NAME:
        dataset = TwitchFixed(root=str(dataset_root), name=region)
        data = dataset[0]
        assert data.x.size(1) == expected_dim, f"{region} feature dim mismatch: {data.x.size(1)} != {expected_dim}"
