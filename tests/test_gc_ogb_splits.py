import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import data_gc


class DummyOGBDataset:
    def get_idx_split(self):
        return {
            'train': torch.tensor([0, 1, 2]),
            'valid': torch.tensor([3]),
            'test': torch.tensor([4, 5]),
        }


def test_ogb_official_splits_used_without_fug_env(monkeypatch):
    monkeypatch.delenv('USE_FUG_EMB', raising=False)

    split_idx = data_gc.create_dataset_splits(DummyOGBDataset(), 'hiv')

    assert split_idx['train'].tolist() == [0, 1, 2]
    assert split_idx['val'].tolist() == [3]
    assert split_idx['test'].tolist() == [4, 5]


def test_hiv_alias_uses_scaffold_fallback_when_official_split_unavailable(monkeypatch):
    calls = []

    monkeypatch.setattr(data_gc, 'load_precomputed_splits', lambda dataset_name, root='./dataset': None)

    def fake_scaffold(dataset, train_ratio, val_ratio, test_ratio, seed, pretraining_mode):
        calls.append('scaffold')
        return {'train': torch.tensor([0]), 'val': torch.tensor([1]), 'test': torch.tensor([2])}

    def fake_random(dataset, train_ratio, val_ratio, test_ratio, seed, pretraining_mode):
        calls.append('random')
        return {'train': torch.tensor([9]), 'val': torch.tensor([8]), 'test': torch.tensor([7])}

    monkeypatch.setattr(data_gc, 'create_scaffold_splits', fake_scaffold)
    monkeypatch.setattr(data_gc, 'create_random_splits', fake_random)

    split_idx = data_gc.create_dataset_splits(object(), 'hiv')

    assert calls == ['scaffold']
    assert split_idx['train'].tolist() == [0]
