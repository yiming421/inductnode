import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import parse_joint_training_args
from src.checkpoint_utils import save_checkpoint, load_checkpoint_config, load_checkpoint_states
from src.data_gc import create_dataset_splits
from train import create_unified_model


def test_checkpoint_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['train.py'])
    args = parse_joint_training_args()
    args.hidden = 16
    args.mplp_signature_dim = 64
    args.checkpoint_dir = str(tmp_path)
    model, predictor, _, _ = create_unified_model(args, 16, 'cpu')
    path = save_checkpoint(model, predictor, None, args, {'best_valid': .5}, 0)
    _, saved = load_checkpoint_config(path)
    other, other_predictor, _, _ = create_unified_model(args, 16, 'cpu')
    load_checkpoint_states(saved, other, other_predictor)
    for original, loaded in ((model, other), (predictor, other_predictor)):
        assert all(torch.equal(tensor, loaded.state_dict()[name]) for name, tensor in original.state_dict().items())


def test_official_ogb_splits_and_source_validation():
    class Dataset:
        def get_idx_split(self):
            return {'train': torch.tensor([0, 1]), 'valid': torch.tensor([2]), 'test': torch.tensor([3, 4])}
    target = create_dataset_splits(Dataset(), 'hiv')
    source = create_dataset_splits(Dataset(), 'bace', pretraining_mode=True)
    assert target['train'].tolist() == [0, 1]
    assert target['test'].tolist() == [3, 4]
    assert source['train'].tolist() == [0, 1, 3, 4]
    assert source['val'].tolist() == [2]
    assert 'test' not in source
