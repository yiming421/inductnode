"""Fetch the fixed released weights; generated files stay outside Git."""
import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.checkpoint_utils import ARCHITECTURE_KEYS

URL = ('https://huggingface.co/fdsajkshf/gilt-checkpoint/resolve/'
       'bc1cd35b4a2fb6dc720d96d32b3a933f68bea864/gilt_model.pt')
SHA256 = '1caebd2210bc7d8d764ee58d9b85b1f736292c68036973f547301567ea97aa70'


def convert(source, output):
    import torch
    digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    if digest != SHA256:
        raise ValueError('Released checkpoint SHA256 mismatch; refusing to load it')
    # This older, hash-pinned release contains NumPy scalar metrics. Its weights
    # are read only after byte-level verification, then saved as tensor-only I/O.
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    keys = (*ARCHITECTURE_KEYS, 'graph_pooling')
    safe = {key: checkpoint[key] for key in ('model_state_dict', 'predictor_state_dict')}
    safe['args'] = {key: checkpoint['args'][key] for key in keys if key in checkpoint['args']}
    with output.open('xb') as handle:
        torch.save(safe, handle)
    loaded = torch.load(output, map_location='cpu', weights_only=True)
    for key in ('model_state_dict', 'predictor_state_dict'):
        assert all(torch.equal(value, loaded[key][name]) for name, value in checkpoint[key].items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('checkpoints/gilt_model.pt'))
    parser.add_argument('--source', type=Path, help='Use a local copy of the same hash-pinned release')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Checkpoint already exists: {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Source plus converted weights peak below 100 MiB; keep temporary data on
    # the destination volume and leave at least 150 MiB of additional headroom.
    if shutil.disk_usage(args.output.parent).free < 250 * 1024**2:
        raise OSError('At least 250 MiB of free space is required')
    with tempfile.TemporaryDirectory(dir=args.output.parent, prefix='.download-') as temporary:
        source = args.source
        if source is None:
            source = Path(temporary) / 'release.pt'
            urllib.request.urlretrieve(URL, source)
        convert(source, args.output)
    print(f'Saved verified weights: {args.output}')


if __name__ == '__main__':
    main()
