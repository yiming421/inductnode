"""NC/LP evaluation with the paper's ordered seeds; one run by default."""
import argparse
from pathlib import Path
import subprocess
import sys
from protocol import NC_LP_SEEDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setting', choices=['nc1', 'nc5-lp5'], required=True)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--runs', default=1, type=int)
    parser.add_argument('--gpu', default='0', help='CUDA index or cpu')
    args = parser.parse_args()
    if not 1 <= args.runs <= len(NC_LP_SEEDS):
        parser.error('runs must be between 1 and 10')
    root = Path(__file__).resolve().parents[1]
    command = [sys.executable, '-B', str(root / 'train.py'),
        '--use_pretrained_model', 'true', '--load_checkpoint', str(args.checkpoint.resolve()),
        '--runs', str(args.runs), '--gpu', args.gpu, '--seed', '42',
        '--enable_nc', 'true', '--enable_lp', str(args.setting == 'nc5-lp5').lower(),
        '--enable_gc', str(args.setting == 'nc5-lp5').lower(),
        '--gc_test_dataset', 'hiv', '--context_graph_num', '5',
        '--context_num', '1' if args.setting == 'nc1' else '5',
        '--unseen_test_context_samples', '3', '--use_pca_cache', 'false',
        '--unseen_test_context_seed_bases', ','.join(map(str, NC_LP_SEEDS[:args.runs]))]
    # The 5-shot run keeps the original NC -> LP -> HIV preparation and RNG
    # sequence. The graph-table protocol has its own evaluate_gc.py entry point.
    subprocess.run(command, cwd=root, check=True)


if __name__ == '__main__':
    main()
