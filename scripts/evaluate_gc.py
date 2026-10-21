"""Regenerate the graph-table protocol from raw datasets and frozen weights."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train
import src.data_gc as data_gc
import src.engine_gc as engine
from src.data_features import load_tu_original_features
from src.checkpoint_utils import load_checkpoint_config, override_args_from_checkpoint, load_checkpoint_states
from protocol import HIV_SEEDS, PCBA_SEEDS, PROTEINS_SEEDS


def molecular_contexts(dataset, train_ids, seed):
    """Five training graphs per class/assay in the original class order."""
    labels = torch.cat([graph.y.reshape(1, -1) for graph in dataset])
    train_ids = np.asarray(train_ids)
    rng = np.random.RandomState(seed)
    contexts = {}
    for task in range(labels.shape[1]):
        contexts[task] = {}
        for cls in (0, 1):
            candidates = train_ids[labels[train_ids, task].numpy() == cls]
            if len(candidates) < 5:
                raise ValueError(f'Assay {task} class {cls} has fewer than five training graphs')
            if labels.shape[1] == 1:
                selected = list(candidates[:5])
                for count, index in enumerate(candidates[5:], start=6):
                    choice = rng.randint(0, count)
                    if choice < 5:
                        selected[choice] = index
            else:
                selected = rng.choice(candidates, size=5, replace=False)
            indices = [int(i) for i in selected]
            contexts[task][cls] = dict(indices=indices, graphs=[dataset[i] for i in indices])
    return contexts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['hiv', 'pcba', 'PROTEINS'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--contexts', type=int, default=1)
    parser.add_argument('--ogb-root', default='./dataset/ogb')
    parser.add_argument('--tu-root', default='./dataset/TU')
    parser.add_argument('--preprocess-device', default='cuda:0', help='PROTEINS PCA device; reference used an NVIDIA A100')
    cli = parser.parse_args()
    seeds = {'hiv': HIV_SEEDS, 'pcba': PCBA_SEEDS, 'PROTEINS': PROTEINS_SEEDS}[cli.dataset]
    if not 1 <= cli.contexts <= len(seeds):
        parser.error(f'contexts must be between 1 and {len(seeds)}')
    seeds = seeds[:cli.contexts]
    torch.set_num_threads(4)
    # The reference TU preparation used CUDA PCA; deterministic CPU inference
    # starts after that preparation, matching the original evaluation sequence.
    torch.use_deterministic_algorithms(cli.dataset != 'PROTEINS')
    sys.argv = [sys.argv[0]]
    args = train.parse_joint_training_args()
    _, checkpoint = load_checkpoint_config(cli.checkpoint)
    override_args_from_checkpoint(args, checkpoint['args'])
    vars(args).update(enable_nc=False, enable_lp=False, enable_gc=True,
        gc_test_dataset=cli.dataset, context_graph_num=5, context_num=5,
        use_pretrained_model=True, use_pca_cache=False, gc_test_batch_size=512,
        pca_device='cpu', gc_metric='auc', ogb_root=cli.ogb_root,
        unseen_test_context_samples=1, unseen_test_context_seed_base=0)
    device = torch.device('cpu')
    if cli.dataset == 'PROTEINS':
        args.gc_metric = 'accuracy'
        args.gc_test_batch_size = 128
        args.pca_device = 'gpu' if cli.preprocess_device.startswith('cuda') else 'cpu'
        preprocessing_device = (torch.cuda.get_device_name(torch.device(cli.preprocess_device))
            if args.pca_device == 'gpu' else 'CPU')
        print(f'PROTEINS preprocessing device: {preprocessing_device}')
        if 'A100' not in preprocessing_device:
            print('PROTEINS reference PCA used an NVIDIA A100. Other devices can change '
                  'PCA signs and near-zero components, so scores may differ from the paper.')
        train.set_all_random_seeds(42)
        loaded = load_tu_original_features('PROTEINS', tu_root=cli.tu_root)
        if loaded is None:
            raise RuntimeError('Could not load PROTEINS. Place its PyG raw/processed cache under --tu-root or check network access.')
        dataset, mapping = loaded
        dataset.name = 'PROTEINS'
        labels = np.array([int(graph.y.item()) for graph in dataset])
        trainval, test = train_test_split(np.arange(len(dataset)), test_size=.2, random_state=42, stratify=labels)
        source, valid = train_test_split(trainval, test_size=.25, random_state=42, stratify=labels[trainval])
        splits = dict(train=np.sort(source), val=np.sort(valid), test=np.sort(test))
        contexts = {}
        for seed in seeds:
            train.set_all_random_seeds(seed)
            info = data_gc.prepare_graph_data_for_pfn(dataset, splits, context_k=5, device=device)
            contexts[seed] = info['context_graphs']
        info['feature_mapping'] = mapping
        train.set_all_random_seeds(42)
        train.process_datasets_for_models([dataset], [info], args, cli.preprocess_device, test_datasets=True)
        model, predictor, _, _ = train.create_unified_model(args, args.hidden, device)
        load_checkpoint_states(checkpoint, model, predictor)
        inference_seed = 42
    else:
        # Preserve preprocessing RNG: support selection uses an independent
        # RandomState, so it cannot perturb the PCA or model initialization.
        contexts = {}
        original_sampler = data_gc.prepare_graph_data_for_pfn
        def fixed_sampler(dataset, split_idx, context_k=5, device='cpu'):
            for seed in seeds:
                contexts[seed] = molecular_contexts(dataset, split_idx['train'], seed)
            tasks = dataset[0].y.numel()
            return dict(dataset=dataset, name=cli.dataset, split_idx=split_idx,
                context_graphs=contexts[seeds[0]], context_labels={t: {0: 0, 1: 1} for t in range(tasks)},
                num_classes=dataset.num_classes, num_features=dataset.num_node_features,
                is_multitask=tasks > 1, num_tasks=tasks)
        data_gc.prepare_graph_data_for_pfn = fixed_sampler
        train.set_all_random_seeds(0)
        try:
            loaded = train.load_and_preprocess_data(args, device, skip_training_data=True)
        finally:
            data_gc.prepare_graph_data_for_pfn = original_sampler
        info = loaded['gc_test'][1][0]
        dataset, splits = info['dataset'], info['split_idx']
        model, predictor, _, _ = train.create_unified_model(args, args.hidden, device)
        load_checkpoint_states(checkpoint, model, predictor)
        inference_seed = 0
    model.eval().requires_grad_(False)
    predictor.eval().requires_grad_(False)
    torch.use_deterministic_algorithms(True)
    multitask = dataset[0].y.numel() > 1
    query = np.asarray(splits['test'])
    batch_size = args.gc_test_batch_size
    if len(query) % batch_size == 1:
        batch_size -= 1
    loaders = data_gc.create_data_loaders(dataset, {'test': query}, batch_size=batch_size,
        shuffle=False, task_idx=None if multitask else 0, use_index_tracking=True)
    name = 'evaluate_graph_classification_multitask_vectorized' if multitask else 'evaluate_graph_classification_single_task'
    base_eval = getattr(engine, name)
    predictions = {seed: [] for seed in seeds}
    def sweep(*unused, **kwargs):
        if not multitask:
            kwargs['task_idx'] = 0
        for seed in seeds:
            info['context_graphs'] = contexts[seed]
            kwargs['return_logits'] = True
            result = base_eval(model, predictor, info, loaders, **kwargs)
            predictions[seed].append(result['test_logits'].cpu())
        return result
    setattr(engine, name, sweep)
    kwargs = dict(pooling_method=args.graph_pooling, device=device,
        normalize_class_h=args.normalize_class_h, dataset_name=cli.dataset, args=args,
        num_augmentations=5, normalize_views=True, use_batchnorm=args.use_batchnorm,
        linear_projection=False, feature_projection=True, edge_dropout_rate=0.)
    if not multitask:
        kwargs['task_idx'] = 0
    train.set_all_random_seeds(inference_seed)
    try:
        with torch.inference_mode():
            getattr(engine, name + '_with_tta')(model, predictor, info, loaders, **kwargs)
    finally:
        setattr(engine, name, base_eval)
    truth = torch.cat([dataset[int(i)].y.reshape(1, -1) for i in query]).numpy()
    rows = []
    for seed, views in predictions.items():
        assert len(views) == 6
        logits = torch.stack(views).mean(0)
        assert torch.isfinite(logits).all() and logits.shape[0] == len(query)
        if multitask:
            scores = []
            for task in range(truth.shape[1]):
                mask = np.isfinite(truth[:, task])
                y = truth[mask, task]
                if len(np.unique(y)) == 2:
                    z = logits[mask, task].numpy()
                    scores.append((roc_auc_score(y, z), average_precision_score(y, z)))
            row = dict(zip(('roc_auc', 'ap'), np.mean(scores, axis=0)))
        else:
            row = dict(roc_auc=roc_auc_score(truth[:, 0], logits.double().softmax(1)[:, 1].numpy()))
            if cli.dataset == 'PROTEINS':
                row['accuracy'] = accuracy_score(truth[:, 0], logits.argmax(1).numpy())
        rows.append(row)
        print(f'{cli.dataset} context_seed={seed}: {row}')
    for metric in rows[0]:
        values = np.array([row[metric] for row in rows]) * 100
        std = values.std(ddof=1) if len(values) > 1 else 0.
        print(f'{cli.dataset} {metric}: {values.mean():.4f} +/- {std:.4f} (contexts={len(rows)})')


if __name__ == '__main__':
    main()
