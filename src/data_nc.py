from __future__ import annotations
from torch_geometric.datasets import Planetoid, WikiCS, Coauthor, Amazon, Flickr, Airports, Actor, DeezerEurope, LastFMAsia, AttributedGraphDataset, CitationFull, FacebookPagePage
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset
from torch_sparse import SparseTensor
import torch
from collections import defaultdict
import torch.serialization
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr, TensorAttr
from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage
import os
try:
    torch.serialization.add_safe_globals([torch.Tensor, torch.Size, torch.dtype, torch.device, Data, DataEdgeAttr, DataTensorAttr, TensorAttr, GlobalStorage, NodeStorage, EdgeStorage, SparseTensor, defaultdict, type(None), tuple, list, dict, int, float, str, bool])
except (ImportError, AttributeError):
    pass

def get_project_root():
    """Get the project root directory"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def get_data_root():
    return os.environ.get('GILT_DATA_ROOT', os.path.join(get_project_root(), 'dataset'))

def build_adj_t(edge_index, num_nodes, make_undirected=True):
    """Build SparseTensor adjacency with optional symmetrization."""
    adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes))
    if make_undirected:
        adj_t = adj_t.to_symmetric()
    return adj_t.coalesce()

def rebalance_splits_for_pretraining(split_idx, strategy='smallest_for_valid'):
    if strategy == 'original':
        return split_idx
    train_size = len(split_idx['train'])
    valid_size = len(split_idx['valid'])
    test_size = len(split_idx['test'])
    if strategy == 'smallest_for_valid':
        sizes = {'train': train_size, 'valid': valid_size, 'test': test_size}
        smallest_split = min(sizes, key=sizes.get)
        print(f'  [Split Rebalance] Original - Train: {train_size:,} | Valid: {valid_size:,} | Test: {test_size:,}')
        print(f"  [Split Rebalance] Using '{smallest_split}' ({sizes[smallest_split]:,} nodes) for validation")
        valid_idx = split_idx[smallest_split]
        all_idx = torch.cat([split_idx['train'], split_idx['valid'], split_idx['test']])
        all_set = set(all_idx.tolist())
        valid_set = set(valid_idx.tolist())
        train_set = all_set - valid_set
        train_idx = torch.tensor(sorted(list(train_set)))
        new_split = {'train': train_idx, 'valid': valid_idx, 'test': valid_idx}
        print(f'  [Split Rebalance] New      - Train: {len(train_idx):,} | Valid: {len(valid_idx):,}')
        return new_split
    elif strategy == 'legacy':
        train_idx = torch.cat([split_idx['train'], split_idx['valid']])
        valid_idx = split_idx['test']
        test_idx = split_idx['test']
        print(f'  [Split Rebalance] Legacy mode - Train: {len(train_idx):,} (train+valid) | Valid: {len(valid_idx):,} (test)')
        return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
    else:
        raise ValueError(f'Unknown strategy: {strategy}')

def load_ogbn_data(dataset, make_undirected=True):
    name = dataset
    dataset = PygNodePropPredDataset(name=dataset, root=os.path.join(get_data_root()))
    data = dataset[0]
    data.adj_t = build_adj_t(data.edge_index, data.num_nodes, make_undirected=make_undirected)
    split_idx = dataset.get_idx_split()
    data.name = name
    data.y = data.y.squeeze()
    return (data, split_idx)

def load_ogbn_data_train(dataset, split_strategy='smallest_for_valid', make_undirected=True):
    name = dataset
    dataset = PygNodePropPredDataset(name=dataset, root=os.path.join(get_data_root()))
    data = dataset[0]
    data.adj_t = build_adj_t(data.edge_index, data.num_nodes, make_undirected=make_undirected)
    split_idx = dataset.get_idx_split()
    data.name = name
    data.y = data.y.squeeze()
    split_idx = rebalance_splits_for_pretraining(split_idx, strategy=split_strategy)
    return (data, split_idx)

def load_data_train(dataset_name: str, make_undirected: bool=True):
    root = os.path.join(get_data_root())
    print(f'Loading dataset {dataset_name} from root {root}', flush=True)
    if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root=root, name=dataset_name)
    elif dataset_name == 'WikiCS':
        dataset = WikiCS(root=os.path.join(root, 'WikiCS'))
    elif dataset_name in ['CS', 'Physics']:
        dataset = Coauthor(root=os.path.join(root, 'Coauthor'), name=dataset_name)
    elif dataset_name in ['Computers', 'Photo']:
        dataset = Amazon(root=os.path.join(root, 'Amazon'), name=dataset_name)
    elif dataset_name == 'Flickr':
        flickr_root = os.path.join(root, 'Flickr')
        dataset = Flickr(root=flickr_root)
    elif dataset_name in ['USA', 'Brazil', 'Europe']:
        dataset = Airports(root=os.path.join(root, 'Airports'), name=dataset_name)
    elif dataset_name == 'Actor':
        dataset = Actor(root=os.path.join(root, 'Actor'))
    elif dataset_name == 'DeezerEurope':
        dataset = DeezerEurope(root=os.path.join(root, 'DeezerEurope'))
    elif dataset_name == 'LastFMAsia':
        dataset = LastFMAsia(root=os.path.join(root, 'LastFMAsia'))
    elif dataset_name in ['Wiki', 'BlogCatalog', 'Facebook', 'TWeibo']:
        dataset = AttributedGraphDataset(root=os.path.join(root, 'AttributedGraph'), name=dataset_name)
    elif dataset_name in ['DBLP']:
        dataset = CitationFull(root=os.path.join(root, 'CitationFull'), name=dataset_name)
    elif dataset_name == 'FacebookPagePage':
        dataset = FacebookPagePage(root=os.path.join(root, 'FacebookPagePage'))
    else:
        raise ValueError(f'Dataset {dataset_name} not supported.')
    data = dataset[0]
    num_nodes = data.num_nodes
    print(f'Number of nodes: {num_nodes}', flush=True)
    print(f'Number of edges: {data.num_edges}', flush=True)
    labels = data.y
    nodes_by_class = defaultdict(list)
    for node_idx, label in enumerate(labels):
        nodes_by_class[label.item()].append(node_idx)
    train_idx = []
    valid_idx = []
    for class_label, indices in nodes_by_class.items():
        class_indices = torch.tensor(indices)
        num_class_nodes = len(class_indices)
        shuffled_indices = class_indices[torch.randperm(num_class_nodes)]
        train_end = int(num_class_nodes * 0.9)
        train_idx.extend(shuffled_indices[:train_end])
        valid_idx.extend(shuffled_indices[train_end:])
    train_idx = torch.tensor(train_idx)[torch.randperm(len(train_idx))]
    valid_idx = torch.tensor(valid_idx)[torch.randperm(len(valid_idx))]
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True
    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask[valid_idx] = True
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.adj_t = build_adj_t(data.edge_index, data.num_nodes, make_undirected=make_undirected)
    split_idx = {'train': train_idx, 'valid': valid_idx, 'test': valid_idx}
    data.name = dataset_name
    return (data, split_idx)

def load_data(dataset, make_undirected: bool=True, split_id: int | None=None):
    name = dataset
    dataset_aliases = {'AmzComp': 'Computers', 'AmzPhoto': 'Photo', 'AmzRatings': 'Amazon-ratings', 'CoCS': 'CS', 'CoPhysics': 'Physics', 'Roman': 'Roman-empire', 'Deezer': 'DeezerEurope', 'AirBrazil': 'Brazil', 'AirEU': 'Europe', 'AirUS': 'USA', 'FullCora': 'Cora-ML'}
    if dataset in dataset_aliases:
        dataset = dataset_aliases[dataset]
    if dataset in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root=os.path.join(get_data_root()), name=dataset)
    elif dataset == 'WikiCS':
        dataset = WikiCS(root=os.path.join(get_data_root(), 'WikiCS'))
    elif dataset in ['CS', 'Physics']:
        dataset = Coauthor(root=os.path.join(get_data_root(), 'Coauthor'), name=dataset)
    elif dataset in ['Computers', 'Photo']:
        dataset = Amazon(root=os.path.join(get_data_root(), 'Amazon'), name=dataset)
    elif dataset in ['USA', 'Brazil', 'Europe']:
        dataset = Airports(root=os.path.join(get_data_root(), 'Airports'), name=dataset)
    elif dataset == 'Actor':
        dataset = Actor(root=os.path.join(get_data_root(), 'Actor'))
    elif dataset == 'DeezerEurope':
        dataset = DeezerEurope(root=os.path.join(get_data_root(), 'DeezerEurope'))
    elif dataset == 'LastFMAsia':
        dataset = LastFMAsia(root=os.path.join(get_data_root(), 'LastFMAsia'))
    elif dataset in ['Wiki', 'BlogCatalog', 'Facebook', 'TWeibo']:
        dataset = AttributedGraphDataset(root=os.path.join(get_data_root(), 'AttributedGraph'), name=dataset)
    elif dataset in ['DBLP']:
        dataset = CitationFull(root=os.path.join(get_data_root(), 'CitationFull'), name=dataset)
    elif dataset == 'FacebookPagePage':
        dataset = FacebookPagePage(root=os.path.join(get_data_root(), 'FacebookPagePage'))
    else:
        raise ValueError(f'Dataset {dataset} not supported.')
    data = dataset[0]
    data.adj_t = build_adj_t(data.edge_index, data.num_nodes, make_undirected=make_undirected)
    if hasattr(data, 'train_mask') and data.train_mask is not None:
        split_idx = dict()
        train_mask = data.train_mask
        val_mask = data.val_mask
        test_mask = data.test_mask
        if train_mask.dim() == 2:
            num_splits = int(train_mask.shape[1])
            selected_split = 0 if split_id is None else int(split_id)
            if selected_split < 0 or selected_split >= num_splits:
                raise ValueError(f'Invalid split_id={selected_split} for dataset {name}. Valid range: [0, {num_splits - 1}]')
            data.predefined_num_splits = num_splits
            data.predefined_split_id = selected_split
            print(f'[load_data] Dataset has 2D train_mask with {num_splits} splits, using split {selected_split}')
            train_mask = train_mask[:, selected_split]
        if val_mask.dim() == 2:
            selected_split = 0 if split_id is None else int(split_id)
            val_mask = val_mask[:, selected_split]
        if test_mask is not None and test_mask.dim() == 2:
            selected_split = 0 if split_id is None else int(split_id)
            test_mask = test_mask[:, selected_split]
        split_idx['train'] = train_mask.nonzero(as_tuple=False).reshape(-1)
        split_idx['valid'] = val_mask.nonzero(as_tuple=False).reshape(-1)
        split_idx['test'] = test_mask.nonzero(as_tuple=False).reshape(-1) if test_mask is not None else torch.tensor([])
        num_nodes = data.num_nodes
        for split_name, split_indices in split_idx.items():
            if len(split_indices) > 0:
                if split_indices.max().item() >= num_nodes:
                    print(f'[CORRUPTION] {name} {split_name}_idx has invalid indices! Max: {split_indices.max().item()}, valid: [0, {num_nodes - 1}]')
                if split_indices.min().item() < 0:
                    print(f'[CORRUPTION] {name} {split_name}_idx has negative indices! Min: {split_indices.min().item()}')
                if len(split_indices) != len(split_indices.unique()):
                    duplicates = len(split_indices) - len(split_indices.unique())
                    print(f'[CORRUPTION] {name} {split_name}_idx has {duplicates} duplicate indices!')
    else:
        from collections import defaultdict
        num_nodes = data.num_nodes
        labels = data.y
        nodes_by_class = defaultdict(list)
        for node_idx, label in enumerate(labels):
            nodes_by_class[label.item()].append(node_idx)
        train_idx = []
        valid_idx = []
        test_idx = []
        for class_label, indices in nodes_by_class.items():
            class_indices = torch.tensor(indices)
            num_class_nodes = len(class_indices)
            perm = torch.randperm(num_class_nodes)
            class_indices = class_indices[perm]
            train_size = min(20, num_class_nodes // 2)
            valid_size = min(30, (num_class_nodes - train_size) // 2)
            train_idx.extend(class_indices[:train_size].tolist())
            valid_idx.extend(class_indices[train_size:train_size + valid_size].tolist())
            test_idx.extend(class_indices[train_size + valid_size:].tolist())
        split_idx = dict()
        split_idx['train'] = torch.tensor(train_idx)
        split_idx['valid'] = torch.tensor(valid_idx)
        split_idx['test'] = torch.tensor(test_idx)
        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.train_mask[split_idx['train']] = True
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask[split_idx['valid']] = True
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask[split_idx['test']] = True
    data.name = name
    return (data, split_idx)

def load_all_data(train_datasets, make_undirected: bool=True, expand_webkb_splits: bool=False):
    data_list = []
    split_idx_list = []
    webkb_datasets = {'Cornell', 'Texas', 'Wisconsin'}
    for dataset in train_datasets:
        print(dataset, flush=True)
        if expand_webkb_splits and dataset in webkb_datasets:
            data_first, split_first = load_data(dataset, make_undirected=make_undirected, split_id=0)
            num_splits = int(getattr(data_first, 'predefined_num_splits', 1))
            if num_splits <= 1:
                data_list.append(data_first)
                split_idx_list.append(split_first)
            else:
                data_first.name = f'{data_first.name}_split0'
                data_list.append(data_first)
                split_idx_list.append(split_first)
                for split_idx in range(1, num_splits):
                    data_split, split_edge = load_data(dataset, make_undirected=make_undirected, split_id=split_idx)
                    data_split.name = f'{data_split.name}_split{split_idx}'
                    data_list.append(data_split)
                    split_idx_list.append(split_edge)
                print(f'[load_all_data] Expanded {dataset} into {num_splits} official splits')
            continue
        if dataset.startswith('ogbn-'):
            data, split_edge = load_ogbn_data(dataset, make_undirected=make_undirected)
        else:
            data, split_edge = load_data(dataset, make_undirected=make_undirected)
        data_list.append(data)
        split_idx_list.append(split_edge)
    return (data_list, split_idx_list)

def expand_dataset_names(train_datasets):
    """Expand special dataset names like 'Twitch' into all regions"""
    expanded_datasets = []
    for dataset in train_datasets:
        expanded_datasets.append(dataset)
    return expanded_datasets

def load_all_data_train(train_datasets, split_strategy='smallest_for_valid', make_undirected: bool=True):
    expanded_datasets = expand_dataset_names(train_datasets)
    data_list = []
    split_idx_list = []
    for dataset in expanded_datasets:
        print(dataset, flush=True)
        if dataset.startswith('ogbn-'):
            data, split_edge = load_ogbn_data_train(dataset, split_strategy=split_strategy, make_undirected=make_undirected)
        else:
            data, split_edge = load_data_train(dataset, make_undirected=make_undirected)
        data_list.append(data)
        split_idx_list.append(split_edge)
    return (data_list, split_idx_list)
