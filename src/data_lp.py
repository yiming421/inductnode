from torch_geometric.datasets import Planetoid, Coauthor, Amazon, Flickr, Airports, AttributedGraphDataset, CitationFull, FacebookPagePage, WikiCS, Actor, DeezerEurope, LastFMAsia
from torch_geometric.utils import to_undirected
from torch_geometric import transforms
from torch_sparse import SparseTensor
from ogb.linkproppred import PygLinkPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
import torch
import numpy as np
import os
try:
    from torch_geometric.data.data import Data as TorchGeometricData
    torch.serialization.add_safe_globals([TorchGeometricData, np.core.multiarray._reconstruct, np.ndarray, np.dtype])
except (ImportError, AttributeError):
    pass
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.environ.get('GILT_DATA_ROOT', os.path.join(PROJECT_ROOT, 'dataset'))

def safe_load_dataset(dataset_class, **kwargs):
    """Safely load PyTorch Geometric datasets with proper error handling."""
    try:
        return dataset_class(**kwargs)
    except Exception as e:
        if 'weights_only' in str(e) or 'WeightsUnpickler' in str(e):
            original_load = torch.load

            def patched_load(*args, **kwargs):
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            torch.load = patched_load
            try:
                result = dataset_class(**kwargs)
                return result
            finally:
                torch.load = original_load
        else:
            raise e

def safe_get_edge_split(dataset):
    """Safely get edge split from OGB dataset with proper error handling."""
    try:
        return dataset.get_edge_split()
    except Exception as e:
        if 'weights_only' in str(e) or 'WeightsUnpickler' in str(e):
            original_load = torch.load

            def patched_load(*args, **kwargs):
                if 'weights_only' not in kwargs:
                    kwargs['weights_only'] = False
                return original_load(*args, **kwargs)
            torch.load = patched_load
            try:
                result = dataset.get_edge_split()
                return result
            finally:
                torch.load = original_load
        else:
            raise e

def random_split_edges(data, val_ratio=0.1, test_ratio=0.2):
    device = data.x.device if hasattr(data, 'x') and data.x is not None else 'cpu'
    if hasattr(data, 'edge_index'):
        data.edge_index = to_undirected(data.edge_index).to(device)
    transform = transforms.RandomLinkSplit(num_val=val_ratio, num_test=test_ratio, is_undirected=True, add_negative_train_samples=False, neg_sampling_ratio=1.0)
    train_data, val_data, test_data = transform(data)
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = train_data.edge_label_index.t()[train_data.edge_label == 1]
    split_edge['valid']['edge'] = val_data.edge_label_index.t()[val_data.edge_label == 1]
    split_edge['valid']['edge_neg'] = val_data.edge_label_index.t()[val_data.edge_label == 0]
    split_edge['test']['edge'] = test_data.edge_label_index.t()[test_data.edge_label == 1]
    split_edge['test']['edge_neg'] = test_data.edge_label_index.t()[test_data.edge_label == 0]
    return split_edge

def load_data(dataset, device='cpu', is_pretraining=False):
    name = dataset
    if dataset in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = safe_load_dataset(Planetoid, root=DATASET_ROOT, name=dataset)
    elif dataset in ['CS', 'Physics']:
        dataset = safe_load_dataset(Coauthor, root=os.path.join(DATASET_ROOT, 'Coauthor'), name=dataset)
    elif dataset in ['Computers', 'Photo']:
        dataset = safe_load_dataset(Amazon, root=os.path.join(DATASET_ROOT, 'Amazon'), name=dataset)
    elif dataset == 'Flickr':
        dataset = safe_load_dataset(Flickr, root=os.path.join(DATASET_ROOT, 'Flickr'))
    elif dataset in ['USA', 'Brazil', 'Europe']:
        dataset = safe_load_dataset(Airports, root=os.path.join(DATASET_ROOT, 'Airports'), name=dataset)
    elif dataset in ['Wiki', 'BlogCatalog']:
        dataset = safe_load_dataset(AttributedGraphDataset, root=os.path.join(DATASET_ROOT, 'AttributedGraph'), name=dataset)
    elif dataset == 'DBLP':
        dataset = safe_load_dataset(CitationFull, root=os.path.join(DATASET_ROOT, 'CitationFull'), name=dataset)
    elif dataset in ['FacebookPage', 'FacebookPagePage']:
        dataset = safe_load_dataset(FacebookPagePage, root=os.path.join(DATASET_ROOT, 'FacebookPagePage'))
    elif dataset == 'ogbn-arxiv':
        dataset = safe_load_dataset(PygNodePropPredDataset, name=dataset, root=DATASET_ROOT)
    elif dataset == 'WikiCS':
        dataset = safe_load_dataset(WikiCS, root=os.path.join(DATASET_ROOT, 'WikiCS'))
    elif dataset == 'Actor':
        dataset = safe_load_dataset(Actor, root=os.path.join(DATASET_ROOT, 'Actor'))
    elif dataset == 'DeezerEurope':
        dataset = safe_load_dataset(DeezerEurope, root=os.path.join(DATASET_ROOT, 'DeezerEurope'))
    elif dataset == 'LastFMAsia':
        dataset = safe_load_dataset(LastFMAsia, root=os.path.join(DATASET_ROOT, 'LastFMAsia'))
    else:
        raise ValueError(f'Dataset {dataset} not supported in load_data')
    data = dataset[0]
    if hasattr(data, 'x') and data.x is not None:
        data.x = data.x.to(device)
    if hasattr(data, 'edge_index'):
        data.edge_index = data.edge_index.to(device)
    split_edge = random_split_edges(data)
    if is_pretraining:
        train_edges = split_edge['train']['edge']
        test_edges = split_edge['test']['edge']
        combined_edges = torch.cat([train_edges, test_edges], dim=0)
        split_edge['train']['edge'] = combined_edges
        data.edge_index = to_undirected(combined_edges.t())
    else:
        data.edge_index = to_undirected(split_edge['train']['edge'].t())
    data.num_nodes = data.x.shape[0]
    data.edge_weight = None
    data.adj_t = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce().to(device)
    data.name = name
    data.default_metric = 'hits@100'
    return (data, split_edge)

def load_ogbl_data(dataset_name, device='cpu'):
    name_d = dataset_name
    dataset = safe_load_dataset(PygLinkPropPredDataset, name=dataset_name, root=DATASET_ROOT)
    data = dataset[0]
    data.edge_weight = None
    if hasattr(data, 'x') and data.x is not None:
        data.x = data.x.to(device)
    if hasattr(data, 'edge_index'):
        data.edge_index = data.edge_index.to(device)
    data.adj_t = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce().to(device)
    split_edge = safe_get_edge_split(dataset)
    if dataset_name == 'ogbl-collab':
        selected_year_index = torch.reshape((split_edge['train']['year'] >= 2011).nonzero(as_tuple=False), (-1,))
        split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
        split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
        split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
        data.edge_index = to_undirected(split_edge['train']['edge'].t()).to(device)
        data.adj_t = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).to(device)
        full_edge_index = torch.cat([split_edge['valid']['edge'].t(), split_edge['train']['edge'].t()], dim=-1)
        full_edge_index = to_undirected(full_edge_index).to(device)
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, sparse_sizes=(data.num_nodes, data.num_nodes)).to(device)
    if dataset_name == 'ogbl-collab':
        data.default_metric = 'hits@50'
    else:
        raise ValueError('Unknown OGB dataset: {}'.format(dataset_name))
    data.name = name_d
    return (data, split_edge)

def load_all_data_link(train_datasets, device='cpu', is_pretraining=False):
    data_list = []
    split_edge_list = []
    for dataset in train_datasets:
        print(dataset, flush=True)
        if dataset.startswith('ogbl'):
            data, split_edge = load_ogbl_data(dataset, device=device)
        else:
            data, split_edge = load_data(dataset, device=device, is_pretraining=is_pretraining)
        if not hasattr(data, 'x') or data.x is None:
            raise ValueError('This release requires numerical node features')
        data.x = data.x.to(device)
        data.adj_t = data.adj_t.to(device)
        if hasattr(data, 'edge_index'):
            data.edge_index = data.edge_index.to(device)
        if hasattr(data, 'full_adj_t') and data.full_adj_t is not None:
            data.full_adj_t = data.full_adj_t.to(device)
        for split_name, split_data in split_edge.items():
            if isinstance(split_data, dict):
                for key, tensor_data in split_data.items():
                    if isinstance(tensor_data, torch.Tensor):
                        split_edge[split_name][key] = tensor_data.to(device)
        data_list.append(data)
        split_edge_list.append(split_edge)
    return (data_list, split_edge_list)
