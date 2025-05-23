from torch_geometric.datasets import Planetoid, WikiCS
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset
from torch_sparse import SparseTensor
import torch

def load_ogbn_data(dataset):
    name = dataset
    dataset = PygNodePropPredDataset(name=dataset)
    data = dataset[0]
    data.adj_t = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    split_idx = dataset.get_idx_split()
    data.name = name
    data.y = data.y.squeeze()
    return data, split_idx

def load_data(dataset):
    name = dataset
    if dataset in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root='dataset', name=dataset)
    elif dataset == 'WikiCS':
        dataset = WikiCS(root='dataset/WikiCS')
    elif dataset == 'Products':
        dict_products = torch.load('dataset/Products/ogbn-products_subset.pt')
        data = Data(
            x = dict_products['x'],
            y = dict_products['y'].squeeze(),
            train_mask = dict_products['train_mask'],
            val_mask = dict_products['val_mask'],
            test_mask = dict_products['test_mask'],
            adj_t = dict_products['adj_t'],
        )
        data.edge_index = torch.stack([data.adj_t.coo()[0], data.adj_t.coo()[1]], dim=0)
        print(data.x.shape)
        print(data.y.shape)
        print(data.edge_index.shape)
        print(data.train_mask.shape)
        print(data.val_mask.shape)
        print(data.test_mask.shape)
        print(data.train_mask.sum())
        print(data.val_mask.sum())
        print(data.test_mask.sum())

        dataset = [data]
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    data = dataset[0]
    data.adj_t = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    split_idx = dict()
    split_idx['train'] = data.train_mask.nonzero(as_tuple=False).reshape(-1)
    split_idx['valid'] = data.val_mask.nonzero(as_tuple=False).reshape(-1)
    split_idx['test'] = data.test_mask.nonzero(as_tuple=False).reshape(-1)
    data.name = name
    return data, split_idx

def load_all_data(train_datasets):
    data_list = []
    split_idx_list = []
    for dataset in train_datasets:
        print(dataset, flush=True)
        if dataset.startswith('ogbn-'):
            data, split_edge = load_ogbn_data(dataset)
        else:
            data, split_edge = load_data(dataset)
        data_list.append(data)
        split_idx_list.append(split_edge)
    return data_list, split_idx_list
