import torch
from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from ogb.nodeproppred import PygNodePropPredDataset
from torch_sparse import SparseTensor


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
    else:
        raise ValueError(f"Dataset {dataset} not supported.")
    data = dataset[0]
    data.adj_t = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    split_idx = dict()
    split_idx['train'] = data.train_mask.nonzero(as_tuple=False).view(-1)
    split_idx['valid'] = data.val_mask.nonzero(as_tuple=False).view(-1)
    split_idx['test'] = data.test_mask.nonzero(as_tuple=False).view(-1)
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
