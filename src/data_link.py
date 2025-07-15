from torch_geometric.datasets import Planetoid, Coauthor, Amazon
from torch_geometric.utils import to_undirected, train_test_split_edges, add_self_loops, degree
from torch_sparse import SparseTensor
from ogb.linkproppred import PygLinkPropPredDataset
import torch
import scipy.io as sio
import scipy.sparse as ssp
from torch_geometric.data import Data
import numpy as np

def random_split_edges(data, val_ratio=0.1, test_ratio=0.2):
    result = train_test_split_edges(data, val_ratio=val_ratio, test_ratio=test_ratio)

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = result.train_pos_edge_index.t()
    split_edge['valid']['edge'] = result.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = result.val_neg_edge_index.t()
    split_edge['test']['edge'] = result.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = result.test_neg_edge_index.t()
    return split_edge

def load_data(dataset):
    name = dataset
    if dataset in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(root='../dataset', name=dataset)
    elif dataset in ['CS', 'Physics']:
        dataset = Coauthor(root='../dataset', name=dataset)
    elif dataset in ['Computers', 'Photo']:
        dataset = Amazon(root='../dataset', name=dataset)
    data = dataset[0]
    split_edge = random_split_edges(data)
    data.edge_index = to_undirected(split_edge['train']['edge'].t())
    #data.edge_index = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
    data.num_nodes = data.x.shape[0]
    data.edge_weight = None
    data.adj_t = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    data.name = name
    data.default_metric = "hits@100"
    return data, split_edge

def load_ogbl_data(dataset_name): 
    name_d = dataset_name
    dataset = PygLinkPropPredDataset(name=dataset_name, root='../dataset')
    data = dataset[0]
    # data.edge_index = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
    data.edge_weight = None
    data.adj_t = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    split_edge = dataset.get_edge_split()
    if dataset_name == 'ogbl-collab':
        selected_year_index = torch.reshape(
            (split_edge['train']['year'] >= 2011).nonzero(as_tuple=False), (-1,))
        split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
        split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
        split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]

        data.edge_index = to_undirected(split_edge['train']['edge'].t())
        # data.edge_index = add_self_loops(edge, num_nodes=data.num_nodes)[0]
        data.adj_t = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))

        full_edge_index = torch.cat([split_edge['valid']['edge'].t(), split_edge['train']['edge'].t()], dim=-1)
        full_edge_index = to_undirected(full_edge_index)
        # full_edge_index = add_self_loops(full_edge_index, num_nodes=data.num_nodes)[0]
        data.full_adj_t = SparseTensor.from_edge_index(full_edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    if dataset_name == 'ogbl-citation2':
        for name in ['train','valid','test']:
            u=split_edge[name]["source_node"]
            v=split_edge[name]["target_node"]
            split_edge[name]['edge']=torch.stack((u,v),dim=0).t()
        for name in ['valid','test']:
            u=split_edge[name]["source_node"].repeat(1, 1000).view(-1)
            v=split_edge[name]["target_node_neg"].view(-1)
            split_edge[name]['edge_neg']=torch.stack((u,v),dim=0).t()   
    if dataset_name == 'ogbl-ppa':
        data.x = None

    # Set default metric based on dataset
    if dataset_name == 'ogbl-ddi':
        data.default_metric = "hits@20"
    elif dataset_name == 'ogbl-collab':
        data.default_metric = "hits@50"
    elif dataset_name == 'ogbl-citation2':
        data.default_metric = "mrr"
    elif dataset_name == 'ogbl-ppa':
        data.default_metric = "hits@100"
    else:
        raise ValueError("Unknown OGB dataset: {}".format(dataset_name))

    data.name = name_d
        
    return data, split_edge

def load_all_data_link(train_datasets):
    data_list = []
    split_edge_list = []
    for dataset in train_datasets:
        print(dataset, flush=True)
        if dataset.startswith('ogbl'):
            data, split_edge = load_ogbl_data(dataset)
        else:
            data, split_edge = load_data(dataset)
        data_list.append(data)
        split_edge_list.append(split_edge)
    return data_list, split_edge_list