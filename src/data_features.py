import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import TUDataset

def load_ogb_original_features(name, ogb_root='./dataset/ogb'):
    ogb_names = {'bace': 'ogbg-molbace', 'bbbp': 'ogbg-molbbbp', 'hiv': 'ogbg-molhiv', 'chemhiv': 'ogbg-molhiv', 'pcba': 'ogbg-molpcba', 'chempcba': 'ogbg-molpcba', 'molpcba': 'ogbg-molpcba', 'tox21': 'ogbg-moltox21', 'clintox': 'ogbg-molclintox', 'muv': 'ogbg-molmuv', 'sider': 'ogbg-molsider', 'toxcast': 'ogbg-moltoxcast', 'ogbg-molbace': 'ogbg-molbace', 'ogbg-molbbbp': 'ogbg-molbbbp', 'ogbg-molhiv': 'ogbg-molhiv', 'ogbg-molpcba': 'ogbg-molpcba', 'ogbg-moltox21': 'ogbg-moltox21', 'ogbg-molclintox': 'ogbg-molclintox', 'ogbg-molmuv': 'ogbg-molmuv', 'ogbg-molsider': 'ogbg-molsider', 'ogbg-moltoxcast': 'ogbg-moltoxcast'}
    if name not in ogb_names:
        print(f'[Original-Features] Unknown dataset: {name}')
        return None
    full_ogb_name = ogb_names[name]
    try:
        print(f"[Original-Features] Loading OGB dataset '{full_ogb_name}'...")
        dataset = PygGraphPropPredDataset(name=full_ogb_name, root=ogb_root)
        print(f'[Original-Features] Loaded {len(dataset)} graphs')
    except Exception as e:
        print(f'[Original-Features] Failed to load OGB dataset: {e}')
        return None
    print(f'[Original-Features] Collecting original 9-dim features from all graphs...')
    all_node_features = []
    for i in range(len(dataset)):
        graph = dataset[i]
        all_node_features.append(graph.x.float())
    node_embs = torch.cat(all_node_features, dim=0)
    total_nodes = sum((graph.num_nodes for graph in dataset))
    if node_embs.size(0) != total_nodes:
        print(f'[Original-Features] Size mismatch: {total_nodes} nodes vs {node_embs.size(0)} features')
        return None
    node_idx = 0
    sample_graph = dataset[0]
    is_multitask = sample_graph.y.numel() > 1
    if is_multitask:
        print(f'[Original-Features] Multi-task dataset detected, adding task_mask for {sample_graph.y.numel()} tasks')
    node_index_mapping = {}
    for i in range(len(dataset)):
        graph = dataset[i]
        n_nodes = graph.num_nodes
        node_index_mapping[i] = torch.arange(node_idx, node_idx + n_nodes, dtype=torch.long)
        node_idx += n_nodes
        if is_multitask:
            if graph.y.dtype.is_floating_point:
                graph.task_mask = (~torch.isnan(graph.y)).float()
            else:
                graph.task_mask = torch.ones_like(graph.y, dtype=torch.float)
    print(f'[Original-Features] Created external node index mapping for {len(dataset)} graphs')
    original_features_mapping = {'node_index_mapping': node_index_mapping, 'node_embs': node_embs, 'uses_fug_embeddings': False, 'uses_original_features': True, 'name': name, 'is_multitask': is_multitask}
    print(f"[Original-Features] Ready! External mapping for '{name}' with {node_embs.size(0)} nodes")
    print(f'[Original-Features] Original features: {node_embs.shape} (9-dim) - will be processed with PCA/padding to hidden_dim')
    return (dataset, original_features_mapping)

def load_tu_original_features(name, tu_root='./dataset/TU'):
    try:
        print(f"[TU-Original-Features] Loading TU dataset '{name}'...")
        dataset = TUDataset(root=tu_root, name=name)
        print(f'[TU-Original-Features] Loaded {len(dataset)} graphs')
    except Exception as e:
        print(f'[TU-Original-Features] Failed to load TU dataset: {e}')
        return None
    if dataset.num_node_features == 0:
        print(f"[TU-Original-Features] Dataset '{name}' has no node features - cannot be used")
        return None
    print(f'[TU-Original-Features] Collecting original {dataset.num_node_features}-dim features from all graphs...')
    all_node_features = []
    for i in range(len(dataset)):
        graph = dataset[i]
        if graph.x is None:
            print(f'[TU-Original-Features] Graph {i} has no node features - cannot be used')
            return None
        all_node_features.append(graph.x.float())
    node_embs = torch.cat(all_node_features, dim=0)
    total_nodes = sum((graph.num_nodes for graph in dataset))
    if node_embs.size(0) != total_nodes:
        print(f'[TU-Original-Features] Size mismatch: {total_nodes} nodes vs {node_embs.size(0)} features')
        return None
    node_idx = 0
    sample_graph = dataset[0]
    is_multitask = sample_graph.y.numel() > 1
    if is_multitask:
        print(f'[TU-Original-Features] Multi-task dataset detected, adding task_mask for {sample_graph.y.numel()} tasks')
    node_index_mapping = {}
    for i in range(len(dataset)):
        graph = dataset[i]
        n_nodes = graph.num_nodes
        node_index_mapping[i] = torch.arange(node_idx, node_idx + n_nodes, dtype=torch.long)
        node_idx += n_nodes
        if is_multitask:
            if graph.y.dtype.is_floating_point:
                graph.task_mask = (~torch.isnan(graph.y)).float()
            else:
                graph.task_mask = torch.ones_like(graph.y, dtype=torch.float)
    print(f'[TU-Original-Features] Created external node index mapping for {len(dataset)} graphs')
    original_features_mapping = {'node_index_mapping': node_index_mapping, 'node_embs': node_embs, 'uses_fug_embeddings': False, 'uses_original_features': True, 'name': name, 'is_multitask': is_multitask}
    print(f"[TU-Original-Features] Ready! External mapping for '{name}' with {node_embs.size(0)} nodes")
    print(f'[TU-Original-Features] Original features: {node_embs.shape} ({node_embs.shape[1]}-dim) - will be processed with PCA/padding to hidden_dim')
    return (dataset, original_features_mapping)
