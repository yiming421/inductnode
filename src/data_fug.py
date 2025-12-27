import os
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.datasets import TUDataset

from torch_geometric.datasets import GNNBenchmarkDataset, MNISTSuperpixels, ModelNet
from torch_geometric.transforms import FaceToEdge


def load_ogb_fug_dataset(name, ogb_root='./dataset/ogb', fug_root='./fug'):
    """
    Simple FUG dataset loader - just loads one unified node embedding file.
    Under the new setting, all datasets have unified embeddings, no complex logic needed.
    
    Args:
        name (str): Dataset name (e.g., 'bace', 'bbbp', 'hiv', 'pcba')
        ogb_root (str): Root to download/store OGB datasets
        fug_root (str): Root where FUG embeddings exist
        
    Returns:
        dataset with node_embs attribute, or None if failed
    """
    # Dataset name mapping
    ogb_names = {
        'bace': 'ogbg-molbace',
        'bbbp': 'ogbg-molbbbp',
        'hiv': 'ogbg-molhiv',
        'chemhiv': 'ogbg-molhiv',
        'pcba': 'ogbg-molpcba',
        'chempcba': 'ogbg-molpcba',
        'molpcba': 'ogbg-molpcba',
        'tox21': 'ogbg-moltox21',
        'clintox': 'ogbg-molclintox',
        'muv': 'ogbg-molmuv',
        'sider': 'ogbg-molsider',
        'toxcast': 'ogbg-moltoxcast',
        # Also accept full OGB names directly
        'ogbg-molbace': 'ogbg-molbace',
        'ogbg-molbbbp': 'ogbg-molbbbp',
        'ogbg-molhiv': 'ogbg-molhiv',
        'ogbg-molpcba': 'ogbg-molpcba',
        'ogbg-moltox21': 'ogbg-moltox21',
        'ogbg-molclintox': 'ogbg-molclintox',
        'ogbg-molmuv': 'ogbg-molmuv',
        'ogbg-molsider': 'ogbg-molsider',
        'ogbg-moltoxcast': 'ogbg-moltoxcast',
    }
    
    if name not in ogb_names:
        print(f"[FUG-Simple] Unknown dataset: {name}")
        return None
        
    full_ogb_name = ogb_names[name]
    
    # Load OGB dataset
    try:
        print(f"[FUG-Simple] Loading OGB dataset '{full_ogb_name}'...")
        dataset = PygGraphPropPredDataset(name=full_ogb_name, root=ogb_root)
        print(f"[FUG-Simple] Loaded {len(dataset)} graphs")
    except Exception as e:
        print(f"[FUG-Simple] Failed to load OGB dataset: {e}")
        return None
    
    # Load the single unified node embeddings file
    embedding_file = os.path.join(fug_root, name, f'{full_ogb_name}_node_embeddings.pt')
    if not os.path.exists(embedding_file):
        print(f"[FUG-Simple] Embedding file not found: {embedding_file}")
        return None
        
    try:
        node_embs = torch.load(embedding_file, map_location='cpu')
        print(f"[FUG-Simple] Loaded embeddings: {node_embs.shape}")
    except Exception as e:
        print(f"[FUG-Simple] Failed to load embeddings: {e}")
        return None
    
    # Count total nodes to verify embedding size
    total_nodes = sum(graph.num_nodes for graph in dataset)
    if node_embs.size(0) != total_nodes:
        print(f"[FUG-Simple] Size mismatch: {total_nodes} nodes vs {node_embs.size(0)} embeddings")
        return None
    
    # Create external node index mapping (don't modify graph.x!)
    node_idx = 0
    sample_graph = dataset[0]
    is_multitask = sample_graph.y.numel() > 1
    
    if is_multitask:
        print(f"[FUG-Simple] Multi-task dataset detected, adding task_mask for {sample_graph.y.numel()} tasks")
    
    # Create external mapping instead of modifying graphs
    node_index_mapping = {}
    for i in range(len(dataset)):
        graph = dataset[i]
        n_nodes = graph.num_nodes
        
        # Store the node index range for this graph (external mapping)
        node_index_mapping[i] = torch.arange(node_idx, node_idx + n_nodes, dtype=torch.long)
        node_idx += n_nodes
        
        # Add task_mask for multi-task datasets (this is safe to add)
        if is_multitask:
            if graph.y.dtype.is_floating_point:
                graph.task_mask = (~torch.isnan(graph.y)).float()
            else:
                graph.task_mask = (graph.y != -1).float()
    
    print(f"[FUG-Simple] Created external node index mapping for {len(dataset)} graphs")
    
    # Create completely external FUG mapping (don't modify dataset at all!)
    fug_mapping = {
        'node_index_mapping': node_index_mapping,
        'node_embs': node_embs,
        'uses_fug_embeddings': True,
        'name': name,
        'is_multitask': is_multitask
    }
    
    print(f"[FUG-Simple] Ready! External mapping for '{name}' with {node_embs.size(0)} unified embeddings")

    # Return both pristine dataset and external FUG mapping
    return dataset, fug_mapping


def load_ogb_original_features(name, ogb_root='./dataset/ogb'):
    """
    Load OGB dataset and create node_embs from original raw features (9-dim).
    Uses exactly the same structure as FUG loader, just constructs node_embs differently.

    Args:
        name (str): Dataset name (e.g., 'bace', 'bbbp', 'hiv', 'pcba')
        ogb_root (str): Root to download/store OGB datasets

    Returns:
        tuple: (dataset, original_features_mapping) - same format as FUG loader
    """
    # Dataset name mapping
    ogb_names = {
        'bace': 'ogbg-molbace',
        'bbbp': 'ogbg-molbbbp',
        'hiv': 'ogbg-molhiv',
        'chemhiv': 'ogbg-molhiv',
        'pcba': 'ogbg-molpcba',
        'chempcba': 'ogbg-molpcba',
        'molpcba': 'ogbg-molpcba',
        'tox21': 'ogbg-moltox21',
        'clintox': 'ogbg-molclintox',
        'muv': 'ogbg-molmuv',
        'sider': 'ogbg-molsider',
        'toxcast': 'ogbg-moltoxcast',
        # Also accept full OGB names directly
        'ogbg-molbace': 'ogbg-molbace',
        'ogbg-molbbbp': 'ogbg-molbbbp',
        'ogbg-molhiv': 'ogbg-molhiv',
        'ogbg-molpcba': 'ogbg-molpcba',
        'ogbg-moltox21': 'ogbg-moltox21',
        'ogbg-molclintox': 'ogbg-molclintox',
        'ogbg-molmuv': 'ogbg-molmuv',
        'ogbg-molsider': 'ogbg-molsider',
        'ogbg-moltoxcast': 'ogbg-moltoxcast',
    }

    if name not in ogb_names:
        print(f"[Original-Features] Unknown dataset: {name}")
        return None

    full_ogb_name = ogb_names[name]

    # Load OGB dataset
    try:
        print(f"[Original-Features] Loading OGB dataset '{full_ogb_name}'...")
        dataset = PygGraphPropPredDataset(name=full_ogb_name, root=ogb_root)
        print(f"[Original-Features] Loaded {len(dataset)} graphs")
    except Exception as e:
        print(f"[Original-Features] Failed to load OGB dataset: {e}")
        return None

    # Collect all node features from all graphs (same as FUG but from data.x instead of file)
    print(f"[Original-Features] Collecting original 9-dim features from all graphs...")
    all_node_features = []
    for i in range(len(dataset)):
        graph = dataset[i]
        # Convert int64 -> float32 for PCA processing
        all_node_features.append(graph.x.float())

    # Concatenate into single embedding table (same as FUG)
    node_embs = torch.cat(all_node_features, dim=0)  # [total_nodes, 9]

    # Verify total nodes
    total_nodes = sum(graph.num_nodes for graph in dataset)
    if node_embs.size(0) != total_nodes:
        print(f"[Original-Features] Size mismatch: {total_nodes} nodes vs {node_embs.size(0)} features")
        return None

    # Create external node index mapping (exactly same as FUG)
    node_idx = 0
    sample_graph = dataset[0]
    is_multitask = sample_graph.y.numel() > 1

    if is_multitask:
        print(f"[Original-Features] Multi-task dataset detected, adding task_mask for {sample_graph.y.numel()} tasks")

    # Create external mapping instead of modifying graphs (exactly same as FUG)
    node_index_mapping = {}
    for i in range(len(dataset)):
        graph = dataset[i]
        n_nodes = graph.num_nodes

        # Store the node index range for this graph (external mapping)
        node_index_mapping[i] = torch.arange(node_idx, node_idx + n_nodes, dtype=torch.long)
        node_idx += n_nodes

        # Add task_mask for multi-task datasets (this is safe to add)
        if is_multitask:
            if graph.y.dtype.is_floating_point:
                graph.task_mask = (~torch.isnan(graph.y)).float()
            else:
                graph.task_mask = (graph.y != -1).float()

    print(f"[Original-Features] Created external node index mapping for {len(dataset)} graphs")

    # Create external mapping (exactly same structure as FUG)
    original_features_mapping = {
        'node_index_mapping': node_index_mapping,
        'node_embs': node_embs,
        'uses_fug_embeddings': False,  # NOT using FUG
        'uses_original_features': True,  # Using original features
        'name': name,
        'is_multitask': is_multitask
    }

    print(f"[Original-Features] Ready! External mapping for '{name}' with {node_embs.size(0)} nodes")
    print(f"[Original-Features] Original features: {node_embs.shape} (9-dim) - will be processed with PCA/padding to hidden_dim")

    # Return both pristine dataset and external FUG mapping (same format as FUG)
    return dataset, original_features_mapping


def load_tu_original_features(name, tu_root='./dataset/TU'):
    """
    Load TU dataset and create node_embs from original node features.
    Uses EXACTLY the same structure as OGB original features loader.

    Args:
        name (str): TU dataset name (e.g., 'MUTAG', 'PROTEINS', 'NCI1', etc.)
        tu_root (str): Root to download/store TU datasets

    Returns:
        tuple: (dataset, original_features_mapping) - same format as OGB/FUG loader
    """
    # Load TU dataset
    try:
        print(f"[TU-Original-Features] Loading TU dataset '{name}'...")
        dataset = TUDataset(root=tu_root, name=name)
        print(f"[TU-Original-Features] Loaded {len(dataset)} graphs")
    except Exception as e:
        print(f"[TU-Original-Features] Failed to load TU dataset: {e}")
        return None

    # Check if dataset has node features
    if dataset.num_node_features == 0:
        print(f"[TU-Original-Features] Dataset '{name}' has no node features - cannot be used")
        return None

    # Collect all node features from all graphs (exactly same as OGB)
    print(f"[TU-Original-Features] Collecting original {dataset.num_node_features}-dim features from all graphs...")
    all_node_features = []
    for i in range(len(dataset)):
        graph = dataset[i]
        if graph.x is None:
            print(f"[TU-Original-Features] Graph {i} has no node features - cannot be used")
            return None
        # Convert to float32 for PCA processing
        all_node_features.append(graph.x.float())

    # Concatenate into single embedding table (same as OGB)
    node_embs = torch.cat(all_node_features, dim=0)  # [total_nodes, feature_dim]

    # Verify total nodes
    total_nodes = sum(graph.num_nodes for graph in dataset)
    if node_embs.size(0) != total_nodes:
        print(f"[TU-Original-Features] Size mismatch: {total_nodes} nodes vs {node_embs.size(0)} features")
        return None

    # Create external node index mapping (exactly same as OGB)
    node_idx = 0
    sample_graph = dataset[0]

    # TU datasets are typically single-task, but check just in case
    is_multitask = sample_graph.y.numel() > 1

    if is_multitask:
        print(f"[TU-Original-Features] Multi-task dataset detected, adding task_mask for {sample_graph.y.numel()} tasks")

    # Create external mapping instead of modifying graphs (exactly same as OGB)
    node_index_mapping = {}
    for i in range(len(dataset)):
        graph = dataset[i]
        n_nodes = graph.num_nodes

        # Store the node index range for this graph (external mapping)
        node_index_mapping[i] = torch.arange(node_idx, node_idx + n_nodes, dtype=torch.long)
        node_idx += n_nodes

        # Add task_mask for multi-task datasets (this is safe to add)
        if is_multitask:
            if graph.y.dtype.is_floating_point:
                graph.task_mask = (~torch.isnan(graph.y)).float()
            else:
                graph.task_mask = (graph.y != -1).float()

    print(f"[TU-Original-Features] Created external node index mapping for {len(dataset)} graphs")

    # Create external mapping (exactly same structure as OGB)
    original_features_mapping = {
        'node_index_mapping': node_index_mapping,
        'node_embs': node_embs,
        'uses_fug_embeddings': False,  # NOT using FUG
        'uses_original_features': True,  # Using original features
        'name': name,
        'is_multitask': is_multitask
    }

    print(f"[TU-Original-Features] Ready! External mapping for '{name}' with {node_embs.size(0)} nodes")
    print(f"[TU-Original-Features] Original features: {node_embs.shape} ({node_embs.shape[1]}-dim) - will be processed with PCA/padding to hidden_dim")

    # Return both pristine dataset and external mapping (same format as OGB/FUG)
    return dataset, original_features_mapping


def load_gnn_benchmark_original_features(name, gnn_benchmark_root='./dataset/GNN_Benchmark', split='train'):
    """
    Load GNN Benchmark dataset and create node_embs from original node features.
    Uses EXACTLY the same structure as TU original features loader.

    Args:
        name (str): GNN Benchmark dataset name (e.g., 'MNIST', 'CIFAR10', 'PATTERN', 'CLUSTER')
                   Case-insensitive - will be converted to uppercase
        gnn_benchmark_root (str): Root to download/store GNN Benchmark datasets
        split (str): Which split to load ('train', 'val', 'test')

    Returns:
        tuple: (dataset, original_features_mapping) - same format as OGB/TU/FUG loader
    """
    # GNNBenchmarkDataset expects uppercase names
    name_upper = name.upper()
    
    # Load GNN Benchmark dataset
    try:
        print(f"[GNNBenchmark-Original-Features] Loading GNN Benchmark dataset '{name_upper}' ({split})...")
        dataset = GNNBenchmarkDataset(root=gnn_benchmark_root, name=name_upper, split=split)
        print(f"[GNNBenchmark-Original-Features] Loaded {len(dataset)} graphs")
    except Exception as e:
        print(f"[GNNBenchmark-Original-Features] Failed to load GNN Benchmark dataset: {e}")
        return None

    # Check if dataset has node features
    if dataset.num_node_features == 0:
        print(f"[GNNBenchmark-Original-Features] Dataset '{name}' has no node features - cannot be used")
        return None

    # Collect all node features from all graphs (exactly same as TU)
    # For GNN Benchmark datasets, concatenate pos with x when available
    sample_graph = dataset[0]
    has_pos = hasattr(sample_graph, 'pos') and sample_graph.pos is not None
    
    if has_pos:
        feature_dim = dataset.num_node_features + sample_graph.pos.shape[1]
        print(f"[GNNBenchmark-Original-Features] Collecting {dataset.num_node_features}-dim features + {sample_graph.pos.shape[1]}-dim pos = {feature_dim}-dim total from all graphs...")
    else:
        feature_dim = dataset.num_node_features
        print(f"[GNNBenchmark-Original-Features] Collecting original {dataset.num_node_features}-dim features from all graphs...")
    
    all_node_features = []
    for i in range(len(dataset)):
        graph = dataset[i]
        if graph.x is None:
            print(f"[GNNBenchmark-Original-Features] Graph {i} has no node features - cannot be used")
            return None
        
        # Concatenate pos with x if available
        if has_pos and graph.pos is not None:
            features = torch.cat([graph.x.float(), graph.pos.float()], dim=1)
        else:
            features = graph.x.float()
        
        all_node_features.append(features)

    # Concatenate into single embedding table (same as TU)
    node_embs = torch.cat(all_node_features, dim=0)  # [total_nodes, feature_dim]

    # Verify total nodes
    total_nodes = sum(graph.num_nodes for graph in dataset)
    if node_embs.size(0) != total_nodes:
        print(f"[GNNBenchmark-Original-Features] Size mismatch: {total_nodes} nodes vs {node_embs.size(0)} features")
        return None

    # Create external node index mapping (exactly same as TU)
    node_idx = 0
    sample_graph = dataset[0]

    # GNN Benchmark datasets are typically single-task, but check just in case
    is_multitask = sample_graph.y.numel() > 1

    if is_multitask:
        print(f"[GNNBenchmark-Original-Features] Multi-task dataset detected, adding task_mask for {sample_graph.y.numel()} tasks")

    # Create external mapping instead of modifying graphs (exactly same as TU)
    node_index_mapping = {}
    for i in range(len(dataset)):
        graph = dataset[i]
        n_nodes = graph.num_nodes

        # Store the node index range for this graph (external mapping)
        node_index_mapping[i] = torch.arange(node_idx, node_idx + n_nodes, dtype=torch.long)
        node_idx += n_nodes

        # Add task_mask for multi-task datasets (this is safe to add)
        if is_multitask:
            if graph.y.dtype.is_floating_point:
                graph.task_mask = (~torch.isnan(graph.y)).float()
            else:
                graph.task_mask = (graph.y != -1).float()

    print(f"[GNNBenchmark-Original-Features] Created external node index mapping for {len(dataset)} graphs")

    # Create external mapping (exactly same structure as TU)
    original_features_mapping = {
        'node_index_mapping': node_index_mapping,
        'node_embs': node_embs,
        'uses_fug_embeddings': False,  # NOT using FUG
        'uses_original_features': True,  # Using original features
        'name': name_upper,
        'is_multitask': is_multitask
    }

    print(f"[GNNBenchmark-Original-Features] Ready! External mapping for '{name_upper}' with {node_embs.size(0)} nodes")
    print(f"[GNNBenchmark-Original-Features] Original features: {node_embs.shape} ({node_embs.shape[1]}-dim) - will be processed with PCA/padding to hidden_dim")

    # Return both pristine dataset and external mapping (same format as OGB/TU/FUG)
    return dataset, original_features_mapping


def load_mnist_superpixels_original_features(root='./dataset/MNISTSuperpixels', train=True):
    """
    Load MNISTSuperpixels dataset and create node_embs from original node features.
    Uses EXACTLY the same structure as TU/GNN Benchmark original features loader.
    
    Args:
        root (str): Root to download/store MNISTSuperpixels dataset
        train (bool): Whether to load train split (True) or test split (False)
        
    Returns:
        tuple: (dataset, original_features_mapping) - same format as OGB/TU/GNN Benchmark/FUG loader
    """
    from torch_geometric.transforms import Compose, NormalizeFeatures
    
    # Transform to concatenate node features and positions
    class ConcatPosTransform:
        def __call__(self, data):
            if hasattr(data, 'pos') and data.pos is not None:
                # Concatenate node features (intensity) with 2D positions
                data.x = torch.cat([data.x, data.pos], dim=1)
            return data
    
    # Combine transforms
    transform = Compose([ConcatPosTransform(), NormalizeFeatures()])
    
    # Load MNISTSuperpixels dataset
    try:
        split_name = 'train' if train else 'test'
        print(f"[MNISTSuperpixels-Original-Features] Loading MNISTSuperpixels dataset ({split_name})...")
        dataset = MNISTSuperpixels(root=root, train=train, transform=transform)
        print(f"[MNISTSuperpixels-Original-Features] Loaded {len(dataset)} graphs")
    except Exception as e:
        print(f"[MNISTSuperpixels-Original-Features] Failed to load MNISTSuperpixels dataset: {e}")
        return None
    
    # Check if dataset has node features
    if dataset.num_node_features == 0:
        print(f"[MNISTSuperpixels-Original-Features] Dataset has no node features - cannot be used")
        return None
    
    # Collect all node features into unified embedding table
    print(f"[MNISTSuperpixels-Original-Features] Collecting node features from {len(dataset)} graphs...")
    all_node_features = []
    
    for graph in dataset:
        if graph.x is not None and graph.x.numel() > 0:
            all_node_features.append(graph.x)
    
    # Concatenate to create unified embedding table
    node_embs = torch.cat(all_node_features, dim=0)
    print(f"[MNISTSuperpixels-Original-Features] Unified node embeddings: {node_embs.shape}")
    
    # Verify total nodes
    total_nodes = sum(graph.num_nodes for graph in dataset)
    if node_embs.size(0) != total_nodes:
        print(f"[MNISTSuperpixels-Original-Features] Size mismatch: {total_nodes} nodes vs {node_embs.size(0)} features")
        return None
    
    # Create external node index mapping (exactly same as TU/GNN Benchmark)
    node_idx = 0
    sample_graph = dataset[0]
    
    # MNISTSuperpixels is single-task (10 classes for digits 0-9)
    is_multitask = sample_graph.y.numel() > 1
    
    if is_multitask:
        print(f"[MNISTSuperpixels-Original-Features] Multi-task dataset detected, adding task_mask for {sample_graph.y.numel()} tasks")
    
    # Create external mapping instead of modifying graphs
    node_index_mapping = {}
    for i in range(len(dataset)):
        graph = dataset[i]
        n_nodes = graph.num_nodes
        
        # Store the node index range for this graph (external mapping)
        node_index_mapping[i] = torch.arange(node_idx, node_idx + n_nodes, dtype=torch.long)
        node_idx += n_nodes
        
        # Add task_mask for multi-task datasets (this is safe to add)
        if is_multitask:
            if graph.y.dtype.is_floating_point:
                graph.task_mask = (~torch.isnan(graph.y)).float()
            else:
                graph.task_mask = (graph.y != -1).float()
    
    print(f"[MNISTSuperpixels-Original-Features] Created external node index mapping for {len(dataset)} graphs")
    
    # Create external mapping (exactly same structure as TU/GNN Benchmark)
    original_features_mapping = {
        'node_index_mapping': node_index_mapping,
        'node_embs': node_embs,
        'uses_fug_embeddings': False,  # NOT using FUG
        'uses_original_features': True,  # Using original features
        'name': 'MNISTSuperpixels',
        'is_multitask': is_multitask
    }
    
    print(f"[MNISTSuperpixels-Original-Features] Ready! External mapping for 'MNISTSuperpixels' with {node_embs.size(0)} nodes")
    print(f"[MNISTSuperpixels-Original-Features] Original features: {node_embs.shape} ({node_embs.shape[1]}-dim) - will be processed with PCA/padding to hidden_dim")

    # Return both pristine dataset and external mapping (same format as OGB/TU/GNN Benchmark/FUG)
    return dataset, original_features_mapping


def load_modelnet_original_features(name='10', modelnet_root='./dataset/ModelNet', train=True):
    """
    Load ModelNet dataset (3D mesh classification) and create node_embs from original node features.
    Uses EXACTLY the same structure as TU/GNN Benchmark/MNISTSuperpixels original features loader.

    ModelNet is a 3D mesh dataset that needs to be converted to graph format using FaceToEdge transform.
    Node features are 3D position coordinates.

    Args:
        name (str): Dataset variant ('10' for ModelNet10, '40' for ModelNet40)
        modelnet_root (str): Root to download/store ModelNet datasets
        train (bool): Whether to load train split (True) or test split (False)

    Returns:
        tuple: (dataset, original_features_mapping) - same format as OGB/TU/GNN Benchmark/MNISTSuperpixels/FUG loader
    """
    from torch_geometric.transforms import Compose, NormalizeFeatures

    # Transform to convert mesh faces to graph edges
    transform = Compose([
        FaceToEdge(remove_faces=False),  # Convert 3D mesh to graph
        NormalizeFeatures()              # Normalize node features
    ])

    # Load ModelNet dataset with corrected URLs
    try:
        split_name = 'train' if train else 'test'
        print(f"[ModelNet-Original-Features] Loading ModelNet{name} dataset ({split_name})...")

        # Monkey-patch PyG's ModelNet URLs to use the correct domain (3dvision instead of vision)
        # PyG Issue: https://github.com/pyg-team/pytorch_geometric/issues/XXX
        original_urls = ModelNet.urls
        ModelNet.urls = {
            '10': 'http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip',
            '40': 'http://modelnet.cs.princeton.edu/ModelNet40.zip'
        }
        print(f"[ModelNet-Original-Features] Using corrected URL: {ModelNet.urls[name]}")

        try:
            dataset = ModelNet(root=modelnet_root, name=name, train=train, transform=transform)
            print(f"[ModelNet-Original-Features] Loaded {len(dataset)} graphs")
        finally:
            # Restore original URLs
            ModelNet.urls = original_urls

    except Exception as e:
        print(f"[ModelNet-Original-Features] Failed to load ModelNet dataset: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Check if dataset has node features
    if dataset.num_node_features == 0:
        print(f"[ModelNet-Original-Features] Dataset has no node features - cannot be used")
        return None

    # Collect all node features (3D positions) from all graphs
    print(f"[ModelNet-Original-Features] Collecting 3D position features ({dataset.num_node_features}-dim) from {len(dataset)} graphs...")
    all_node_features = []

    for i, graph in enumerate(dataset):
        if graph.x is None:
            print(f"[ModelNet-Original-Features] Graph {i} has no node features - cannot be used")
            return None
        # Convert to float32 for PCA processing
        all_node_features.append(graph.x.float())

    # Concatenate into single embedding table (same as TU/GNN Benchmark/MNISTSuperpixels)
    node_embs = torch.cat(all_node_features, dim=0)  # [total_nodes, 3]
    print(f"[ModelNet-Original-Features] Unified node embeddings: {node_embs.shape}")

    # Verify total nodes
    total_nodes = sum(graph.num_nodes for graph in dataset)
    if node_embs.size(0) != total_nodes:
        print(f"[ModelNet-Original-Features] Size mismatch: {total_nodes} nodes vs {node_embs.size(0)} features")
        return None

    # Create external node index mapping (exactly same as TU/GNN Benchmark/MNISTSuperpixels)
    node_idx = 0
    sample_graph = dataset[0]

    # ModelNet is single-task classification (10 or 40 classes)
    is_multitask = sample_graph.y.numel() > 1

    if is_multitask:
        print(f"[ModelNet-Original-Features] Multi-task dataset detected, adding task_mask for {sample_graph.y.numel()} tasks")

    # Create external mapping instead of modifying graphs
    node_index_mapping = {}
    for i in range(len(dataset)):
        graph = dataset[i]
        n_nodes = graph.num_nodes

        # Store the node index range for this graph (external mapping)
        node_index_mapping[i] = torch.arange(node_idx, node_idx + n_nodes, dtype=torch.long)
        node_idx += n_nodes

        # Add task_mask for multi-task datasets (this is safe to add)
        if is_multitask:
            if graph.y.dtype.is_floating_point:
                graph.task_mask = (~torch.isnan(graph.y)).float()
            else:
                graph.task_mask = (graph.y != -1).float()

    print(f"[ModelNet-Original-Features] Created external node index mapping for {len(dataset)} graphs")

    # Create external mapping (exactly same structure as TU/GNN Benchmark/MNISTSuperpixels)
    original_features_mapping = {
        'node_index_mapping': node_index_mapping,
        'node_embs': node_embs,
        'uses_fug_embeddings': False,  # NOT using FUG
        'uses_original_features': True,  # Using original features
        'name': f'ModelNet{name}',
        'is_multitask': is_multitask
    }

    print(f"[ModelNet-Original-Features] Ready! External mapping for 'ModelNet{name}' with {node_embs.size(0)} nodes")
    print(f"[ModelNet-Original-Features] Original features: {node_embs.shape} (3D positions) - will be processed with PCA/padding to hidden_dim")

    # Return both pristine dataset and external mapping (same format as OGB/TU/GNN Benchmark/MNISTSuperpixels/FUG)
    return dataset, original_features_mapping
