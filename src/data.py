from torch_geometric.datasets import Planetoid, WikiCS, Coauthor, Amazon, Reddit, Flickr, AmazonProducts, Airports, WebKB, WikipediaNetwork, Actor, DeezerEurope, LastFMAsia, AttributedGraphDataset, EllipticBitcoinDataset, CitationFull, Twitch, FacebookPagePage
from torch_geometric.data import Data
from ogb.nodeproppred import PygNodePropPredDataset
from torch_sparse import SparseTensor
import torch
from collections import defaultdict
# Fix for PyTorch 2.6+ weights_only compatibility with PyG
import torch.serialization
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr, TensorAttr
from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage
import pickle
import os

def get_project_root():
    """Get the project root directory"""
    # The root is 'inductnode', which is one level up from 'inductnode/src'
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def load_ogbn_data(dataset):
    name = dataset
    dataset = PygNodePropPredDataset(name=dataset, root=os.path.join(get_project_root(), 'dataset'))
    data = dataset[0]
    data.adj_t = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    split_idx = dataset.get_idx_split()
    data.name = name
    data.y = data.y.squeeze()
    return data, split_idx

def load_ogbn_data_train(dataset):
    name = dataset
    dataset = PygNodePropPredDataset(name=dataset, root=os.path.join(get_project_root(), 'dataset'))
    data = dataset[0]
    data.adj_t = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    split_idx = dataset.get_idx_split()
    data.name = name
    data.y = data.y.squeeze()
    train_idx = torch.cat([split_idx['train'], split_idx['valid']])
    valid_idx = split_idx['test']
    test_idx = split_idx['test']
    split_idx = {'train': train_idx, 'valid': valid_idx, 'test': test_idx}
    return data, split_idx

def load_text_enhanced_dataset(dataset_name: str):
    """Load text-enhanced datasets with Qwen-generated node features"""
    dataset_path = os.path.join(get_project_root(), 'dataset', dataset_name)
    
    if not os.path.exists(dataset_path):
        raise ValueError(f"Text-enhanced dataset {dataset_name} not found at {dataset_path}")
    
    # Try to load from raw .pt file first
    pt_file = os.path.join(dataset_path, 'raw', f'{dataset_name}.pt')
    if os.path.exists(pt_file):
        print(f"Loading text-enhanced dataset {dataset_name} from {pt_file}")
        data = torch.load(pt_file, map_location='cpu', weights_only=False)
        
        # Convert masks from lists to tensors if needed
        if hasattr(data, 'train_masks') and isinstance(data.train_masks, list):
            # Use the first mask (index 0) if multiple masks exist
            data.train_mask = data.train_masks[0].clone().detach() if len(data.train_masks) > 0 else torch.zeros(data.num_nodes, dtype=torch.bool)
        if hasattr(data, 'val_masks') and isinstance(data.val_masks, list):
            data.val_mask = data.val_masks[0].clone().detach() if len(data.val_masks) > 0 else torch.zeros(data.num_nodes, dtype=torch.bool)
        if hasattr(data, 'test_masks') and isinstance(data.test_masks, list):
            data.test_mask = data.test_masks[0].clone().detach() if len(data.test_masks) > 0 else torch.zeros(data.num_nodes, dtype=torch.bool)
        
        print(f"Loaded text-enhanced {dataset_name}: {data.num_nodes} nodes, {data.num_edges} edges, feature dim: {data.x.shape[1] if data.x is not None else 'None'}")
        return data
    else:
        raise ValueError(f"Data file not found for text-enhanced dataset {dataset_name} at {pt_file}")

def load_data_train(dataset_name: str):
    
    # --- 1. Check if this is a text-enhanced dataset (lowercase or LLaMA variants) ---
    if dataset_name in ['cora', 'wikics', 'pubmed', 'cora_llama_8b', 'cora_llama_13b']:
        data = load_text_enhanced_dataset(dataset_name)
        # Text-enhanced datasets already have proper splits, return directly
        split_idx = {
            'train': data.train_mask.nonzero(as_tuple=False).reshape(-1),
            'valid': data.val_mask.nonzero(as_tuple=False).reshape(-1),
            'test': data.test_mask.nonzero(as_tuple=False).reshape(-1)
        }
        return data, split_idx
    # --- 2. Load the dataset using its specific loader ---
    root = os.path.join(get_project_root(), 'dataset')
    if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root=root, name=dataset_name)
    elif dataset_name == 'WikiCS':
        dataset = WikiCS(root=os.path.join(root, 'WikiCS'))
    elif dataset_name in ['CS', 'Physics']:
        dataset = Coauthor(root=os.path.join(root, 'Coauthor'), name=dataset_name)
    elif dataset_name in ['Computers', 'Photo']:
        dataset = Amazon(root=os.path.join(root, 'Amazon'), name=dataset_name)
    elif dataset_name == 'Reddit':
        dataset = Reddit(root=os.path.join(root, 'Reddit'))
    elif dataset_name == 'Flickr':
        dataset = Flickr(root=os.path.join(root, 'Flickr'))
    elif dataset_name == 'AmazonProducts':
        dataset = AmazonProducts(root=os.path.join(root, 'AmazonProducts'))
    elif dataset_name in ['USA','Brazil', 'Europe']:
        dataset = Airports(root=os.path.join(root, 'Airports'), name=dataset_name)
    elif dataset_name in ['Cornell', 'Texas', 'Wisconsin']:
        dataset = WebKB(root=os.path.join(root, 'WebKB'), name=dataset_name)
    elif dataset_name in ['Chameleon', 'Squirrel']:
        dataset = WikipediaNetwork(root=os.path.join(root, 'WikipediaNetwork'), name=dataset_name, geom_gcn_preprocess=True)
    elif dataset_name == 'Actor':
        dataset = Actor(root=os.path.join(root, 'Actor'))
    elif dataset_name == 'DeezerEurope':
        dataset = DeezerEurope(root=os.path.join(root, 'DeezerEurope'))
    elif dataset_name == 'LastFMAsia':
        dataset = LastFMAsia(root=os.path.join(root, 'LastFMAsia'))
    elif dataset_name in ['Wiki', 'BlogCatalog', 'Facebook', 'TWeibo']:
        dataset = AttributedGraphDataset(root=os.path.join(root, 'AttributedGraph'), name=dataset_name)
    elif dataset_name == 'EllipticBitcoin':
        dataset = EllipticBitcoinDataset(root=os.path.join(root, 'EllipticBitcoin'))
    elif dataset_name in ['DBLP']:
        dataset = CitationFull(root=os.path.join(root, 'CitationFull'), name=dataset_name)
    elif dataset_name.startswith('Twitch-'):
        # Extract region from dataset name (e.g., 'Twitch-DE' -> 'DE')
        region = dataset_name.split('-')[1]
        dataset = Twitch(root=os.path.join(root, 'Twitch'), name=region)
    elif dataset_name == 'FacebookPagePage':
        dataset = FacebookPagePage(root=os.path.join(root, 'FacebookPagePage'))
    elif dataset_name == 'MAG240M':
        # Load MAG240M subset directly
        data, split_idx = load_mag240m_subset()
        data.adj_t = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
        data.adj_t = data.adj_t.to_symmetric().coalesce()
        data.name = dataset_name
        return data, split_idx
    elif dataset_name == 'Products':
        # Load Products subset directly
        data, split_idx = load_products_subset()
        data.name = dataset_name
        return data, split_idx
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
        
    data = dataset[0]

    # --- 2. Create the new 90/10 stratified split ---
    num_nodes = data.num_nodes
    print(f"Number of nodes: {num_nodes}", flush=True)
    print(f"Number of edges: {data.num_edges}", flush=True)
    labels = data.y

    # Group nodes by their class label
    nodes_by_class = defaultdict(list)
    for node_idx, label in enumerate(labels):
        nodes_by_class[label.item()].append(node_idx)

    train_idx = []
    valid_idx = []

    # Perform stratified sampling for each class
    for class_label, indices in nodes_by_class.items():
        class_indices = torch.tensor(indices)
        num_class_nodes = len(class_indices)
        
        # Shuffle indices within the class
        shuffled_indices = class_indices[torch.randperm(num_class_nodes)]
        
        # Split according to the 90/10 ratio
        train_end = int(num_class_nodes * 0.9)
        
        train_idx.extend(shuffled_indices[:train_end])
        valid_idx.extend(shuffled_indices[train_end:])

    # Shuffle the combined training and validation indices
    train_idx = torch.tensor(train_idx)[torch.randperm(len(train_idx))]
    valid_idx = torch.tensor(valid_idx)[torch.randperm(len(valid_idx))]

    # Create new boolean masks and attach them to the data object
    data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.train_mask[train_idx] = True

    data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    data.val_mask[valid_idx] = True

    # Create an empty test mask to prevent downstream errors
    data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # --- 3. Perform other processing (e.g., creating adjacency tensor) ---
    data.adj_t = SparseTensor.from_edge_index(
        data.edge_index, 
        sparse_sizes=(data.num_nodes, data.num_nodes)
    )
    data.adj_t = data.adj_t.to_symmetric().coalesce()

    # --- 4. Create the split_idx dictionary with the correct indices ---
    split_idx = {
        'train': train_idx,
        'valid': valid_idx,
        'test': valid_idx  
    }

    data.name = dataset_name
    return data, split_idx

def load_data(dataset):
    name = dataset
    
    # Check if this is a text-enhanced dataset (including LLaMA variants)
    if dataset in ['cora', 'wikics', 'pubmed', 'cora_llama_8b', 'cora_llama_13b']:
        data = load_text_enhanced_dataset(dataset)
        dataset = [data]
    elif dataset in ['Cora', 'Citeseer', 'Pubmed']:
        dataset = Planetoid(root=os.path.join(get_project_root(), 'dataset'), name=dataset)
    elif dataset == 'WikiCS':
        dataset = WikiCS(root=os.path.join(get_project_root(), 'dataset', 'WikiCS'))
    elif dataset == 'Products':
        # Load Products subset directly
        data, split_idx = load_products_subset()
        data.name = name
        return data, split_idx
    elif dataset == 'FacebookPagePage':
        dataset = FacebookPagePage(root=os.path.join(get_project_root(), 'dataset', 'FacebookPagePage'))
    elif dataset == 'MAG240M':
        # Load MAG240M subset directly
        data, split_idx = load_mag240m_subset()
        data.adj_t = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
        data.adj_t = data.adj_t.to_symmetric().coalesce()
        data.name = name
        return data, split_idx
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

def expand_dataset_names(train_datasets):
    """Expand special dataset names like 'Twitch' into all regions"""
    expanded_datasets = []
    for dataset in train_datasets:
        if dataset == 'Twitch':
            # Expand Twitch into all 6 regions
            twitch_regions = ['Twitch-DE', 'Twitch-EN', 'Twitch-ES', 'Twitch-FR', 'Twitch-PT', 'Twitch-RU']
            expanded_datasets.extend(twitch_regions)
            print(f"Expanded 'Twitch' into: {', '.join(twitch_regions)}")
        else:
            expanded_datasets.append(dataset)
    return expanded_datasets

def load_all_data_train(train_datasets):
    # Expand any special dataset names first
    expanded_datasets = expand_dataset_names(train_datasets)
    
    data_list = []
    split_idx_list = []
    for dataset in expanded_datasets:
        print(dataset, flush=True)
        if dataset.startswith('ogbn-'):
            data, split_edge = load_ogbn_data_train(dataset)
        else:
            data, split_edge = load_data_train(dataset)
        data_list.append(data)
        split_idx_list.append(split_edge)
    return data_list, split_idx_list

def load_products_subset():
    """Load Products subset dataset"""
    print("Loading Products subset...")
    
    try:
        product_path = os.path.join(get_project_root(), 'dataset', 'products', 'raw', 'ogbn-products_subset.pt')
        dict_products = torch.load(product_path, map_location='cpu', weights_only=False)
        data = Data(
            x=dict_products['x'],
            y=dict_products['y'].squeeze(),
            train_mask=dict_products['train_mask'],
            val_mask=dict_products['val_mask'],
            test_mask=dict_products['test_mask'],
            adj_t=dict_products['adj_t'],
        )
        data.edge_index = torch.stack([data.adj_t.coo()[0], data.adj_t.coo()[1]], dim=0)
        
        print(f"Products subset loaded: {data.num_nodes} nodes, {data.num_edges} edges")
        print(f"Feature dimension: {data.x.shape[1] if data.x is not None else 'None'}")
        print(f"Train nodes: {data.train_mask.sum().item()}")
        print(f"Val nodes: {data.val_mask.sum().item()}")
        print(f"Test nodes: {data.test_mask.sum().item()}")
        print(f"Number of classes: {data.y.max().item() + 1}")
        
        # Create split_idx dictionary
        split_idx = {
            'train': data.train_mask.nonzero(as_tuple=False).reshape(-1),
            'valid': data.val_mask.nonzero(as_tuple=False).reshape(-1),
            'test': data.test_mask.nonzero(as_tuple=False).reshape(-1)
        }
        
        return data, split_idx
        
    except Exception as e:
        print(f"Error loading Products subset: {e}")
        raise e

def load_mag240m_subset():
    """Load MAG240M subset dataset with ~5.9M nodes and ~52.8M edges"""
    base_path = os.path.join(get_project_root(), 'dataset', 'mag240m')
    processed_path = os.path.join(base_path, 'processed')
    raw_path = os.path.join(base_path, 'raw')
    
    print("Loading MAG240M subset...")
    
    try:
        # Load edge index - try torch.load first, then pickle
        try:
            edge_index = torch.load(os.path.join(processed_path, 'edge_index_subset.pkl'), map_location='cpu', weights_only=False)
        except:
            with open(os.path.join(processed_path, 'edge_index_subset.pkl'), 'rb') as f:
                edge_index = pickle.load(f)
        
        # Load node mapping to understand which nodes are in the subset
        try:
            node_map = torch.load(os.path.join(processed_path, 'node_map_subset.pkl'), map_location='cpu', weights_only=False)
        except:
            with open(os.path.join(processed_path, 'node_map_subset.pkl'), 'rb') as f:
                node_map = pickle.load(f)
        
        # Load label mapping
        try:
            label_map = torch.load(os.path.join(processed_path, 'label_map_subset.pkl'), map_location='cpu', weights_only=False)
        except:
            with open(os.path.join(processed_path, 'label_map_subset.pkl'), 'rb') as f:
                label_map = pickle.load(f)
        
        # Load the actual processed numeric features from the original dataset
        print("Loading processed 768-dim embeddings...")
        import numpy as np
        
        # Load full features and extract subset
        full_features = np.load(os.path.join(raw_path, 'processed/paper/node_feat.npy'), mmap_mode='r')
        
        # Extract features for nodes in our subset
        subset_features = []
        for node_id in node_map:
            if node_id.item() < len(full_features):
                subset_features.append(full_features[node_id.item()])
            else:
                # For non-paper nodes (authors, institutions), create zero features
                subset_features.append(np.zeros(768, dtype=np.float16))
        
        # Convert to tensor
        x = torch.from_numpy(np.stack(subset_features)).float()
        print(f"Loaded {x.shape[0]} node features with dimension {x.shape[1]}")
        
        # Load split indices
        split_dict = torch.load(os.path.join(raw_path, 'split_dict.pt'), map_location='cpu', weights_only=False)
        
        # Create labels tensor - MAG240M has 153 classes
        # We need to map the global indices to subset indices
        y = torch.full((len(node_map),), -1, dtype=torch.long)  # Initialize with -1 (unlabeled)
        
        # Map split indices to subset indices
        subset_node_to_idx = {node_map[i].item(): i for i in range(len(node_map))}
        
        train_mask = torch.zeros(len(node_map), dtype=torch.bool)
        val_mask = torch.zeros(len(node_map), dtype=torch.bool)
        test_mask = torch.zeros(len(node_map), dtype=torch.bool)
        
        train_idx = []
        val_idx = []
        test_idx = []
        
        # Map original splits to subset indices
        for orig_idx in split_dict['train']:
            if orig_idx.item() in subset_node_to_idx:
                subset_idx = subset_node_to_idx[orig_idx.item()]
                train_mask[subset_idx] = True
                train_idx.append(subset_idx)
                # Set label for this node
                y[subset_idx] = label_map[subset_idx]
        
        for orig_idx in split_dict['valid']:
            if orig_idx.item() in subset_node_to_idx:
                subset_idx = subset_node_to_idx[orig_idx.item()]
                val_mask[subset_idx] = True
                val_idx.append(subset_idx)
                y[subset_idx] = label_map[subset_idx]
        
        for orig_idx in split_dict['test']:
            if orig_idx.item() in subset_node_to_idx:
                subset_idx = subset_node_to_idx[orig_idx.item()]
                test_mask[subset_idx] = True
                test_idx.append(subset_idx)
                y[subset_idx] = label_map[subset_idx]
        
        # Features are already processed as tensor from numpy array
        # No additional processing needed
        
        # Create the data object
        data = Data(
            x=x,
            edge_index=edge_index,
            y=y,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            num_nodes=len(node_map)
        )
        
        # Store metadata about the features
        data.feature_type = "processed_embeddings"  # 768-dim embeddings from language model
        
        print(f"MAG240M subset loaded: {data.num_nodes} nodes, {data.num_edges} edges")
        print(f"Feature dimension: {data.x.shape[1] if data.x.dim() > 1 else 'None'}")
        print(f"Train nodes: {train_mask.sum().item()}")
        print(f"Val nodes: {val_mask.sum().item()}")
        print(f"Test nodes: {test_mask.sum().item()}")
        print(f"Labeled nodes: {(y >= 0).sum().item()}")
        print(f"Number of classes: {y.max().item() + 1 if y.max() >= 0 else 'Unknown'}")
        
        # Create split_idx dictionary
        split_idx = {
            'train': torch.tensor(train_idx),
            'valid': torch.tensor(val_idx),
            'test': torch.tensor(test_idx)
        }
        
        return data, split_idx
        
    except Exception as e:
        print(f"Error loading MAG240M subset: {e}")
        raise e