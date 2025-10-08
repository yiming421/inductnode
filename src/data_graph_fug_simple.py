import os
import torch
from ogb.graphproppred import PygGraphPropPredDataset


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
