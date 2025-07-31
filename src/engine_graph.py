"""
Training and evaluation engine for graph classification using PFN predictor.
Reuses the existing PFNPredictorNodeCls by treating pooled graph embeddings as node embeddings.
"""

import time
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch.utils.data import DataLoader
import numpy as np
from .data_graph import create_graph_batch


def pool_graph_embeddings(node_embeddings, batch, pooling_method='mean'):
    """
    Pool node embeddings to create graph-level embeddings.
    
    Args:
        node_embeddings (torch.Tensor): Node embeddings [num_nodes, hidden_dim]
        batch (torch.Tensor): Batch assignment for each node
        pooling_method (str): Pooling method ('mean', 'max', 'sum')
        
    Returns:
        torch.Tensor: Graph-level embeddings [num_graphs, hidden_dim]
    """
    if pooling_method == 'mean':
        return global_mean_pool(node_embeddings, batch)
    elif pooling_method == 'max':
        return global_max_pool(node_embeddings, batch)
    elif pooling_method == 'sum':
        return global_add_pool(node_embeddings, batch)
    else:
        raise ValueError(f"Unsupported pooling method: {pooling_method}")


def create_context_embeddings(model, context_structure, task_idx=0, pooling_method='mean', device='cuda'):
    """
    Create context embeddings from task-aware context structure.
    
    Args:
        model: GNN model
        context_structure (dict): Task-aware context {task_idx: {class: [graphs]}}
        task_idx (int): Specific task to get context for
        pooling_method (str): Graph pooling method
        device (str): Device for computation
        
    Returns:
        tuple: (context_embeddings, context_labels)
    """
    # Get context graphs for the specific task
    if task_idx not in context_structure:
        raise ValueError(f"Task {task_idx} not found in context structure")
    
    task_context = context_structure[task_idx]
    
    # Note: No model.eval() and no torch.no_grad() here because we need gradients
    # to flow through context embeddings during training
    context_embeddings = []
    context_labels = []
    
    for class_label, graphs in task_context.items():
        if not graphs:  # Skip empty classes
            continue
            
        # Create batch from context graphs
        batch_data = create_graph_batch(graphs, device)
        
        # Get node embeddings from GNN using SparseTensor format
        node_embeddings = model(batch_data.x, batch_data.adj_t)
        
        # Pool to get graph embeddings
        graph_embeddings = pool_graph_embeddings(node_embeddings, batch_data.batch, pooling_method)
        
        # Store embeddings and labels
        context_embeddings.append(graph_embeddings)
        context_labels.extend([class_label] * len(graphs))
    
    # Concatenate all context embeddings
    if context_embeddings:
        context_embeddings = torch.cat(context_embeddings, dim=0)
        context_labels = torch.tensor(context_labels, device=device, dtype=torch.long)
    else:
        # Handle case where no context graphs are available
        hidden_dim = 256  # Default fallback
        context_embeddings = torch.empty(0, hidden_dim, device=device)
        context_labels = torch.empty(0, dtype=torch.long, device=device)
    
    return context_embeddings, context_labels


def prepare_pfn_data_structure(context_embeddings, context_labels, num_classes, device='cuda'):
    """
    Prepare a data structure compatible with PFN predictor.
    This mimics the node classification data structure but for graphs.
    
    Args:
        context_embeddings (torch.Tensor): Context graph embeddings
        context_labels (torch.Tensor): Context graph labels
        num_classes (int): Number of classes
        device (str): Device for computation
        
    Returns:
        object: Data structure compatible with PFN predictor
    """
    class GraphDataForPFN:
        def __init__(self, context_embeddings, context_labels, num_classes):
            self.context_sample = torch.arange(len(context_embeddings), device=device)
            self.y = context_labels  # Context labels
            self.num_classes = num_classes
            self.name = "graph_classification_data"
    
    return GraphDataForPFN(context_embeddings, context_labels, num_classes)


def train_graph_classification_single_task(model, predictor, dataset_info, data_loaders, optimizer, task_idx,
                                         pooling_method='mean', device='cuda', clip_grad=1.0,
                                         orthogonal_push=0.0, normalize_class_h=True):
    """
    Train graph classification for one epoch on a single task with prefiltered data.
    All samples in each batch are guaranteed to have valid labels for the specified task.
    
    Args:
        model: GNN model
        predictor: PFN predictor
        dataset_info (dict): Dataset information including context graphs
        data_loaders (dict): Task-specific data loaders (prefiltered)
        optimizer: Optimizer
        task_idx (int): Current task index
        pooling_method (str): Graph pooling method
        device (str): Device for computation
        clip_grad (float): Gradient clipping value
        orthogonal_push (float): Orthogonal loss weight
        normalize_class_h (bool): Whether to normalize class embeddings
        
    Returns:
        float: Average training loss for this task
    """
    model.train()
    predictor.train()
    
    total_loss = 0
    num_batches = 0
    
    # Training loop for this specific task
    for batch_graphs in data_loaders['train']:
        batch_data = batch_graphs.to(device)
        
        # Get node embeddings from GNN using SparseTensor format
        node_embeddings = model(batch_data.x, batch_data.adj_t)
        
        # Pool to get graph embeddings
        target_embeddings = pool_graph_embeddings(node_embeddings, batch_data.batch, pooling_method)
        
        # Get labels for this batch - all samples are valid for this task
        batch_labels = batch_data.y
        if batch_labels.dim() > 1:
            # Multi-task format: extract labels for current task
            batch_labels = batch_labels[:, task_idx]
        else:
            # Single-task format: use labels directly
            batch_labels = batch_labels
        
        # Ensure labels are long integers
        if batch_labels.dtype != torch.long:
            batch_labels = batch_labels.to(torch.long)
        
        # Create task-specific context embeddings
        context_embeddings, context_labels = create_context_embeddings(
            model, dataset_info['context_graphs'], task_idx=task_idx, 
            pooling_method=pooling_method, device=device
        )
        
        # Prepare PFN data structure
        pfn_data = prepare_pfn_data_structure(context_embeddings, context_labels, 
                                            dataset_info['num_classes'], device)
        
        # Process context embeddings to create class prototypes
        from .utils import process_node_features
        class_h = process_node_features(
            context_embeddings, pfn_data,
            degree_normalize=False,  # Not applicable for graphs
            attention_pool_module=None,
            mlp_module=None,
            normalize=normalize_class_h
        )
        
        # Use PFN predictor (all samples are valid, no masking needed)
        scores = predictor(pfn_data, context_embeddings, target_embeddings, context_labels, class_h)
        scores = F.log_softmax(scores, dim=1)
        
        # Compute loss - no masking needed since all samples are valid
        nll_loss = F.nll_loss(scores, batch_labels)
        
        # Orthogonal loss for class prototypes
        if orthogonal_push > 0:
            class_h_norm = F.normalize(class_h, p=2, dim=1)
            class_matrix = class_h_norm @ class_h_norm.T
            mask = ~torch.eye(class_matrix.size(0), device=class_matrix.device, dtype=torch.bool)
            orthogonal_loss = torch.sum(class_matrix[mask]**2)
        else:
            orthogonal_loss = torch.tensor(0.0, device=device)
        
        loss = nll_loss + orthogonal_push * orthogonal_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def train_graph_classification(model, predictor, dataset_info, data_loaders, optimizer, 
                             pooling_method='mean', device='cuda', clip_grad=1.0,
                             orthogonal_push=0.0, normalize_class_h=True):
    """
    Train graph classification for one epoch.
    
    Args:
        model: GNN model
        predictor: PFN predictor
        dataset_info (dict): Dataset information including context graphs
        data_loaders (dict): Data loaders for train/val/test
        optimizer: Optimizer
        pooling_method (str): Graph pooling method
        device (str): Device for computation
        clip_grad (float): Gradient clipping value
        orthogonal_push (float): Orthogonal loss weight
        normalize_class_h (bool): Whether to normalize class embeddings
        
    Returns:
        float: Average training loss
    """
    model.train()
    predictor.train()
    
    # Check if this is a multi-task dataset
    sample_graph = dataset_info['dataset'][0]
    is_multitask = hasattr(sample_graph, 'task_mask') and sample_graph.y.numel() > 1
    num_tasks = sample_graph.y.numel() if is_multitask else 1
    
    # We'll compute context embeddings fresh for each batch to avoid gradient issues
    
    total_loss = 0
    num_batches = 0
    
    # Training loop
    for batch_graphs in data_loaders['train']:
        batch_data = batch_graphs.to(device)
        
        # Context will be created per task inside the task loop
        
        # Get node embeddings from GNN using SparseTensor format
        node_embeddings = model(batch_data.x, batch_data.adj_t)
        
        # Pool to get graph embeddings (these are our "target" embeddings)
        target_embeddings = pool_graph_embeddings(node_embeddings, batch_data.batch, pooling_method)
        
        # Get labels and task masks for this batch
        batch_labels = batch_data.y  # Shape: [batch_size * num_tasks] or [batch_size] when batched
        
        if is_multitask:
            # Multi-task classification
            batch_task_masks = batch_data.task_mask  # Shape: [batch_size * num_tasks] when batched
            
            # Determine batch size from target embeddings
            batch_size = target_embeddings.size(0)
            
            # Reshape labels and masks from flattened format back to [batch_size, num_tasks]
            if batch_labels.dim() == 1:
                batch_labels = batch_labels.view(batch_size, num_tasks)
            if batch_task_masks.dim() == 1:
                batch_task_masks = batch_task_masks.view(batch_size, num_tasks)
            
            total_task_loss = 0
            valid_tasks = 0
            
            # Process each task separately
            for task_idx in range(num_tasks):
                task_labels = batch_labels[:, task_idx]  # [batch_size]
                task_mask = batch_task_masks[:, task_idx]  # [batch_size]
                
                # Skip if no valid samples for this task
                valid_mask = task_mask.bool()
                if not valid_mask.any():
                    continue
                
                # Create task-specific context embeddings
                task_context_embeddings, task_context_labels = create_context_embeddings(
                    model, dataset_info['context_graphs'], task_idx=task_idx, 
                    pooling_method=pooling_method, device=device
                )
                
                # Prepare task-specific PFN data structure
                task_pfn_data = prepare_pfn_data_structure(task_context_embeddings, task_context_labels, 
                                                         dataset_info['num_classes'], device)
                
                # Process task-specific context embeddings to create class prototypes
                from .utils import process_node_features
                task_class_h = process_node_features(
                    task_context_embeddings, task_pfn_data,
                    degree_normalize=False,  # Not applicable for graphs
                    attention_pool_module=None,
                    mlp_module=None,
                    normalize=normalize_class_h
                )
                
                # Filter to valid samples only
                valid_target_embeddings = target_embeddings[valid_mask]
                # Convert to long here after filtering
                valid_task_labels = task_labels[valid_mask]
                if valid_task_labels.dtype != torch.long:
                    valid_task_labels = valid_task_labels.to(torch.long)
                
                # Use PFN predictor with task-specific context
                print(task_context_embeddings.shape, valid_target_embeddings.shape, task_context_labels.shape, task_class_h.shape, flush=True)
                scores = predictor(task_pfn_data, task_context_embeddings, valid_target_embeddings, task_context_labels, task_class_h)
                scores = F.log_softmax(scores, dim=1)
                
                # Compute loss for this task
                task_loss = F.nll_loss(scores, valid_task_labels)
                total_task_loss += task_loss
                valid_tasks += 1
            
            # Average loss across valid tasks
            if valid_tasks > 0:
                nll_loss = total_task_loss / valid_tasks
            else:
                nll_loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            # Single-task classification (task_idx = 0)
            if batch_labels.dim() > 1:
                batch_labels = batch_labels.squeeze()
            
            # Create context embeddings for single task (task_idx = 0)
            context_embeddings, context_labels = create_context_embeddings(
                model, dataset_info['context_graphs'], task_idx=0, 
                pooling_method=pooling_method, device=device
            )
            
            # Prepare PFN data structure
            pfn_data = prepare_pfn_data_structure(context_embeddings, context_labels, 
                                                dataset_info['num_classes'], device)
            
            # Process context embeddings to create class prototypes
            from .utils import process_node_features
            class_h = process_node_features(
                context_embeddings, pfn_data,
                degree_normalize=False,  # Not applicable for graphs
                attention_pool_module=None,
                mlp_module=None,
                normalize=normalize_class_h
            )
            
            # Use PFN predictor (treating graph embeddings as node embeddings)
            scores = predictor(pfn_data, context_embeddings, target_embeddings, context_labels, class_h)
            scores = F.log_softmax(scores, dim=1)
            
            # Compute loss - ensure labels are long integers
            if batch_labels.dtype != torch.long:
                batch_labels = batch_labels.to(torch.long)
            nll_loss = F.nll_loss(scores, batch_labels)
        
        # Orthogonal loss for class prototypes
        if orthogonal_push > 0:
            class_h_norm = F.normalize(class_h, p=2, dim=1)
            class_matrix = class_h_norm @ class_h_norm.T
            mask = ~torch.eye(class_matrix.size(0), device=class_matrix.device, dtype=torch.bool)
            orthogonal_loss = torch.sum(class_matrix[mask]**2)
        else:
            orthogonal_loss = torch.tensor(0.0, device=device)
        
        loss = nll_loss + orthogonal_push * orthogonal_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0.0


@torch.no_grad()
def evaluate_graph_classification(model, predictor, dataset_info, data_loaders, 
                                pooling_method='mean', device='cuda', 
                                normalize_class_h=True):
    """
    Evaluate graph classification performance.
    
    Args:
        model: GNN model
        predictor: PFN predictor
        dataset_info (dict): Dataset information including context graphs
        data_loaders (dict): Data loaders for train/val/test
        pooling_method (str): Graph pooling method
        device (str): Device for computation
        normalize_class_h (bool): Whether to normalize class embeddings
        
    Returns:
        dict: Dictionary with train/val/test accuracies (or ROC-AUC for multi-task)
    """
    model.eval()
    predictor.eval()
    
    # Check if this is a multi-task dataset
    sample_graph = dataset_info['dataset'][0]
    is_multitask = hasattr(sample_graph, 'task_mask') and sample_graph.y.numel() > 1
    num_tasks = sample_graph.y.numel() if is_multitask else 1
    
    # Context will be created per task during evaluation
    
    results = {}
    
    # Evaluate on each split
    for split_name, data_loader in data_loaders.items():
        if is_multitask:
            # Multi-task evaluation: collect per-task scores and labels
            task_predictions = [[] for _ in range(num_tasks)]
            task_labels = [[] for _ in range(num_tasks)]
            task_valid_samples = [0 for _ in range(num_tasks)]
            
            for batch_graphs in data_loader:
                batch_data = batch_graphs.to(device)
                
                # Get node embeddings from GNN using SparseTensor format
                node_embeddings = model(batch_data.x, batch_data.adj_t)
                
                # Pool to get graph embeddings
                target_embeddings = pool_graph_embeddings(node_embeddings, batch_data.batch, pooling_method)
                
                # Get labels and task masks
                batch_labels = batch_data.y  # [batch_size * num_tasks] when batched
                batch_task_masks = batch_data.task_mask  # [batch_size * num_tasks] when batched
                
                # Determine batch size from target embeddings
                batch_size = target_embeddings.size(0)
                
                # Reshape labels and masks from flattened format back to [batch_size, num_tasks]
                if batch_labels.dim() == 1:
                    batch_labels = batch_labels.view(batch_size, num_tasks)
                if batch_task_masks.dim() == 1:
                    batch_task_masks = batch_task_masks.view(batch_size, num_tasks)
                
                # Process each task separately
                for task_idx in range(num_tasks):
                    task_mask = batch_task_masks[:, task_idx]  # [batch_size]
                    valid_mask = task_mask.bool()
                    
                    if not valid_mask.any():
                        continue
                    
                    # Create task-specific context embeddings for this task
                    task_context_embeddings, task_context_labels = create_context_embeddings(
                        model, dataset_info['context_graphs'], task_idx=task_idx, 
                        pooling_method=pooling_method, device=device
                    )
                    
                    # Prepare task-specific PFN data structure
                    task_pfn_data = prepare_pfn_data_structure(task_context_embeddings, task_context_labels, 
                                                             dataset_info['num_classes'], device)
                    
                    # Process task-specific context embeddings to create class prototypes
                    from .utils import process_node_features
                    task_class_h = process_node_features(
                        task_context_embeddings, task_pfn_data,
                        degree_normalize=False,
                        attention_pool_module=None,
                        mlp_module=None,
                        normalize=normalize_class_h
                    )
                    
                    # Filter to valid samples
                    valid_target_embeddings = target_embeddings[valid_mask]
                    valid_task_labels = batch_labels[:, task_idx][valid_mask].long()
                    
                    # Get predictions using PFN predictor with task-specific context
                    scores = predictor(task_pfn_data, task_context_embeddings, valid_target_embeddings, task_context_labels, task_class_h)
                    
                    # For binary classification, use probability of positive class
                    if dataset_info['num_classes'] == 2:
                        probs = F.softmax(scores, dim=1)
                        task_predictions[task_idx].append(probs[:, 1])  # Probability of class 1
                    else:
                        predictions = scores.argmax(dim=1)
                        task_predictions[task_idx].append(predictions)
                    
                    task_labels[task_idx].append(valid_task_labels)
                    task_valid_samples[task_idx] += valid_mask.sum().item()
            
            # Compute metrics for each task
            task_metrics = []
            for task_idx in range(num_tasks):
                if task_valid_samples[task_idx] == 0:
                    task_metrics.append(0.0)  # No valid samples for this task
                    continue
                
                # Concatenate predictions and labels for this task
                task_preds = torch.cat(task_predictions[task_idx], dim=0)
                task_true = torch.cat(task_labels[task_idx], dim=0)
                
                if dataset_info['num_classes'] == 2:
                    # Binary classification: use ROC-AUC
                    try:
                        from sklearn.metrics import roc_auc_score
                        # Convert to numpy for sklearn
                        task_preds_np = task_preds.cpu().numpy()
                        task_true_np = task_true.cpu().numpy()
                        
                        # Check if we have both classes
                        if len(np.unique(task_true_np)) > 1:
                            auc = roc_auc_score(task_true_np, task_preds_np)
                            task_metrics.append(auc)
                        else:
                            # Only one class present, can't compute AUC
                            task_metrics.append(0.5)  # Random performance
                    except ImportError:
                        # Fallback to accuracy if sklearn not available
                        predictions = (task_preds > 0.5).long()
                        accuracy = (predictions == task_true).float().mean().item()
                        task_metrics.append(accuracy)
                else:
                    # Multi-class: use accuracy
                    accuracy = (task_preds == task_true).float().mean().item()
                    task_metrics.append(accuracy)
            
            # Average metric across all tasks (only count tasks with valid samples)
            valid_task_metrics = [m for i, m in enumerate(task_metrics) if task_valid_samples[i] > 0]
            if valid_task_metrics:
                avg_metric = sum(valid_task_metrics) / len(valid_task_metrics)
            else:
                avg_metric = 0.0
            
            results[split_name] = avg_metric
            
        else:
            # Single-task evaluation
            all_predictions = []
            all_labels = []
            
            # Create context embeddings for single task (task_idx = 0)
            context_embeddings, context_labels = create_context_embeddings(
                model, dataset_info['context_graphs'], task_idx=0, 
                pooling_method=pooling_method, device=device
            )
            
            # Prepare PFN data structure
            pfn_data = prepare_pfn_data_structure(context_embeddings, context_labels, 
                                                dataset_info['num_classes'], device)
            
            # Process context embeddings to create class prototypes
            from .utils import process_node_features
            class_h = process_node_features(
                context_embeddings, pfn_data,
                degree_normalize=False,
                attention_pool_module=None,
                mlp_module=None,
                normalize=normalize_class_h
            )
            
            for batch_graphs in data_loader:
                batch_data = batch_graphs.to(device)
                
                # Get node embeddings from GNN using SparseTensor format
                node_embeddings = model(batch_data.x, batch_data.adj_t)
                
                # Pool to get graph embeddings
                target_embeddings = pool_graph_embeddings(node_embeddings, batch_data.batch, pooling_method)
                
                # Get predictions using PFN predictor
                scores = predictor(pfn_data, context_embeddings, target_embeddings, context_labels, class_h)
                predictions = scores.argmax(dim=1)
                
                all_predictions.append(predictions)
                batch_labels = batch_data.y
                if batch_labels.dim() > 1:
                    batch_labels = batch_labels.squeeze()
                all_labels.append(batch_labels.long())
            
            # Concatenate all predictions and labels
            all_predictions = torch.cat(all_predictions, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            
            # Calculate accuracy
            accuracy = (all_predictions == all_labels).float().mean().item()
            results[split_name] = accuracy
    
    return results


def train_and_evaluate_graph_classification(model, predictor, train_datasets, train_processed_data_list, args, 
                                          optimizer, scheduler=None, device='cuda', test_datasets=None, test_processed_data_list=None):
    """
    Complete training and evaluation pipeline for graph classification.
    
    Args:
        model: GNN model
        predictor: PFN predictor
        train_datasets: List of training PyTorch Geometric datasets
        train_processed_data_list: List of processed training data information
        args: Training arguments
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        device: Device for computation
        test_datasets: Optional list of test datasets for evaluation at eval intervals
        test_processed_data_list: Optional list of processed test data information
        
    Returns:
        dict: Final results for all datasets
    """
    from .data_graph import create_data_loaders
    
    if len(train_datasets) == 0:
        raise ValueError("No training datasets provided")
    
    # Get dataset names for logging
    dataset_names = []
    for train_dataset_info in train_processed_data_list:
        name = train_dataset_info['dataset'].name if hasattr(train_dataset_info['dataset'], 'name') else f'dataset_{len(dataset_names)}'
        dataset_names.append(name)
    
    print(f"\nTraining on {len(train_datasets)} datasets: {', '.join(dataset_names)}")
    
    # Create task-filtered data loaders for all training datasets
    all_train_data_loaders = []
    all_task_filtered_splits = []
    
    for train_dataset_info in train_processed_data_list:
        # Create task-filtered datasets
        from .data_graph import create_task_filtered_datasets
        task_filtered_splits = create_task_filtered_datasets(
            train_dataset_info['dataset'], 
            train_dataset_info['split_idx']
        )
        all_task_filtered_splits.append(task_filtered_splits)
        
        # For now, create loaders for task 0 (will be updated per task during training)
        data_loaders = create_data_loaders(
            train_dataset_info['dataset'], 
            task_filtered_splits[0],  # Start with task 0
            batch_size=args.batch_size,
            shuffle=True,
            task_idx=0
        )
        all_train_data_loaders.append(data_loaders)
    
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    patience = 20  # Early stopping patience
    best_results = {}
    
    # Training loop
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Training step on all datasets with task-specific data loaders
        total_train_loss = 0
        for dataset_idx, (train_dataset_info, task_filtered_splits) in enumerate(zip(train_processed_data_list, all_task_filtered_splits)):
            dataset_loss = 0
            dataset_tasks = 0
            
            # Train on each task separately using prefiltered data
            for task_idx, task_splits in task_filtered_splits.items():
                # Create task-specific data loaders
                task_data_loaders = create_data_loaders(
                    train_dataset_info['dataset'], 
                    task_splits,
                    batch_size=args.batch_size,
                    shuffle=True,
                    task_idx=task_idx
                )
                
                # Train on this specific task (all samples in batch are valid for this task)
                task_loss = train_graph_classification_single_task(
                    model, predictor, train_dataset_info, task_data_loaders, optimizer, task_idx,
                    pooling_method=args.graph_pooling, device=device,
                    clip_grad=args.clip_grad, orthogonal_push=args.orthogonal_push,
                    normalize_class_h=args.normalize_class_h
                )
                dataset_loss += task_loss
                dataset_tasks += 1
            
            # Average loss across tasks for this dataset
            if dataset_tasks > 0:
                avg_dataset_loss = dataset_loss / dataset_tasks
                total_train_loss += avg_dataset_loss
                print(f"Epoch {epoch:3d}, Dataset {dataset_names[dataset_idx]}: Avg Task Loss = {avg_dataset_loss:.4f} ({dataset_tasks} tasks)")
            else:
                print(f"Epoch {epoch:3d}, Dataset {dataset_names[dataset_idx]}: No valid tasks")
        
        avg_train_loss = total_train_loss / len(train_processed_data_list)
        
        # Validation on all training datasets every epoch (for model selection)
        # NOTE: For now, using the original evaluation which handles masking internally
        # This could be optimized later to use task-specific evaluation
        all_val_results = {}
        total_val_acc = 0
        
        for dataset_idx, (train_dataset_info, task_filtered_splits) in enumerate(zip(train_processed_data_list, all_task_filtered_splits)):
            dataset_name = dataset_names[dataset_idx]
            
            # Create combined data loaders for evaluation (using task 0 for now)
            eval_data_loaders = create_data_loaders(
                train_dataset_info['dataset'], 
                train_dataset_info['split_idx'],  # Use original splits for evaluation
                batch_size=args.batch_size,
                shuffle=False,
                task_idx="eval"
            )
            
            val_results = evaluate_graph_classification(
                model, predictor, train_dataset_info, eval_data_loaders,
                pooling_method=args.graph_pooling, device=device,
                normalize_class_h=args.normalize_class_h
            )
            all_val_results[dataset_name] = val_results
            total_val_acc += val_results['val']
        
        # Use average validation accuracy across all datasets for early stopping
        avg_val_acc = total_val_acc / len(train_processed_data_list)
        
        # Update best model based on average validation performance
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            best_epoch = epoch
            patience_counter = 0
            
            # Save current results as best for all datasets
            best_results.update(all_val_results)
        else:
            patience_counter += 1
        
        # Evaluation on unseen test datasets at eval intervals
        if epoch % args.eval_interval == 0:
            # Log current training progress
            if epoch % args.log_interval == 0:
                # Log average metrics across all training datasets
                avg_train_acc = sum(results['train'] for results in all_val_results.values()) / len(all_val_results)
                avg_test_acc = sum(results['test'] for results in all_val_results.values()) / len(all_val_results)
                
                print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | "
                      f"Avg Train Acc: {avg_train_acc:.4f} | Avg Val Acc: {avg_val_acc:.4f} | "
                      f"Avg Test Acc: {avg_test_acc:.4f} | Time: {time.time()-start_time:.2f}s")
                
                # Log individual dataset performance
                for dataset_name, results in all_val_results.items():
                    print(f"  {dataset_name}: Train={results['train']:.4f}, Val={results['val']:.4f}, Test={results['test']:.4f}")
            
            # Evaluate on unseen test datasets if provided
            if test_datasets is not None and test_processed_data_list is not None:
                print(f"  Evaluating on unseen test datasets at epoch {epoch}...")
                for test_dataset, test_dataset_info in zip(test_datasets, test_processed_data_list):
                    test_name = test_dataset_info['dataset'].name if hasattr(test_dataset_info['dataset'], 'name') else f'test_dataset_{len(best_results)}'
                    
                    # Create data loaders for test dataset
                    test_data_loaders = create_data_loaders(
                        test_dataset_info['dataset'], 
                        test_dataset_info['split_idx'], 
                        batch_size=args.test_batch_size,
                        shuffle=False
                    )
                    
                    # Evaluate on test dataset
                    test_results = evaluate_graph_classification(
                        model, predictor, test_dataset_info, test_data_loaders,
                        pooling_method=args.graph_pooling, device=device,
                        normalize_class_h=args.normalize_class_h
                    )
                    
                    # Store test results (will be overwritten each eval interval)
                    best_results[f"test_{test_name}"] = test_results.copy()
                    
                    print(f"    {test_name}: Train={test_results['train']:.4f}, "
                          f"Val={test_results['val']:.4f}, Test={test_results['test']:.4f}")
        
        # Early stopping based on validation performance on training dataset
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
    
    # Final summary
    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    
    return best_results
