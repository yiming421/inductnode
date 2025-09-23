"""
Training and evaluation engine for graph classification using PFN predictor.
Reuses the existing PFNPredictorNodeCls by treating pooled graph embeddings as node embeddings.
"""

import time
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_sparse import SparseTensor
import gc
from sklearn.metrics import roc_auc_score, average_precision_score
from .data_graph import create_graph_batch, create_task_filtered_datasets
from .utils import process_node_features
from .data_utils import batch_edge_dropout, feature_dropout
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity


def apply_feature_dropout_if_enabled(x, args, rank=0):
    """
    Apply feature dropout if enabled in args (after projection only).

    Args:
        x (torch.Tensor): Input features
        args: Arguments containing feature dropout configuration
        rank (int): Process rank for logging

    Returns:
        torch.Tensor: Features with dropout applied
    """
    if (args is not None and
        hasattr(args, 'feature_dropout_enabled') and args.feature_dropout_enabled and
        hasattr(args, 'feature_dropout_rate') and args.feature_dropout_rate > 0):

        dropout_type = getattr(args, 'feature_dropout_type', 'element_wise')
        verbose = getattr(args, 'verbose_feature_dropout', False) and rank == 0

        return feature_dropout(x, args.feature_dropout_rate, training=True,
                             dropout_type=dropout_type, verbose=verbose)
    return x

def refresh_gc_context_if_needed(dataset_info, batch_idx, epoch, args, device='cuda', task_idx=None):
    """
    Refresh graph classification context if batch-level refresh is enabled.
    
    Args:
        dataset_info (dict): Dataset information including context structure
        batch_idx (int): Current batch index
        epoch (int): Current epoch
        args: Arguments containing refresh settings
        device (str): Device for computation
        task_idx (int, optional): Specific task to refresh. If None, refresh all tasks
    """
    # Check if batch refresh is enabled and it's time to refresh
    if getattr(args, 'context_batch_refresh_interval', 0) <= 0:
        return
        
    if batch_idx > 0 and batch_idx % args.context_batch_refresh_interval == 0:
        # Refresh context for this dataset
        refresh_seed = args.seed + epoch * 10000 + batch_idx
        torch.manual_seed(refresh_seed)
        
        dataset = dataset_info['dataset']
        dataset_name = getattr(dataset, 'name', 'unknown')
        
        # Refresh context_graphs structure
        if 'context_graphs' in dataset_info:
            refresh_gc_context_graphs(dataset_info, args, device, task_idx)
            if task_idx is not None:
                print(f"🔄 GC Dataset {dataset_name} context refreshed for task {task_idx} at batch {batch_idx}")
            else:
                print(f"🔄 GC Dataset {dataset_name} context refreshed (all tasks) at batch {batch_idx}")

def refresh_gc_context_graphs(dataset_info, args, device):
    """
    Refresh context graphs structure by resampling context graphs for each task/class.
    """
    from .data_graph import prepare_graph_data_for_pfn
    
    if 'context_graphs' not in dataset_info:
        return
    
    # Get context size from args
    context_k = getattr(args, 'context_k', 5)
    
    # Use the same function that originally created the context structure
    refreshed_data = prepare_graph_data_for_pfn(
        dataset_info['dataset'], 
        dataset_info['split_idx'], 
        context_k, 
        device
    )
    
    # Update the context_graphs in the dataset_info
    dataset_info['context_graphs'] = refreshed_data['context_graphs']

def _get_node_embedding_table(dataset, task_idx, device, dataset_info=None):
    """
    Get the node embedding table for a dataset.
    Under the new unified setting, all datasets just have dataset.node_embs.
    For FUG datasets with external mapping, use the FUG embedding table.
    
    Args:
        dataset: Dataset with node_embs attribute
        task_idx: Task index (unused)
        device: Target device
        dataset_info: Optional dataset info containing FUG mapping
        
    Returns:
        torch.Tensor: Node embedding table [N_nodes, emb_dim]
    """
    # Check for FUG external mapping first
    if dataset_info and 'fug_mapping' in dataset_info:
        fug_mapping = dataset_info['fug_mapping']
        if fug_mapping and 'node_embs' in fug_mapping:
            return fug_mapping['node_embs']  # Keep on CPU, move batches as needed
    
    # Fallback to regular dataset embeddings
    if hasattr(dataset, 'node_embs') and dataset.node_embs is not None:
        return dataset.node_embs.to(device)
    
    print(f"[ERROR] No node_embs found in dataset - expected under unified setting")
    return None

# ---- ADDED: safe lookup helper to prevent silent CUDA device-side asserts ----
def _safe_lookup_node_embeddings_micro_optimized(node_emb_table: torch.Tensor, x: torch.Tensor, context: str="",
                                                batch_data=None, dataset_info=None) -> torch.Tensor:
    """
    Micro-transfer optimized version - eliminates thousands of GPU-CPU transfers.
    Same algorithm as original but with only 2 transfers total instead of 40,000+.
    """
    import numpy as np
    
    # Case 4: FUG external mapping - handle original OGB features  
    if (dataset_info and 'fug_mapping' in dataset_info and 
        x.dim() == 2 and x.size(1) > 1 and not x.dtype.is_floating_point):
        
            
        if batch_data is None or not hasattr(batch_data, 'original_graph_indices'):
            raise ValueError(f"[FUG] Batch data with original_graph_indices required for external mapping in {context}")
        
        fug_mapping = dataset_info['fug_mapping']
        node_mapping = fug_mapping['node_index_mapping']
        
        # 1. Single GPU->CPU transfer for original graph indices
        original_indices_cpu = batch_data.original_graph_indices.cpu()

        # 2. Build the list of FUG index tensors. This happens on the CPU since the
        #    source tensors in 'node_mapping' are on the CPU.
        try:
            indices_list = [node_mapping[g_idx.item()] for g_idx in original_indices_cpu]
        except KeyError as e:
            raise ValueError(f"[FUG] Original graph index {e.args[0]} not found.") from e

        # 3. Concatenate into a single index tensor on the CPU.
        all_fug_indices_cpu = torch.cat(indices_list)

        # 4. Perform the SINGLE, vectorized embedding lookup on the CPU table.
        processed_embeddings_cpu = node_emb_table[all_fug_indices_cpu]
        
        # --- Optional Sanity Check ---
        if processed_embeddings_cpu.shape[0] != x.shape[0]:
             raise ValueError(f"[FUG] Size mismatch after mapping.")
             
        # 5. Transfer the final result to the GPU in one go.
        target_device = x.device if x.numel() > 0 else 'cuda'
        result = processed_embeddings_cpu.to(target_device)
        
        
        return result
    
    # For non-FUG cases, fall back to original implementation
    return _safe_lookup_node_embeddings_original(node_emb_table, x, context, batch_data, dataset_info)


def _safe_lookup_node_embeddings(node_emb_table: torch.Tensor, x: torch.Tensor, context: str="", 
                                 batch_data=None, dataset_info=None) -> torch.Tensor:
    """Return embedded node features ensuring indices are valid.
    Handles cases:
      1) x already is an embedding matrix (float, 2D) -> return as-is.
      2) x is Long indices (1D or [N,1]) -> validate range then index.
      3) x is numeric but not long -> cast after verifying integral values.
      4) FUG: x is original OGB features (2D, non-float) -> use external mapping.
    Raises a clear Python exception instead of triggering a CUDA device-side assert.
    """
    
    # For FUG cases, use the micro-optimized version
    if (dataset_info and 'fug_mapping' in dataset_info and 
        x.dim() == 2 and x.size(1) > 1 and not x.dtype.is_floating_point):
        return _safe_lookup_node_embeddings_micro_optimized(node_emb_table, x, context, batch_data, dataset_info)
    
    # Case 1: already embedded (float & 2D & width not 1)
    if x.dim() == 2 and x.size(1) > 1 and x.dtype.is_floating_point:
        return x  # Already features / embeddings
    # Squeeze [N,1]
    if x.dim() == 2 and x.size(1) == 1:
        x = x.squeeze(1)
    # Ensure integer type
    if x.dtype != torch.long:
        if not torch.allclose(x, x.round()):
            raise ValueError(f"[SAFE_LOOKUP]{context} x contains non-integer values; cannot treat as indices. dtype={x.dtype}")
        x = x.long()
    if x.dim() != 1:
        raise ValueError(f"[SAFE_LOOKUP]{context} Expected index tensor 1D after processing, got shape {tuple(x.shape)}")
    if x.numel() == 0:
        return x.new_empty((0, node_emb_table.size(1)))
    min_idx = int(x.min().item())
    max_idx = int(x.max().item())
    table_size = node_emb_table.size(0)
    if min_idx < 0 or max_idx >= table_size:
        raise IndexError(
            f"[SAFE_LOOKUP]{context} Index out of bounds: min={min_idx} max={max_idx} table_size={table_size}. "
            f"This would have triggered a CUDA device assert during gather."
        )
    return node_emb_table[x]
# ---- END ADDED ----


class MultiApr:
    """
    Multi-task Average Precision metric equivalent to TSGFM's MultiApr.
    Computes AP for each task individually and returns the mean.
    """
    def __init__(self, num_tasks):
        self.num_tasks = num_tasks
        self.predictions = []
        self.targets = []
    
    def update(self, preds, targets, valid_mask=None):
        """
        Update with predictions and targets.
        
        Args:
            preds (torch.Tensor): Predictions [batch_size, num_tasks]
            targets (torch.Tensor): Targets [batch_size, num_tasks] 
            valid_mask (torch.Tensor, optional): Validity mask [batch_size, num_tasks]
        """
        if valid_mask is not None:
            # Keep original shapes; mark invalid positions as NaN so we can mask per-task at compute time
            preds = preds.clone()
            targets = targets.clone()
            preds[~valid_mask] = float('nan')
            targets[~valid_mask] = float('nan')
            self.predictions.append(preds.detach().cpu())
            self.targets.append(targets.detach().cpu())
        else:
            self.predictions.append(preds.detach().cpu())
            self.targets.append(targets.detach().cpu())
    
    def compute(self):
        """
        Compute average precision across all tasks.
        
        Returns:
            float: Mean AP across all tasks
        """
        if not self.predictions:
            return 0.0
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(self.predictions, dim=0)  # [total_samples, num_tasks]
        all_targets = torch.cat(self.targets, dim=0)   # [total_samples, num_tasks]
        
        task_aps = []
        for task_idx in range(self.num_tasks):
            task_preds = all_preds[:, task_idx].numpy()
            task_targets = all_targets[:, task_idx].numpy()
            
            # Skip tasks with no valid samples or only one class
            valid_mask = ~np.isnan(task_targets)
            if valid_mask.sum() == 0 or len(np.unique(task_targets[valid_mask])) < 2:
                continue
                
            try:
                ap = average_precision_score(task_targets[valid_mask], task_preds[valid_mask])
                task_aps.append(ap)
            except ValueError:
                # Handle edge cases
                continue
        
        return np.mean(task_aps) if task_aps else 0.0
    
    def reset(self):
        """Reset the metric state."""
        self.predictions = []
        self.targets = []


def get_dataset_metric(dataset_name, is_multitask=None):
    """
    Get the appropriate evaluation metric(s) for a given dataset.
    Uses intelligent defaults based on task type with specific overrides.
    
    Args:
        dataset_name (str): Name of the dataset
        is_multitask (bool, optional): Whether this is a multi-task dataset
        
    Returns:
        str or list: Metric name(s) ('auc', 'ap', 'accuracy') or list of metrics for PCBA
    """
    dataset_name = dataset_name.lower()
    
    # Specific overrides for known datasets
    if 'chemhiv' in dataset_name or 'hiv' in dataset_name:
        return 'auc'
    elif 'chempcba' in dataset_name or 'pcba' in dataset_name:
        # PCBA uses AUC metric
        return 'auc'
    
    # Intelligent defaults based on task type
    if is_multitask is not None:
        if is_multitask:
            # Multi-task datasets typically use AP (average precision)
            return 'ap'
        else:
            # Single-task datasets typically use AUC for binary classification
            return 'auc'
    
    # Fallback to accuracy if task type is unknown
    return 'accuracy'


def calculate_multiple_metrics(predictions, labels, probabilities, metric_types):
    """
    Calculate multiple metrics for a single evaluation.
    
    Args:
        predictions (torch.Tensor): Predicted class labels
        labels (torch.Tensor): True labels
        probabilities (torch.Tensor): Class probabilities (for AUC/AP)
        metric_types (list): List of metric types to calculate
        
    Returns:
        dict: Dictionary mapping metric names to calculated values
    """
    results = {}
    for metric_type in metric_types:
        results[metric_type] = calculate_metric(predictions, labels, probabilities, metric_type)
    return results


def aggregate_task_metrics(metric_values):
    """
    Aggregate metric values across multiple tasks, handling both single and multiple metrics.
    
    Args:
        metric_values (list): List of metric values (can be mixed int/float and dict types)
        
    Returns:
        float or dict: Aggregated metric value(s)
    """
    if not metric_values:
        return 0.0
        
    if all(isinstance(val, dict) for val in metric_values):
        # All are dicts with multiple metrics - aggregate each metric separately
        aggregated_metrics = {}
        for metric_name in metric_values[0].keys():
            metric_vals = [val[metric_name] for val in metric_values]
            aggregated_metrics[metric_name] = sum(metric_vals) / len(metric_vals)
        return aggregated_metrics
    elif all(isinstance(val, (int, float)) for val in metric_values):
        # All are single metrics - simple average
        return sum(metric_values) / len(metric_values)
    else:
        # Mixed types - extract primary metric from dicts, average all
        primary_values = []
        for val in metric_values:
            if isinstance(val, dict):
                # Extract primary metric (AUC if available, otherwise AP, otherwise first value)
                primary_val = val.get('auc', val.get('ap', next(iter(val.values()))))
                primary_values.append(primary_val)
            else:
                primary_values.append(val)
        return sum(primary_values) / len(primary_values)


def format_metric_results(metric_results, dataset_name=None):
    """
    Format metric results for display, handling both single and multiple metrics.
    
    Args:
        metric_results (float or dict): Single metric value or dict of multiple metrics
        dataset_name (str, optional): Dataset name for context
        
    Returns:
        str: Formatted string for display
    """
    if isinstance(metric_results, dict):
        # Multiple metrics (e.g., PCBA with AUC and AP)
        parts = []
        for metric_name, value in metric_results.items():
            parts.append(f"{metric_name.upper()}={value:.4f}")
        return ", ".join(parts)
    else:
        # Single metric - still need to determine the name
        return f"{metric_results:.4f}"


def calculate_metric(predictions, labels, probabilities, metric_type):
    """
    Calculate the specified metric.
    
    Args:
        predictions (torch.Tensor): Predicted class labels
        labels (torch.Tensor): True labels
        probabilities (torch.Tensor): Class probabilities (for AUC/AP)
        metric_type (str): Type of metric ('auc', 'ap', 'accuracy')
        
    Returns:
        float: Calculated metric value
    """
    if metric_type == 'accuracy':
        return (predictions == labels).float().mean().item()
    elif metric_type == 'auc':
        # For binary classification, use probabilities of positive class
        if probabilities.shape[1] == 2:
            probs = probabilities[:, 1].cpu().numpy()
        else:
            probs = probabilities.cpu().numpy()
        
        labels_np = labels.cpu().numpy()
        
        # Debug: Check for unusual cases
        unique_labels = set(labels_np)
        if len(unique_labels) < 2:
            pass
            return 0.0
        
        try:
            auc_score = roc_auc_score(labels_np, probs)
            if auc_score != auc_score:  # Check for NaN
                pass
                return 0.0
            return auc_score
        except ValueError as e:
            pass
            return 0.0
    elif metric_type == 'ap':
        # For binary classification, use probabilities of positive class
        if probabilities.shape[1] == 2:
            probs = probabilities[:, 1].cpu().numpy()
        else:
            probs = probabilities.cpu().numpy()
        try:
            labels_np = labels.cpu().numpy()
            probs_np = probs
            ap_score = average_precision_score(labels_np, probs_np)
            return ap_score
        except ValueError as e:
            # Handle case where only one class is present
            return 0.0
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


def get_gpu_memory_usage():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
            'reserved': torch.cuda.memory_reserved() / 1024**3,  # GB
            'max_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
        }
    return {'allocated': 0, 'reserved': 0, 'max_allocated': 0}


def log_gpu_memory(stage):
    """Log GPU memory usage for a specific stage."""
    mem = get_gpu_memory_usage()
    print(f"[GPU-MEM {stage}] {mem['allocated']:.4f}GB allocated, {mem['reserved']:.4f}GB reserved, {mem['max_allocated']:.4f}GB max allocated")

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


def _create_context_embeddings_computed(model, context_structure, dataset, task_idx, pooling_method, device, identity_projection, dataset_info):
    """Compute context embeddings normally (fallback for datasets without pre-computed embeddings)."""
    # Get context graphs for the specific task
    if task_idx not in context_structure:
        raise ValueError(f"Task {task_idx} not found in context structure")
    
    task_context = context_structure[task_idx]
    
    context_embeddings = []
    context_labels = []
    
    for class_label, context_data in task_context.items():
        if not context_data:  # Skip empty classes
            continue
            
        # Extract graphs and indices from the new context structure
        if isinstance(context_data, dict) and 'graphs' in context_data and 'indices' in context_data:
            graphs = context_data['graphs']
            graph_indices = context_data['indices']
        else:
            # Fallback for old format (direct graph list)
            graphs = context_data
            graph_indices = None
            
        # Create batch from context graphs
        batch_data = create_graph_batch(graphs, device)
        
        # Add original graph indices for FUG external mapping
        if graph_indices is not None:
            batch_data.original_graph_indices = torch.tensor(graph_indices, dtype=torch.long)
        
        # Convert node indices to embeddings using unified node_embs
        node_emb_table = _get_node_embedding_table(dataset, task_idx, device, dataset_info)
        if node_emb_table is not None:
            batch_data.x = _safe_lookup_node_embeddings(node_emb_table, batch_data.x, context=f"context_task{task_idx}", 
                                                        batch_data=batch_data, dataset_info=dataset_info)
        else:
            raise ValueError("Expected node embedding table under unified setting")
        
        # Apply identity projection if needed
        if dataset_info and dataset_info.get('needs_identity_projection', False) and identity_projection is not None:
            x_input = identity_projection(batch_data.x)
        else:
            x_input = batch_data.x
        
        batch_data.adj_t = SparseTensor.from_edge_index(
            batch_data.edge_index,
            sparse_sizes=(batch_data.num_nodes, batch_data.num_nodes)
        ).to_symmetric().coalesce()
        
        # Get node embeddings from GNN using SparseTensor format
        node_embeddings = model(x_input, batch_data.adj_t)
        
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


def train_graph_classification_full_batch(model, predictor, train_dataset_info, unfiltered_splits, optimizer,
                                         pooling_method='mean', device='cuda', batch_size=4096, clip_grad=1.0,
                                         orthogonal_push=0.0, normalize_class_h=True, identity_projection=None, context_k=None, args=None):
    """
    Full batch training: NO pre-filtering by tasks. Load ALL graphs, check ALL 128 tasks dynamically.
    Accumulate losses from all valid tasks before single parameter update.
    """
    model.train() 
    predictor.train()
    if identity_projection is not None:
        identity_projection.train()
    
    dataset_loss = 0
    dataset_batches = 0
    
    # Create unfiltered data loader - use ALL graphs in training set
    from .data_graph import create_data_loaders
    # Check if FUG mapping is present to use index tracking
    use_fug_tracking = 'fug_mapping' in train_dataset_info
    data_loaders = create_data_loaders(
        train_dataset_info['dataset'],
        unfiltered_splits,  # Use original splits, no task filtering
        batch_size=batch_size,
        shuffle=True,
        task_idx=None,       # No specific task - we want all graphs
        use_index_tracking=use_fug_tracking
    )
    
    # Determine number of tasks from dataset
    sample_graph = train_dataset_info['dataset'][0]
    num_tasks = sample_graph.y.shape[0] if sample_graph.y.dim() > 0 else 1
    
    print(f"Full batch training: Processing ALL {num_tasks} tasks simultaneously per batch (no pre-filtering)")
    
    batch_idx = 0
    for batch in data_loaders['train']:
        # Batch-level context refresh
        if args is not None:
            refresh_gc_context_if_needed(train_dataset_info, batch_idx, 0, args, device)
        
        optimizer.zero_grad()
        accumulated_loss = 0.0
        valid_task_count = 0
        batch_idx += 1
        
        # Process ALL tasks for this unfiltered batch
        for task_idx in range(num_tasks):
            task_loss = train_graph_classification_single_task_no_update(
                model, predictor, train_dataset_info, batch, task_idx,
                pooling_method, device, orthogonal_push, normalize_class_h, identity_projection, context_k, args
            )
            
            if task_loss > 0:
                accumulated_loss += task_loss
                valid_task_count += 1
        
        # Single parameter update after processing all tasks
        if valid_task_count > 0:
            accumulated_loss.backward()
            
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
                if identity_projection:
                    torch.nn.utils.clip_grad_norm_(identity_projection.parameters(), clip_grad)
            
            optimizer.step()
            dataset_loss += accumulated_loss.item()
            dataset_batches += 1
    
    return dataset_loss / dataset_batches if dataset_batches > 0 else 0.0


@torch.no_grad()
def evaluate_graph_classification_full_batch(model, predictor, dataset_info, data_loaders,
                                           pooling_method='mean', device='cuda', 
                                           normalize_class_h=True, dataset_name=None, identity_projection=None):
    """
    Evaluate graph classification using full batch approach: NO pre-filtering by tasks.
    Process ALL graphs and check ALL tasks dynamically, then aggregate results.
    Uses the same metric calculation as single task version.
    """
    model.eval()
    predictor.eval()
    if identity_projection is not None:
        identity_projection.eval()
    
    # Determine number of tasks from dataset
    sample_graph = dataset_info['dataset'][0]
    num_tasks = sample_graph.y.shape[0] if sample_graph.y.dim() > 0 else 1
    
    # Determine the appropriate metric for this dataset
    is_multitask = sample_graph.y.numel() > 1
    metric_type = get_dataset_metric(dataset_name, is_multitask) if dataset_name else 'accuracy'
    
    split_results = {}
    
    # Evaluate each split (train, val, test)
    for split_name in ['train', 'val', 'test']:
        if split_name not in data_loaders:
            continue
            
        all_task_metrics = []
        
        # For each task, collect predictions across all graphs in this split
        for task_idx in range(num_tasks):
            all_predictions = []
            all_labels = []
            all_probabilities = []
            
            for batch in data_loaders[split_name]:
                batch_data = batch.to(device)
                batch_size = batch_data.num_graphs
                
                if batch_size == 0:
                    continue
                
                # Convert node indices to embeddings using unified node_embs
                node_emb_table = _get_node_embedding_table(dataset_info['dataset'], task_idx, device, dataset_info)
                if node_emb_table is not None:
                    batch_data.x = _safe_lookup_node_embeddings(node_emb_table, batch_data.x, context=f"eval_full_task{task_idx}", 
                                                                batch_data=batch_data, dataset_info=dataset_info)
                else:
                    raise ValueError("Expected node embedding table under unified setting")
                
                # Apply identity projection if needed
                if dataset_info.get('needs_identity_projection', False) and identity_projection is not None:
                    x_input = identity_projection(batch_data.x)
                else:
                    x_input = batch_data.x

                # Apply feature dropout AFTER projection (evaluation doesn't need dropout but kept for consistency)
                # x_input = apply_feature_dropout_if_enabled(x_input, args, rank=0)  # Commented out for evaluation

                batch_data.adj_t = SparseTensor.from_edge_index(
                    batch_data.edge_index,
                    sparse_sizes=(batch_data.num_nodes, batch_data.num_nodes)
                ).to_symmetric().coalesce()
                
                # Get node embeddings from GNN
                node_embeddings = model(x_input, batch_data.adj_t)
                
                # Pool to get graph embeddings
                target_embeddings = pool_graph_embeddings(node_embeddings, batch_data.batch, pooling_method)
                
                # Get labels for this batch
                batch_labels = batch_data.y
                
                # Handle label shapes
                if is_multitask:
                    # Multi-task format
                    if batch_labels.dim() == 1 and len(batch_labels) == batch_size * num_tasks:
                        batch_labels = batch_labels.view(batch_size, num_tasks)
                    
                    # Extract labels for current task
                    task_labels = batch_labels[:, task_idx]
                else:
                    # Single-task format
                    if batch_labels.dim() > 1:
                        task_labels = batch_labels.squeeze()
                    else:
                        task_labels = batch_labels
                
                # Filter valid labels for this task
                valid_mask = ~torch.isnan(task_labels)
                if valid_mask.sum() == 0:
                    continue
                
                valid_embeddings = target_embeddings[valid_mask]
                valid_labels = task_labels[valid_mask].long()
                
                if valid_embeddings.size(0) == 0:
                    continue
                
                # Create task-specific context embeddings
                context_embeddings, context_labels = _create_context_embeddings_computed(
                    model, dataset_info['context_graphs'], dataset_info['dataset'], task_idx,
                    pooling_method, device, identity_projection, dataset_info
                )
                
                # Prepare PFN data structure
                pfn_data = prepare_pfn_data_structure(context_embeddings, context_labels,
                                                    dataset_info['num_classes'], device)
                
                # Process context embeddings to create class prototypes
                class_h = process_node_features(
                    context_embeddings, pfn_data,
                    degree_normalize=False, attention_pool_module=None,
                    mlp_module=None, normalize=normalize_class_h
                )
                
                # Use PFN predictor
                scores, _ = predictor(pfn_data, context_embeddings, valid_embeddings, context_labels, class_h)
                
                # Get predictions and probabilities (same as single task version)
                probabilities = F.softmax(scores, dim=1)
                predictions = torch.argmax(scores, dim=1)
                
                all_predictions.append(predictions.cpu())
                all_labels.append(valid_labels.cpu())
                all_probabilities.append(probabilities.cpu())
            
            # Calculate metric(s) for this task
            if all_predictions:
                all_predictions_task = torch.cat(all_predictions, dim=0)
                all_labels_task = torch.cat(all_labels, dim=0)
                all_probabilities_task = torch.cat(all_probabilities, dim=0)
                
                if isinstance(metric_type, list):
                    # Multiple metrics (e.g., PCBA with both AUC and AP)
                    task_metrics = calculate_multiple_metrics(all_predictions_task, all_labels_task, all_probabilities_task, metric_type)
                    all_task_metrics.append(task_metrics)
                else:
                    # Single metric
                    task_metric = calculate_metric(all_predictions_task, all_labels_task, all_probabilities_task, metric_type)
                    all_task_metrics.append(task_metric)
        
        # Aggregate across all tasks for this split
        if all_task_metrics:
            if isinstance(metric_type, list):
                # Multiple metrics: aggregate each metric separately
                aggregated_metrics = {}
                for metric_name in metric_type:
                    metric_values = [task_metrics[metric_name] for task_metrics in all_task_metrics]
                    aggregated_metrics[metric_name] = sum(metric_values) / len(metric_values)
                split_results[split_name] = aggregated_metrics
            else:
                # Single metric: simple average
                split_results[split_name] = sum(all_task_metrics) / len(all_task_metrics)
        else:
            if isinstance(metric_type, list):
                split_results[split_name] = {metric_name: 0.0 for metric_name in metric_type}
            else:
                split_results[split_name] = 0.0
    
    return split_results


def train_graph_classification_single_task_no_update(model, predictor, dataset_info, batch, task_idx,
                                                     pooling_method, device, orthogonal_push, normalize_class_h, identity_projection, context_k=None, args=None):
    """
    Process single task without parameter update - for accumulating losses.
    Returns loss tensor that can be accumulated.
    """
    batch_data = batch.to(device)
    
    # Convert node indices to embeddings using unified node_embs
    node_emb_table = _get_node_embedding_table(dataset_info['dataset'], task_idx, device, dataset_info)
    if node_emb_table is not None:
        batch_data_x = _safe_lookup_node_embeddings(node_emb_table, batch_data.x.long(), context=f"train_full_task{task_idx}", 
                                                    batch_data=batch_data, dataset_info=dataset_info)
    else:
        raise ValueError("Expected node embedding table under unified setting")
    
    # Apply identity projection if needed
    if dataset_info.get('needs_identity_projection', False) and identity_projection is not None:
        x_input = identity_projection(batch_data_x)
    else:
        x_input = batch_data_x

    # Apply feature dropout AFTER projection
    x_input = apply_feature_dropout_if_enabled(x_input, args, rank=0)

    # Apply edge dropout if enabled (before creating adj_t)
    if args is not None and hasattr(args, 'edge_dropout_enabled') and args.edge_dropout_enabled and hasattr(args, 'edge_dropout_rate'):
        batch_data = batch_edge_dropout(batch_data, args.edge_dropout_rate, training=model.training)

    batch_data.adj_t = SparseTensor.from_edge_index(
        batch_data.edge_index,
        sparse_sizes=(batch_data.num_nodes, batch_data.num_nodes)
    ).to_symmetric().coalesce()

    # Get node embeddings from GNN
    node_embeddings = model(x_input, batch_data.adj_t)
    
    # Pool to get graph embeddings
    target_embeddings = pool_graph_embeddings(node_embeddings, batch_data.batch, pooling_method)

    # Get labels for this batch
    batch_labels = batch_data.y
    batch_size = target_embeddings.size(0)
    
    # Skip single-sample batches
    if batch_size == 1:
        return 0.0
    
    # Check if this is a multi-task dataset
    sample_graph = dataset_info['dataset'][0]
    is_multitask = sample_graph.y.numel() > 1
    
    if is_multitask:
        # Multi-task format
        num_tasks = sample_graph.y.numel()
        if batch_labels.dim() == 1 and len(batch_labels) == batch_size * num_tasks:
            batch_labels = batch_labels.view(batch_size, num_tasks)
        
        # Extract labels for current task
        batch_labels = batch_labels[:, task_idx]
    else:
        # Single-task format
        if batch_labels.dim() > 1:
            batch_labels = batch_labels.squeeze()
    
    # Filter valid labels  
    valid_mask = ~torch.isnan(batch_labels)
    if valid_mask.sum() <= 1:
        return 0.0
    
    valid_embeddings = target_embeddings[valid_mask]
    valid_labels = batch_labels[valid_mask].long()
    
    # Create task-specific context embeddings
    context_embeddings, context_labels = _create_context_embeddings_computed(
        model, dataset_info['context_graphs'], dataset_info['dataset'], task_idx,
        pooling_method, device, identity_projection, dataset_info
    )
    
    # Prepare PFN data structure
    pfn_data = prepare_pfn_data_structure(context_embeddings, context_labels,
                                        dataset_info['num_classes'], device)
    
    # Process context embeddings to create class prototypes
    class_h = process_node_features(
        context_embeddings, pfn_data,
        degree_normalize=False, attention_pool_module=None,
        mlp_module=None, normalize=normalize_class_h
    )
    
    # Use PFN predictor
    scores, _ = predictor(pfn_data, context_embeddings, valid_embeddings, context_labels, class_h)
    scores = F.log_softmax(scores, dim=1)
    
    # Compute loss
    nll_loss = F.nll_loss(scores, valid_labels)
    
    return nll_loss


def train_graph_classification_single_task(model, predictor, dataset_info, data_loaders, optimizer, task_idx,
                                         pooling_method='mean', device='cuda', clip_grad=1.0,
                                         orthogonal_push=0.0, normalize_class_h=True, identity_projection=None, context_k=None, args=None):
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

    # Pre-extract all batches to eliminate DataLoader iterator overhead
    train_batches = list(data_loaders['train'])

    total_batches = len(train_batches)
    for batch_idx, batch_graphs in enumerate(train_batches):
        # Batch-level context refresh
        if args is not None:
            refresh_gc_context_if_needed(dataset_info, batch_idx, 0, args, device)
            
        print(f"\rTraining batch {batch_idx+1}/{total_batches}", end="", flush=True)
        batch_data = batch_graphs.to(device)
        
        # Convert node indices to embeddings using unified node_embs
        node_emb_table = _get_node_embedding_table(dataset_info['dataset'], task_idx, device, dataset_info)
        if node_emb_table is not None:
            batch_data.x = _safe_lookup_node_embeddings(node_emb_table, batch_data.x, context=f"train_task{task_idx}", 
                                                        batch_data=batch_data, dataset_info=dataset_info)
        else:
            raise ValueError("Expected node embedding table under unified setting")
        
        # Apply identity projection if needed
        needs_proj = dataset_info.get('needs_identity_projection', False)
        has_proj = identity_projection is not None

        if needs_proj and has_proj:
            x_input = identity_projection(batch_data.x)
        else:
            x_input = batch_data.x

        # Apply feature dropout AFTER projection
        x_input = apply_feature_dropout_if_enabled(x_input, args, rank=0)
        
        # Apply edge dropout if enabled (before creating adj_t)
        if args is not None and hasattr(args, 'edge_dropout_enabled') and args.edge_dropout_enabled and hasattr(args, 'edge_dropout_rate'):
            batch_data = batch_edge_dropout(batch_data, args.edge_dropout_rate, training=model.training)

        batch_data.adj_t = SparseTensor.from_edge_index(
            batch_data.edge_index,
            sparse_sizes=(batch_data.num_nodes, batch_data.num_nodes)
        ).to_symmetric().coalesce()

        # Get node embeddings from GNN using SparseTensor format
        node_embeddings = model(x_input, batch_data.adj_t)
        
        # Pool to get graph embeddings
        target_embeddings = pool_graph_embeddings(node_embeddings, batch_data.batch, pooling_method)

        # Get labels for this batch - all samples are valid for this task
        batch_labels = batch_data.y
        batch_size = target_embeddings.size(0)
        
        # Skip single-sample batches to avoid dimension issues with PFN predictor
        if batch_size == 1:
            print(f"Warning: Skipping training batch with only 1 sample to avoid PFN predictor dimension issues")
            continue
        
        # Check if this is a multi-task dataset
        sample_graph = dataset_info['dataset'][0]
        is_multitask = sample_graph.y.numel() > 1
        
        if is_multitask:
            # Multi-task format: labels are flattened [batch_size * num_tasks]
            num_tasks = sample_graph.y.numel()
            if batch_labels.dim() == 1 and len(batch_labels) == batch_size * num_tasks:
                # Reshape from flattened format back to [batch_size, num_tasks]
                batch_labels = batch_labels.view(batch_size, num_tasks)
            
            # Extract labels for current task
            batch_labels = batch_labels[:, task_idx]
        else:
            # Single-task format: labels might be squeezed to 1D
            if batch_labels.dim() > 1:
                batch_labels = batch_labels.squeeze()
        
        # Ensure labels are long integers
        if batch_labels.dtype != torch.long:
            batch_labels = batch_labels.to(torch.long)
        
        # Create task-specific context embeddings
        context_embeddings, context_labels = _create_context_embeddings_computed(
            model, dataset_info['context_graphs'], dataset_info['dataset'], task_idx, 
            pooling_method, device, identity_projection, dataset_info
        )
        
        # Prepare PFN data structure
        pfn_data = prepare_pfn_data_structure(context_embeddings, context_labels, 
                                            dataset_info['num_classes'], device)
        
        # Process context embeddings to create class prototypes
        class_h = process_node_features(
            context_embeddings, pfn_data,
            degree_normalize=False,  # Not applicable for graphs
            attention_pool_module=None,
            mlp_module=None,
            normalize=normalize_class_h
        )
        
        # Use PFN predictor (all samples are valid, no masking needed)
        scores, refined_class_h = predictor(pfn_data, context_embeddings, target_embeddings, context_labels, class_h)
        scores = F.log_softmax(scores, dim=1)
        
        # Compute loss - no masking needed since all samples are valid
        nll_loss = F.nll_loss(scores, batch_labels)
        
        # Orthogonal loss for refined class prototypes
        if orthogonal_push > 0:
            refined_class_h_norm = F.normalize(refined_class_h, p=2, dim=1)
            class_matrix = refined_class_h_norm @ refined_class_h_norm.T
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
            if identity_projection: torch.nn.utils.clip_grad_norm_(identity_projection.parameters(), clip_grad)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        print(f"\rTraining batch {batch_idx+1}/{total_batches} completed (loss: {loss.item():.4f})", end="", flush=True)
    
    print(f"\nTraining completed: {num_batches} batches, avg loss: {total_loss/num_batches if num_batches > 0 else 0.0:.4f}")
    return total_loss / num_batches if num_batches > 0 else 0.0

@torch.no_grad()
def evaluate_graph_classification_single_task(model, predictor, dataset_info, data_loaders, task_idx,
                                             pooling_method='mean', device='cuda', 
                                             normalize_class_h=True, dataset_name=None, identity_projection=None, context_k=None):
    """
    Evaluate graph classification performance for a single task with prefiltered data.
    All samples in each batch are guaranteed to have valid labels for the specified task.
    
    Args:
        model: GNN model
        predictor: PFN predictor
        dataset_info (dict): Dataset information including context graphs
        data_loaders (dict): Task-specific data loaders (prefiltered)
        task_idx (int): Current task index
        pooling_method (str): Graph pooling method
        device (str): Device for computation
        normalize_class_h (bool): Whether to normalize class embeddings
        dataset_name (str): Name of the dataset (for metric selection)
        
    Returns:
        dict: Dictionary with train/val/test metrics for this task (AUC for chemhiv, AUC for chempcba, accuracy for others)
    """
    model.eval()
    predictor.eval()
    
    results = {}
    total_start = time.perf_counter()
    
    # Time context creation
    context_start = time.perf_counter()
    # Create task-specific context embeddings once for this task
    context_embeddings, context_labels = _create_context_embeddings_computed(
        model, dataset_info['context_graphs'], dataset_info['dataset'], task_idx, 
        pooling_method, device, identity_projection, dataset_info
    )
    context_time = time.perf_counter() - context_start
    
    # Time PFN data preparation
    pfn_prep_start = time.perf_counter()
    # Prepare PFN data structure
    pfn_data = prepare_pfn_data_structure(context_embeddings, context_labels, 
                                        dataset_info['num_classes'], device)
    
    # Process context embeddings to create class prototypes
    class_h = process_node_features(
        context_embeddings, pfn_data,
        degree_normalize=False,
        attention_pool_module=None,
        mlp_module=None,
        normalize=normalize_class_h
    )
    pfn_prep_time = time.perf_counter() - pfn_prep_start

    # Determine the appropriate metric for this dataset
    # Check if this is a multi-task dataset
    sample_graph = dataset_info['dataset'][0]
    is_multitask = sample_graph.y.numel() > 1
    metric_type = get_dataset_metric(dataset_name, is_multitask) if dataset_name else 'accuracy'

    # Evaluate on each split
    split_times = {}
    for split_name, data_loader in data_loaders.items():
        split_start = time.perf_counter()
        if len(data_loader.dataset) == 0:
            results[split_name] = 0.0
            split_times[split_name] = 0.0
            continue
            
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        # Time batch processing (direct iteration, no pre-extraction)
        batch_processing_start = time.perf_counter()
        total_samples = len(data_loader.dataset)
        
        for batch_idx, batch_graphs in enumerate(data_loader):
            print(f"\rEvaluating batch {batch_idx+1}...", end="", flush=True)
            batch_data = batch_graphs.to(device)
            node_emb_table = _get_node_embedding_table(dataset_info['dataset'], task_idx, device, dataset_info)
            
            if node_emb_table is not None:
                batch_data.x = _safe_lookup_node_embeddings(node_emb_table, batch_data.x, context=f"eval_task{task_idx}_batch{batch_idx}", 
                                                            batch_data=batch_data, dataset_info=dataset_info)
            else:
                raise ValueError("Expected node embedding table under unified setting")
            
            # Apply identity projection if needed
            if dataset_info.get('needs_identity_projection', False) and identity_projection is not None:
                x_input = identity_projection(batch_data.x)
            else:
                x_input = batch_data.x

            # Apply feature dropout AFTER projection (evaluation doesn't need dropout but kept for consistency)
            # x_input = apply_feature_dropout_if_enabled(x_input, args, rank=0)  # Commented out for evaluation

            batch_data.adj_t = SparseTensor.from_edge_index(
                batch_data.edge_index,
                sparse_sizes=(batch_data.num_nodes, batch_data.num_nodes)
            ).to_symmetric().coalesce()
            node_embeddings = model(x_input, batch_data.adj_t)
            
            target_embeddings = pool_graph_embeddings(node_embeddings, batch_data.batch, pooling_method)
            
            # Get labels for this batch - all samples are valid for this task
            batch_labels = batch_data.y
            batch_size = target_embeddings.size(0)
            
            # Skip single-sample batches to avoid dimension issues with PFN predictor
            if batch_size == 1:
                print(f"Warning: Skipping batch with only 1 sample to avoid PFN predictor dimension issues")
                continue
            
            # Check if this is a multi-task dataset
            sample_graph = dataset_info['dataset'][0]
            is_multitask = sample_graph.y.numel() > 1
            
            if is_multitask:
                # Multi-task format: reshape and extract task labels
                num_tasks = sample_graph.y.numel()
                if batch_labels.dim() == 1 and len(batch_labels) == batch_size * num_tasks:
                    batch_labels = batch_labels.view(batch_size, num_tasks)
                batch_labels = batch_labels[:, task_idx]
            else:
                # Single-task format
                if batch_labels.dim() > 1:
                    batch_labels = batch_labels.squeeze()
        
            # Ensure labels are long integers
            if batch_labels.dtype != torch.long:
                batch_labels = batch_labels.to(torch.long)
            
            scores, _ = predictor(pfn_data, context_embeddings, target_embeddings, context_labels, class_h)
            probabilities = F.softmax(scores, dim=1)
            predictions = scores.argmax(dim=1)
            
            all_predictions.append(predictions.cpu())
            all_labels.append(batch_labels.cpu())
            all_probabilities.append(probabilities.cpu())
            print(f"\rEvaluating batch {batch_idx+1} completed ({len(predictions)} samples)", end="", flush=True)

        # Concatenate all predictions and labels
        if all_predictions:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_probabilities = torch.cat(all_probabilities, dim=0)
            print(f"\n{split_name} evaluation: {len(all_predictions)} total samples processed", end="")
            
            # Calculate the appropriate metric(s)
            if isinstance(metric_type, list):
                # Multiple metrics (e.g., PCBA with both AUC and AP)
                metric_values = calculate_multiple_metrics(all_predictions, all_labels, all_probabilities, metric_type)
                results[split_name] = metric_values
            else:
                # Single metric
                metric_value = calculate_metric(all_predictions, all_labels, all_probabilities, metric_type)
                results[split_name] = metric_value
        else:
            if isinstance(metric_type, list):
                results[split_name] = {metric_name: 0.0 for metric_name in metric_type}
            else:
                results[split_name] = 0.0
        
        # Time split completion
        batch_processing_time = time.perf_counter() - batch_processing_start
        split_times[split_name] = time.perf_counter() - split_start
        
        print(f" | processing: {batch_processing_time:.3f}s")

    # Final timing summary
    total_time = time.perf_counter() - total_start
    print(f"Task {task_idx} timing: context: {context_time:.3f}s, pfn_prep: {pfn_prep_time:.3f}s, total: {total_time:.3f}s")
    if split_times:
        split_breakdown = ", ".join([f"{split}: {t:.3f}s" for split, t in split_times.items()])
        print(f"  Split breakdown: {split_breakdown}")
    
    return results


def train_and_evaluate_graph_classification(model, predictor, train_datasets, train_processed_data_list, args, 
                                          optimizer, scheduler=None, device='cuda', test_datasets=None, test_processed_data_list=None, identity_projection=None):
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
    
    # Determine training approach based on configuration
    print(args)
    use_full_batch_training = getattr(args, 'full_batch_training', False)
    
    if use_full_batch_training:
        print("Full batch training: using unfiltered data loaders (no pre-filtering)")
        # For full batch training, we DON'T pre-filter by tasks
        # We want all graphs in each batch, then dynamically check all 128 tasks
        all_task_filtered_splits = []  # Keep variable for compatibility, but won't use task filtering
        
        for train_dataset_info in train_processed_data_list:
            # Use original splits WITHOUT task filtering
            all_task_filtered_splits.append(train_dataset_info['split_idx'])
    else:
        print("Task-specific training: using task-filtered data loaders")
        # Create task-filtered data loaders for task-specific training
        all_task_filtered_splits = []
        
        for train_dataset_info in train_processed_data_list:
            # Create task-filtered datasets
            task_filtered_splits = create_task_filtered_datasets(
                train_dataset_info['dataset'], 
                train_dataset_info['split_idx']
            )
            all_task_filtered_splits.append(task_filtered_splits)
    
    # Pre-filter test datasets once (if provided) - but skip filtering for full batch training
    all_test_task_filtered_splits = []
    if test_datasets is not None and test_processed_data_list is not None:
        if use_full_batch_training:
            print("Full batch training: using unfiltered test datasets")
            for test_dataset_info in test_processed_data_list:
                # Use original splits WITHOUT task filtering
                all_test_task_filtered_splits.append(test_dataset_info['split_idx'])
        else:
            print("Pre-filtering test datasets (one-time operation)...")
            for test_dataset_info in test_processed_data_list:
                test_name = test_dataset_info['dataset'].name if hasattr(test_dataset_info['dataset'], 'name') else f'test_dataset_{len(all_test_task_filtered_splits)}'
                filter_start_time = time.perf_counter()
                
                # Create task-filtered datasets for test evaluation (one-time)
                test_task_filtered_splits = create_task_filtered_datasets(
                    test_dataset_info['dataset'], 
                    test_dataset_info['split_idx'],
                    "test"  # Only need test split
                )
                all_test_task_filtered_splits.append(test_task_filtered_splits)
                
                print(f"Pre-filtered test dataset {test_name} in {time.perf_counter() - filter_start_time:.2f}s")
            print("Test dataset pre-filtering completed.")
    
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    patience = 50  # Early stopping patience
    best_results = {}
    
    # Training loop
    for epoch in range(args.epochs):
        epoch_start_time = time.perf_counter()
        
        # Training step 
        total_train_loss = 0
        
        if use_full_batch_training:
            # Full batch training - no pre-filtering, process all tasks dynamically
            for dataset_idx, (train_dataset_info, unfiltered_splits) in enumerate(zip(train_processed_data_list, all_task_filtered_splits)):
                log_gpu_memory(f"EPOCH_{epoch}, DATASET_{dataset_idx}")
                dataset_start_time = time.perf_counter()
                
                # Train using full batch approach (unfiltered splits)
                dataset_loss = train_graph_classification_full_batch(
                    model, predictor, train_dataset_info, unfiltered_splits, optimizer,
                    pooling_method=args.graph_pooling, device=device, batch_size=args.batch_size,
                    clip_grad=args.clip_grad, orthogonal_push=args.orthogonal_push,
                    normalize_class_h=args.normalize_class_h, identity_projection=identity_projection,
                    context_k=getattr(args, 'context_k', None), args=args
                )
                
                total_train_loss += dataset_loss
                print(f"Epoch {epoch:3d}, Dataset {dataset_names[dataset_idx]} (Full Batch, No Pre-filtering): Loss = {dataset_loss:.4f}, Time: {time.perf_counter() - dataset_start_time:.2f}s)")
        else:
            # Task-specific training (original approach)
            for dataset_idx, (train_dataset_info, task_filtered_splits) in enumerate(zip(train_processed_data_list, all_task_filtered_splits)):
                log_gpu_memory(f"EPOCH_{epoch}, DATASET_{dataset_idx}")
                dataset_start_time = time.perf_counter()
                dataset_loss = 0
                dataset_tasks = 0
                
                # Train on each task separately using prefiltered data
                for task_idx, task_splits in task_filtered_splits.items():
                    # Create task-specific data loaders
                    # Check if FUG mapping is present to use index tracking
                    use_fug_tracking = 'fug_mapping' in train_dataset_info
                    task_data_loaders = create_data_loaders(
                        train_dataset_info['dataset'], 
                        task_splits,
                        batch_size=args.batch_size,
                        shuffle=True,
                        task_idx=task_idx,
                        use_index_tracking=use_fug_tracking
                    )
                    
                    # Train on this specific task (all samples in batch are valid for this task)
                    task_loss = train_graph_classification_single_task(
                        model, predictor, train_dataset_info, task_data_loaders, optimizer, task_idx,
                        pooling_method=args.graph_pooling, device=device,
                        clip_grad=args.clip_grad, orthogonal_push=args.orthogonal_push,
                        normalize_class_h=args.normalize_class_h, identity_projection=identity_projection,
                        context_k=getattr(args, 'context_k', None), args=args
                    )
                    
                    dataset_loss += task_loss
                    dataset_tasks += 1
                    print(f"Task {task_idx} completed: loss={task_loss:.4f}")
                
                # Average loss across tasks for this dataset
                if dataset_tasks > 0:
                    avg_dataset_loss = dataset_loss / dataset_tasks
                    total_train_loss += avg_dataset_loss
                    print(f"Epoch {epoch:3d}, Dataset {dataset_names[dataset_idx]}: Avg Task Loss = {avg_dataset_loss:.4f} ({dataset_tasks} tasks), Time: {time.perf_counter() - dataset_start_time:.2f}s)")
                else:
                    print(f"Epoch {epoch:3d}, Dataset {dataset_names[dataset_idx]}: No valid tasks")
        
        avg_train_loss = total_train_loss / len(train_processed_data_list)
        
        # Validation evaluation (consistent with training approach)
        all_val_results = {}
        total_val_acc = 0
        
        if use_full_batch_training:
            # Full batch evaluation - use unfiltered evaluation, check all tasks dynamically
            for dataset_idx, (train_dataset_info, unfiltered_splits) in enumerate(zip(train_processed_data_list, all_task_filtered_splits)):
                dataset_start_time = time.perf_counter()
                dataset_name = dataset_names[dataset_idx]
                log_gpu_memory(f"EPOCH_{epoch}, DATASET_{dataset_idx}")
                
                # Create unfiltered evaluation loaders
                # Check if FUG mapping is present to use index tracking
                use_fug_tracking = 'fug_mapping' in train_dataset_info
                eval_loaders = create_data_loaders(
                    train_dataset_info['dataset'], 
                    unfiltered_splits,  # No task filtering
                    batch_size=args.batch_size,
                    shuffle=False,
                    task_idx=None,  # No specific task
                    use_index_tracking=use_fug_tracking
                )
                
                # Evaluate using unfiltered approach - check all tasks dynamically
                val_results = evaluate_graph_classification_full_batch(
                    model, predictor, train_dataset_info, eval_loaders,
                    pooling_method=args.graph_pooling, device=device,
                    normalize_class_h=args.normalize_class_h, dataset_name=dataset_name, 
                    identity_projection=identity_projection
                )
                
                all_val_results[dataset_name] = val_results
                if 'val' in val_results:
                    if isinstance(val_results['val'], dict):
                        # Multiple metrics: use AUC for validation tracking (primary metric for PCBA)
                        total_val_acc += val_results['val'].get('auc', val_results['val'].get('ap', 0.0))
                    else:
                        total_val_acc += val_results['val']
                
                sample_graph = train_dataset_info['dataset'][0]
                is_multitask = sample_graph.y.numel() > 1
                metric_type = get_dataset_metric(dataset_name, is_multitask)
                
                if isinstance(metric_type, list):
                    # Multiple metrics (e.g., PCBA with AUC and AP)
                    train_str = format_metric_results(val_results['train'])
                    val_str = format_metric_results(val_results['val'])
                    print(f"Epoch {epoch:3d}, Dataset {dataset_name} (Full Batch, No Pre-filtering): Train {train_str}, Val {val_str}, Time: {time.perf_counter() - dataset_start_time:.2f}s")
                else:
                    # Single metric
                    metric_name = metric_type.upper()
                    print(f"Epoch {epoch:3d}, Dataset {dataset_name} (Full Batch, No Pre-filtering): Train {metric_name} = {val_results['train']:.4f}, Val {metric_name} = {val_results['val']:.4f}, Time: {time.perf_counter() - dataset_start_time:.2f}s")
        else:
            # Task-specific evaluation (original approach)
            for dataset_idx, (train_dataset_info, task_filtered_splits) in enumerate(zip(train_processed_data_list, all_task_filtered_splits)):
                dataset_start_time = time.perf_counter()
                dataset_name = dataset_names[dataset_idx]
                log_gpu_memory(f"EPOCH_{epoch}, DATASET_{dataset_idx}")
                
                # Evaluate each task separately and aggregate results
                task_val_accs = []
                task_train_accs = []
                
                for task_idx, task_splits in task_filtered_splits.items():
                    # Create task-specific data loaders for evaluation
                    # Check if FUG mapping is present to use index tracking
                    use_fug_tracking = 'fug_mapping' in train_dataset_info
                    task_eval_loaders = create_data_loaders(
                        train_dataset_info['dataset'], 
                        task_splits,
                        batch_size=args.batch_size,
                        shuffle=False,
                        task_idx=task_idx,
                        use_index_tracking=use_fug_tracking
                    )
                    
                    # Evaluate this specific task
                    task_results = evaluate_graph_classification_single_task(
                        model, predictor, train_dataset_info, task_eval_loaders, task_idx,
                        pooling_method=args.graph_pooling, device=device,
                        normalize_class_h=args.normalize_class_h, dataset_name=dataset_name, identity_projection=identity_projection,
                        context_k=getattr(args, 'context_k', None)
                    )
                    task_train_accs.append(task_results['train'])
                    task_val_accs.append(task_results['val'])
                    print(f"Task {task_idx} evaluation completed: train={task_results['train']:.4f}, val={task_results['val']:.4f}")
                
                # Aggregate results across tasks
                if task_val_accs:
                    val_results = {
                        'train': aggregate_task_metrics(task_train_accs),
                        'val': aggregate_task_metrics(task_val_accs),
                    }
                    all_val_results[dataset_name] = val_results
                    # Handle multiple metrics for total validation tracking
                    if isinstance(val_results['val'], dict):
                        total_val_acc += val_results['val'].get('auc', val_results['val'].get('ap', 0.0))
                    else:
                        total_val_acc += val_results['val']
                else:
                    # No valid tasks
                    val_results = {'train': 0.0, 'val': 0.0}
                    all_val_results[dataset_name] = val_results
                
                # Check if this is a multi-task dataset
                sample_graph = train_dataset_info['dataset'][0]
                is_multitask = sample_graph.y.numel() > 1
                metric_type = get_dataset_metric(dataset_name, is_multitask)
                
                if isinstance(metric_type, list):
                    # Multiple metrics (e.g., PCBA with AUC and AP)
                    train_str = format_metric_results(val_results['train'])
                    val_str = format_metric_results(val_results['val'])
                    print(f"Epoch {epoch:3d}, Dataset {dataset_name}: Train {train_str}, Val {val_str}, Time: {time.perf_counter() - dataset_start_time:.2f}s")
                else:
                    # Single metric
                    metric_name = metric_type.upper()
                    print(f"Epoch {epoch:3d}, Dataset {dataset_name}: Train {metric_name} = {val_results['train']:.4f}, Val {metric_name} = {val_results['val']:.4f}, Time: {time.perf_counter() - dataset_start_time:.2f}s")
        
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

        # Prepare metrics for wandb logging (accumulate all metrics for this epoch)
        epoch_wandb_metrics = {}
        
        if epoch % args.log_interval == 0:
            # Log average metrics across all training datasets
            avg_train_acc = sum(results['train'] for results in all_val_results.values()) / len(all_val_results)
            avg_valid_acc = sum(results['val'] for results in all_val_results.values()) / len(all_val_results)

            epoch_time = time.perf_counter() - epoch_start_time
            gpu_mem = get_gpu_memory_usage()
            print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | "
                    f"Avg Train Acc: {avg_train_acc:.4f} | Avg Val Acc: {avg_valid_acc:.4f} | "
                    f"Epoch Time: {epoch_time:.2f}s | "
                    f"GPU: {gpu_mem['allocated']:.2f}GB")
            
            # Accumulate epoch metrics for wandb
            epoch_wandb_metrics.update({
                'train_loss': avg_train_loss,
                'avg_train_metric': avg_train_acc,
                'avg_val_metric': avg_valid_acc,
                'best_val_metric': best_val_acc,
                'lr': optimizer.param_groups[0]['lr']
            })
            
            # Add per-dataset metrics
            for dataset_name, results in all_val_results.items():
                epoch_wandb_metrics[f'{dataset_name}_train_metric'] = results['train']
                epoch_wandb_metrics[f'{dataset_name}_val_metric'] = results['val']
        
        # Evaluation on unseen test datasets at eval intervals
        if epoch % args.eval_interval == 0:
            
            # Evaluate on unseen test datasets if provided
            if test_datasets is not None and test_processed_data_list is not None:
                print(f"  Evaluating on unseen test datasets at epoch {epoch}...")
                
                
                avg_test_metric = 0.0
                
                if use_full_batch_training:
                    # Full batch evaluation on unseen datasets - use unfiltered approach
                    for dataset_idx, (test_dataset_info, test_unfiltered_splits) in enumerate(zip(test_processed_data_list, all_test_task_filtered_splits)):
                        dataset_start_time = time.perf_counter()
                        test_name = test_dataset_info['dataset'].name if hasattr(test_dataset_info['dataset'], 'name') else f'test_dataset_{dataset_idx}'
                        
                        # Create unfiltered test loaders (only test split)
                        test_only_splits = {'test': test_unfiltered_splits['test']} if 'test' in test_unfiltered_splits else test_unfiltered_splits
                        # Check if FUG mapping is present to use index tracking
                        use_fug_tracking = 'fug_mapping' in test_dataset_info
                        test_loaders = create_data_loaders(
                            test_dataset_info['dataset'], 
                            test_only_splits,
                            batch_size=args.test_batch_size,
                            shuffle=False,
                            task_idx=None,  # No task filtering
                            use_index_tracking=use_fug_tracking
                        )
                        
                        # Evaluate using full batch approach
                        test_eval_results = evaluate_graph_classification_full_batch(
                            model, predictor, test_dataset_info, test_loaders,
                            pooling_method=args.graph_pooling, device=device,
                            normalize_class_h=args.normalize_class_h, dataset_name=test_name, 
                            identity_projection=identity_projection
                        )
                        
                        test_results = {'test': test_eval_results.get('test', 0.0)}
                        best_results[f"test_{test_name}"] = test_results.copy()
                        
                        test_sample_graph = test_dataset_info['dataset'][0]
                        test_is_multitask = test_sample_graph.y.numel() > 1
                        test_metric_type = get_dataset_metric(test_name, test_is_multitask)
                        
                        if isinstance(test_metric_type, list):
                            # Multiple metrics (e.g., PCBA with AUC and AP)
                            test_str = format_metric_results(test_results['test'])
                            print(f"    {test_name} (Full Batch, No Pre-filtering): Test {test_str}", 
                                  f"Time: {time.perf_counter() - dataset_start_time:.2f}s")
                            # Use AUC for averaging if available, otherwise AP
                            test_value = test_results['test'].get('auc', test_results['test'].get('ap', 0.0))
                            avg_test_metric += test_value
                        else:
                            # Single metric
                            test_metric_name = test_metric_type.upper()
                            print(f"    {test_name} (Full Batch, No Pre-filtering): Test {test_metric_name}={test_results['test']:.4f}", 
                                  f"Time: {time.perf_counter() - dataset_start_time:.2f}s")
                            avg_test_metric += test_results['test']
                        if isinstance(test_results['test'], dict):
                            # Multiple metrics: log each separately
                            for metric_name, value in test_results['test'].items():
                                epoch_wandb_metrics[f'test_{test_name}_{metric_name}'] = value
                        else:
                            epoch_wandb_metrics[f'test_{test_name}_metric'] = test_results['test']
                else:
                    # Task-specific evaluation (original): only test split per unseen dataset
                    for dataset_idx, (test_dataset_info, test_task_filtered_splits) in enumerate(zip(test_processed_data_list, all_test_task_filtered_splits)):
                        dataset_start_time = time.perf_counter()
                        test_name = test_dataset_info['dataset'].name if hasattr(test_dataset_info['dataset'], 'name') else f'test_dataset_{dataset_idx}'
                        
                        # Evaluate each task separately and aggregate - ONLY TEST SPLIT for unseen datasets
                        task_test_results = {'test': []}
                        
                        for task_idx, task_splits in test_task_filtered_splits.items():
                            # Create data loaders ONLY for test split of unseen dataset
                            test_only_splits = {'test': task_splits['test']}
                            # Check if FUG mapping is present to use index tracking
                            use_fug_tracking = 'fug_mapping' in test_dataset_info
                            task_test_loaders = create_data_loaders(
                                test_dataset_info['dataset'], 
                                test_only_splits,
                                batch_size=args.test_batch_size,
                                shuffle=False,
                                task_idx=task_idx,
                                use_index_tracking=use_fug_tracking
                            )
                            
                            # Debug: Check profiling setting
                            profiling_enabled = getattr(args, 'enable_profiling', False)
                            print(f"      [DEBUG] Task {task_idx}: enable_profiling = {profiling_enabled}")
                            
                            # Per-task profiling if enabled
                            if profiling_enabled:
                                prof = profile(
                                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                                    record_shapes=True,
                                    profile_memory=True,
                                    with_stack=True
                                )
                                prof.start()
                                with record_function(f"gc_eval_{test_name}_task_{task_idx}"):
                                    # Evaluate this specific task - only test split
                                    task_results = evaluate_graph_classification_single_task(
                                        model, predictor, test_dataset_info, task_test_loaders, task_idx,
                                        pooling_method=args.graph_pooling, device=device,
                                        normalize_class_h=args.normalize_class_h, dataset_name=test_name, identity_projection=identity_projection,
                                        context_k=getattr(args, 'context_k', None)
                                    )
                                prof.stop()
                                
                                # Save per-task profiling results
                                profile_filename = f"gc_eval_{test_name}_task_{task_idx}_epoch_{epoch}.json"
                                prof.export_chrome_trace(profile_filename)
                                print(f"      [PROFILING] Task {task_idx}: Saved profiling to {profile_filename}")
                                
                                # Print top CPU functions for this task
                                print(f"      [PROFILING] Task {task_idx} Top CPU ops:")
                                cpu_table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=5)
                                for line in cpu_table.split('\n')[:7]:  # Header + top 5 rows
                                    if line.strip():
                                        print(f"        {line}")
                            else:
                                # Evaluate this specific task - only test split  
                                task_results = evaluate_graph_classification_single_task(
                                    model, predictor, test_dataset_info, task_test_loaders, task_idx,
                                    pooling_method=args.graph_pooling, device=device,
                                    normalize_class_h=args.normalize_class_h, dataset_name=test_name, identity_projection=identity_projection,
                                    context_k=getattr(args, 'context_k', None)
                                )
                            
                            # Only collect test results for unseen datasets
                            task_test_results['test'].append(task_results['test'])

                        # Aggregate results across tasks - only test split
                        if task_test_results['test']:
                            test_results = {'test': aggregate_task_metrics(task_test_results['test'])}
                        else:
                            test_results = {'test': 0.0}
                        
                        # Store test results (will be overwritten each eval interval)
                        best_results[f"test_{test_name}"] = test_results.copy()
                        
                        # Check if this is a multi-task dataset for label only
                        test_sample_graph = test_dataset_info['dataset'][0]
                        test_is_multitask = test_sample_graph.y.numel() > 1
                        test_metric_type = get_dataset_metric(test_name, test_is_multitask)
                        
                        if isinstance(test_metric_type, list):
                            # Multiple metrics (e.g., PCBA with AUC and AP)
                            test_str = format_metric_results(test_results['test'])
                            print(f"    {test_name}: Test {test_str}", 
                                  f"Time: {time.perf_counter() - dataset_start_time:.2f}s")
                            # Use AUC for averaging if available, otherwise AP
                            test_value = test_results['test'].get('auc', test_results['test'].get('ap', 0.0))
                            avg_test_metric += test_value
                        else:
                            # Single metric
                            test_metric_name = test_metric_type.upper()
                            print(f"    {test_name}: Test {test_metric_name}={test_results['test']:.4f}", 
                                  f"Time: {time.perf_counter() - dataset_start_time:.2f}s")
                            avg_test_metric += test_results['test']
                        if isinstance(test_results['test'], dict):
                            # Multiple metrics: log each separately
                            for metric_name, value in test_results['test'].items():
                                epoch_wandb_metrics[f'test_{test_name}_{metric_name}'] = value
                        else:
                            epoch_wandb_metrics[f'test_{test_name}_metric'] = test_results['test']
                
                # Average test metric across all unseen datasets
                avg_test_metric /= len(test_datasets)
                print(f"  Average Test Metric across all unseen datasets: {avg_test_metric:.4f}")
                epoch_wandb_metrics['avg_test_metric'] = avg_test_metric
        
        # Early stopping based on validation performance on training dataset
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Log all accumulated metrics for this epoch with explicit step
        if epoch_wandb_metrics:
            try:
                import wandb
                wandb.log(epoch_wandb_metrics)
            except ImportError:
                pass  # wandb not available
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
    
    # Final summary
    final_gpu_mem = get_gpu_memory_usage()
    print(f"Training completed. Best validation metric: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Final GPU memory usage: {final_gpu_mem['allocated']:.2f}GB allocated, "
          f"{final_gpu_mem['max_allocated']:.2f}GB peak")
    
    return best_results
