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
from .data_gc import create_graph_batch, create_task_filtered_datasets
from .utils import process_node_features
from .data_utils import batch_edge_dropout, feature_dropout
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity
from .training_monitor import TrainingMonitor


def build_anchor_meta_graph(graph_embeddings, anchor_indices, k_neighbors=5, sim='cos', weight_sharpening=1.0):
    """
    Build a sparse meta-graph using anchor-based similarity.
    Instead of computing N×N similarity, we compute N×K similarity to K anchors.
    Then connect each graph to its top-k similar anchors.

    Args:
        graph_embeddings: [num_graphs, dim] tensor of all graph embeddings
        anchor_indices: Indices of anchor graphs (subset of context set)
        k_neighbors: Number of anchors to connect each graph to
        sim: Similarity metric ('cos', 'tanimoto', 'dot')
        weight_sharpening: Power to raise edge weights to (e.g., 2.0 makes strong edges stronger)

    Returns:
        adj: Sparse adjacency matrix [num_graphs, num_graphs]
    """
    num_graphs = graph_embeddings.size(0)
    num_anchors = len(anchor_indices)
    device = graph_embeddings.device

    anchor_embeddings = graph_embeddings[anchor_indices]  # [K, dim]

    # Compute similarity from all graphs to anchors: [N, K] matrix
    if sim == 'cos':
        norm_all = F.normalize(graph_embeddings, dim=-1)
        norm_anchors = F.normalize(anchor_embeddings, dim=-1)
        similarity_to_anchors = norm_all @ norm_anchors.t()  # [N, K]
    elif sim == 'tanimoto':
        dot_product = graph_embeddings @ anchor_embeddings.t()  # [N, K]
        all_norm_sq = (graph_embeddings ** 2).sum(dim=1, keepdim=True)  # [N, 1]
        anchor_norm_sq = (anchor_embeddings ** 2).sum(dim=1, keepdim=True).t()  # [1, K]
        similarity_to_anchors = dot_product / (all_norm_sq + anchor_norm_sq - dot_product + 1e-8)
    else:  # dot
        similarity_to_anchors = graph_embeddings @ anchor_embeddings.t()

    # Connect each graph to its top-k similar anchors
    k = min(k_neighbors, num_anchors)
    top_k_values, top_k_anchor_idx = torch.topk(similarity_to_anchors, k=k, dim=1)

    # Vectorized edge construction
    # Source nodes: each graph i repeated k times
    src_nodes = torch.arange(num_graphs, device=device).unsqueeze(1).repeat(1, k).flatten()  # [N*k]

    # Target nodes: map anchor indices to global indices
    anchor_indices_tensor = torch.tensor(anchor_indices, device=device)
    dst_nodes = anchor_indices_tensor[top_k_anchor_idx.flatten()]  # [N*k]

    # Edge weights - apply sharpening to control influence
    edge_weights = top_k_values.flatten()  # [N*k]

    # Apply weight sharpening: w' = w^k
    if weight_sharpening != 1.0:
        edge_weights = torch.pow(torch.clamp(edge_weights, min=0), weight_sharpening)

    # Create bidirectional edges
    edge_index = torch.stack([
        torch.cat([src_nodes, dst_nodes]),  # sources
        torch.cat([dst_nodes, src_nodes])   # targets
    ], dim=0)
    edge_weights = torch.cat([edge_weights, edge_weights])

    # Add self-loops with weight 1.0
    self_loop_index = torch.arange(num_graphs, device=device)
    self_loop_edges = torch.stack([self_loop_index, self_loop_index], dim=0)
    self_loop_weights = torch.ones(num_graphs, device=device)

    # Combine regular edges with self-loops
    edge_index = torch.cat([edge_index, self_loop_edges], dim=1)
    edge_weights = torch.cat([edge_weights, self_loop_weights])

    adj = SparseTensor.from_edge_index(
        edge_index,
        edge_attr=edge_weights,
        sparse_sizes=(num_graphs, num_graphs)
    )

    return adj


def correct_and_smooth_graph(adj, base_logits, train_idx, train_labels, num_classes,
                             num_iters=50, alpha=0.5):
    """
    Correct & Smooth: post-process feature-based predictions with label propagation on meta-graph.

    Args:
        adj: Sparse adjacency matrix [num_graphs, num_graphs]
        base_logits: Base predictions [num_graphs, num_classes]
        train_idx: Indices of training graphs (context set)
        train_labels: Labels for training graphs
        num_classes: Number of classes
        num_iters: Number of propagation iterations
        alpha: Smoothing factor (higher = more emphasis on old predictions)

    Returns:
        smoothed_probs: Smoothed probability predictions [num_graphs, num_classes]
    """
    device = base_logits.device
    num_graphs = base_logits.size(0)

    # Compute symmetric normalization: D^(-1/2) A D^(-1/2)
    deg = adj.sum(dim=1).to_dense()  # Row sum = degree
    deg_inv_sqrt = (deg + 1e-9).pow(-0.5)  # Add epsilon for numerical stability

    # Initialize with base predictions (softmax)
    y_soft = F.softmax(base_logits, dim=1)

    # Create one-hot labels for training set
    y_train = torch.zeros(num_graphs, num_classes, device=device)
    y_train[train_idx] = F.one_hot(train_labels.long(), num_classes).float()

    # Label propagation with residual connection
    for _ in range(num_iters):
        # Symmetric normalization: D^(-1/2) @ A @ D^(-1/2) @ y
        y_normalized = deg_inv_sqrt.view(-1, 1) * y_soft
        y_prop = adj @ y_normalized
        y_prop = deg_inv_sqrt.view(-1, 1) * y_prop

        # Blend: (1-α) * propagated + α * old
        y_soft = (1 - alpha) * y_prop + alpha * y_soft

        # Fix training labels (hard constraint)
        y_soft[train_idx] = y_train[train_idx]

    return y_soft


def apply_feature_dropout_if_enabled(x, args, rank=0, training=True):
    """
    Apply feature dropout if enabled in args (after projection only).

    Args:
        x (torch.Tensor): Input features
        args: Arguments containing feature dropout configuration
        rank (int): Process rank for logging
        training (bool): Whether the model is in training mode

    Returns:
        torch.Tensor: Features with dropout applied
    """
    if (args is not None and
        hasattr(args, 'feature_dropout_enabled') and args.feature_dropout_enabled and
        hasattr(args, 'feature_dropout_rate') and args.feature_dropout_rate > 0):

        dropout_type = getattr(args, 'feature_dropout_type', 'element_wise')
        verbose = getattr(args, 'verbose_feature_dropout', False) and rank == 0

        return feature_dropout(x, args.feature_dropout_rate, training=training,
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
            refresh_gc_context_graphs(dataset_info, args, device)
            if task_idx is not None:
                print(f"🔄 GC Dataset {dataset_name} context refreshed for task {task_idx} at batch {batch_idx}")
            else:
                print(f"🔄 GC Dataset {dataset_name} context refreshed (all tasks) at batch {batch_idx}")

def refresh_gc_context_graphs(dataset_info, args, device):
    """
    Refresh context graphs structure by resampling context graphs for each task/class.
    """
    from .data_gc import prepare_graph_data_for_pfn
    
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
    Handles external mapping for both OGB (integer features) and TU (float features).
    """
    import numpy as np

    # External mapping for FUG/Original Features - works for both OGB integer and TU float features
    # NOTE: x can be ANY shape when using external mapping - we don't use x values, only graph indices
    if dataset_info and 'fug_mapping' in dataset_info:
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
             raise ValueError(f"[FUG] Size mismatch after mapping: got {processed_embeddings_cpu.shape[0]} embeddings for {x.shape[0]} nodes")

        # 5. Transfer the final result to the GPU in one go.
        target_device = x.device if x.numel() > 0 else 'cuda'
        result = processed_embeddings_cpu.to(target_device)


        return result

    # Should never reach here if called correctly from _safe_lookup_node_embeddings
    raise ValueError(f"[FUG] _safe_lookup_node_embeddings_micro_optimized called without FUG mapping in {context}")


def _safe_lookup_node_embeddings(node_emb_table: torch.Tensor, x: torch.Tensor, context: str="",
                                 batch_data=None, dataset_info=None) -> torch.Tensor:
    """Return embedded node features ensuring indices are valid.
    Handles cases:
      1) FUG/Original Features: Use external mapping (both integer OGB and float TU features)
      2) x already is an embedding matrix (float, 2D) -> return as-is.
      3) x is Long indices (1D or [N,1]) -> validate range then index.
      4) x is numeric but not long -> cast after verifying integral values.
    Raises a clear Python exception instead of triggering a CUDA device-side assert.
    """
    # CRITICAL: Check for FUG/original features mapping FIRST (handles both OGB integer and TU float features)
    # NOTE: For datasets with external mapping, x can be ANY shape (including [N,1] for single-feature datasets)
    # The actual embeddings are in fug_mapping['node_embs'], so we use graph indices to lookup
    if dataset_info and 'fug_mapping' in dataset_info:
        # Use micro-optimized version for external mapping (works for both int and float raw features)
        return _safe_lookup_node_embeddings_micro_optimized(node_emb_table, x, context, batch_data, dataset_info)

    # Case 1: already embedded (float & 2D & width not 1) - ONLY if no FUG mapping
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


def get_dataset_metric(dataset_name, num_classes=None, is_multitask=None):
    """
    Get the appropriate evaluation metric(s) for a given dataset.
    Uses intelligent defaults based on task type with specific overrides.
    
    Args:
        dataset_name (str): Name of the dataset
        num_classes (int, optional): Number of classes in the dataset
        is_multitask (bool, optional): Whether this is a multi-task dataset
        
    Returns:
        str or list: Metric name(s) ('auc', 'ap', 'accuracy') or list of metrics for PCBA
    """
    dataset_name = dataset_name.lower()
    
    # Specific overrides for known datasets
    if 'chemhiv' in dataset_name or 'hiv' in dataset_name:
        return 'auc'
    elif 'chempcba' in dataset_name or 'pcba' in dataset_name:
        # PCBA: report both AP (OGB primary) and AUC for completeness
        return ['ap', 'auc']
    elif 'mnist' in dataset_name:
        # MNIST is multi-class classification
        return 'accuracy'
    
    # Intelligent defaults based on number of classes
    if num_classes is not None:
        if num_classes == 2:
            # Binary classification uses AUC
            return 'auc'
        else:
            # Multi-class classification uses accuracy
            return 'accuracy'
    
    # Intelligent defaults based on task type (legacy fallback)
    if is_multitask is not None:
        if is_multitask:
            # Multi-task datasets typically use AP (average precision)
            return 'ap'
        else:
            # Single-task datasets: assume binary for AUC (may not be correct)
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
        
    def _mean_ignore_nan(values):
        filtered = [v for v in values if v == v]  # NaN check
        if not filtered:
            return 0.0
        return sum(filtered) / len(filtered)

    if all(isinstance(val, dict) for val in metric_values):
        # All are dicts with multiple metrics - aggregate each metric separately
        aggregated_metrics = {}
        for metric_name in metric_values[0].keys():
            metric_vals = [val[metric_name] for val in metric_values]
            aggregated_metrics[metric_name] = _mean_ignore_nan(metric_vals)
        return aggregated_metrics
    elif all(isinstance(val, (int, float)) for val in metric_values):
        # All are single metrics - average, ignoring NaN
        return _mean_ignore_nan(metric_values)
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
        accuracy = (predictions == labels).float().mean().item()
        return accuracy
    elif metric_type == 'auc':
        # For binary classification, use probabilities of positive class
        if probabilities.dim() > 1:
            if probabilities.shape[1] == 2:
                probs = probabilities[:, 1].cpu().numpy()
            elif probabilities.shape[1] == 1:
                probs = probabilities.view(-1).cpu().numpy()
            else:
                probs = probabilities.cpu().numpy()
        else:
            probs = probabilities.cpu().numpy()
        
        labels_np = labels.cpu().numpy()
        
        # Debug: Check for unusual cases
        unique_labels = set(labels_np)
        if len(unique_labels) < 2:
            return float('nan')
        
        try:
            auc_score = roc_auc_score(labels_np, probs)
            if auc_score != auc_score:  # Check for NaN
                pass
                return 0.0
            return auc_score
        except ValueError as e:
            return float('nan')
    elif metric_type == 'ap':
        # For binary classification, use probabilities of positive class
        if probabilities.dim() > 1:
            if probabilities.shape[1] == 2:
                probs = probabilities[:, 1].cpu().numpy()
            elif probabilities.shape[1] == 1:
                probs = probabilities.view(-1).cpu().numpy()
            else:
                probs = probabilities.cpu().numpy()
        else:
            probs = probabilities.cpu().numpy()
        try:
            labels_np = labels.cpu().numpy()
            probs_np = probs
            ap_score = average_precision_score(labels_np, probs_np)
            return ap_score
        except ValueError as e:
            # Handle case where only one class is present
            return float('nan')
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

def pool_graph_embeddings(node_embeddings, batch, pooling_method='mean', virtualnode_embeddings=None):
    """
    Pool node embeddings to create graph-level embeddings.

    Args:
        node_embeddings (torch.Tensor or tuple): Node embeddings [num_nodes, hidden_dim],
                                                   or (node_embeddings, virtualnode_embeddings) if virtual node is used
        batch (torch.Tensor): Batch assignment for each node
        pooling_method (str): Pooling method ('mean', 'max', 'sum', 'virtual_node')
        virtualnode_embeddings (torch.Tensor, optional): Virtual node embeddings [num_graphs, hidden_dim]

    Returns:
        torch.Tensor: Graph-level embeddings [num_graphs, hidden_dim]
    """
    # Handle tuple return from model (when use_virtual_node=True)
    if isinstance(node_embeddings, tuple):
        node_embeddings, virtualnode_embeddings = node_embeddings

    # If virtual node embeddings are provided, use them directly
    if virtualnode_embeddings is not None:
        return virtualnode_embeddings

    # Otherwise use standard pooling
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
        node_embeddings = model(x_input, batch_data.adj_t, batch_data.batch)

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


def _create_all_task_context_embeddings(model, context_structure, dataset, pooling_method, device, identity_projection,
                                        dataset_info, return_timing=False, sync_cuda=False):
    """
    Compute context embeddings for ALL tasks in one pass through the context structure.
    Returns per-task tensors for pos/neg classes.

    Returns:
        pos_embeddings_by_task: list[Tensor] (len=num_tasks), each [n_pos, hidden_dim]
        neg_embeddings_by_task: list[Tensor] (len=num_tasks), each [n_neg, hidden_dim]
        timing (optional): dict with encode_time, overhead_time, concat_time, total_time, num_context_batches
    """
    if not context_structure:
        if return_timing:
            return [], [], {
                'encode_time': 0.0,
                'overhead_time': 0.0,
                'concat_time': 0.0,
                'total_time': 0.0,
                'num_context_batches': 0
            }
        return [], []

    task_indices = sorted(context_structure.keys())
    num_tasks = max(task_indices) + 1 if task_indices else 0
    pos_embeddings_by_task = [None] * num_tasks
    neg_embeddings_by_task = [None] * num_tasks

    timing = {
        'encode_time': 0.0,
        'overhead_time': 0.0,
        'concat_time': 0.0,
        'total_time': 0.0,
        'num_context_batches': 0
    }
    total_start = time.perf_counter()
    use_sync = sync_cuda and torch.cuda.is_available()

    for task_idx in task_indices:
        task_context = context_structure[task_idx]

        pos_list = []
        neg_list = []

        for class_label, context_data in task_context.items():
            if not context_data:
                continue
            step_start = time.perf_counter()

            if isinstance(context_data, dict) and 'graphs' in context_data and 'indices' in context_data:
                graphs = context_data['graphs']
                graph_indices = context_data['indices']
            else:
                graphs = context_data
                graph_indices = None

            batch_data = create_graph_batch(graphs, device)
            if graph_indices is not None:
                batch_data.original_graph_indices = torch.tensor(graph_indices, dtype=torch.long)

            node_emb_table = _get_node_embedding_table(dataset, task_idx, device, dataset_info)
            if node_emb_table is not None:
                batch_data.x = _safe_lookup_node_embeddings(
                    node_emb_table, batch_data.x,
                    context=f"vec_context_task{task_idx}", batch_data=batch_data, dataset_info=dataset_info
                )
            else:
                raise ValueError("Expected node embedding table under unified setting")

            if dataset_info and dataset_info.get('needs_identity_projection', False) and identity_projection is not None:
                x_input = identity_projection(batch_data.x)
            else:
                x_input = batch_data.x

            batch_data.adj_t = SparseTensor.from_edge_index(
                batch_data.edge_index,
                sparse_sizes=(batch_data.num_nodes, batch_data.num_nodes)
            ).to_symmetric().coalesce()

            if use_sync:
                torch.cuda.synchronize()
            encode_start = time.perf_counter()
            node_embeddings = model(x_input, batch_data.adj_t, batch_data.batch)
            graph_embeddings = pool_graph_embeddings(node_embeddings, batch_data.batch, pooling_method)
            if use_sync:
                torch.cuda.synchronize()
            encode_end = time.perf_counter()
            timing['encode_time'] += encode_end - encode_start

            if class_label == 1:
                pos_list.append(graph_embeddings)
            else:
                neg_list.append(graph_embeddings)
            step_end = time.perf_counter()
            timing['overhead_time'] += (step_end - step_start) - (encode_end - encode_start)
            timing['num_context_batches'] += 1

        concat_start = time.perf_counter()
        pos_embeddings_by_task[task_idx] = torch.cat(pos_list, dim=0) if pos_list else None
        neg_embeddings_by_task[task_idx] = torch.cat(neg_list, dim=0) if neg_list else None
        timing['concat_time'] += time.perf_counter() - concat_start

    timing['total_time'] = time.perf_counter() - total_start
    if return_timing:
        return pos_embeddings_by_task, neg_embeddings_by_task, timing
    return pos_embeddings_by_task, neg_embeddings_by_task


def _collect_multitask_context_graphs(context_structure, dataset, dataset_info=None):
    """
    Collect a unique set of context graphs across ALL tasks for multi-task ridge regression.

    Returns:
        graphs (list): List of unique Data objects
        graph_indices (list or None): Dataset indices for graphs (required for FUG mapping)
    """
    if not context_structure:
        return [], None

    # Prefer using indices for stable deduplication and FUG mapping
    use_indices = True
    unique_indices = []
    seen = set()
    for task_context in context_structure.values():
        for context_data in task_context.values():
            if not context_data:
                continue
            if isinstance(context_data, dict) and 'indices' in context_data:
                for idx in context_data['indices']:
                    idx_int = idx.item() if torch.is_tensor(idx) else int(idx)
                    if idx_int not in seen:
                        seen.add(idx_int)
                        unique_indices.append(idx_int)
            else:
                use_indices = False
                break
        if not use_indices:
            break

    if use_indices:
        unique_indices = sorted(unique_indices)
        graphs = [dataset[idx] for idx in unique_indices]
        return graphs, unique_indices

    # Fallback: deduplicate by object id (no indices available)
    graphs = []
    seen_ids = set()
    for task_context in context_structure.values():
        for context_data in task_context.values():
            if not context_data:
                continue
            if isinstance(context_data, dict) and 'graphs' in context_data:
                graph_list = context_data['graphs']
            else:
                graph_list = context_data
            for graph in graph_list:
                gid = id(graph)
                if gid in seen_ids:
                    continue
                seen_ids.add(gid)
                graphs.append(graph)

    if dataset_info and 'fug_mapping' in dataset_info:
        raise ValueError("Context indices required for FUG mapping but not found in context_structure")

    return graphs, None


def _extract_multitask_labels_and_masks(graphs):
    """
    Extract multi-task labels and masks from a list of graphs.

    Returns:
        labels (Tensor): [N, T] float
        masks (Tensor): [N, T] bool
    """
    labels = []
    masks = []
    num_tasks = None

    for graph in graphs:
        y = graph.y
        if y.dim() > 1:
            y = y.view(-1)
        else:
            y = y.view(-1)

        if num_tasks is None:
            num_tasks = y.numel()

        if hasattr(graph, 'task_mask') and graph.task_mask is not None:
            mask = graph.task_mask
            mask = mask.view(-1).bool() if mask.dim() > 0 else mask.bool()
        else:
            if y.dtype.is_floating_point:
                mask = ~torch.isnan(y)
            else:
                mask = (y != -1)

        if y.dtype.is_floating_point:
            y = torch.where(mask, y, torch.zeros_like(y))
        else:
            y = torch.where(mask, y, torch.zeros_like(y))

        labels.append(y.float())
        masks.append(mask.bool())

    if num_tasks is None:
        num_tasks = 0

    if labels:
        labels = torch.stack(labels, dim=0)
        masks = torch.stack(masks, dim=0)
    else:
        labels = torch.empty((0, num_tasks), dtype=torch.float32)
        masks = torch.empty((0, num_tasks), dtype=torch.bool)

    return labels, masks


def _build_multitask_support_mask(context_structure, graphs, graph_indices):
    """
    Build per-task support mask from context_structure.

    Returns:
        support_mask (Tensor): [N, T] bool, True if graph is in support set for task t.
    """
    if not context_structure or not graphs:
        return torch.empty((0, 0), dtype=torch.bool)

    task_indices = sorted(context_structure.keys())
    num_tasks = max(task_indices) + 1 if task_indices else 0
    support_mask = torch.zeros((len(graphs), num_tasks), dtype=torch.bool)

    id_to_row = {id(graph): i for i, graph in enumerate(graphs)}

    if graph_indices is not None:
        index_to_row = {int(idx): i for i, idx in enumerate(graph_indices)}
        for task_idx, task_context in context_structure.items():
            for context_data in task_context.values():
                if not context_data:
                    continue
                if isinstance(context_data, dict) and 'indices' in context_data:
                    for idx in context_data['indices']:
                        row = index_to_row.get(int(idx))
                        if row is not None:
                            support_mask[row, task_idx] = True
                else:
                    # Fallback: use object ids if indices missing
                    for graph in (context_data.get('graphs', []) if isinstance(context_data, dict) else context_data):
                        row = id_to_row.get(id(graph))
                        if row is not None:
                            support_mask[row, task_idx] = True
    else:
        for task_idx, task_context in context_structure.items():
            for context_data in task_context.values():
                if not context_data:
                    continue
                graph_list = context_data.get('graphs', []) if isinstance(context_data, dict) else context_data
                for graph in graph_list:
                    row = id_to_row.get(id(graph))
                    if row is not None:
                        support_mask[row, task_idx] = True

    return support_mask


def _create_multitask_context_embeddings(model, context_structure, dataset, pooling_method, device, identity_projection,
                                         dataset_info, batch_size=1024, return_timing=False, sync_cuda=False):
    """
    Compute context embeddings and [N, T] labels/masks for multi-task ridge regression.

    Returns:
        context_embeddings: [N, D]
        context_labels: [N, T]
        context_masks: [N, T] (valid label mask AND per-task support mask)
        timing (optional)
    """
    graphs, graph_indices = _collect_multitask_context_graphs(context_structure, dataset, dataset_info)
    if not graphs:
        empty = torch.empty((0, 0), device=device)
        labels = torch.empty((0, 0), device=device)
        masks = torch.empty((0, 0), dtype=torch.bool, device=device)
        if return_timing:
            return empty, labels, masks, {
                'encode_time': 0.0,
                'overhead_time': 0.0,
                'concat_time': 0.0,
                'total_time': 0.0,
                'num_context_batches': 0
            }
        return empty, labels, masks

    labels_cpu, masks_cpu = _extract_multitask_labels_and_masks(graphs)
    support_mask = _build_multitask_support_mask(context_structure, graphs, graph_indices)
    if support_mask.numel() > 0:
        masks_cpu = masks_cpu & support_mask

    timing = {
        'encode_time': 0.0,
        'overhead_time': 0.0,
        'concat_time': 0.0,
        'total_time': 0.0,
        'num_context_batches': 0
    }
    total_start = time.perf_counter()
    use_sync = sync_cuda and torch.cuda.is_available()

    context_embeddings = []
    node_emb_table = _get_node_embedding_table(dataset, 0, device, dataset_info)
    if node_emb_table is None:
        raise ValueError("Expected node embedding table under unified setting")

    for start in range(0, len(graphs), batch_size):
        end = min(start + batch_size, len(graphs))
        step_start = time.perf_counter()

        batch_graphs = graphs[start:end]
        batch_data = create_graph_batch(batch_graphs, device)
        if graph_indices is not None:
            batch_data.original_graph_indices = torch.tensor(graph_indices[start:end], dtype=torch.long)

        batch_data.x = _safe_lookup_node_embeddings(
            node_emb_table, batch_data.x,
            context="multitask_context", batch_data=batch_data, dataset_info=dataset_info
        )

        if dataset_info and dataset_info.get('needs_identity_projection', False) and identity_projection is not None:
            x_input = identity_projection(batch_data.x)
        else:
            x_input = batch_data.x

        batch_data.adj_t = SparseTensor.from_edge_index(
            batch_data.edge_index,
            sparse_sizes=(batch_data.num_nodes, batch_data.num_nodes)
        ).to_symmetric().coalesce()

        if use_sync:
            torch.cuda.synchronize()
        encode_start = time.perf_counter()
        node_embeddings = model(x_input, batch_data.adj_t, batch_data.batch)
        graph_embeddings = pool_graph_embeddings(node_embeddings, batch_data.batch, pooling_method)
        if use_sync:
            torch.cuda.synchronize()
        encode_end = time.perf_counter()

        timing['encode_time'] += encode_end - encode_start
        timing['overhead_time'] += (time.perf_counter() - step_start) - (encode_end - encode_start)
        timing['num_context_batches'] += 1
        context_embeddings.append(graph_embeddings)

    concat_start = time.perf_counter()
    context_embeddings = torch.cat(context_embeddings, dim=0) if context_embeddings else torch.empty((0, 0), device=device)
    timing['concat_time'] += time.perf_counter() - concat_start
    timing['total_time'] = time.perf_counter() - total_start

    context_labels = labels_cpu.to(device)
    context_masks = masks_cpu.to(device)

    if return_timing:
        return context_embeddings, context_labels, context_masks, timing
    return context_embeddings, context_labels, context_masks


def _compute_multitask_ridge_weights(context_embeddings, context_labels, context_masks, ridge_alpha=1.0):
    """
    Closed-form ridge regression for multi-task labels with per-task masks.

    Args:
        context_embeddings: [N, D]
        context_labels: [N, T]
        context_masks: [N, T] boolean
        ridge_alpha: > 0
    Returns:
        W: [D, T]
    """
    if ridge_alpha <= 0:
        raise ValueError("Ridge alpha must be positive for numerical stability")

    if context_embeddings.numel() == 0 or context_labels.numel() == 0:
        return torch.empty((0, 0), device=context_embeddings.device)

    num_tasks = context_labels.size(1)
    hidden_dim = context_embeddings.size(1)
    device = context_embeddings.device
    dtype = context_embeddings.dtype

    W = torch.zeros(hidden_dim, num_tasks, device=device, dtype=dtype)
    I = torch.eye(hidden_dim, device=device, dtype=dtype)
    failures = 0

    for t in range(num_tasks):
        mask_t = context_masks[:, t]
        if mask_t.sum() < 2:
            continue
        w = mask_t.float()
        XtX = context_embeddings.t() @ (context_embeddings * w.unsqueeze(1))
        XtY = context_embeddings.t() @ (context_labels[:, t] * w)
        try:
            W[:, t] = torch.linalg.solve(XtX + ridge_alpha * I, XtY)
        except torch.linalg.LinAlgError:
            failures += 1
            W[:, t] = torch.zeros(hidden_dim, device=device, dtype=dtype)

    if failures > 0:
        print(f"[GC-VEC] ridge: {failures}/{num_tasks} tasks failed solve; using zero weights.")

    return W


def _vectorized_multitask_logits(target_embeddings, pos_proto, neg_proto, sim='dot'):
    """
    Compute per-task logits in a vectorized way.

    Args:
        target_embeddings: [B, D]
        pos_proto: [T, D] or None
        neg_proto: [T, D] or None
        sim: 'dot' or 'cos'

    Returns:
        logits: [B, T]
    """
    if pos_proto is None or neg_proto is None:
        return None

    if sim == 'cos':
        target_norm = F.normalize(target_embeddings, p=2, dim=-1)
        pos_norm = F.normalize(pos_proto, p=2, dim=-1)
        neg_norm = F.normalize(neg_proto, p=2, dim=-1)
        pos_scores = target_norm @ pos_norm.t()
        neg_scores = target_norm @ neg_norm.t()
    else:
        pos_scores = target_embeddings @ pos_proto.t()
        neg_scores = target_embeddings @ neg_proto.t()

    return pos_scores - neg_scores


def train_graph_classification_multitask_vectorized(model, predictor, dataset_info, data_loaders, optimizer,
                                                   pooling_method='mean', device='cuda', clip_grad=1.0,
                                                   orthogonal_push=0.0, normalize_class_h=True,
                                                   identity_projection=None, args=None, lambda_=1.0):
    """
    Vectorized multi-task training: single forward, build per-task pos/neg prototypes (or ridge weights),
    compute logits [B, T], and apply BCEWithLogits with task_mask.
    """
    model.train()
    predictor.train()
    if identity_projection is not None:
        identity_projection.train()

    dataset = dataset_info['dataset']
    sample_graph = dataset[0]
    is_multitask = sample_graph.y.numel() > 1
    if not is_multitask:
        raise ValueError("Vectorized GC path expects a multi-task dataset (e.g., PCBA)")

    num_tasks = sample_graph.y.numel()

    start_time = time.perf_counter()
    total_loss = 0.0
    batch_count = 0
    batch_time_total = 0.0
    proto_time_total = 0.0
    proto_encode_time_total = 0.0
    proto_overhead_time_total = 0.0
    proto_concat_time_total = 0.0
    proto_context_batches_total = 0
    profile_context = getattr(args, 'gc_profile_context', False) if args is not None else False
    sim_type = getattr(args, 'gc_sim', 'dot') if args is not None else 'dot'
    use_ridge = sim_type == 'ridge'
    ridge_alpha = getattr(args, 'gc_ridge_alpha', 1.0) if args is not None else 1.0
    context_batch_size = getattr(args, 'gc_batch_size', 1024) if args is not None else 1024

    for batch in data_loaders['train']:
        if use_ridge:
            if profile_context:
                context_embeddings, context_labels, context_masks, timing = _create_multitask_context_embeddings(
                    model, dataset_info['context_graphs'], dataset, pooling_method, device, identity_projection, dataset_info,
                    batch_size=context_batch_size, return_timing=True, sync_cuda=True
                )
                proto_time_total += timing['total_time']
                proto_encode_time_total += timing['encode_time']
                proto_overhead_time_total += timing['overhead_time']
                proto_concat_time_total += timing['concat_time']
                proto_context_batches_total += timing['num_context_batches']
            else:
                proto_start = time.perf_counter()
                context_embeddings, context_labels, context_masks = _create_multitask_context_embeddings(
                    model, dataset_info['context_graphs'], dataset, pooling_method, device, identity_projection, dataset_info,
                    batch_size=context_batch_size
                )
                proto_time_total += time.perf_counter() - proto_start

            ridge_start = time.perf_counter()
            W = _compute_multitask_ridge_weights(context_embeddings, context_labels, context_masks, ridge_alpha=ridge_alpha)
            proto_time_total += time.perf_counter() - ridge_start
        else:
            if profile_context:
                # Recompute prototypes EVERY batch (expensive, but requested)
                pos_embeds_by_task, neg_embeds_by_task, timing = _create_all_task_context_embeddings(
                    model, dataset_info['context_graphs'], dataset, pooling_method, device, identity_projection, dataset_info,
                    return_timing=True, sync_cuda=True
                )
                proto_time_total += timing['total_time']
                proto_encode_time_total += timing['encode_time']
                proto_overhead_time_total += timing['overhead_time']
                proto_concat_time_total += timing['concat_time']
                proto_context_batches_total += timing['num_context_batches']
                proto_start = time.perf_counter()
            else:
                proto_start = time.perf_counter()
                # Recompute prototypes EVERY batch (expensive, but requested)
                pos_embeds_by_task, neg_embeds_by_task = _create_all_task_context_embeddings(
                    model, dataset_info['context_graphs'], dataset, pooling_method, device, identity_projection, dataset_info
                )

            # Compute prototypes (mean) for each task, fallback to zeros if missing
            proto_dim = None
            pos_proto = []
            neg_proto = []
            for t in range(num_tasks):
                pos_e = pos_embeds_by_task[t]
                neg_e = neg_embeds_by_task[t]
                if pos_e is None or neg_e is None:
                    if proto_dim is None:
                        for e in pos_embeds_by_task + neg_embeds_by_task:
                            if e is not None:
                                proto_dim = e.size(-1)
                                break
                    if proto_dim is None:
                        proto_dim = 256
                    pos_proto.append(torch.zeros(proto_dim, device=device))
                    neg_proto.append(torch.zeros(proto_dim, device=device))
                    continue
                proto_dim = pos_e.size(-1)
                pos_proto.append(pos_e.mean(dim=0))
                neg_proto.append(neg_e.mean(dim=0))

            pos_proto = torch.stack(pos_proto, dim=0)
            neg_proto = torch.stack(neg_proto, dim=0)
            proto_time_total += time.perf_counter() - proto_start

        batch_start = time.perf_counter()
        optimizer.zero_grad()
        batch_data = batch.to(device)
        if batch_data.num_graphs == 0:
            continue

        node_emb_table = _get_node_embedding_table(dataset_info['dataset'], 0, device, dataset_info)
        if node_emb_table is not None:
            batch_data.x = _safe_lookup_node_embeddings(
                node_emb_table, batch_data.x,
                context="train_vec_multitask", batch_data=batch_data, dataset_info=dataset_info
            )
        else:
            raise ValueError("Expected node embedding table under unified setting")

        if dataset_info.get('needs_identity_projection', False) and identity_projection is not None:
            x_input = identity_projection(batch_data.x)
        else:
            x_input = batch_data.x

        x_input = apply_feature_dropout_if_enabled(x_input, args, rank=0, training=model.training)

        if args is not None and hasattr(args, 'edge_dropout_enabled') and args.edge_dropout_enabled and hasattr(args, 'edge_dropout_rate'):
            batch_data = batch_edge_dropout(batch_data, args.edge_dropout_rate, training=model.training)

        batch_data.adj_t = SparseTensor.from_edge_index(
            batch_data.edge_index,
            sparse_sizes=(batch_data.num_nodes, batch_data.num_nodes)
        ).to_symmetric().coalesce()

        node_embeddings = model(x_input, batch_data.adj_t, batch_data.batch)
        target_embeddings = pool_graph_embeddings(node_embeddings, batch_data.batch, pooling_method)

        batch_labels = batch_data.y
        if batch_labels.dim() == 1 and batch_labels.numel() == batch_data.num_graphs * num_tasks:
            batch_labels = batch_labels.view(batch_data.num_graphs, num_tasks)

        # task_mask from graph if present, otherwise derive from labels
        if hasattr(batch_data, 'task_mask') and batch_data.task_mask is not None:
            task_mask = batch_data.task_mask
            if task_mask.dim() == 1 and task_mask.numel() == batch_data.num_graphs * num_tasks:
                task_mask = task_mask.view(batch_data.num_graphs, num_tasks)
        else:
            if batch_labels.dtype.is_floating_point:
                task_mask = ~torch.isnan(batch_labels)
            else:
                task_mask = batch_labels != -1

        # Replace invalid labels with 0 to keep BCE stable
        labels = batch_labels.float()
        labels = torch.where(task_mask, labels, torch.zeros_like(labels))

        # Vectorized logits [B, T]
        if use_ridge:
            logits = target_embeddings @ W
        else:
            logits = _vectorized_multitask_logits(target_embeddings, pos_proto, neg_proto, sim=sim_type)
        if logits is None:
            continue

        loss_matrix = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        valid = task_mask.float()
        loss = (loss_matrix * valid).sum() / valid.sum().clamp_min(1.0)

        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
            if identity_projection:
                torch.nn.utils.clip_grad_norm_(identity_projection.parameters(), clip_grad)

        optimizer.step()
        total_loss += loss.item()
        batch_count += 1
        batch_time_total += time.perf_counter() - batch_start

    total_time = time.perf_counter() - start_time
    avg_batch_time = batch_time_total / max(batch_count, 1)
    avg_proto_time = proto_time_total / max(batch_count, 1)
    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    method_name = "ridge" if use_ridge else "proto"
    print(f"[GC-VEC] train ({method_name}): avg_loss={avg_loss:.4f} ctx_avg={avg_proto_time:.2f}s batches={batch_count} avg_batch={avg_batch_time:.3f}s total={total_time:.2f}s")
    if profile_context:
        avg_encode = proto_encode_time_total / max(batch_count, 1)
        avg_overhead = (proto_overhead_time_total + proto_concat_time_total) / max(batch_count, 1)
        overhead_pct = (avg_overhead / max(avg_proto_time, 1e-9)) * 100.0
        avg_context_batches = proto_context_batches_total / max(batch_count, 1)
        print(f"[GC-VEC] ctx_breakdown: encode_avg={avg_encode:.2f}s overhead_avg={avg_overhead:.2f}s overhead_pct={overhead_pct:.1f}% context_batches={avg_context_batches:.1f}")

    return avg_loss


@torch.no_grad()
def evaluate_graph_classification_multitask_vectorized(model, predictor, dataset_info, data_loaders,
                                                       pooling_method='mean', device='cuda',
                                                       normalize_class_h=True, dataset_name=None,
                                                       identity_projection=None, args=None):
    """
    Vectorized multi-task evaluation: compute logits [B, T] and AUC/AP with task_mask.
    """
    model.eval()
    predictor.eval()
    if identity_projection is not None:
        identity_projection.eval()

    dataset = dataset_info['dataset']
    sample_graph = dataset[0]
    is_multitask = sample_graph.y.numel() > 1
    if not is_multitask:
        raise ValueError("Vectorized GC eval expects a multi-task dataset (e.g., PCBA)")

    num_tasks = sample_graph.y.numel()
    metric_type = get_dataset_metric(dataset_name, num_classes=dataset_info.get('num_classes', None), is_multitask=True)
    sim_type = getattr(args, 'gc_sim', 'dot') if args is not None else 'dot'
    use_ridge = sim_type == 'ridge'
    ridge_alpha = getattr(args, 'gc_ridge_alpha', 1.0) if args is not None else 1.0
    context_batch_size = getattr(args, 'gc_test_batch_size', 4096) if args is not None else 4096
    max_eval_batches = getattr(args, 'gc_train_eval_max_batches', 0) if args is not None else 0

    eval_start = time.perf_counter()
    proto_start = time.perf_counter()
    if use_ridge:
        context_embeddings, context_labels, context_masks = _create_multitask_context_embeddings(
            model, dataset_info['context_graphs'], dataset, pooling_method, device, identity_projection, dataset_info,
            batch_size=context_batch_size
        )
        W = _compute_multitask_ridge_weights(context_embeddings, context_labels, context_masks, ridge_alpha=ridge_alpha)
    else:
        pos_embeds_by_task, neg_embeds_by_task = _create_all_task_context_embeddings(
            model, dataset_info['context_graphs'], dataset, pooling_method, device, identity_projection, dataset_info
        )

        proto_dim = None
        pos_proto = []
        neg_proto = []
        for t in range(num_tasks):
            pos_e = pos_embeds_by_task[t]
            neg_e = neg_embeds_by_task[t]
            if pos_e is None or neg_e is None:
                if proto_dim is None:
                    for e in pos_embeds_by_task + neg_embeds_by_task:
                        if e is not None:
                            proto_dim = e.size(-1)
                            break
                if proto_dim is None:
                    proto_dim = 256
                pos_proto.append(torch.zeros(proto_dim, device=device))
                neg_proto.append(torch.zeros(proto_dim, device=device))
                continue
            proto_dim = pos_e.size(-1)
            pos_proto.append(pos_e.mean(dim=0))
            neg_proto.append(neg_e.mean(dim=0))

        pos_proto = torch.stack(pos_proto, dim=0)
        neg_proto = torch.stack(neg_proto, dim=0)
    proto_time = time.perf_counter() - proto_start

    split_results = {}
    for split_name in ['train', 'val', 'test']:
        if split_name not in data_loaders:
            continue

        split_start = time.perf_counter()
        split_batch_time = 0.0
        all_logits = []
        all_labels = []
        all_masks = []

        for batch_idx, batch in enumerate(data_loaders[split_name]):
            if split_name == 'train' and max_eval_batches > 0 and batch_idx >= max_eval_batches:
                break
            batch_start = time.perf_counter()
            batch_data = batch.to(device)
            if batch_data.num_graphs == 0:
                continue

            node_emb_table = _get_node_embedding_table(dataset_info['dataset'], 0, device, dataset_info)
            if node_emb_table is not None:
                batch_data.x = _safe_lookup_node_embeddings(
                    node_emb_table, batch_data.x,
                    context=f"eval_vec_{split_name}", batch_data=batch_data, dataset_info=dataset_info
                )
            else:
                raise ValueError("Expected node embedding table under unified setting")

            if dataset_info.get('needs_identity_projection', False) and identity_projection is not None:
                x_input = identity_projection(batch_data.x)
            else:
                x_input = batch_data.x

            batch_data.adj_t = SparseTensor.from_edge_index(
                batch_data.edge_index,
                sparse_sizes=(batch_data.num_nodes, batch_data.num_nodes)
            ).to_symmetric().coalesce()

            node_embeddings = model(x_input, batch_data.adj_t, batch_data.batch)
            target_embeddings = pool_graph_embeddings(node_embeddings, batch_data.batch, pooling_method)

            batch_labels = batch_data.y
            if batch_labels.dim() == 1 and batch_labels.numel() == batch_data.num_graphs * num_tasks:
                batch_labels = batch_labels.view(batch_data.num_graphs, num_tasks)

            if hasattr(batch_data, 'task_mask') and batch_data.task_mask is not None:
                task_mask = batch_data.task_mask
                if task_mask.dim() == 1 and task_mask.numel() == batch_data.num_graphs * num_tasks:
                    task_mask = task_mask.view(batch_data.num_graphs, num_tasks)
            else:
                if batch_labels.dtype.is_floating_point:
                    task_mask = ~torch.isnan(batch_labels)
                else:
                    task_mask = batch_labels != -1

            labels = batch_labels.float()
            labels = torch.where(task_mask, labels, torch.zeros_like(labels))

            if use_ridge:
                logits = target_embeddings @ W
            else:
                logits = _vectorized_multitask_logits(target_embeddings, pos_proto, neg_proto, sim=sim_type)
            if logits is None:
                continue

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_masks.append(task_mask.detach().cpu())
            split_batch_time += time.perf_counter() - batch_start

        if not all_logits:
            if isinstance(metric_type, list):
                split_results[split_name] = {metric_name: 0.0 for metric_name in metric_type}
            else:
                split_results[split_name] = 0.0
            continue

        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        all_masks = torch.cat(all_masks, dim=0)

        # Compute per-task metrics using mask
        task_metrics = []
        for t in range(num_tasks):
            mask_t = all_masks[:, t].bool()
            if mask_t.sum() == 0:
                continue
            labels_t = all_labels[mask_t, t]
            probs_t = torch.sigmoid(all_logits[mask_t, t]).view(-1)
            preds_t = (probs_t >= 0.5).long()
            if isinstance(metric_type, list):
                metrics_t = calculate_multiple_metrics(preds_t, labels_t.long(), probs_t, metric_type)
                task_metrics.append(metrics_t)
            else:
                metric_t = calculate_metric(preds_t, labels_t.long(), probs_t, metric_type)
                task_metrics.append(metric_t)

        split_results[split_name] = aggregate_task_metrics(task_metrics) if task_metrics else 0.0
        split_time = time.perf_counter() - split_start
        avg_batch_time = split_batch_time / max(len(data_loaders[split_name]), 1)
        method_name = "ridge" if use_ridge else "proto"
        print(f"[GC-VEC] eval {split_name} ({method_name}): ctx={proto_time:.2f}s split_time={split_time:.2f}s avg_batch={avg_batch_time:.3f}s")

    total_time = time.perf_counter() - eval_start
    print(f"[GC-VEC] eval total: {total_time:.2f}s")
    return split_results

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
    from .data_gc import create_data_loaders
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
                                           normalize_class_h=True, dataset_name=None, identity_projection=None, args=None):
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
    num_classes = dataset_info.get('num_classes', None)
    metric_type = get_dataset_metric(dataset_name, num_classes=num_classes, is_multitask=is_multitask) if dataset_name else 'accuracy'
    
    split_results = {}
    
    # Evaluate each split (train, val, test)
    for split_name in ['train', 'val', 'test']:
        if split_name not in data_loaders:
            continue
        all_task_metrics = []
        
        # For each task, collect predictions across all graphs in this split
        max_eval_batches = getattr(args, 'gc_train_eval_max_batches', 0) if args is not None else 0
        for task_idx in range(num_tasks):
            all_predictions = []
            all_labels = []
            all_probabilities = []
            
            for batch_idx, batch in enumerate(data_loaders[split_name]):
                if split_name == 'train' and max_eval_batches and batch_idx >= max_eval_batches:
                    break
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
                node_embeddings = model(x_input, batch_data.adj_t, batch_data.batch)

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

                # Use PFN predictor with graph classification head
                pred_output = predictor(pfn_data, context_embeddings, valid_embeddings, context_labels, class_h, task_type='graph_classification')
                if len(pred_output) == 3:  # MoE case with auxiliary loss
                    scores, _, _ = pred_output  # Discard auxiliary loss during evaluation
                else:  # Standard case
                    scores, _ = pred_output
                
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
            split_results[split_name] = aggregate_task_metrics(all_task_metrics)
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
    x_input = apply_feature_dropout_if_enabled(x_input, args, rank=0, training=model.training)

    # Apply edge dropout if enabled (before creating adj_t)
    if args is not None and hasattr(args, 'edge_dropout_enabled') and args.edge_dropout_enabled and hasattr(args, 'edge_dropout_rate'):
        batch_data = batch_edge_dropout(batch_data, args.edge_dropout_rate, training=model.training)

    batch_data.adj_t = SparseTensor.from_edge_index(
        batch_data.edge_index,
        sparse_sizes=(batch_data.num_nodes, batch_data.num_nodes)
    ).to_symmetric().coalesce()

    # Get node embeddings from GNN
    node_embeddings = model(x_input, batch_data.adj_t, batch_data.batch)

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
    
    # Use PFN predictor with graph classification head
    scores, _ = predictor(pfn_data, context_embeddings, valid_embeddings, context_labels, class_h, task_type='graph_classification')
    scores = F.log_softmax(scores, dim=1)

    # Compute loss
    total_loss = F.nll_loss(scores, valid_labels)

    return total_loss


def train_graph_classification_single_task(model, predictor, dataset_info, data_loaders, optimizer, task_idx,
                                         pooling_method='mean', device='cuda', clip_grad=1.0,
                                         orthogonal_push=0.0, normalize_class_h=True, identity_projection=None, context_k=None, args=None, lambda_=1.0):
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

    # Create monitor for detailed logging (disabled by default)
    # Set enable_training_monitor=True in args to enable
    enable_monitoring = getattr(args, 'enable_training_monitor', False) if args else False
    log_interval = getattr(args, 'monitor_log_interval', 1) if args else 1  # Log every batch by default
    monitor = TrainingMonitor(log_interval=log_interval, detailed=False) if enable_monitoring else None

    if monitor is not None:
        print(f"[MONITOR] Training monitoring ENABLED (log_interval={log_interval})")

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
        x_input = apply_feature_dropout_if_enabled(x_input, args, rank=0, training=model.training)
        
        # Apply edge dropout if enabled (before creating adj_t)
        if args is not None and hasattr(args, 'edge_dropout_enabled') and args.edge_dropout_enabled and hasattr(args, 'edge_dropout_rate'):
            batch_data = batch_edge_dropout(batch_data, args.edge_dropout_rate, training=model.training)

        batch_data.adj_t = SparseTensor.from_edge_index(
            batch_data.edge_index,
            sparse_sizes=(batch_data.num_nodes, batch_data.num_nodes)
        ).to_symmetric().coalesce()

        # Get node embeddings from GNN using SparseTensor format
        node_embeddings = model(x_input, batch_data.adj_t, batch_data.batch)

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

        # Use PFN predictor with graph classification head (all samples are valid, no masking needed)
        scores_raw, refined_class_h = predictor(pfn_data, context_embeddings, target_embeddings, context_labels, class_h, task_type='graph_classification')

        # Apply log_softmax for loss computation
        scores_log = F.log_softmax(scores_raw, dim=1)

        # Compute loss - no masking needed since all samples are valid
        nll_loss = F.nll_loss(scores_log, batch_labels)

        # Orthogonal loss for refined class prototypes
        if orthogonal_push > 0:
            refined_class_h_norm = F.normalize(refined_class_h, p=2, dim=1)
            class_matrix = refined_class_h_norm @ refined_class_h_norm.T
            mask = ~torch.eye(class_matrix.size(0), device=class_matrix.device, dtype=torch.bool)
            orthogonal_loss = torch.sum(class_matrix[mask]**2)
        else:
            orthogonal_loss = torch.tensor(0.0, device=device)

        loss = nll_loss + orthogonal_push * orthogonal_loss
        loss = loss * lambda_  # Apply lambda scaling (same as NC and LP)

        # ===== MONITORING: Collect statistics before backward =====
        if monitor is not None:
            stats = {}

            # Check predictor outputs (use RAW scores before log_softmax for proper statistics)
            stats.update(monitor.check_predictor_outputs(scores_raw, batch_labels))

            # Check batch statistics
            stats.update(monitor.check_batch_statistics(batch_data, labels=batch_labels))

            # Check context embedding statistics
            stats.update(monitor.check_embeddings(context_embeddings, "context_emb"))
            stats.update(monitor.check_embeddings(target_embeddings, "target_emb"))

            # Check loss components
            stats.update(monitor.check_loss_components(nll_loss, auxiliary_loss, orthogonal_loss, loss))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # ===== MONITORING: Check gradients before clipping =====
        if monitor is not None:
            grad_stats = monitor.check_gradients(model, predictor, identity_projection, prefix="before_clip")
            stats.update(grad_stats)

        # Gradient clipping
        total_grad_norm_before = 0.0
        if clip_grad > 0:
            # Compute total norm before clipping for monitoring
            if monitor is not None:
                for p in list(model.parameters()) + list(predictor.parameters()) + (list(identity_projection.parameters()) if identity_projection else []):
                    if p.grad is not None:
                        total_grad_norm_before += p.grad.data.norm(2).item() ** 2
                total_grad_norm_before = total_grad_norm_before ** 0.5
                stats['grad_norm_before_clip'] = total_grad_norm_before

            # Apply clipping (current per-module approach)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
            if identity_projection: torch.nn.utils.clip_grad_norm_(identity_projection.parameters(), clip_grad)

        # ===== MONITORING: Check gradients after clipping =====
        if monitor is not None:
            grad_stats_after = monitor.check_gradients(model, predictor, identity_projection, prefix="after_clip")
            for key, val in grad_stats_after.items():
                stats[key.replace("before_clip", "after_clip")] = val

            # Log all statistics for this batch
            monitor.log_batch_stats(stats, task_idx=task_idx, batch_idx=batch_idx)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Simpler progress message when monitoring is disabled
        if monitor is None:
            print(f"\rTraining batch {batch_idx+1}/{total_batches} completed (loss: {loss.item():.4f})", end="", flush=True)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f"\nTraining completed: {num_batches} batches, avg loss: {avg_loss:.4f}")

    # Print epoch summary if monitoring is enabled
    if monitor is not None:
        monitor.print_epoch_summary(epoch=0)  # epoch passed from outer loop would be better

    return avg_loss

@torch.no_grad()
def evaluate_graph_classification_single_task(model, predictor, dataset_info, data_loaders, task_idx,
                                             pooling_method='mean', device='cuda',
                                             normalize_class_h=True, dataset_name=None, identity_projection=None, context_k=None, args=None):
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
        args: Training arguments (for meta-graph C&S config)

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
    num_classes = dataset_info.get('num_classes', None)
    metric_type = get_dataset_metric(dataset_name, num_classes=num_classes, is_multitask=is_multitask) if dataset_name else 'accuracy'

    # Check if meta-graph C&S is enabled
    use_graph_cs = getattr(args, 'use_graph_cs', False) if args is not None else False

    # Storage for meta-graph C&S (only if enabled)
    all_test_embeddings = []
    all_test_logits = []
    all_test_labels = []
    split_sizes = {}  # Track size of each split for reconstruction

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
        split_embeddings = []
        split_logits = []

        # Time batch processing (direct iteration, no pre-extraction)
        batch_processing_start = time.perf_counter()
        total_samples = len(data_loader.dataset)

        max_eval_batches = getattr(args, 'gc_train_eval_max_batches', 0) if args is not None else 0
        for batch_idx, batch_graphs in enumerate(data_loader):
            if split_name == 'train' and max_eval_batches > 0 and batch_idx >= max_eval_batches:
                break
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
            node_embeddings = model(x_input, batch_data.adj_t, batch_data.batch)

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

            pred_output = predictor(pfn_data, context_embeddings, target_embeddings, context_labels, class_h, task_type='graph_classification')
            if len(pred_output) == 3:  # MoE case with auxiliary loss
                scores, _, _ = pred_output  # Discard auxiliary loss during evaluation
            else:  # Standard case
                scores, _ = pred_output
            probabilities = F.softmax(scores, dim=1)
            predictions = scores.argmax(dim=1)

            all_predictions.append(predictions.cpu())
            all_labels.append(batch_labels.cpu())
            all_probabilities.append(probabilities.cpu())

            # Store embeddings and logits for C&S if enabled
            if use_graph_cs:
                split_embeddings.append(target_embeddings.cpu())
                split_logits.append(scores.cpu())

            print(f"\rEvaluating batch {batch_idx+1} completed ({len(predictions)} samples)", end="", flush=True)

        # Concatenate all predictions and labels for this split
        if all_predictions:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
            all_probabilities = torch.cat(all_probabilities, dim=0)

            if use_graph_cs:
                split_emb = torch.cat(split_embeddings, dim=0)
                split_log = torch.cat(split_logits, dim=0)
                all_test_embeddings.append(split_emb)
                all_test_logits.append(split_log)
                all_test_labels.append(all_labels)
                split_sizes[split_name] = len(all_labels)

            print(f"\n{split_name} evaluation: {len(all_predictions)} total samples processed", end="")

            # Calculate the appropriate metric(s) - base predictions
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

    # Apply meta-graph C&S if enabled
    if use_graph_cs and all_test_embeddings:
        print("\n[Meta-Graph C&S] Building meta-graph with context + test graphs...")
        cs_start = time.perf_counter()

        # Combine context embeddings + all test embeddings
        all_embeddings = torch.cat([context_embeddings.cpu()] + all_test_embeddings, dim=0).to(device)
        all_logits = torch.cat([torch.zeros(len(context_labels), dataset_info['num_classes']).cpu()] + all_test_logits, dim=0).to(device)
        all_labels_combined = torch.cat([context_labels.cpu()] + all_test_labels, dim=0)

        num_context = len(context_labels)

        # Use context graphs as anchors (or sample if too many)
        num_anchors = min(getattr(args, 'num_anchors', 1000), num_context)
        if num_anchors < num_context:
            anchor_indices = np.random.choice(num_context, size=num_anchors, replace=False).tolist()
        else:
            anchor_indices = list(range(num_context))

        # Build meta-graph
        k_neighbors = getattr(args, 'cs_k_neighbors', 10)
        weight_sharpening = getattr(args, 'weight_sharpening', 1.0)
        meta_graph_sim = getattr(args, 'meta_graph_sim', 'cos')

        adj = build_anchor_meta_graph(
            all_embeddings,
            anchor_indices,
            k_neighbors=k_neighbors,
            sim=meta_graph_sim,
            weight_sharpening=weight_sharpening
        )

        # For C&S, we need proper logits for context graphs
        # Re-compute context predictions
        context_pred_output = predictor(pfn_data, context_embeddings, context_embeddings, context_labels, class_h, task_type='graph_classification')
        if len(context_pred_output) == 3:
            context_scores, _, _ = context_pred_output
        else:
            context_scores, _ = context_pred_output

        # Update logits with context predictions
        all_logits[:num_context] = context_scores

        # Apply C&S
        context_idx = torch.arange(num_context, device=device)
        num_classes = dataset_info['num_classes']
        cs_num_iters = getattr(args, 'cs_num_iters', 50)
        cs_alpha = getattr(args, 'cs_alpha', 0.5)

        smoothed_logits = correct_and_smooth_graph(
            adj, all_logits, context_idx, context_labels.to(device), num_classes,
            num_iters=cs_num_iters, alpha=cs_alpha
        )

        # Extract smoothed predictions for each test split
        offset = num_context
        for split_name in split_sizes.keys():
            split_size = split_sizes[split_name]
            split_smoothed = smoothed_logits[offset:offset + split_size].cpu()
            split_labels = all_labels_combined[offset:offset + split_size]

            smoothed_predictions = split_smoothed.argmax(dim=1)

            # Recalculate metrics with smoothed predictions
            if isinstance(metric_type, list):
                metric_values = calculate_multiple_metrics(smoothed_predictions, split_labels, split_smoothed, metric_type)
                results[split_name] = metric_values
            else:
                metric_value = calculate_metric(smoothed_predictions, split_labels, split_smoothed, metric_type)
                results[split_name] = metric_value

            offset += split_size

        cs_time = time.perf_counter() - cs_start
        print(f"[Meta-Graph C&S] Completed in {cs_time:.3f}s")

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
    from .data_gc import create_data_loaders
    
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
        
        for idx, train_dataset_info in enumerate(train_processed_data_list):
            # Use original splits WITHOUT task filtering
            splits = train_dataset_info['split_idx']
            all_task_filtered_splits.append(splits)
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
                        context_k=getattr(args, 'context_k', None), args=args, lambda_=1.0
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
                    identity_projection=identity_projection, args=args
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
                metric_type = get_dataset_metric(dataset_name, num_classes=train_dataset_info.get("num_classes", None), is_multitask=is_multitask)
                
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
                        context_k=getattr(args, 'context_k', None), args=args
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
                metric_type = get_dataset_metric(dataset_name, num_classes=train_dataset_info.get("num_classes", None), is_multitask=is_multitask)
                
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
                            identity_projection=identity_projection, args=args
                        )
                        
                        test_results = {'test': test_eval_results.get('test', 0.0)}
                        best_results[f"test_{test_name}"] = test_results.copy()
                        
                        test_sample_graph = test_dataset_info['dataset'][0]
                        test_is_multitask = test_sample_graph.y.numel() > 1
                        test_metric_type = get_dataset_metric(test_name, num_classes=test_dataset_info.get("num_classes", None), is_multitask=test_is_multitask)
                        
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
                                        context_k=getattr(args, 'context_k', None), args=args
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
                                    context_k=getattr(args, 'context_k', None), args=args
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
                        test_metric_type = get_dataset_metric(test_name, num_classes=test_dataset_info.get("num_classes", None), is_multitask=test_is_multitask)
                        
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
