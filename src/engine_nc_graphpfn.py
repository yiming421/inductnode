"""
Training and evaluation engine for GraphPFN (foundational model with ICL).

This is a simplified engine specifically for GraphPFN, which has a different
interface than the standard PFN predictor:
- GraphPFN: foundational model, fixed decoder, in-context learning
- Standard PFN: task-specific heads, prototype-based prediction

Key differences:
1. No PCA padding - GraphPFN handles raw GNN output dimension
2. Different forward signature: (gnn_embeddings, labels, context_mask, num_classes)
3. No class prototypes - direct logit prediction
4. Fixed n_out=10, sliced at inference (or decomposed for >10 classes)
5. Output space decomposition for >10 classes using ECOC strategy
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from .utils import acc
from .data_utils import select_k_shot_context
from .graphpfn.decomposition import create_decomposer


def graphpfn_forward_with_decomposition_training(
    graphpfn_predictor,
    gnn_embeddings,
    labels,
    context_mask,
    num_classes,
    decomposer,
    optimizer,
    query_indices_batch,  # Only the query indices from current training batch (GRAPH indices)
    batch_nodes,  # Mapping from subset to graph indices
    args=None,  # For gradient clipping
):
    """Training-optimized forward pass with per-subproblem gradient descent.

    For training, we perform gradient descent after each subproblem to avoid
    accumulating large tensors that cause OOM.

    Note: We do NOT compute accuracy during decomposition training because each
    sub-problem uses a different local label space (e.g., all 0-9). Aggregating
    accuracies across sub-problems would be meaningless. Use validation metrics
    (wandb: valid_seen/nc_metric) to track actual performance with proper global
    label space evaluation.

    Args:
        graphpfn_predictor: GraphPFN predictor module
        gnn_embeddings: [N_subset, hidden] GNN output features for subset
        labels: [N_subset] node labels for subset
        context_mask: [N_subset] boolean mask for support/context nodes in subset
        num_classes: Total number of classes
        decomposer: OutputSpaceDecomposer for >10 classes
        optimizer: Optimizer for gradient descent
        query_indices_batch: [B] query indices from current training batch (GRAPH node IDs)
        batch_nodes: [N_subset] mapping from subset index to graph node ID
        args: Training arguments (for gradient clipping)

    Returns:
        avg_loss: Average loss across all samples in batch
    """
    if decomposer is None:
        raise ValueError("This function is only for decomposed training (>10 classes)")

    device = gnn_embeddings.device

    # Create query mask for the subset (needed later)
    query_mask = ~context_mask

    # Map query_indices_batch (GRAPH indices) to subset indices
    # batch_nodes[i] = graph_node_id, we need to find which i corresponds to query_indices_batch
    batch_query_mask_in_batch_nodes = torch.isin(batch_nodes, query_indices_batch)
    batch_query_subset_indices = torch.where(batch_query_mask_in_batch_nodes)[0]

    # Get labels for these batch query nodes (using subset indices)
    batch_query_labels = labels[batch_query_subset_indices]

    total_loss = 0.0
    total_samples = 0

    # Process each sub-problem with immediate gradient descent
    for sub_idx in range(decomposer.config.num_subproblems):
        # Get which original classes belong to this sub-problem
        original_classes_in_sub = decomposer.get_original_classes_for_subproblem(sub_idx)
        original_classes_tensor = torch.tensor(original_classes_in_sub, device=device)

        # Filter batch query nodes: only those with labels in this sub-problem
        batch_query_in_sub_mask = torch.isin(batch_query_labels.long(), original_classes_tensor)
        if not batch_query_in_sub_mask.any():
            continue  # No batch query nodes for this sub-problem
        
        # Zero gradients for this sub-problem
        optimizer.zero_grad()
        
        # Create modified labels: context + only relevant query nodes
        modified_labels = labels.clone().float()
        modified_context_mask = context_mask.clone()
        
        # Set irrelevant query nodes to NaN (so they won't be selected)
        irrelevant_query_mask = query_mask & (~torch.isin(labels.long(), original_classes_tensor))
        modified_labels[irrelevant_query_mask] = float('nan')
        
        # Prepare sub-problem data (training mode - only queries with labels in this partition)
        sub_labels, sub_context_mask, node_selection_mask = decomposer.prepare_subproblem_data(
            labels=modified_labels,
            context_mask=modified_context_mask,
            sub_idx=sub_idx,
            training=True,  # Training mode: filter query nodes by label
        )
        
        # Select embeddings
        selected_embeddings = gnn_embeddings[node_selection_mask]
        sub_num_classes = decomposer.get_subproblem_num_classes(sub_idx)
        
        # Forward through GraphPFN for this sub-problem
        sub_logits = graphpfn_predictor(
            gnn_embeddings=selected_embeddings,
            labels=sub_labels,
            context_mask=sub_context_mask,
            num_classes=sub_num_classes,
        )  # [N_query_sub, sub_num_classes]
        
        # Find which nodes in this sub-problem are from our training batch
        # Note: selected_indices are SUBSET indices, we need to map to graph IDs
        selected_subset_indices = node_selection_mask.nonzero(as_tuple=True)[0]
        selected_graph_indices = batch_nodes[selected_subset_indices]  # Map to graph IDs

        selected_query_mask = ~sub_context_mask
        selected_query_graph_indices = selected_graph_indices[selected_query_mask]

        # Get batch query nodes that are in this subproblem (compare graph IDs)
        batch_in_selected_mask = torch.isin(selected_query_graph_indices, query_indices_batch)
        if not batch_in_selected_mask.any():
            continue
            
        batch_logits_sub = sub_logits[batch_in_selected_mask]

        # Get the actual query nodes (GRAPH IDs) that produced these logits
        actual_batch_nodes_in_sub_graph_ids = selected_query_graph_indices[batch_in_selected_mask]

        # Get labels for these specific nodes
        # Map graph IDs back to subset indices to get labels from subset
        batch_nodes_subset_mask = torch.isin(batch_nodes, actual_batch_nodes_in_sub_graph_ids)
        batch_nodes_subset_indices = torch.where(batch_nodes_subset_mask)[0]
        batch_labels_sub = labels[batch_nodes_subset_indices].long()
        
        # Map global labels to sub-problem labels (vectorized)
        batch_labels_local = decomposer.class_to_local[batch_labels_sub]

        # Compute loss for this sub-problem
        if len(batch_logits_sub) > 0:
            sub_loss = F.cross_entropy(batch_logits_sub, batch_labels_local)

            # Backward pass for this sub-problem only
            sub_loss.backward()

            # Gradient clipping
            if args is not None and hasattr(args, 'clip_grad'):
                torch.nn.utils.clip_grad_norm_(graphpfn_predictor.parameters(), args.clip_grad)

            optimizer.step()

            # Accumulate loss
            total_loss += sub_loss.item() * len(batch_logits_sub)
            total_samples += len(batch_logits_sub)

    avg_loss = total_loss / max(total_samples, 1)
    return avg_loss


def graphpfn_forward_with_decomposition(
    graphpfn_predictor,
    gnn_embeddings,
    labels,
    context_mask,
    num_classes,
    decomposer=None,
):
    """Forward pass through GraphPFN with optional output space decomposition.

    Args:
        graphpfn_predictor: GraphPFN predictor module
        gnn_embeddings: [N, hidden] GNN output features
        labels: [N] node labels (NaN for query nodes)
        context_mask: [N] boolean mask for support/context nodes
        num_classes: Total number of classes
        decomposer: Optional OutputSpaceDecomposer for >10 classes

    Returns:
        probs: [N_query, num_classes] probability distribution (NOT logits!)
            For compatibility with aggregation, returns probabilities
    """
    if decomposer is None:
        # Direct forward (num_classes <= 10)
        logits = graphpfn_predictor(
            gnn_embeddings=gnn_embeddings,
            labels=labels,
            context_mask=context_mask,
            num_classes=num_classes,
        )
        probs = F.softmax(logits, dim=-1)
        return probs

    # Decomposition path (num_classes > 10)
    import time
    decomp_start = time.time()

    device = gnn_embeddings.device
    N = gnn_embeddings.shape[0]
    num_subproblems = decomposer.config.num_subproblems

    # Run each sub-problem independently
    sub_probabilities = []

    for sub_idx in range(num_subproblems):
        # Prepare data for this sub-problem (inference mode - include ALL query nodes)
        # Returns: filtered labels, filtered context mask, and which nodes to pass
        sub_labels, sub_context_mask, node_selection_mask = decomposer.prepare_subproblem_data(
            labels=labels,
            context_mask=context_mask,
            sub_idx=sub_idx,
            training=False,  # Inference mode: include all query nodes for aggregation
        )

        # Select only relevant nodes: context from this partition + ALL query nodes
        selected_embeddings = gnn_embeddings[node_selection_mask]

        # Get sub-problem num_classes
        sub_num_classes = decomposer.get_subproblem_num_classes(sub_idx)

        # Forward through GraphPFN for this sub-problem
        # sub_labels and sub_context_mask are already filtered to selected nodes
        sub_logits = graphpfn_predictor(
            gnn_embeddings=selected_embeddings,
            labels=sub_labels,
            context_mask=sub_context_mask,
            num_classes=sub_num_classes,
        )  # [N_query, sub_num_classes] - In inference mode, N_query is THE SAME for all sub-problems

        # Convert to probabilities
        sub_probs = F.softmax(sub_logits, dim=-1)
        sub_probabilities.append(sub_probs)

    # Aggregate predictions from all sub-problems
    final_probs = decomposer.aggregate_predictions(sub_probabilities)

    decomp_time = time.time() - decomp_start
    print(f"  [Decomposition] {num_classes} classes → {num_subproblems} sub-problems, time: {decomp_time:.3f}s")

    # CORRUPTION CHECK 7: Validate decomposition output shape
    N_query = (~context_mask).sum().item()
    expected_shape = (N_query, num_classes)
    if final_probs.shape != expected_shape:
        print(f"[CORRUPTION] Decomposition output shape mismatch! Got {final_probs.shape}, expected {expected_shape}")

    # CORRUPTION CHECK 8: Check if aggregation is reasonable
    # Each query node should have probabilities summing to 1.0
    prob_sums = final_probs.sum(dim=-1)
    if not torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5):
        print(f"[CORRUPTION] Aggregated probabilities don't sum to 1! Range: [{prob_sums.min().item():.6f}, {prob_sums.max().item():.6f}]")

    return final_probs


def correct_and_smooth(adj, base_logits, context_idx, context_labels, num_classes,
                       num_iters=50, alpha=0.5):
    """
    Correct & Smooth: post-process feature-based predictions with label propagation.

    Args:
        adj: SparseTensor adjacency matrix
        base_logits: [num_nodes, num_classes] initial predictions from model
        context_idx: Tensor of context/support node indices (FEW-SHOT samples only!)
        context_labels: Tensor of context node labels
        num_classes: Number of classes
        num_iters: Number of propagation iterations (default: 50)
        alpha: Blending factor (default: 0.5)

    Returns:
        Y: [num_nodes, num_classes] refined predictions after C&S
    """
    device = adj.device()
    num_nodes = base_logits.size(0)

    # Compute normalized adjacency WITH self-loops: D^{-1/2} (A + I) D^{-1/2}
    deg = adj.sum(dim=1).to_dense()
    deg_with_selfloop = deg + 1  # Add 1 for self-loop
    deg_inv_sqrt = (deg_with_selfloop + 1e-9).pow(-0.5)

    # Start with softmax of base logits (feature-based estimate)
    Y = F.softmax(base_logits, dim=-1)

    # Ground truth for context/support set (few-shot samples only!)
    Y_support = F.one_hot(context_labels.long(), num_classes=num_classes).float()

    # Propagate and clamp
    for _ in range(num_iters):
        # Propagate with self-loop: D^{-1/2} (A*Y + Y) D^{-1/2}
        Y_new = deg_inv_sqrt.view(-1, 1) * Y
        Y_new = adj @ Y_new + Y_new  # A*Y + Y (implicit self-loop)
        Y_new = deg_inv_sqrt.view(-1, 1) * Y_new

        # Blend with previous
        Y = (1 - alpha) * Y_new + alpha * Y

        # Clamp context set to ground truth (force truth to flow outward from few-shot samples)
        Y[context_idx] = Y_support

    return Y


def train_graphpfn(model, data, train_idx, optimizer, graphpfn_predictor, batch_size,
                   projector=None, rank=0, epoch=0, identity_projection=None,
                   lambda_=1.0, args=None, external_embeddings=None):
    """
    Train GraphPFN on a single dataset for one epoch.

    GraphPFN uses in-context learning:
    - Context nodes (with labels) provide examples
    - Target nodes (train_idx) are predicted
    - No gradient updates to class prototypes (foundational model)

    Args:
        model: GNN backbone
        data: Graph dataset
        train_idx: Training node indices
        optimizer: Optimizer for GNN + GraphPFN
        graphpfn_predictor: GraphPFN predictor module
        batch_size: Batch size
        projector: Optional projector module (not used for GraphPFN)
        rank: Process rank for distributed training
        epoch: Current epoch number
        identity_projection: Optional identity projection (not used for GraphPFN)
        lambda_: Loss weight (not used for GraphPFN)
        args: Training arguments
        external_embeddings: External node embeddings (not used for GraphPFN)

    Returns:
        dict with 'total', 'nll', 'de' loss values
    """
    model.train()
    graphpfn_predictor.train()

    # Get number of classes from data
    num_classes = int(data.y.max().item()) + 1

    # Create decomposer if needed (for >10 classes)
    device = data.x.device if hasattr(data.x, 'device') else 'cpu'
    decomposer = create_decomposer(num_classes, max_classes_per_subproblem=graphpfn_predictor.n_out, device=str(device))
    if decomposer is not None and rank == 0:
        print(f"[GraphPFN] Using output space decomposition: {num_classes} classes → "
              f"{decomposer.config.num_subproblems} sub-problems")
    elif num_classes > graphpfn_predictor.n_out:
        raise ValueError(f"Dataset has {num_classes} classes but GraphPFN only supports {graphpfn_predictor.n_out}. "
                        f"Decomposition failed to initialize!")

    # Use distributed sampler for DDP
    if dist.is_initialized():
        indices = torch.arange(train_idx.size(0))
        sampler = DistributedSampler(indices, shuffle=True)
        sampler.set_epoch(epoch)
        dataloader = DataLoader(indices, batch_size=batch_size, sampler=sampler)
    else:
        dataloader = DataLoader(range(train_idx.size(0)), batch_size, shuffle=True)

    # Print dataset statistics for decomposition training
    if rank == 0 and decomposer is not None:
        print(f"[Decomposition Training] {num_classes} classes → {decomposer.config.num_subproblems} sub-problems, "
              f"{train_idx.size(0)} training nodes, {len(dataloader)} batches")

    total_loss = 0
    total_samples = 0
    total_correct = 0  # Only used for non-decomposition path (<=10 classes)

    for batch_idx, perm in enumerate(dataloader):
        if isinstance(perm, torch.Tensor):
            perm = perm.tolist()
        train_perm_idx = train_idx[perm]

        # GraphPFN uses raw GNN output (no PCA padding)
        # Just use data.x directly
        x_input = data.x

        # GNN forward pass on full graph
        h = model(x_input, data.adj_t)  # [num_nodes, hidden]

        # CRITICAL: Only pass context + current batch to GraphPFN (not entire graph!)
        # This prevents OOM from computing attention over all nodes

        # Step 1: Combine context nodes and current batch target nodes
        # Remove any overlap (in case batch contains context nodes)
        batch_nodes = torch.cat([data.context_sample, train_perm_idx])
        batch_nodes = torch.unique(batch_nodes, sorted=True)  # [num_context + num_batch_unique]

        # Step 2: Extract embeddings and labels for this subset
        h_subset = h[batch_nodes]  # [num_subset, hidden]
        labels_subset = data.y.squeeze()[batch_nodes]  # [num_subset]

        # Step 3: Create context mask for subset (vectorized)
        # True for nodes that are in context, False for targets
        is_context = torch.isin(batch_nodes, data.context_sample)  # [num_subset]

        # Step 4: GraphPFN forward on subset only
        if decomposer is not None:
            # Training-optimized path: per-subproblem gradient descent
            # Get query nodes from the current batch that are in the subset
            target_nodes_in_subset = batch_nodes[~is_context]
            batch_query_in_subset = torch.isin(target_nodes_in_subset, train_perm_idx)

            if batch_query_in_subset.any():
                query_indices_batch = target_nodes_in_subset[batch_query_in_subset]

                # This function handles gradient descent internally and returns loss only
                avg_loss = graphpfn_forward_with_decomposition_training(
                    graphpfn_predictor=graphpfn_predictor,
                    gnn_embeddings=h_subset,
                    labels=labels_subset,
                    context_mask=is_context,
                    num_classes=num_classes,
                    decomposer=decomposer,
                    optimizer=optimizer,
                    query_indices_batch=query_indices_batch,
                    batch_nodes=batch_nodes,  # Pass mapping for index conversion
                    args=args,
                )

                # Track loss only (no need for additional loss.backward() or optimizer.step())
                batch_size = len(query_indices_batch)
                total_loss += avg_loss * batch_size
                total_samples += batch_size
            # Skip the rest of the loop since gradient descent already happened
            continue
        else:
            # Direct forward for <=10 classes
            optimizer.zero_grad()
            
            logits_subset = graphpfn_predictor(
                gnn_embeddings=h_subset,
                labels=labels_subset,
                context_mask=is_context,
                num_classes=num_classes,
            )  # [num_targets_in_subset, num_classes] LOGITS

            # Step 5: Map train_perm_idx to positions in output logits (vectorized)
            # GraphPFN returns logits for batch_nodes[~is_context] in that order
            target_nodes_in_subset = batch_nodes[~is_context]  # Nodes that got predictions

            # Find which positions in target_nodes_in_subset correspond to train_perm_idx
            # Use broadcasting: target_nodes_in_subset[i] == train_perm_idx[j] ?
            matches = target_nodes_in_subset.unsqueeze(1) == train_perm_idx.unsqueeze(0)  # [num_targets, num_batch]
            target_positions = matches.any(dim=1).nonzero(as_tuple=True)[0]  # Positions in logits_subset
            batch_positions_in_perm = matches.any(dim=0).nonzero(as_tuple=True)[0]  # Positions in train_perm_idx

            # Step 6: Extract logits and labels for train_perm_idx only
            batch_logits = logits_subset[target_positions]  # [num_valid_batch, num_classes]
            batch_labels = labels_subset[~is_context][target_positions]  # [num_valid_batch]

            # Cross-entropy loss on BATCH ONLY
            loss = F.cross_entropy(batch_logits, batch_labels.long())

            loss.backward()

            # Gradient clipping (only GraphPFN predictor has parameters)
            if args is not None and hasattr(args, 'clip_grad'):
                torch.nn.utils.clip_grad_norm_(graphpfn_predictor.parameters(), args.clip_grad)

            optimizer.step()

            # Track metrics
            total_loss += loss.item() * len(batch_logits)

            with torch.no_grad():
                batch_preds = batch_logits.argmax(dim=-1)
                total_correct += (batch_preds == batch_labels).sum().item()
                total_samples += len(batch_logits)

    avg_loss = total_loss / max(total_samples, 1)

    if rank == 0:
        if decomposer is not None:
            # Decomposition: Don't print accuracy (meaningless across different label spaces)
            print(f"GraphPFN Train - Loss: {avg_loss:.4f}")
        else:
            # Non-decomposition: Print accuracy
            train_acc = total_correct / max(total_samples, 1) if total_samples > 0 else 0.0
            print(f"GraphPFN Train - Loss: {avg_loss:.4f}, Acc: {train_acc:.4f}")

    return {
        'total': avg_loss,
        'nll': avg_loss,
        'de': 0.0,  # No DE loss for GraphPFN
    }


@torch.no_grad()
def test_graphpfn(model, graphpfn_predictor, data, train_idx, valid_idx, test_idx,
                  batch_size, projector=None, rank=0, identity_projection=None,
                  external_embeddings=None, args=None):
    """
    Evaluate GraphPFN on a single dataset.

    Args:
        model: GNN backbone
        graphpfn_predictor: GraphPFN predictor module
        data: Graph dataset
        train_idx, valid_idx, test_idx: Split indices
        batch_size: Batch size
        projector: Optional projector (not used)
        rank: Process rank
        identity_projection: Optional identity projection (not used)
        external_embeddings: External embeddings (not used)
        args: Arguments

    Returns:
        tuple: (train_acc, valid_acc, test_acc, inference_time)
    """
    st = time.time()
    model.eval()
    graphpfn_predictor.eval()

    # Get number of classes
    num_classes = int(data.y.max().item()) + 1

    # DEBUG: Check if context_sample exists
    if not hasattr(data, 'context_sample'):
        dataset_name = getattr(data, 'name', 'unknown')
        print(f"[DEBUG] ERROR: Dataset {dataset_name} has no context_sample!")
        print(f"[DEBUG] Dataset has {data.num_nodes} nodes, {num_classes} classes")
        print(f"[DEBUG] train_idx size: {len(train_idx)}, valid_idx size: {len(valid_idx)}, test_idx size: {len(test_idx)}")
        raise AttributeError(f"Dataset {dataset_name} missing context_sample - must be set before evaluation!")

    # Create decomposer if needed (for >10 classes)
    device = data.x.device if hasattr(data.x, 'device') else 'cpu'
    decomposer = create_decomposer(num_classes, max_classes_per_subproblem=graphpfn_predictor.n_out, device=str(device))
    if decomposer is not None and rank == 0:
        print(f"[GraphPFN Eval] Using output space decomposition: {num_classes} classes → "
              f"{decomposer.config.num_subproblems} sub-problems")

    # GraphPFN uses raw GNN output
    x_input = data.x

    # GNN forward pass on full graph
    h = model(x_input, data.adj_t)
    labels = data.y.squeeze()

    # DEBUG: Print context info (after labels is defined)
    if rank == 0:
        dataset_name = getattr(data, 'name', 'unknown')
        print(f"[DEBUG] {dataset_name}: context_sample size = {len(data.context_sample)}, "
              f"num_classes = {num_classes}, total_nodes = {data.num_nodes}")
        print(f"[DEBUG] {dataset_name}: train={len(train_idx)}, valid={len(valid_idx)}, test={len(test_idx)}")

        # CORRUPTION CHECK 1: Validate index ranges
        max_node_id = data.num_nodes - 1
        if len(data.context_sample) > 0:
            context_max = data.context_sample.max().item()
            context_min = data.context_sample.min().item()
            if context_max > max_node_id or context_min < 0:
                print(f"[CORRUPTION] context_sample has invalid indices! Range: [{context_min}, {context_max}], valid: [0, {max_node_id}]")

        if len(train_idx) > 0:
            train_max = train_idx.max().item()
            if train_max > max_node_id:
                print(f"[CORRUPTION] train_idx has invalid indices! Max: {train_max}, valid max: {max_node_id}")

        if len(test_idx) > 0:
            test_max = test_idx.max().item()
            if test_max > max_node_id:
                print(f"[CORRUPTION] test_idx has invalid indices! Max: {test_max}, valid max: {max_node_id}")

        # CORRUPTION CHECK 2: Validate label ranges
        label_max = labels.max().item()
        label_min = labels.min().item()
        if label_max >= num_classes or label_min < 0:
            print(f"[CORRUPTION] Labels out of range! Range: [{label_min}, {label_max}], expected: [0, {num_classes-1}]")

        # Check context distribution across classes
        context_labels = labels[data.context_sample]
        for c in range(num_classes):
            count = (context_labels == c).sum().item()
            print(f"[DEBUG]   Class {c}: {count} context nodes")

        # CORRUPTION CHECK 3: Check if context has all classes
        missing_classes = []
        for c in range(num_classes):
            if (context_labels == c).sum().item() == 0:
                missing_classes.append(c)
        if missing_classes:
            print(f"[CORRUPTION] Context missing classes: {missing_classes} - model cannot predict these!")

    # Process in batches to avoid OOM (context + batch of query nodes)
    all_logits = torch.zeros(h.size(0), num_classes, device=h.device)

    # Get all non-context nodes (nodes to predict)
    all_nodes = torch.arange(h.size(0), device=h.device)
    is_context_full = torch.isin(all_nodes, data.context_sample)
    query_nodes = all_nodes[~is_context_full]  # All nodes except context

    # Process query nodes in batches
    for batch_start in range(0, len(query_nodes), batch_size):
        batch_end = min(batch_start + batch_size, len(query_nodes))
        query_batch = query_nodes[batch_start:batch_end]

        # Combine context + current query batch
        batch_nodes = torch.cat([data.context_sample, query_batch])
        batch_nodes = torch.unique(batch_nodes, sorted=True)

        # Extract subset
        h_subset = h[batch_nodes]
        labels_subset = labels[batch_nodes]

        # Create context mask for subset
        is_context = torch.isin(batch_nodes, data.context_sample)

        # GraphPFN forward on subset (with optional decomposition)
        probs_subset = graphpfn_forward_with_decomposition(
            graphpfn_predictor=graphpfn_predictor,
            gnn_embeddings=h_subset,
            labels=labels_subset,
            context_mask=is_context,
            num_classes=num_classes,
            decomposer=decomposer,
        )

        # Convert to logits for consistency
        logits_subset = torch.log(probs_subset + 1e-9)

        # Map back to original indices
        target_nodes_in_subset = batch_nodes[~is_context]
        # Ensure logits_subset is on the same device as all_logits
        all_logits[target_nodes_in_subset] = logits_subset.to(all_logits.device)

    # Apply C&S if enabled (validation-based selection)
    use_cs = False
    if args is not None and hasattr(args, 'use_cs') and args.use_cs:
        # Apply C&S
        cs_logits = correct_and_smooth(
            data.adj_t,
            all_logits,
            data.context_sample,
            labels[data.context_sample],
            num_classes,
            num_iters=getattr(args, 'cs_num_iters', 50),
            alpha=getattr(args, 'cs_alpha', 0.5),
        )

        # Validate on valid set to choose between base and C&S
        base_valid_acc = acc(all_logits[valid_idx].argmax(dim=-1), labels[valid_idx])
        cs_valid_acc = acc(cs_logits[valid_idx].argmax(dim=-1), labels[valid_idx])

        if cs_valid_acc > base_valid_acc:
            all_logits = cs_logits
            use_cs = True
            if rank == 0:
                print(f"Using C&S (valid acc: {cs_valid_acc:.4f} vs base {base_valid_acc:.4f})")
        elif rank == 0:
            print(f"Not using C&S (valid acc: {cs_valid_acc:.4f} vs base {base_valid_acc:.4f})")

    pred_labels = all_logits.argmax(dim=-1)

    # DEBUG: Check prediction distribution
    if rank == 0:
        dataset_name = getattr(data, 'name', 'unknown')
        print(f"[DEBUG] {dataset_name} prediction distribution:")
        for c in range(num_classes):
            pred_count = (pred_labels == c).sum().item()
            true_count = (labels == c).sum().item()
            print(f"[DEBUG]   Class {c}: predicted={pred_count}, actual={true_count}")

        # CORRUPTION CHECK 4: Check if collapsed to single class
        unique_preds = pred_labels.unique()
        if len(unique_preds) == 1:
            print(f"[CORRUPTION] Model collapsed to single class {unique_preds[0].item()}!")
        elif len(unique_preds) < num_classes / 2:
            print(f"[CORRUPTION] Model predicting only {len(unique_preds)}/{num_classes} classes!")

        # CORRUPTION CHECK 5: Check test set predictions specifically
        test_preds = pred_labels[test_idx]
        test_labels_true = labels[test_idx]
        test_unique_preds = test_preds.unique()
        test_unique_labels = test_labels_true.unique()

        print(f"[DEBUG] Test set: {len(test_idx)} nodes, {len(test_unique_labels)} unique true labels, {len(test_unique_preds)} unique predictions")

        # Check if test predictions are diverse
        if len(test_unique_preds) == 1:
            print(f"[CORRUPTION] Test predictions collapsed to single class {test_unique_preds[0].item()}!")
            print(f"[CORRUPTION] Test true label distribution:")
            for c in range(num_classes):
                count = (test_labels_true == c).sum().item()
                if count > 0:
                    print(f"[CORRUPTION]   Class {c}: {count} nodes")

        # CORRUPTION CHECK 6: Check logits sanity
        test_logits = all_logits[test_idx]
        logit_mean = test_logits.mean().item()
        logit_std = test_logits.std().item()
        logit_max = test_logits.max().item()
        logit_min = test_logits.min().item()

        print(f"[DEBUG] Test logits: mean={logit_mean:.2f}, std={logit_std:.2f}, range=[{logit_min:.2f}, {logit_max:.2f}]")

        # Check if logits are degenerate (all same)
        if logit_std < 0.01:
            print(f"[CORRUPTION] Test logits have near-zero variance ({logit_std:.6f}) - all predictions identical!")

        # Check confidence distribution (max prob per sample)
        test_probs = torch.softmax(test_logits, dim=-1)
        test_confidence = test_probs.max(dim=-1)[0]
        print(f"[DEBUG] Test confidence: mean={test_confidence.mean().item():.4f}, min={test_confidence.min().item():.4f}, max={test_confidence.max().item():.4f}")

        if test_confidence.mean().item() > 0.95:
            print(f"[CORRUPTION] Model extremely confident (avg {test_confidence.mean().item():.4f}) but low accuracy - possible label mismatch!")

    # Compute accuracy
    train_acc = acc(pred_labels[train_idx], labels[train_idx])
    valid_acc = acc(pred_labels[valid_idx], labels[valid_idx])
    test_acc = acc(pred_labels[test_idx], labels[test_idx])

    inference_time = time.time() - st

    if rank == 0:
        print(f'GraphPFN Test - Train: {train_acc:.4f}, Valid: {valid_acc:.4f}, Test: {test_acc:.4f}, Time: {inference_time:.2f}s')

    return train_acc, valid_acc, test_acc, inference_time


def train_all_graphpfn(model, data_list, split_idx_list, optimizer, graphpfn_predictor,
                       batch_size, projector=None, rank=0, epoch=0,
                       identity_projection=None, lambda_=1.0, args=None,
                       external_embeddings_list=None):
    """Train GraphPFN on all datasets."""
    tot_loss = 0
    tot_nll_loss = 0

    for i, (data, split_idx) in enumerate(zip(data_list, split_idx_list)):
        train_idx = split_idx['train']
        external_embeddings = external_embeddings_list[i] if external_embeddings_list else None

        loss_dict = train_graphpfn(
            model, data, train_idx, optimizer, graphpfn_predictor, batch_size,
            projector, rank, epoch, identity_projection, lambda_, args, external_embeddings
        )

        tot_loss += loss_dict['total']
        tot_nll_loss += loss_dict['nll']

        if rank == 0:
            print(f"Dataset {data.name} - Loss: {loss_dict['total']:.4f}", flush=True)

    return {
        'total': tot_loss / len(data_list),
        'nll': tot_nll_loss / len(data_list),
        'de': 0.0,
    }


@torch.no_grad()
def test_all_graphpfn(model, graphpfn_predictor, data_list, split_idx_list, batch_size,
                      projector=None, rank=0, identity_projection=None,
                      external_embeddings_list=None, args=None):
    """Test GraphPFN on all datasets (transductive setting - returns geometric mean)."""
    tot_train_metric = 1
    tot_valid_metric = 1
    tot_test_metric = 1

    for i, (data, split_idx) in enumerate(zip(data_list, split_idx_list)):
        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']
        external_embeddings = external_embeddings_list[i] if external_embeddings_list else None

        train_acc, valid_acc, test_acc, _ = test_graphpfn(
            model, graphpfn_predictor, data, train_idx, valid_idx, test_idx, batch_size,
            projector, rank, identity_projection, external_embeddings, args=args
        )

        tot_train_metric *= train_acc
        tot_valid_metric *= valid_acc
        tot_test_metric *= test_acc

    # Geometric mean
    return (tot_train_metric ** (1/len(data_list)),
            tot_valid_metric ** (1/len(data_list)),
            tot_test_metric ** (1/len(data_list)))


@torch.no_grad()
def test_all_induct_graphpfn(model, graphpfn_predictor, data_list, split_idx_list, batch_size,
                              projector=None, rank=0, identity_projection=None,
                              external_embeddings_list=None, args=None):
    """Test GraphPFN on all datasets (inductive setting - returns individual metrics)."""
    import time

    train_metric_list, valid_metric_list, test_metric_list = [], [], []
    for dataset_idx, (data, split_idx) in enumerate(zip(data_list, split_idx_list)):
        dataset_start_time = time.time()

        train_idx = split_idx['train']
        valid_idx = split_idx['valid']
        test_idx = split_idx['test']

        external_embeddings = external_embeddings_list[dataset_idx] if external_embeddings_list else None
        train_metric, valid_metric, test_metric, _ = test_graphpfn(
            model, graphpfn_predictor, data, train_idx, valid_idx, test_idx, batch_size,
            projector, rank, identity_projection, external_embeddings, args=args
        )

        dataset_time = time.time() - dataset_start_time
        if rank == 0:
            print(f"    Dataset {dataset_idx + 1}/{len(data_list)}: "
                  f"Train={train_metric:.4f}, Valid={valid_metric:.4f}, Test={test_metric:.4f} "
                  f"({dataset_time:.2f}s)")

        train_metric_list.append(train_metric)
        valid_metric_list.append(valid_metric)
        test_metric_list.append(test_metric)

    return train_metric_list, valid_metric_list, test_metric_list
