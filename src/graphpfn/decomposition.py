"""
Output Space Decomposition for handling >10 classes in GraphPFN.

Based on classic ECOC approach: decompose C-class problem into B sub-problems,
each with ≤10 classes (to satisfy GraphPFN's constraint).

Key Design Principles:
1. Each sub-problem sees ONLY its partition's classes in the support set
2. Query nodes from other partitions are OOD to this sub-problem
3. Use probability space (not logits) for aggregation to handle OOD gracefully
4. Well-calibrated models output flat distributions for OOD queries
5. FULLY VECTORIZED: all operations use tensor indexing, no Python loops

Example (C=40):
    Sub-problem 0 (classes 0-9):
        Support: nodes with labels in {0,1,2,3,4,5,6,7,8,9}
        Query: ALL unlabeled nodes (including those that are actually class 39)
        Output: For class-39 node → flat distribution ~[0.1, 0.1, ..., 0.1]

    Sub-problem 3 (classes 30-39):
        Support: nodes with labels in {30,31,32,33,34,35,36,37,38,39}
        Query: ALL unlabeled nodes
        Output: For class-39 node → peaked distribution [0.0, ..., 0.9]

    Aggregation:
        Combine all sub-problem probabilities (renormalize to C classes)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional


@dataclass
class DecompositionConfig:
    """Configuration for output space decomposition.

    Args:
        num_classes: Total number of classes (C)
        max_classes_per_subproblem: Maximum classes per sub-problem (default: 10)
        aggregation_method: How to combine sub-problem predictions
            - "uniform_prior": Weighted by partition size (default)
            - "entropy_weighted": Weight by prediction confidence (future)
    """
    num_classes: int
    max_classes_per_subproblem: int = 10
    aggregation_method: str = "uniform_prior"

    def __post_init__(self):
        assert self.num_classes > 0
        assert self.max_classes_per_subproblem > 0

    @property
    def num_subproblems(self) -> int:
        """Number of sub-problems needed: B = ⌈C / max_classes_per_subproblem⌉"""
        return (self.num_classes + self.max_classes_per_subproblem - 1) // self.max_classes_per_subproblem

    @property
    def needs_decomposition(self) -> bool:
        """Whether decomposition is needed (C > max_classes_per_subproblem)"""
        # Skip decomposition if num_classes <= 10 (no benefit, adds overhead)
        return self.num_classes > 10 and self.num_classes > self.max_classes_per_subproblem


class OutputSpaceDecomposer:
    """Decomposes C-class problem into B sub-problems with ≤10 classes each.

    Example (C=40, max=10):
        Sub-problem 0: classes {0-9}   → 10-way classification
        Sub-problem 1: classes {10-19} → 10-way classification
        Sub-problem 2: classes {20-29} → 10-way classification
        Sub-problem 3: classes {30-39} → 10-way classification

    Key Features:
    1. Support set filtering: Each sub-problem only sees its own classes
    2. OOD handling: Query nodes from other partitions get flat distributions
    3. Probability aggregation: Combine using normalized probabilities
    4. FULLY VECTORIZED: Pre-computes all index mappings at init time
    """

    def __init__(self, config: DecompositionConfig, device: str = 'cpu'):
        self.config = config
        self.device = device
        self._build_class_partitions()

    def _build_class_partitions(self):
        """Build class partitions and pre-compute ALL index mappings.

        Pre-computed tensors (all vectorized operations):
        - class_to_subproblem: [C] which sub-problem each class belongs to
        - class_to_local: [C] local index within sub-problem
        - partition_sizes: [B] number of classes in each partition
        - scatter_indices: [B, max_size] for scattering sub-probs to final probs
        """
        C = self.config.num_classes
        max_per_sub = self.config.max_classes_per_subproblem
        B = self.config.num_subproblems

        # Keep list form for compatibility (but won't use in hot path)
        self.partitions = []

        # Pre-allocate all mapping tensors
        self.class_to_subproblem = torch.zeros(C, dtype=torch.long, device=self.device)
        self.class_to_local = torch.zeros(C, dtype=torch.long, device=self.device)
        self.partition_sizes = torch.zeros(B, dtype=torch.long, device=self.device)

        # Scatter indices for aggregation: [B, max_per_sub]
        # scatter_indices[b, i] = global class index for sub-problem b, local class i
        # Use -1 for padding (will be masked out)
        self.scatter_indices = torch.full((B, max_per_sub), -1, dtype=torch.long, device=self.device)

        # Build partitions using vectorized operations
        for sub_idx in range(B):
            start = sub_idx * max_per_sub
            end = min(start + max_per_sub, C)
            size = end - start

            # Create partition as tensor
            partition_tensor = torch.arange(start, end, dtype=torch.long, device=self.device)
            self.partitions.append(partition_tensor.tolist())  # For compatibility

            # Fill mapping tensors (vectorized)
            self.class_to_subproblem[start:end] = sub_idx
            self.class_to_local[start:end] = torch.arange(size, dtype=torch.long, device=self.device)
            self.partition_sizes[sub_idx] = size
            self.scatter_indices[sub_idx, :size] = partition_tensor

    def prepare_subproblem_data(
        self,
        labels: torch.Tensor,
        context_mask: torch.Tensor,
        sub_idx: int,
        training: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare data for a specific sub-problem (FULLY VECTORIZED).

        CRITICAL: Only pass relevant nodes to GraphPFN!
        - Context nodes from THIS partition only
        - Query nodes:
            * Training mode (training=True): Only queries whose true labels belong to this partition
            * Inference mode (training=False): ALL query nodes (for aggregation)

        Args:
            labels: Original class labels [N], values in {0, ..., C-1}
                NaN for unlabeled nodes
            context_mask: Boolean mask [N], True for support/context nodes, False for query
            sub_idx: Which sub-problem (0 to B-1)
            training: If True, filter query nodes by label; if False, include all query nodes

        Returns:
            sub_labels: Remapped labels [N_subset], values in {0, ..., |partition|-1} or NaN
            sub_context_mask: Context mask for subset [N_subset], True for context from this partition
            node_selection_mask: Boolean mask [N], True for nodes to pass to GraphPFN
        """
        N = labels.shape[0]
        device = labels.device

        # Move pre-computed tensors to same device if needed
        if self.class_to_local.device != device:
            self.class_to_subproblem = self.class_to_subproblem.to(device)
            self.class_to_local = self.class_to_local.to(device)

        # Step 1: Identify which context nodes belong to this partition
        context_indices = torch.where(context_mask)[0]

        # Filter context nodes by partition
        partition_context_mask = torch.zeros(N, dtype=torch.bool, device=device)
        if len(context_indices) > 0:
            context_labels = labels[context_indices].long()
            belongs_to_partition = (self.class_to_subproblem[context_labels] == sub_idx)
            partition_context_indices = context_indices[belongs_to_partition]
            partition_context_mask[partition_context_indices] = True

        # Step 2: Node selection mask - context from partition + query nodes
        query_mask = ~context_mask

        if training:
            # Training mode: Filter query nodes by partition (only those with labels in this partition)
            partition_query_mask = torch.zeros(N, dtype=torch.bool, device=device)
            query_indices = torch.where(query_mask)[0]
            if len(query_indices) > 0:
                query_labels = labels[query_indices]
                # Only include queries with labels belonging to this partition
                has_label = ~torch.isnan(query_labels)
                if has_label.any():
                    labeled_query_indices = query_indices[has_label]
                    labeled_query_labels = query_labels[has_label].long()
                    belongs_to_partition = (self.class_to_subproblem[labeled_query_labels] == sub_idx)
                    partition_query_indices = labeled_query_indices[belongs_to_partition]
                    partition_query_mask[partition_query_indices] = True

            node_selection_mask = partition_context_mask | partition_query_mask
        else:
            # Inference mode: Include ALL query nodes (for aggregation)
            node_selection_mask = partition_context_mask | query_mask

        # Step 3: Extract selected nodes
        selected_indices = torch.where(node_selection_mask)[0]
        N_subset = len(selected_indices)

        # Step 4: Create sub_labels for selected nodes (remap classes to local indices)
        # Force float dtype to support NaN values
        sub_labels = torch.full((N_subset,), float('nan'), dtype=torch.float32, device=device)
        selected_labels = labels[selected_indices]
        # Convert to float if needed
        if selected_labels.dtype != torch.float32:
            selected_labels = selected_labels.float()

        # Remap labels for nodes that have labels
        is_labeled = ~torch.isnan(selected_labels)
        if is_labeled.any():
            labeled_indices_in_subset = torch.where(is_labeled)[0]
            labeled_classes = selected_labels[labeled_indices_in_subset].long()

            # Check which belong to this partition
            belongs_to_partition = (self.class_to_subproblem[labeled_classes] == sub_idx)

            # Remap
            if belongs_to_partition.any():
                in_partition_indices = labeled_indices_in_subset[belongs_to_partition]
                in_partition_classes = labeled_classes[belongs_to_partition]
                sub_labels[in_partition_indices] = self.class_to_local[in_partition_classes].float()

        # Step 5: Create sub_context_mask for selected nodes
        # True for partition context nodes, False for query nodes
        sub_context_mask = partition_context_mask[selected_indices]

        return sub_labels, sub_context_mask, node_selection_mask

    def aggregate_predictions(
        self,
        sub_probabilities: list[torch.Tensor],
    ) -> torch.Tensor:
        """Aggregate predictions from B sub-problems into C-class distribution (FULLY VECTORIZED).

        Uses probability space (not logits) to handle OOD gracefully:
        - Sub-problems output flat distributions for OOD queries
        - In-distribution sub-problems output peaked distributions
        - Aggregation combines and renormalizes

        Args:
            sub_probabilities: List of B tensors, each [N, ≤10] probabilities
                sub_probabilities[i] corresponds to partition i
                MUST be probabilities (sum to 1), not logits!

        Returns:
            final_probs: [N, C] probability distribution over all original classes
        """
        assert len(sub_probabilities) == self.config.num_subproblems

        N = sub_probabilities[0].shape[0]
        C = self.config.num_classes
        device = sub_probabilities[0].device
        dtype = sub_probabilities[0].dtype

        # Move scatter_indices to same device if needed
        if self.scatter_indices.device != device:
            self.scatter_indices = self.scatter_indices.to(device)
            self.partition_sizes = self.partition_sizes.to(device)

        # Initialize with zeros (not -inf, we're in probability space)
        final_probs = torch.zeros((N, C), device=device, dtype=dtype)

        # FULLY VECTORIZED scatter using index_add_ (NO LOOPS!)
        max_per_sub = self.config.max_classes_per_subproblem
        B = self.config.num_subproblems

        # Pad each sub_prob to max_per_sub and stack (still need this loop, but it's small)
        padded_sub_probs = []
        for sub_idx, sub_probs in enumerate(sub_probabilities):
            num_classes = self.partition_sizes[sub_idx].item()
            if num_classes < max_per_sub:
                # Pad with zeros
                padding = torch.zeros((N, max_per_sub - num_classes), device=device, dtype=dtype)
                padded = torch.cat([sub_probs, padding], dim=1)
            else:
                padded = sub_probs
            padded_sub_probs.append(padded)

        # Stack: [B, N, max_per_sub]
        stacked_probs = torch.stack(padded_sub_probs, dim=0)

        # Create mask for valid indices: [B, max_per_sub]
        valid_mask = (self.scatter_indices >= 0)

        # Flatten for scatter: [B*max_per_sub, N]
        # We'll use advanced indexing to scatter all at once
        flat_indices = self.scatter_indices[valid_mask]  # [total_valid]
        flat_probs = stacked_probs.permute(1, 0, 2).reshape(N, -1)  # [N, B*max_per_sub]
        flat_probs = flat_probs[:, valid_mask.flatten()]  # [N, total_valid]

        # Scatter using index_copy (vectorized)
        # final_probs shape: [N, C]
        # We need to place flat_probs[:, i] into final_probs[:, flat_indices[i]]
        final_probs.scatter_(dim=1, index=flat_indices.unsqueeze(0).expand(N, -1), src=flat_probs)

        # Renormalize (in case of numerical issues)
        final_probs = final_probs / (final_probs.sum(dim=1, keepdim=True) + 1e-9)

        return final_probs

    def get_subproblem_num_classes(self, sub_idx: int) -> int:
        """Get number of classes in a specific sub-problem."""
        return len(self.partitions[sub_idx])

    def get_original_classes_for_subproblem(self, sub_idx: int) -> list[int]:
        """Get the original class indices for a specific sub-problem."""
        return self.partitions[sub_idx]

    def __repr__(self) -> str:
        return (f"OutputSpaceDecomposer(num_classes={self.config.num_classes}, "
                f"num_subproblems={self.config.num_subproblems}, "
                f"partitions={self.partitions})")


def create_decomposer(num_classes: int, max_classes_per_subproblem: int = 10, device: str = 'cpu') -> OutputSpaceDecomposer | None:
    """Factory function to create decomposer only if needed.

    Args:
        num_classes: Total number of classes
        max_classes_per_subproblem: Maximum classes per sub-problem (default: 10)
        device: Device to store pre-computed tensors ('cpu' or 'cuda')

    Returns:
        OutputSpaceDecomposer if num_classes > max_classes_per_subproblem, else None
    """
    config = DecompositionConfig(
        num_classes=num_classes,
        max_classes_per_subproblem=max_classes_per_subproblem
    )

    if config.needs_decomposition:
        return OutputSpaceDecomposer(config, device=device)
    else:
        return None
