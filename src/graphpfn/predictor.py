"""
GraphPFN predictor wrapper for graph node classification.

Adapted from TabPFN's PerFeatureTransformer:
https://github.com/automl/TabPFN/blob/main/src/tabpfn/architectures/base/transformer.py
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import einops
import torch
from torch import nn

from .encoders import (
    FourierFeatureEncoderStep,
    LinearInputEncoderStep,
    NanHandlingEncoderStep,
    SequentialEncoder,
)
from .layer import PerFeatureEncoderLayer

if TYPE_CHECKING:
    from .config import GraphPFNConfig


class GraphPFNPredictor(nn.Module):
    """GraphPFN foundational model for node classification via in-context learning.

    This is a TabPFN-style foundational model adapted for graphs:
    - **Foundational model**: Pre-trained once, then frozen for zero-shot inference
    - **Fixed decoder**: Always outputs n_out logits (max 10 classes), sliced at inference
    - **In-context learning**: No task-specific training, only context examples
    - Takes GNN embeddings as input (instead of raw tabular data)
    - Groups features using features_per_group (default=2)
    - Applies feature positional embeddings to distinguish groups (CRITICAL!)
    - Uses dual attention (features + items) from TabPFN
    - Supports KV caching for efficient inference

    Like TabPFN:
    - Maximum 10 classes supported (n_out <= 10)
    - For tasks with <10 classes, logits are sliced: logits[:, :num_classes]
    - Decoder weights are shared across all tasks (no task-specific heads)

    Skipped from TabPFN:
    - DAG positional encodings (dag_pos_enc_dim)
    - Thinking tokens
    - Separate decoder (use_separate_decoder)
    - Bar distribution for regression
    - Layer dropout
    - Encoder compression layer

    Args:
        config: GraphPFN configuration (config.n_out defines max classes, default 10)
        cache_trainset_representation: If True, cache K,V for context nodes (inference speedup)

    Note:
        GNN output dimension is determined dynamically at forward time.
        GraphPFN accepts arbitrary dimensions and pads minimally to divisible by features_per_group.
    """

    def __init__(
        self,
        *,
        config: GraphPFNConfig,
        cache_trainset_representation: bool = False,
    ):
        super().__init__()

        self.config = config
        self.emsize = config.emsize
        self.features_per_group = config.features_per_group
        self.cache_trainset_representation = cache_trainset_representation

        # Fixed n_out for foundational model (like TabPFN)
        # This is the MAXIMUM number of classes the model can handle
        self.n_out = config.n_out
        assert self.n_out <= 10, \
            f"GraphPFN foundational model supports at most 10 classes (got n_out={self.n_out})"

        # X encoder: GNN embeddings -> emsize
        # Input is [num_nodes, features_per_group] after grouping
        # Use Fourier Features when features_per_group=1 for coordinate-free encoding
        # Otherwise use learnable Linear projection (standard TabPFN approach)
        if config.features_per_group == 1:
            # Fourier Feature encoding: element-wise, no cross-feature mixing
            # Output: [num_nodes, emsize] per feature
            self.x_encoder = SequentialEncoder(
                FourierFeatureEncoderStep(
                    num_features=config.features_per_group,
                    emsize=config.emsize,
                    replace_nan_by_zero=False,
                    scale=getattr(config, 'fourier_feature_scale', 1.0),
                    in_keys=("main",),
                    out_keys=("output",),
                ),
                output_key="output",
            )
        else:
            # Linear encoding: learnable projection (TabPFN default)
            # Encoder uses SAME weights for all groups (key insight from TabPFN!)
            self.x_encoder = SequentialEncoder(
                LinearInputEncoderStep(
                    num_features=config.features_per_group,
                    emsize=config.emsize,
                    replace_nan_by_zero=False,
                    bias=config.encoder_use_bias,
                    in_keys=("main",),
                    out_keys=("output",),
                ),
                output_key="output",
            )

        # Y encoder: labels -> emsize
        # Handles NaN for target nodes (which don't have labels during inference)
        # Input is [num_nodes, 1] label + [num_nodes, 1] nan_indicator = 2 features
        self.y_encoder = SequentialEncoder(
            NanHandlingEncoderStep(
                keep_nans=True,
                in_keys=("main",),
                out_keys=("main", "nan_indicators"),
            ),
            LinearInputEncoderStep(
                num_features=2,  # label + nan_indicator
                emsize=config.emsize,
                replace_nan_by_zero=False,
                bias=config.encoder_use_bias,
                in_keys=("main", "nan_indicators"),
                out_keys=("output",),
            ),
            output_key="output",
        )

        # Feature positional embeddings (CRITICAL!)
        # Without this, all feature groups are indistinguishable because
        # they all pass through the SAME encoder weights
        self.feature_positional_embedding = config.feature_positional_embedding
        if self.feature_positional_embedding == "learned":
            self.feature_positional_embedding_embeddings = nn.Embedding(
                1000, config.emsize
            )
        elif self.feature_positional_embedding == "subspace":
            # Default mode: project from emsize//4 subspace
            self.feature_positional_embedding_embeddings = nn.Linear(
                config.emsize // 4, config.emsize
            )
        else:
            self.feature_positional_embedding_embeddings = None

        self.random_embedding_seed = config.seed
        self.cached_feature_positional_embeddings: torch.Tensor | None = None

        # Transformer layers
        nhid = config.emsize * config.nhid_factor
        self.transformer_layers = nn.ModuleList([
            PerFeatureEncoderLayer(
                config=config,
                dim_feedforward=nhid,
                activation="gelu",
                attention_between_features=config.attention_between_features,
            )
            for _ in range(config.nlayers)
        ])

        # Decoder: emsize -> n_out (FIXED, foundational model)
        # Like TabPFN, we have a fixed output dimension (max 10 classes)
        # For tasks with fewer classes, we slice the logits
        self.decoder = nn.Sequential(
            nn.Linear(config.emsize, nhid),
            nn.GELU(),
            nn.Linear(nhid, self.n_out),  # FIXED output dimension
        )

    def forward(
        self,
        gnn_embeddings: torch.Tensor,
        labels: torch.Tensor,
        context_mask: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        """Forward pass through GraphPFN (in-context learning).

        This is a foundational model that does NOT train task-specific heads.
        The decoder always outputs n_out logits (max 10), which are sliced
        to the actual number of classes for the task.

        Args:
            gnn_embeddings: GNN output features [num_nodes, gnn_output_dim]
            labels: Node labels [num_nodes], can contain NaN for target nodes
            context_mask: Boolean mask [num_nodes], True for context nodes with labels
            num_classes: Actual number of classes for this task (must be <= n_out)

        Returns:
            Logits for target nodes [num_target_nodes, num_classes] (sliced from n_out)
        """
        # Validate num_classes
        assert num_classes <= self.n_out, \
            f"Task has {num_classes} classes but model only supports {self.n_out} classes"

        num_nodes, gnn_output_dim = gnn_embeddings.shape
        device = gnn_embeddings.device
        dtype = gnn_embeddings.dtype

        # Validate inputs
        assert labels.shape == (num_nodes,), \
            f"Expected shape [{num_nodes}], got {labels.shape}"
        assert context_mask.shape == (num_nodes,), \
            f"Expected shape [{num_nodes}], got {context_mask.shape}"

        # single_eval_pos = number of context nodes (nodes with labels)
        single_eval_pos = context_mask.sum().item()

        # Reorder: context nodes first, then target nodes
        # This matches TabPFN's convention where y[:single_eval_pos] are context
        context_indices = torch.where(context_mask)[0]
        target_indices = torch.where(~context_mask)[0]
        reorder_indices = torch.cat([context_indices, target_indices])

        gnn_embeddings = gnn_embeddings[reorder_indices]
        labels = labels[reorder_indices]

        # Normalize input features if enabled (like TabPFN)
        if self.config.normalize_x:
            # Standardize: zero mean, unit variance
            mean = gnn_embeddings.mean(dim=0, keepdim=True)
            std = gnn_embeddings.std(dim=0, keepdim=True)
            gnn_embeddings = (gnn_embeddings - mean) / (std + 1e-9)
            print(f"[DEBUG NORM] After normalization: shape={gnn_embeddings.shape}, norm={gnn_embeddings.norm().item():.2f}")

        # Pad GNN embeddings to be divisible by features_per_group (like TabPFN)
        # This is done DYNAMICALLY based on actual input dimension
        missing_to_next = (
            self.features_per_group - (gnn_output_dim % self.features_per_group)
        ) % self.features_per_group

        if missing_to_next > 0:
            padding = torch.zeros(
                num_nodes, missing_to_next, device=device, dtype=dtype
            )
            gnn_embeddings = torch.cat([gnn_embeddings, padding], dim=1)

        # Now gnn_embeddings is padded to be divisible by features_per_group
        padded_dim = gnn_embeddings.shape[1]
        num_feature_groups = padded_dim // self.features_per_group

        # Reshape to [batch=1, seq_len=num_nodes, num_feature_groups, features_per_group]
        # TabPFN convention: batch first, then sequence (nodes)
        x = einops.rearrange(
            gnn_embeddings,
            "n (f g) -> 1 n f g",
            g=self.features_per_group,
        )  # [1, num_nodes, num_feature_groups, features_per_group]

        # Prepare labels: [num_nodes] -> [1, num_nodes, 1]
        # Set target labels to NaN (will be handled by NanHandlingEncoderStep)
        # Convert to float to support NaN
        y = labels.clone().float().unsqueeze(0).unsqueeze(-1)  # [1, num_nodes, 1]
        y[:, single_eval_pos:] = torch.nan

        # Encode X
        # Flatten batch and feature_groups for encoding (same encoder for all groups!)
        x_flat = einops.rearrange(x, "b n f g -> n (b f) g")
        # Shape: [num_nodes, batch*num_feature_groups, features_per_group]

        x_encoded = self.x_encoder(
            {"main": x_flat},
            single_eval_pos=single_eval_pos,
            cache_trainset_representation=self.cache_trainset_representation,
        )  # [num_nodes, batch*num_feature_groups, emsize]

        # Reshape back
        x_encoded = einops.rearrange(
            x_encoded,
            "n (b f) e -> b n f e",
            b=1,
        )  # [1, num_nodes, num_feature_groups, emsize]

        # Encode Y
        y_encoded = self.y_encoder(
            {"main": y.transpose(0, 1)},  # [num_nodes, 1, 1]
            single_eval_pos=single_eval_pos,
            cache_trainset_representation=self.cache_trainset_representation,
        )  # [num_nodes, 1, emsize]

        y_encoded = y_encoded.transpose(0, 1)  # [1, num_nodes, emsize]

        # Add feature positional embeddings to x_encoded
        x_encoded = self._add_feature_positional_embeddings(
            x_encoded,
            cache_embeddings=(
                self.cache_trainset_representation and single_eval_pos > 0
            ),
            use_cached_embeddings=(
                self.cache_trainset_representation and single_eval_pos == 0
            ),
        )

        # Concatenate x and y: [1, num_nodes, num_feature_groups, emsize] + [1, num_nodes, 1, emsize]
        # -> [1, num_nodes, num_feature_groups+1, emsize]
        state = torch.cat([x_encoded, y_encoded.unsqueeze(2)], dim=2)

        # Pass through transformer layers
        for layer in self.transformer_layers:
            state = layer(
                state,
                single_eval_pos=single_eval_pos,
                cache_trainset_representation=self.cache_trainset_representation,
            )

        # Extract output for target nodes
        # state: [1, num_nodes, num_feature_groups+1, emsize]
        # We want the y token (last in feature dimension) for target nodes
        target_embeddings = state[0, single_eval_pos:, -1, :]  # [num_target_nodes, emsize]

        # Decode to logits (FIXED n_out dimension)
        logits_full = self.decoder(target_embeddings)  # [num_target_nodes, n_out]

        # Slice to actual number of classes (like TabPFN does)
        # This is the key: foundational model with fixed decoder, slice at inference
        logits = logits_full[:, :num_classes]  # [num_target_nodes, num_classes]

        return logits

    def _add_feature_positional_embeddings(
        self,
        x: torch.Tensor,
        *,
        cache_embeddings: bool = False,
        use_cached_embeddings: bool = False,
    ) -> torch.Tensor:
        """Add feature positional embeddings to distinguish feature groups.

        This is CRITICAL because all feature groups are encoded with SAME weights.
        Without positional embeddings, the model cannot distinguish between groups.

        Args:
            x: Encoded features [batch, num_nodes, num_feature_groups, emsize]
            cache_embeddings: If True, cache the embeddings for reuse
            use_cached_embeddings: If True, use cached embeddings

        Returns:
            x with positional embeddings added [batch, num_nodes, num_feature_groups, emsize]
        """
        # Use cached if available
        if use_cached_embeddings and self.cached_feature_positional_embeddings is not None:
            x = x + self.cached_feature_positional_embeddings[None, None]
            return x

        # Create RNG with fixed seed for reproducibility
        positional_embedding_rng = torch.Generator(device=x.device).manual_seed(
            self.random_embedding_seed
        )

        # Generate embeddings based on mode
        embs: torch.Tensor | None = None

        if self.feature_positional_embedding == "normal_rand_vec":
            # Random Gaussian vectors
            embs = torch.randn(
                (x.shape[2], x.shape[3]),  # [num_feature_groups, emsize]
                device=x.device,
                dtype=x.dtype,
                generator=positional_embedding_rng,
            )
        elif self.feature_positional_embedding == "uni_rand_vec":
            # Random uniform vectors in [-1, 1]
            embs = (
                torch.rand(
                    (x.shape[2], x.shape[3]),  # [num_feature_groups, emsize]
                    device=x.device,
                    dtype=x.dtype,
                    generator=positional_embedding_rng,
                )
                * 2 - 1
            )
        elif self.feature_positional_embedding == "learned":
            # Sample from learned embedding table
            assert self.feature_positional_embedding_embeddings is not None
            w = self.feature_positional_embedding_embeddings.weight
            indices = torch.randint(
                0,
                w.shape[0],
                (x.shape[2],),  # [num_feature_groups]
                generator=positional_embedding_rng,
            )
            embs = w[indices]  # [num_feature_groups, emsize]
        elif self.feature_positional_embedding == "subspace":
            # Default mode: random vectors in low-dimensional subspace
            # This provides enough distinguishability while being memory-efficient
            assert self.feature_positional_embedding_embeddings is not None
            embs = torch.randn(
                (x.shape[2], x.shape[3] // 4),  # [num_feature_groups, emsize//4]
                device=x.device,
                dtype=x.dtype,
                generator=positional_embedding_rng,
            )
            # Project to full emsize
            embs = self.feature_positional_embedding_embeddings(embs)  # [num_feature_groups, emsize]
        elif self.feature_positional_embedding is None:
            # No positional embeddings (NOT RECOMMENDED!)
            embs = None
        else:
            raise ValueError(f"Unknown feature_positional_embedding: {self.feature_positional_embedding}")

        # Add to x
        if embs is not None:
            x = x + embs[None, None]  # Broadcast over batch and num_nodes

        # Cache if requested
        if cache_embeddings and embs is not None:
            self.cached_feature_positional_embeddings = embs

        return x

    def empty_trainset_representation_cache(self) -> None:
        """Clear cached K,V representations and feature positional embeddings."""
        self.cached_feature_positional_embeddings = None
        for layer in self.transformer_layers:
            layer.empty_trainset_representation_cache()
