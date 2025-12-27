"""
Configuration for GraphPFN architecture.

Simplified from TabPFN's ModelConfig, keeping only essential parameters.
"""

from dataclasses import dataclass


@dataclass
class GraphPFNConfig:
    """Configuration for GraphPFN architecture.

    This is a simplified version of TabPFN's ModelConfig,
    adapted for graph learning tasks.
    """

    # ============================================================================
    # Core architecture parameters (from TabPFN defaults)
    # ============================================================================

    emsize: int = 192
    """The embedding dimension. TabPFN default: 192"""

    nhead: int = 6
    """Number of attention heads for both between-item and between-feature attention.
    TabPFN default: 6"""

    nlayers: int = 12
    """Number of transformer layers in the encoder.
    TabPFN default: 12"""

    nhid_factor: int = 4
    """MLP hidden dimension = emsize * nhid_factor.
    TabPFN default: 4 (so 192 * 4 = 768)"""

    features_per_group: int = 2
    """Number of features per group for dual attention.
    The GNN output dimension must be divisible by this.
    TabPFN default: 2"""

    # ============================================================================
    # Encoder parameters
    # ============================================================================

    dropout: float = 0.0
    """Dropout probability. TabPFN default: 0.0 (no dropout)"""

    encoder_type: str = "linear"
    """Type of input encoder. Only 'linear' is supported for now."""

    encoder_use_bias: bool = True
    """Whether to use bias in encoder linear layers. TabPFN default: False
    but we use True for simplicity."""

    fourier_feature_scale: float = 1.0
    """Scale parameter for Fourier Feature encoder (used when features_per_group=1).
    Controls the standard deviation of random frequencies. Higher values make
    the encoding more sensitive to fine-grained value differences."""

    # ============================================================================
    # NaN handling
    # ============================================================================

    nan_handling_enabled: bool = True
    """Enable NaN handling for input features. TabPFN default: True"""

    nan_handling_y_encoder: bool = True
    """Enable NaN handling for target labels. TabPFN default: True"""

    # ============================================================================
    # Normalization
    # ============================================================================

    normalize_x: bool = True
    """Whether to normalize input features. TabPFN default: True"""

    # ============================================================================
    # Attention parameters
    # ============================================================================

    attention_between_features: bool = True
    """Enable attention between feature groups (dual attention).
    This is the key innovation from TabPFN."""

    attention_init_gain: float = 1.0
    """Gain for attention weight initialization. TabPFN default: 1.0"""

    # ============================================================================
    # Feature positional embeddings (IMPORTANT!)
    # ============================================================================

    feature_positional_embedding: str = "subspace"
    """Type of feature positional embedding. TabPFN default: 'subspace'
    Options: 'subspace', 'learned', 'normal_rand_vec', 'uni_rand_vec', None
    This is CRITICAL for distinguishing feature groups since encoder weights are shared."""

    seed: int = 0
    """Random seed for feature positional embeddings. TabPFN default: 0"""

    # ============================================================================
    # Task-specific parameters
    # ============================================================================

    n_out: int = 10
    """Number of output classes (for classification).
    This will be set dynamically based on the dataset."""

    # ============================================================================
    # Parameters we SKIP from TabPFN
    # ============================================================================

    # NOT INCLUDED (using simplified versions or omitting):
    # - multiquery_item_attention: Skip GQA optimization
    # - multiquery_item_attention_for_test_set: Skip optimization
    # - recompute_attn: Skip gradient checkpointing
    # - recompute_layer: Skip gradient checkpointing
    # - remove_empty_features: Can handle at preprocessing
    # - remove_duplicate_features: Can handle at preprocessing
    # - remove_outliers: Can handle at preprocessing
    # - normalize_by_used_features: Skip advanced normalization
    # - normalize_on_train_only: Skip (always normalize on train only)
    # - normalize_to_ranking: Skip advanced normalization
    # - use_separate_decoder: Skip (use simple decoder)
    # - dag_pos_enc_dim: Not relevant for graphs
    # - num_thinking_rows: Skip thinking tokens
    # - item_attention_type: Only support "full"
    # - feature_attention_type: Only support "full"
    # - use_separate_decoder: Skip encoder-decoder architecture

    def __post_init__(self):
        """Validate configuration."""
        assert self.emsize % self.nhead == 0, \
            f"emsize ({self.emsize}) must be divisible by nhead ({self.nhead})"
        assert self.features_per_group > 0, \
            f"features_per_group must be positive, got {self.features_per_group}"
        assert self.nhid_factor > 0, \
            f"nhid_factor must be positive, got {self.nhid_factor}"
        assert self.nlayers > 0, \
            f"nlayers must be positive, got {self.nlayers}"
        assert self.encoder_type == "linear", \
            f"Only 'linear' encoder is supported, got {self.encoder_type}"
