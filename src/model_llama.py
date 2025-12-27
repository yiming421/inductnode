"""
Llama-based Transformer Predictor for fixing anisotropy problem.

Uses Hugging Face's LlamaModel with:
- RMSNorm instead of LayerNorm
- SwiGLU activation instead of ReLU
- Small initialization (std=0.02) instead of Xavier
- No positional embeddings (disabled RoPE)

Supports full token formulation like the original model:
- Token concatenation: [node_emb, label_emb] for context, [node_emb, padding] for target
- Zero padding only (simplified, no MLP padding)
- Multiple embedding extraction modes: full, first_half, second_half
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoOpRotaryEmbedding(nn.Module):
    """
    No-Op Rotary Position Embedding that returns identity transformation.

    Used to disable positional encoding in Llama when we don't need it
    (e.g., for set-based in-context learning where order doesn't matter).

    Returns cos=1, sin=0 so that:
        q_embed = (q * cos) + (rotate(q) * sin) = q * 1 + rotate(q) * 0 = q
        k_embed = (k * cos) + (rotate(k) * sin) = k
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        # No inv_freq needed since we're returning identity

    def forward(self, x, seq_len=None):
        """
        Args:
            x: Input tensor [batch, num_heads, seq_len, head_dim]
            seq_len: Sequence length (optional)

        Returns:
            cos: Tensor of ones [1, 1, seq_len, head_dim]
            sin: Tensor of zeros [1, 1, seq_len, head_dim]
        """
        device = x.device
        dtype = x.dtype

        # Get actual sequence length from input
        if seq_len is None:
            seq_len = x.shape[2] if x.dim() == 4 else x.shape[1]

        # Return identity rotation: cos=1, sin=0
        # Shape: [1, 1, seq_len, head_dim] for broadcasting
        cos = torch.ones((1, 1, seq_len, self.dim), device=device, dtype=dtype)
        sin = torch.zeros((1, 1, seq_len, self.dim), device=device, dtype=dtype)

        return cos, sin


class LlamaPFNPredictorNodeCls(nn.Module):
    """
    Node classification predictor using Llama Transformer architecture.

    Uses masked self-attention to implement cross-attention pattern:
    - Context tokens can attend to all context tokens (bidirectional)
    - Target tokens can attend to context tokens (cross-attention)
    - Target tokens attend only to themselves, not other targets

    This fixes the anisotropy problem by using:
    - RMSNorm instead of LayerNorm
    - SwiGLU activation instead of ReLU
    - Proper small initialization (std=0.02)
    - NO positional embeddings (disabled RoPE for set-based learning)

    Supports full token formulation like original:
    - Token concatenation with class prototypes
    - Zero padding only (simplified)
    - Multiple embedding extraction modes
    """

    def __init__(self, hidden_dim, nhead=1, num_layers=2, mlp_layers=2, dropout=0.2,
                 norm=False, degree=False, att=None, mlp=None, sim='dot',
                 norm_affine=True, normalize=False,
                 # Node classification ridge regression
                 nc_sim='dot', nc_ridge_alpha=1.0,
                 head_num_layers=2,
                 # Llama-specific parameters
                 llama_num_heads=1, llama_intermediate_size=None,
                 # Token formulation parameters (matching original)
                 skip_token_formulation=False,
                 use_full_embedding=False,
                 use_first_half_embedding=False):
        """
        Args:
            hidden_dim: Hidden dimension size
            nhead: Number of heads (legacy, not used - use llama_num_heads instead)
            num_layers: Number of Llama layers
            mlp_layers: MLP layers (legacy, not used in Llama)
            dropout: Dropout rate
            norm: Normalization flag (legacy)
            degree: Degree normalization flag (passed to task head)
            att: Attention pooling module (passed to task head)
            mlp: MLP pooling module (passed to task head)
            sim: Similarity metric (legacy)
            norm_affine: Norm affine flag (passed to task head)
            normalize: Normalization flag (passed to task head)
            nc_sim: Node classification similarity metric (dot/cos/euclidean/mlp/ridge)
            nc_ridge_alpha: Ridge regression alpha
            head_num_layers: Number of layers in task head
            llama_num_heads: Number of attention heads in Llama
            llama_intermediate_size: FFN intermediate size (default: 4*hidden_dim)
            skip_token_formulation: Skip token formulation (use GNN embeddings directly)
            use_full_embedding: Use full 2*hidden_dim output
            use_first_half_embedding: Use first half of output
        """
        super().__init__()
        from transformers import LlamaConfig, LlamaModel

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.nc_sim = nc_sim
        self.nc_ridge_alpha = nc_ridge_alpha
        self.head_num_layers = head_num_layers

        # Token formulation options (matching original)
        self.skip_token_formulation = skip_token_formulation
        self.use_full_embedding = use_full_embedding
        self.use_first_half_embedding = use_first_half_embedding

        # Store parameters for task head (exactly as original)
        self.degree = degree
        self.att = att
        self.mlp_pool = mlp
        self.normalize = normalize

        # Determine Llama's actual hidden size
        if skip_token_formulation:
            llama_hidden_size = hidden_dim
        else:
            llama_hidden_size = hidden_dim * 2  # Token formulation doubles dimension

        # Llama configuration
        if llama_intermediate_size is None:
            llama_intermediate_size = llama_hidden_size * 4  # SwiGLU standard expansion

        self.config = LlamaConfig(
            vocab_size=1,  # Minimal vocab (we don't use embedding layer, avoid wasting 128MB VRAM)
            hidden_size=llama_hidden_size,
            intermediate_size=llama_intermediate_size,
            num_attention_heads=llama_num_heads,
            num_hidden_layers=num_layers,
            num_key_value_heads=llama_num_heads,  # Can be < num_heads for GQA
            rms_norm_eps=1e-6,
            initializer_range=0.02,  # Small initialization (vs Xavier's ~0.088)
            rope_theta=10000.0,
            attention_bias=False,  # Modern architecture: bias-free attention (Llama 3, DeepSeek)
            attention_dropout=dropout,
            max_position_embeddings=2048,  # Max sequence length
            use_cache=False,  # Don't cache KV for training
            hidden_act='silu',  # SwiGLU uses SiLU
        )

        # Llama model (no embedding layer, we provide embeddings directly)
        self.llama = LlamaModel(self.config)

        # Replace RoPE with No-Op version (we don't need positional info for set-based learning)
        for layer in self.llama.layers:
            head_dim = layer.self_attn.head_dim
            layer.self_attn.rotary_emb = NoOpRotaryEmbedding(
                dim=head_dim,
                max_position_embeddings=self.config.max_position_embeddings,
                base=self.config.rope_theta
            )

        # Task-specific head (same as original)
        from .model import NodeClassificationHead

        # Determine head input dimension
        if skip_token_formulation:
            head_hidden_dim = hidden_dim
        else:
            if use_full_embedding:
                head_hidden_dim = hidden_dim * 2
            else:
                head_hidden_dim = hidden_dim

        self.nc_head = NodeClassificationHead(
            hidden_dim=head_hidden_dim,
            dropout=dropout,
            norm=norm,
            norm_affine=norm_affine,
            sim=self.nc_sim,
            ridge_alpha=self.nc_ridge_alpha,
            head_num_layers=self.head_num_layers
        )

    def forward(self, data, context_x, target_x, context_y, class_x, task_type='node_classification'):
        """
        Forward pass for node classification.

        Args:
            data: PyG batch object (for task head compatibility)
            context_x: [num_context, hidden_dim] - Context node embeddings from GNN
            target_x: [num_target, hidden_dim] - Target node embeddings from GNN
            context_y: [num_context] - Context labels
            class_x: [num_classes, hidden_dim] - Class prototypes
            task_type: Task type (only 'node_classification' supported)

        Returns:
            logits: [num_target, num_classes] - Classification logits
            class_emb: [num_classes, hidden_dim] - Class prototypes
        """
        if task_type != 'node_classification':
            raise NotImplementedError(f"LlamaPFNPredictorNodeCls only supports node_classification, got {task_type}")

        num_context = context_x.size(0)
        num_target = target_x.size(0)

        # Step 1: Token formulation (matching original model)
        if self.skip_token_formulation:
            # Simple path: Use GNN embeddings directly
            context_tokens = context_x  # [num_context, hidden_dim]
            target_tokens = target_x    # [num_target, hidden_dim]
        else:
            # Full token formulation path
            # Context tokens: [context_x, class_x[context_y]]
            class_x_y = class_x[context_y]  # [num_context, hidden_dim]
            context_tokens = torch.cat([context_x, class_x_y], dim=1)  # [num_context, 2*hidden_dim]

            # Target tokens: [target_x, zero_padding]
            padding = torch.zeros_like(target_x)  # [num_target, hidden_dim]
            target_tokens = torch.cat([target_x, padding], dim=1)  # [num_target, 2*hidden_dim]

        # Step 2: Concatenate into single sequence and add batch dimension
        all_tokens = torch.cat([context_tokens, target_tokens], dim=0)  # [num_context+num_target, hidden]
        all_tokens = all_tokens.unsqueeze(0)  # [1, seq_len, hidden] - add batch dimension

        # Step 3: Create attention mask for cross-attention pattern
        attention_mask = self._create_attention_mask(num_context, num_target, device=all_tokens.device)

        # Step 4: Forward through Llama
        outputs = self.llama(
            inputs_embeds=all_tokens,
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=True,
        )

        # Step 5: Extract embeddings (split back to context and target)
        all_embeddings = outputs.last_hidden_state[0]  # [seq_len, hidden]
        context_tokens_out = all_embeddings[:num_context, :]  # [num_context, hidden]
        target_tokens_out = all_embeddings[num_context:, :]  # [num_target, hidden]

        # Step 6: Extract label embeddings (matching original logic)
        if self.skip_token_formulation:
            # Simple path: embeddings are already the right size
            context_label_emb = context_tokens_out  # [num_context, hidden_dim]
            target_label_emb = target_tokens_out    # [num_target, hidden_dim]
        else:
            # Token formulation path: choose which part to use
            if self.use_full_embedding:
                # Use full embedding (both halves)
                context_label_emb = context_tokens_out  # [num_context, 2*hidden_dim]
                target_label_emb = target_tokens_out    # [num_target, 2*hidden_dim]
            elif self.use_first_half_embedding:
                # Use first half (node features part)
                context_label_emb = context_tokens_out[:, :self.hidden_dim]  # [num_context, hidden_dim]
                target_label_emb = target_tokens_out[:, :self.hidden_dim]    # [num_target, hidden_dim]
            else:
                # Use second half (label embedding part) - default behavior
                context_label_emb = context_tokens_out[:, self.hidden_dim:]  # [num_context, hidden_dim]
                target_label_emb = target_tokens_out[:, self.hidden_dim:]    # [num_target, hidden_dim]

        # Step 7: Task head (exactly as original)
        logits, class_emb = self.nc_head(
            target_label_emb=target_label_emb,
            context_label_emb=context_label_emb,
            context_y=context_y,
            data=data,
            degree_normalize=self.degree,
            attention_pool_module=self.att,
            mlp_module=self.mlp_pool,
            normalize=self.normalize
        )

        return logits, class_emb

    def _create_attention_mask(self, num_context, num_target, device):
        """
        Create attention mask for cross-attention pattern.

        Mask structure:
        - Context tokens [0:num_context]: Can attend to ALL context tokens ONLY (not targets)
        - Target token i [num_context+i]: Can attend to ALL context + itself ONLY (not other targets)

        Args:
            num_context: Number of context tokens
            num_target: Number of target tokens
            device: Device to create mask on

        Returns:
            attention_mask: [1, 1, seq_len, seq_len]
                           0.0 = attend, -inf = mask out
        """
        total_len = num_context + num_target

        # Start with all positions masked (1 = cannot attend)
        mask = torch.ones(total_len, total_len, dtype=torch.float32, device=device)

        # Context can attend to context
        mask[:num_context, :num_context] = 0

        # Target can attend to all context
        mask[num_context:, :num_context] = 0

        # Target can attend to itself only (diagonal)
        # Use in-place fill_diagonal_ to avoid creating torch.eye(num_target) matrix
        # First, set target-target region to 1 (masked), then set diagonal to 0 (attend)
        mask[num_context:, num_context:].fill_diagonal_(0)

        # Convert to attention mask: 0 -> 0.0 (attend), 1 -> -inf (mask)
        attention_mask = mask.masked_fill(mask == 1, float('-inf'))
        attention_mask = attention_mask.masked_fill(mask == 0, 0.0)

        # Shape: [batch_size, 1, seq_len, seq_len]
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)

        return attention_mask

    def get_target_embeddings(self, context_x, target_x, context_y, class_x):
        """
        Extract Transformer-generated target embeddings without task head.
        Used for contrastive augmentation loss.

        Args:
            context_x: [num_context, hidden_dim] - Context embeddings
            target_x: [num_target, hidden_dim] - Target embeddings
            context_y: [num_context] - Context labels
            class_x: [num_classes, hidden_dim] - Class prototypes

        Returns:
            target_embeddings: [num_target, hidden_dim or 2*hidden_dim] - Transformer output embeddings
        """
        num_context = context_x.size(0)
        num_target = target_x.size(0)

        # Token formulation (same as forward)
        if self.skip_token_formulation:
            context_tokens = context_x
            target_tokens = target_x
        else:
            class_x_y = class_x[context_y]
            context_tokens = torch.cat([context_x, class_x_y], dim=1)

            padding = torch.zeros_like(target_x)
            target_tokens = torch.cat([target_x, padding], dim=1)

        all_tokens = torch.cat([context_tokens, target_tokens], dim=0).unsqueeze(0)
        attention_mask = self._create_attention_mask(num_context, num_target, device=all_tokens.device)

        outputs = self.llama(
            inputs_embeds=all_tokens,
            attention_mask=attention_mask,
            return_dict=True,
        )

        target_tokens_out = outputs.last_hidden_state[0, num_context:, :]

        # Extract embeddings (same logic as forward)
        if self.skip_token_formulation:
            target_embeddings = target_tokens_out
        else:
            if self.use_full_embedding:
                target_embeddings = target_tokens_out
            elif self.use_first_half_embedding:
                target_embeddings = target_tokens_out[:, :self.hidden_dim]
            else:
                target_embeddings = target_tokens_out[:, self.hidden_dim:]

        return target_embeddings
