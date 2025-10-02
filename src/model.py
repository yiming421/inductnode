import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, MessagePassing, SAGEConv, GATConv
from torch_geometric.utils import degree
from torch_sparse.matmul import spmm_add
from torch_scatter import scatter_softmax
from .utils import process_node_features

def get_activation_fn(activation_name):
    """Get activation function by name."""
    if activation_name == 'relu':
        return F.relu
    elif activation_name == 'gelu':
        return F.gelu
    elif activation_name == 'silu':
        return F.silu
    else:
        raise ValueError(f"Unsupported activation function: {activation_name}")

class MoELayer(nn.Module):
    """
    Mixture of Experts Layer for Transformer FFN.

    Args:
        hidden_dim: Input/output dimension
        expert_dim: Hidden dimension of each expert (typically 4 * hidden_dim)
        num_experts: Number of expert networks
        top_k: Number of experts to route to (typically 1 or 2)
        mlp_layers: Number of layers in each expert MLP
        dropout: Dropout rate
        norm: Whether to use normalization in experts
        norm_affine: Whether to use learnable affine parameters in norm
        expert_capacity_factor: Controls expert capacity for load balancing
        use_auxiliary_loss: Whether to add auxiliary loss for load balancing
    """
    def __init__(self, hidden_dim, expert_dim=None, num_experts=4, top_k=2, mlp_layers=2,
                 dropout=0.2, norm=False, norm_affine=True, expert_capacity_factor=1.25,
                 use_auxiliary_loss=True, gating_bias=True):
        super(MoELayer, self).__init__()

        self.hidden_dim = hidden_dim
        self.expert_dim = expert_dim or (4 * hidden_dim)
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity_factor = expert_capacity_factor
        self.use_auxiliary_loss = use_auxiliary_loss

        # Gating network - learns which experts to route to
        self.gating = nn.Linear(hidden_dim, num_experts, bias=gating_bias)

        # Expert networks - multiple specialized FFNs
        self.experts = nn.ModuleList([
            MLP(
                in_channels=hidden_dim,
                hidden_channels=self.expert_dim,
                out_channels=hidden_dim,
                num_layers=mlp_layers,
                dropout=dropout,
                norm=norm,
                tailact=False,
                norm_affine=norm_affine
            ) for _ in range(num_experts)
        ])

        # Load balancing
        self.expert_counts = None
        self.expert_gates = None

    def forward(self, x):
        """
        Forward pass with expert routing.

        Args:
            x: Input tensor [seq_len, batch_size, hidden_dim] or [batch_size, hidden_dim]

        Returns:
            output: MoE output [same shape as x]
            auxiliary_loss: Load balancing loss (if enabled)
        """
        original_shape = x.shape
        # Flatten to [batch_size * seq_len, hidden_dim] for easier processing
        x_flat = x.view(-1, self.hidden_dim)
        batch_size = x_flat.size(0)

        # Gating network: decide which experts to route to
        gate_logits = self.gating(x_flat)  # [batch_size, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)

        # Top-k gating: select top_k experts for each token
        top_k_gates, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        top_k_gates = top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)  # Renormalize

        # Initialize output
        output = torch.zeros_like(x_flat)

        # Route to experts
        for i in range(self.num_experts):
            # Find tokens that should be routed to expert i
            expert_mask = (top_k_indices == i).any(dim=-1)

            if expert_mask.any():
                # Get tokens for this expert
                expert_tokens = x_flat[expert_mask]

                # Get corresponding gates (weights) for this expert
                expert_gate_indices = (top_k_indices[expert_mask] == i).nonzero(as_tuple=True)[1]
                expert_gates = top_k_gates[expert_mask, expert_gate_indices]

                # Process through expert
                expert_output = self.experts[i](expert_tokens)

                # Apply gating weights and accumulate
                weighted_output = expert_output * expert_gates.unsqueeze(-1)
                output[expert_mask] += weighted_output

        # Reshape back to original shape
        output = output.view(original_shape)

        # Auxiliary loss for load balancing (optional)
        auxiliary_loss = self._compute_auxiliary_loss(gate_probs) if self.use_auxiliary_loss else 0.0

        return output, auxiliary_loss

    def _compute_auxiliary_loss(self, gate_probs):
        """
        Compute auxiliary loss to encourage load balancing across experts.
        Based on the original Switch Transformer paper.
        """
        # Fraction of tokens routed to each expert
        expert_counts = gate_probs.sum(dim=0)  # [num_experts]
        expert_fractions = expert_counts / expert_counts.sum()

        # Gate probabilities mean across all tokens
        gate_means = gate_probs.mean(dim=0)  # [num_experts]

        # Auxiliary loss: encourage uniform distribution
        auxiliary_loss = self.num_experts * torch.sum(expert_fractions * gate_means)

        return auxiliary_loss

    def get_expert_usage_stats(self):
        """Get statistics about expert usage for monitoring."""
        if self.expert_counts is not None:
            return {
                'expert_counts': self.expert_counts.cpu().numpy(),
                'expert_utilization': (self.expert_counts / self.expert_counts.sum()).cpu().numpy(),
                'max_expert_usage': self.expert_counts.max().item(),
                'min_expert_usage': self.expert_counts.min().item(),
            }
        return None

class PureGCNConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, adj_t):
        norm = torch.rsqrt_((1+adj_t.sum(dim=-1))).reshape(-1, 1)
        x = norm * x
        x = spmm_add(adj_t, x) + x
        x = norm * x
        return x

class PureGCN(nn.Module):
    def __init__(self, num_layers=2) -> None:
        super().__init__()
        self.conv = PureGCNConv()
        self.num_layers = num_layers

    def forward(self, x, adj_t):
        for _ in range(self.num_layers):
            x = self.conv(x, adj_t)
        return x

class PureGCN_v1(nn.Module):
    def __init__(self, input_dim, num_layers=2, hidden=256, dp=0, norm=False, res=False,
                 relu=False, norm_affine=True, activation='relu'):
        super().__init__()

        # Input projection
        self.lin = nn.Linear(input_dim, hidden) if input_dim != hidden else nn.Identity()

        # GCN Convolution Layer
        self.conv = PureGCNConv()
        self.num_layers = num_layers
        self.dp = dp
        self.norm = norm
        self.res = res
        self.relu = relu  # Keep for backward compatibility

        # Handle activation function
        if relu:  # If relu is True, use activation function
            self.activation_fn = get_activation_fn(activation)
        else:  # If relu is False, no activation (linear GNN)
            self.activation_fn = None

        # Use separate LayerNorm instances per layer if normalization is enabled
        if self.norm:
            self.norms = nn.ModuleList([nn.LayerNorm(hidden, elementwise_affine=norm_affine) for _ in range(num_layers)])

    def forward(self, x, adj_t):
        x = self.lin(x)  # Apply input projection
        ori = x
        for i in range(self.num_layers):
            if i != 0:
                if self.res:
                    x = x + ori  # Memory-efficient: reuses x's memory when possible
                if self.norm:
                    x = self.norms[i](x)  # Apply per-layer normalization
                if self.activation_fn is not None:
                    x = self.activation_fn(x)
                if self.dp > 0:
                    x = F.dropout(x, p=self.dp, training=self.training)  # Safe dropout
            x = self.conv(x, adj_t)  # Apply GCN convolution
        return x

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2,
                 norm=False, tailact=False, norm_affine=True):
        super(MLP, self).__init__()
        self.lins = torch.nn.Sequential()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        if norm:
            self.lins.append(nn.LayerNorm(hidden_channels, elementwise_affine=norm_affine))
        self.lins.append(nn.ReLU())
        if dropout > 0:
            self.lins.append(nn.Dropout(dropout))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            if norm:
                self.lins.append(nn.LayerNorm(hidden_channels, elementwise_affine=norm_affine))
            self.lins.append(nn.ReLU())
            if dropout > 0:
                self.lins.append(nn.Dropout(dropout))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        if tailact:
            self.lins.append(nn.LayerNorm(out_channels, elementwise_affine=norm_affine))
            self.lins.append(nn.ReLU())
            self.lins.append(nn.Dropout(dropout))

    def forward(self, x):
        x = self.lins(x)
        return x.squeeze()
    
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, norm=False, relu=False, prop_step=2, dropout=0.2,
                 multilayer=False, use_gin=False, res=False, norm_affine=True, activation='relu'):
        super(GCN, self).__init__()
        self.lin = nn.Linear(in_feats, h_feats) if in_feats != h_feats else nn.Identity()
        self.multilayer = multilayer
        self.use_gin = use_gin
        if multilayer:
            self.convs = nn.ModuleList()
            for _ in range(prop_step):
                if use_gin:
                    self.convs.append(GINConv(MLP(h_feats, h_feats, h_feats, 2, dropout, norm, 
                                                  False, norm_affine)))
                self.convs.append(GCNConv(h_feats, h_feats))
        else:
            if use_gin:
                self.conv = GINConv(MLP(h_feats, h_feats, h_feats, 2, dropout, norm, False, norm_affine))
            else:
                self.conv = GCNConv(h_feats, h_feats)
        self.norm = norm
        self.relu = relu
        self.prop_step = prop_step

        # Handle activation function
        if relu:  # If relu is True, use activation function
            self.activation_fn = get_activation_fn(activation)
        else:  # If relu is False, no activation (linear GNN)
            self.activation_fn = None

        if norm:
            self.norms = nn.ModuleList([nn.LayerNorm(h_feats, elementwise_affine=norm_affine) \
                                        for _ in range(prop_step)])
            self.dp = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.res = res

    def _apply_norm_and_activation(self, x, i):
        if self.norm:
            x = self.norms[i](x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        if self.norm:
            x = self.dp(x)
        return x
    
    def forward(self, in_feat, g):
        h = self.lin(in_feat)
        ori = h
        for i in range(self.prop_step):
            if i != 0:
                if self.res:
                    h = h + ori  # Safe addition - PyTorch optimizes memory when possible
                h = self._apply_norm_and_activation(h, i)
            if self.multilayer:
                h = self.convs[i](h, g)
            else:
                h = self.conv(h, g)
        return h

class EnhancedGCN(nn.Module):
    """
    Enhanced GCN with OGB-style root embeddings and degree normalization tricks.
    This should bridge the performance gap with OGB GCN implementation.
    """
    def __init__(self, in_feats, h_feats, norm=False, relu=False, prop_step=2, dropout=0.2, 
                 multilayer=False, res=False, norm_affine=True, use_root_emb=True):
        super(EnhancedGCN, self).__init__()
        self.lin = nn.Linear(in_feats, h_feats) if in_feats != h_feats else nn.Identity()
        self.multilayer = multilayer
        self.prop_step = prop_step
        self.norm = norm
        self.relu = relu
        self.res = res
        self.use_root_emb = use_root_emb
        
        # OGB-style root embeddings (key representation trick!)
        if use_root_emb:
            self.root_emb = nn.Embedding(1, h_feats)
        
        # GCN layers
        if multilayer:
            self.convs = nn.ModuleList()
            for _ in range(prop_step):
                self.convs.append(GCNConv(h_feats, h_feats))
        else:
            self.conv = GCNConv(h_feats, h_feats)
            
        # Normalization layers
        if norm:
            self.norms = nn.ModuleList([nn.LayerNorm(h_feats, elementwise_affine=norm_affine) 
                                        for _ in range(prop_step)])
            self.dp = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def _apply_norm_and_activation(self, x, i):
        if self.norm:
            x = self.norms[i](x)
        if self.relu:
            x = F.relu(x)
        if self.norm:
            x = self.dp(x)
        return x
    
    def _compute_degree_norm(self, edge_index, num_nodes):
        """Compute OGB-style degree normalization with +1 trick."""
        from torch_geometric.utils import degree
        row, col = edge_index
        deg = degree(row, num_nodes, dtype=torch.float) + 1  # OGB +1 trick
        deg_inv = 1.0 / deg
        deg_inv[deg_inv == float('inf')] = 0
        return deg_inv

    def forward(self, in_feat, adj_t):
        h = self.lin(in_feat)  # Input projection
        
        # Convert SparseTensor to edge_index for degree computation
        edge_index = adj_t.coo()[:2]  # Get edge_index from SparseTensor
        num_nodes = h.size(0)
        
        # Compute degree normalization (OGB style with +1)
        deg_inv = self._compute_degree_norm(edge_index, num_nodes)
        
        ori = h
        for i in range(self.prop_step):
            if i != 0:
                if self.res:
                    h = h + ori
                h = self._apply_norm_and_activation(h, i)
            
            # Standard GCN message passing
            if self.multilayer:
                h_msg = self.convs[i](h, adj_t)
            else:
                h_msg = self.conv(h, adj_t)
            
            # OGB representation trick: Add root embedding residual!
            if self.use_root_emb:
                root_contrib = F.relu(h + self.root_emb.weight) * deg_inv.view(-1, 1)
                h = h_msg + root_contrib
            else:
                h = h_msg
                
        return h


class AblationGCN(nn.Module):
    """
    Configurable GCN for systematic ablation study.
    Can toggle each architectural component to match OGB GCN exactly.
    """
    def __init__(self, in_feats, h_feats, prop_step=2, dropout=0.2, norm_affine=True, 
                 use_root_emb=True, use_ogb_message_passing=False, use_batch_norm=False, 
                 use_layer_linear=False):
        super(AblationGCN, self).__init__()
        
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.prop_step = prop_step
        self.dropout = dropout
        self.use_root_emb = use_root_emb
        self.use_ogb_message_passing = use_ogb_message_passing
        self.use_batch_norm = use_batch_norm
        self.use_layer_linear = use_layer_linear
        
        # Input projection
        self.input_proj = nn.Linear(in_feats, h_feats) if in_feats != h_feats else nn.Identity()
        
        if use_ogb_message_passing:
            # Use OGB-style custom message passing
            self.convs = nn.ModuleList([
                self.OGBGCNConv(h_feats, use_root_emb, use_layer_linear) 
                for _ in range(prop_step)
            ])
        else:
            # Use standard PyG GCNConv
            from torch_geometric.nn import GCNConv
            self.convs = nn.ModuleList([GCNConv(h_feats, h_feats) for _ in range(prop_step)])
            
            # Root embeddings for standard GCN (applied externally)
            if use_root_emb:
                self.root_emb = nn.Embedding(1, h_feats)
        
        # Normalization choice
        if use_batch_norm:
            self.norms = nn.ModuleList([nn.BatchNorm1d(h_feats) for _ in range(prop_step)])
        else:
            self.norms = nn.ModuleList([nn.LayerNorm(h_feats, elementwise_affine=norm_affine) 
                                        for _ in range(prop_step)])
    
    class OGBGCNConv(MessagePassing):
        """Exact replica of OGB GCN layer for ablation"""
        def __init__(self, emb_dim, use_root_emb=True, use_layer_linear=True):
            super().__init__(aggr='add')
            self.use_layer_linear = use_layer_linear
            self.use_root_emb = use_root_emb
            
            if use_layer_linear:
                self.linear = nn.Linear(emb_dim, emb_dim)
                
            if use_root_emb:
                self.root_emb = nn.Embedding(1, emb_dim)
                
        def forward(self, x, edge_index):
            if self.use_layer_linear:
                x = self.linear(x)  # OGB layer-wise projection
            
            # No edge features (zeros)
            edge_embedding = torch.zeros(edge_index.size(1), x.size(-1), device=x.device, dtype=x.dtype)
            
            row, col = edge_index
            
            # OGB normalization with +1 trick  
            deg = degree(row, x.size(0), dtype=x.dtype) + 1
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            
            # Message passing
            msg_result = self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm)
            
            # Root embedding contribution (if enabled)
            if self.use_root_emb:
                root_contrib = F.relu(x + self.root_emb.weight) * (1.0 / deg.view(-1, 1))
                return msg_result + root_contrib
            else:
                return msg_result
                
        def message(self, x_j, edge_attr, norm):
            # OGB message function: norm * F.relu(x_j + edge_attr)
            # Since edge_attr is zeros: norm * F.relu(x_j)
            return norm.view(-1, 1) * F.relu(x_j + edge_attr)
        
        def update(self, aggr_out):
            return aggr_out
    
    def forward(self, in_feat, adj_t):
        # Convert SparseTensor to edge_index
        edge_index = adj_t.coo()[:2]
        
        h = self.input_proj(in_feat)
        
        if self.use_ogb_message_passing:
            # OGB-style processing (root emb handled internally)
            for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
                h = conv(h, edge_index)
                h = norm(h)
                
                if i == len(self.convs) - 1:
                    h = F.dropout(h, self.dropout, training=self.training)
                else:
                    h = F.dropout(F.relu(h), self.dropout, training=self.training)
        else:
            # Standard processing with external root embeddings
            for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
                h_msg = conv(h, adj_t)
                
                # Add root embedding if using standard GCN
                if self.use_root_emb:
                    deg_inv = self._compute_degree_norm(edge_index, h.size(0))
                    root_contrib = F.relu(h + self.root_emb.weight) * deg_inv.view(-1, 1)
                    h = h_msg + root_contrib
                else:
                    h = h_msg
                    
                h = norm(h)
                
                if i == len(self.convs) - 1:
                    h = F.dropout(h, self.dropout, training=self.training)
                else:
                    h = F.dropout(F.relu(h), self.dropout, training=self.training)
        
        return h
    
    def _compute_degree_norm(self, edge_index, num_nodes):
        """OGB-style degree normalization"""
        row, col = edge_index
        deg = degree(row, num_nodes, dtype=torch.float) + 1
        deg_inv = 1.0 / deg
        deg_inv[deg_inv == float('inf')] = 0
        return deg_inv


class LightGCN(nn.Module):
    def __init__(self, in_feats, h_feats, prop_step=2, dropout = 0.2, relu = False, norm=False):
        super(LightGCN, self).__init__()
        self.lin = nn.Linear(in_feats, h_feats) if in_feats != h_feats else nn.Identity()
        self.relu = relu
        self.alphas = nn.Parameter(torch.ones(prop_step))
        self.norm = norm
        
        if self.norm:
            self.norms = nn.ModuleList([nn.LayerNorm(h_feats) for _ in range(prop_step)])
            self.dp = nn.Dropout(dropout)

    def _apply_norm_and_activation(self, x, i):
        if self.norm:
            x = self.norms[i](x)
        if self.relu:
            x = F.relu(x)
        if self.norm:
            x = self.dp(x)
        return x

    def forward(self, in_feat, g):
        in_feat = self.lin(in_feat)

        alpha = F.softmax(self.alphas, dim=0)
        h = self.conv1(in_feat, g).flatten(1)
        res = h * alpha[0]
        for i in range(1, self.prop_step):
            h = self._apply_norm_and_activation(h, i)
            h = self.conv2(h, g).flatten(1)
            res += h * alpha[i]
        return res
    
class MLPPredictor(nn.Module):
    def __init__(self, h_feats, out_feats, dropout, layer=2, res=False, norm=False, scale=False):
        super().__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(h_feats, h_feats))
        for _ in range(layer - 2):
            self.lins.append(torch.nn.Linear(h_feats, h_feats))
        self.lins.append(torch.nn.Linear(h_feats, out_feats))
        self.dropout = dropout
        self.res = res
        self.scale = scale
        if scale:
            self.scale_norm = nn.LayerNorm(h_feats)
        self.norm = norm
        if norm:
            self.norms = torch.nn.ModuleList()
            for _ in range(layer - 1):
                self.norms.append(nn.LayerNorm(h_feats))

    def forward(self, x):
        if self.scale:
            x = self.scale_norm(x)
        ori = x
        for i in range(len(self.lins) - 1):
            x = self.lins[i](x)
            if self.res:
                x += ori
            if self.norm:
                x = self.norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x.squeeze()

class Prodigy_Predictor(nn.Module):
    def __init__(self, norm=False):
        super().__init__()
        self.norm = norm

    def forward(self, x, class_x):
        if self.norm:
            x = F.normalize(x, p=2, dim=-1)
            class_x = F.normalize(class_x, p=2, dim=-1)
        x = torch.matmul(x, class_x.t())
        return x

class Prodigy_Predictor_mlp(nn.Module):
    def __init__(self, h_feats, dropout, layer=2, norm=False, scale=False, seperate=False):
        super().__init__()
        self.seperate = seperate
        if self.seperate:
            self.mlp_x = MLP(h_feats, h_feats, h_feats, layer, dropout, norm)
            self.mlp_class_x = MLP(h_feats, h_feats, h_feats, layer, dropout, norm)
        else:
            self.mlp = MLP(h_feats, h_feats, h_feats, layer, dropout, norm)
        self.norm = norm
        self.scale = scale
        if self.scale:
            if seperate:
                self.scale_norm_x = nn.LayerNorm(h_feats)
                self.scale_norm_class_x = nn.LayerNorm(h_feats)
            else:
                self.scale_norm = nn.LayerNorm(h_feats)

    def forward(self, x, class_x):
        if self.scale:
            x = self.scale_norm(x)
            class_x = self.scale_norm(class_x)
        if self.seperate:
            if self.scale:
                x = self.scale_norm_x(x)
                class_x = self.scale_norm_class_x(class_x)
            x = self.mlp_x(x)
            class_x = self.mlp_class_x(class_x)
        else:
            if self.scale:
                x = self.scale_norm(x)
            x = self.mlp(x)
            class_x = self.mlp(class_x)
        x = torch.matmul(x, class_x.t())
        return x

class PFNTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, n_head=1, mlp_layers=2, dropout=0.2, norm=False,
                 separate_att=False, unsqueeze=False, norm_affine=True, norm_type='post',
                 use_moe=False, num_experts=4, moe_top_k=2, moe_auxiliary_loss_weight=0.01):
        super(PFNTransformerLayer, self).__init__()
        self.hidden_dim = hidden_dim
        if separate_att:
            self.self_att = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=n_head,
                dropout=dropout,
            )
            self.cross_att = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=n_head,
                dropout=dropout,
            )
        else:
            self.self_att = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=n_head,
                dropout=dropout,
            )
        # FFN or MoE layer
        self.use_moe = use_moe
        self.moe_auxiliary_loss_weight = moe_auxiliary_loss_weight

        if use_moe:
            self.ffn = MoELayer(
                hidden_dim=hidden_dim,
                expert_dim=4 * hidden_dim,
                num_experts=num_experts,
                top_k=moe_top_k,
                mlp_layers=mlp_layers,
                dropout=dropout,
                norm=norm,
                norm_affine=norm_affine,
                use_auxiliary_loss=True
            )
        else:
            self.ffn = MLP(
                in_channels=hidden_dim,
                hidden_channels=4 * hidden_dim,
                out_channels=hidden_dim,
                num_layers=mlp_layers,
                dropout=dropout,
                norm=norm,
                tailact=False,
                norm_affine=norm_affine
            )

        self.context_norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=norm_affine)
        self.context_norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=norm_affine)
        self.tar_norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=norm_affine)
        self.tar_norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=norm_affine)
        self.separate_att = separate_att
        self.unsqueeze = unsqueeze
        self.norm_type = norm_type

    def forward(self, x_context, x_target):
        auxiliary_losses = []

        if self.norm_type == 'pre':
            # Pre-norm: LayerNorm before sublayers
            # Context self-attention
            x_context_norm = self.context_norm1(x_context)
            x_context_att, _ = self.self_att(x_context_norm, x_context_norm, x_context_norm)
            x_context = x_context_att + x_context

            # Context FFN
            x_context_norm = self.context_norm2(x_context)
            if self.use_moe:
                x_context_fnn, aux_loss = self.ffn(x_context_norm)
                auxiliary_losses.append(aux_loss)
            else:
                # Store original shape to preserve it
                orig_shape = x_context_norm.shape
                x_context_fnn = self.ffn(x_context_norm)

                # If FFN changed the shape, restore it
                if x_context_fnn.shape != orig_shape:
                    x_context_fnn = x_context_fnn.view(orig_shape)

            x_context = x_context_fnn + x_context

            # Target cross/self-attention (context should now be 3D)
            x_target_norm = self.tar_norm1(x_target)
            context_for_att = x_context

            if self.separate_att:
                x_target_att, _ = self.cross_att(x_target_norm, context_for_att, context_for_att)
            else:
                x_target_att, _ = self.self_att(x_target_norm, context_for_att, context_for_att)
            x_target = x_target_att + x_target

            # Target FFN
            x_target_norm = self.tar_norm2(x_target)
            if self.use_moe:
                x_target_fnn, aux_loss = self.ffn(x_target_norm)
                auxiliary_losses.append(aux_loss)
            else:
                # Store original shape to preserve it
                orig_shape = x_target_norm.shape
                x_target_fnn = self.ffn(x_target_norm)

                # If FFN changed the shape, restore it
                if x_target_fnn.shape != orig_shape:
                    x_target_fnn = x_target_fnn.view(orig_shape)

            x_target = x_target_fnn + x_target

        else:  # post-norm (original behavior)
            # Post-norm: LayerNorm after sublayers
            # Context self-attention
            x_context_att, _ = self.self_att(x_context, x_context, x_context)
            x_context = x_context_att + x_context
            x_context = self.context_norm1(x_context)

            # Context FFN
            if self.use_moe:
                x_context_fnn, aux_loss = self.ffn(x_context)
                auxiliary_losses.append(aux_loss)
            else:
                # Store original shape to preserve it
                orig_shape = x_context.shape
                x_context_fnn = self.ffn(x_context)

                # If FFN changed the shape, restore it
                if x_context_fnn.shape != orig_shape:
                    x_context_fnn = x_context_fnn.view(orig_shape)

            x_context = x_context_fnn + x_context
            x_context = self.context_norm2(x_context)

            # Target cross/self-attention (context should now be 3D)
            context_for_att = x_context

            if self.separate_att:
                x_target_att, _ = self.cross_att(x_target, context_for_att, context_for_att)
            else:
                x_target_att, _ = self.self_att(x_target, context_for_att, context_for_att)
            x_target = x_target_att + x_target
            x_target = self.tar_norm1(x_target)

            # Target FFN
            if self.use_moe:
                x_target_fnn, aux_loss = self.ffn(x_target)
                auxiliary_losses.append(aux_loss)
            else:
                # Store original shape to preserve it
                orig_shape = x_target.shape
                x_target_fnn = self.ffn(x_target)

                # If FFN changed the shape, restore it
                if x_target_fnn.shape != orig_shape:
                    x_target_fnn = x_target_fnn.view(orig_shape)

            if self.unsqueeze:
                x_target_fnn = x_target_fnn.unsqueeze(1)
                # For residual connection, need to match dimensions temporarily
                x_target_expanded = x_target.unsqueeze(1)
                x_target_residual = x_target_fnn + x_target_expanded
                # Squeeze back to 3D for consistency
                x_target = x_target_residual.squeeze(1)
            else:
                x_target = x_target_fnn + x_target
            x_target = self.tar_norm2(x_target)

        # Return auxiliary loss for MoE training
        total_auxiliary_loss = sum(auxiliary_losses) * self.moe_auxiliary_loss_weight if auxiliary_losses else 0.0

        if self.use_moe:
            return x_context, x_target, total_auxiliary_loss
        else:
            return x_context, x_target

class PFNPredictorNodeCls(nn.Module):
    def __init__(self, hidden_dim, nhead=1, num_layers=2, mlp_layers=2, dropout=0.2,
                 norm=False, separate_att=False, degree=False, att=None, mlp=None, sim='dot',
                 padding='zero', norm_affine=True, normalize=False,
                 use_first_half_embedding=False, use_full_embedding=False, norm_type='post',
                 use_moe=False, moe_num_experts=4, moe_top_k=2, moe_auxiliary_loss_weight=0.01):
        super(PFNPredictorNodeCls, self).__init__()
        self.hidden_dim = hidden_dim
        self.d_label = hidden_dim  # Label embedding has the same dimension as node features
        self.embed_dim = hidden_dim + self.d_label  # Token dimension after concatenation
        self.sim = sim
        self.padding = padding
        
        if self.padding == 'mlp':
            self.pad_mlp = MLP(
                in_channels=self.hidden_dim,
                hidden_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                num_layers=2,
                dropout=dropout,
                norm=norm,
                norm_affine=norm_affine
            )
        if self.sim == 'mlp':
            self.sim_mlp = MLP(
                in_channels=self.hidden_dim,
                hidden_channels=self.hidden_dim,
                out_channels=self.hidden_dim,
                num_layers=2,
                dropout=dropout,
                norm=norm,
                norm_affine=norm_affine
            )
        
        # Transformer layers, similar to the original PFNPredictor
        self.transformer_row = nn.ModuleList([
            PFNTransformerLayer(
                hidden_dim=self.embed_dim,
                n_head=nhead,
                mlp_layers=mlp_layers,
                dropout=dropout,
                norm=norm,
                separate_att=separate_att,
                unsqueeze=False,
                norm_type=norm_type,
                use_moe=use_moe,
                num_experts=moe_num_experts,
                moe_top_k=moe_top_k,
                moe_auxiliary_loss_weight=moe_auxiliary_loss_weight
            ) for _ in range(num_layers)
        ])
        self.degree = degree
        self.att = att
        self.mlp_pool = mlp
        self.normalize = normalize
        self.use_first_half_embedding = use_first_half_embedding
        self.use_full_embedding = use_full_embedding
        self.use_moe = use_moe

        # Validate embedding options (only one should be True)
        if sum([use_first_half_embedding, use_full_embedding]) > 1:
            raise ValueError("Only one of use_first_half_embedding or use_full_embedding can be True")
    
    def forward(self, data, context_x, target_x, context_y, class_x, task_type='node_classification'):
        """
        Unified forward pass supporting both node classification and link prediction.
        
        For node classification:
        - data: Graph data with .y and .context_sample attributes
        - context_x: Context node embeddings
        - target_x: Target node embeddings  
        - context_y: Context node labels
        - class_x: Class prototypes
        
        For link prediction:
        - data: Graph data (may not have .y and .context_sample)
        - context_x: Context edge embeddings
        - target_x: Target edge embeddings
        - context_y: Context edge labels (0: no-link, 1: link)
        - class_x: Link prototypes [2, hidden_dim]
        """
        # Step 1: Create context tokens

        class_x_y = class_x[context_y]  # [num_context, hidden_dim]
        context_tokens = torch.cat([context_x, class_x_y], dim=1)  # [num_context, 2*hidden_dim]

        # Step 2: Create target tokens
        if self.padding == 'zero':
            padding = torch.zeros_like(target_x)  # [num_target, hidden_dim]
        elif self.padding == 'mlp':
            padding = self.pad_mlp(target_x)
        else:
            raise ValueError("Invalid padding type. Choose 'zero' or 'mlp'.")
        target_tokens = torch.cat([target_x, padding], dim=1)
        
        # Step 3: Prepare for transformer (add sequence dimension)
        context_tokens = context_tokens.unsqueeze(1)  # [num_context, 1, 2*hidden_dim]
        target_tokens = target_tokens.unsqueeze(1)    # [num_target, 1, 2*hidden_dim]
        
        # Step 4: Process through transformer layers
        total_auxiliary_loss = 0.0
        for layer in self.transformer_row:
            if layer.use_moe:
                context_tokens, target_tokens, aux_loss = layer(context_tokens, target_tokens)
                total_auxiliary_loss += aux_loss
            else:
                context_tokens, target_tokens = layer(context_tokens, target_tokens)
        
        # Step 5: Extract refined label embeddings
        context_tokens = context_tokens.squeeze(1)  # [num_context, 2*hidden_dim]
        target_tokens = target_tokens.squeeze(1)    # [num_target, 2*hidden_dim]

        # Choose which part of the embedding to use for prototype calculation
        if self.use_full_embedding:
            # Use full embedding (both halves) - will be [num_context/target, 2*hidden_dim]
            context_label_emb = context_tokens  # [num_context, 2*hidden_dim]
            target_label_emb = target_tokens    # [num_target, 2*hidden_dim]
        elif self.use_first_half_embedding:
            # Use first half (node features part)
            context_label_emb = context_tokens[:, :self.hidden_dim]  # [num_context, hidden_dim]
            target_label_emb = target_tokens[:, :self.hidden_dim]    # [num_target, hidden_dim]
        else:
            # Use second half (label embedding part) - default behavior
            context_label_emb = context_tokens[:, self.hidden_dim:]  # [num_context, hidden_dim]
            target_label_emb = target_tokens[:, self.hidden_dim:]    # [num_target, hidden_dim]
        
        # Step 6: Compute refined class embeddings (different approaches for different tasks)
        if task_type == 'node_classification':
            # Use process_node_features for node classification
            class_emb = process_node_features(
                context_label_emb,
                data,
                degree_normalize=self.degree,
                attention_pool_module=self.att,
                mlp_module=self.mlp_pool,
                normalize=self.normalize
            )
        elif task_type == 'link_prediction':
            # Use custom pooling for link prediction (no dependency on data.y or data.context_sample)
            device = target_label_emb.device
            num_classes = class_x.size(0)  # Should be 2 for link prediction
            # Use the actual dimension of target_label_emb instead of self.hidden_dim
            actual_hidden_dim = target_label_emb.size(-1)
            class_emb = torch.zeros(num_classes, actual_hidden_dim, device=device, dtype=target_label_emb.dtype)
            
            # Pool refined embeddings for each class
            for class_idx in range(num_classes):
                class_mask = (context_y == class_idx)
                if class_mask.any():
                    if self.att is not None:
                        # Use attention pooling if available
                        class_embeddings = context_label_emb[class_mask]
                        class_labels = torch.zeros(class_embeddings.size(0), dtype=torch.long, device=device)
                        class_emb[class_idx] = self.att(class_embeddings, class_labels, num_classes=1).squeeze(0)
                    else:
                        # Use mean pooling as fallback
                        class_emb[class_idx] = context_label_emb[class_mask].mean(dim=0)
            
            # Apply MLP if specified
            if self.mlp_pool is not None:
                class_emb = self.mlp_pool(class_emb)
            
            # Normalize if specified
            if self.normalize:
                class_emb = F.normalize(class_emb, p=2, dim=-1)
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}. Choose 'node_classification' or 'link_prediction'.")

        # Step 7: Compute logits using similarity
        if self.sim == 'dot':
            logits = torch.matmul(target_label_emb, class_emb.t())
        elif self.sim == 'cos':
            target_label_emb = F.normalize(target_label_emb, p=2, dim=-1)
            class_emb = F.normalize(class_emb, p=2, dim=-1)
            logits = torch.matmul(target_label_emb, class_emb.t())
        elif self.sim == 'mlp':
            target_label_emb = self.sim_mlp(target_label_emb)
            class_emb = self.sim_mlp(class_emb)
            logits = torch.matmul(target_label_emb, class_emb.t())
        else:
            raise ValueError("Invalid similarity type. Choose 'dot', 'cos', or 'mlp'.")
        if self.use_moe and total_auxiliary_loss > 0:
            return logits, class_emb, total_auxiliary_loss
        else:
            return logits, class_emb
    
class PFNPredictorBinaryCls(nn.Module):
    def __init__(self, hidden_dim, nhead=1, num_layers=2, mlp_layers=2, dropout=0.2, norm=False, scale=False,
                padding='zeros', output_target=False, norm_affine=True, norm_type='post'):
        super(PFNPredictorBinaryCls, self).__init__()
        # Store original hidden_dim for feature splitting
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # MLP for initial edge prediction (outputs 1 dimension for label prediction)
        if padding == 'mlp':
            self.mlp = MLP(hidden_dim, hidden_dim, 1, 2, dropout, norm, False, norm_affine)
        
        # Feature scaling
        self.scale = nn.LayerNorm(hidden_dim) if scale else None
        
        # Shared transformer components (dimension = hidden_dim + 1 for label concatenation)

        self.transformer_row = nn.ModuleList([
            PFNTransformerLayer(hidden_dim + 1, n_head=nhead, mlp_layers=2, dropout=dropout,
                                norm=norm, norm_affine=norm_affine, norm_type=norm_type)
            for _ in range(num_layers)
        ])

        # Final prediction head
        self.head = MLP(
            in_channels=hidden_dim + 1,
            hidden_channels=hidden_dim + 1,
            out_channels=1,
            num_layers=mlp_layers,
            dropout=dropout,
            norm=norm,
            tailact=False,
            norm_affine=norm_affine
        )
        self.padding = padding
        self.output_target = output_target
        self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, context_pos_x, context_neg_x, target_x):
        # Feature normalization
        if self.scale is not None:
            context_pos_x = self.scale(context_pos_x)
            context_neg_x = self.scale(context_neg_x)
            target_x = self.scale(target_x)
        
        # 2. Label Concatenation ----------------------------------------------
        # Add label indicators to context
        context_pos_x = torch.cat([
            context_pos_x, 
            torch.ones(context_pos_x.size(0), 1, device=context_pos_x.device)
        ], dim=-1)
        context_neg_x = torch.cat([
            context_neg_x, 
            torch.zeros(context_neg_x.size(0), 1, device=context_neg_x.device)
        ], dim=-1)

        if self.padding == 'mlp':
            target_x_label = self.mlp(target_x).unsqueeze(-1)  # Predict label for target edges
        elif self.padding == 'zeros':
            target_x_label = torch.zeros(target_x.size(0), 1, device=target_x.device)
        else:
            raise ValueError('Unknown padding method:', self.padding)

        target_x = torch.cat([target_x, target_x_label], dim=-1)  # [num_edges, hidden_dim+1]
        context_x = torch.cat([context_pos_x, context_neg_x], dim=0)  # [num_context, hidden_dim+1]

        for layer in self.transformer_row:
            context_x, target_x = layer(context_x, target_x)

        if self.output_target:
            return target_x.squeeze(1)[:, -1].squeeze(-1)
        else:
            return self.head(target_x.squeeze(1)).squeeze(-1)
        
class AttentionPool(nn.Module):
    def __init__(self, in_channels, out_channels, nhead=1, dp=0.2):
        super(AttentionPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nhead = nhead
        self.dp = dp

        self.lin = nn.Linear(in_channels, nhead * out_channels)
        self.att = nn.Linear(out_channels, 1)

    def forward(self, context_h_input, context_y, num_classes=None):
        if num_classes is None:
            if context_y.numel() == 0:
                return torch.empty(0, self.nhead * self.in_channels, device=context_h_input.device)
            num_classes = context_y.max().item() + 1
        
        if context_h_input.numel() == 0:
             return torch.zeros(num_classes, self.nhead * self.in_channels, device=context_h_input.device)

        context_h_ori_dropout = F.dropout(context_h_input, p=self.dp, training=self.training)
        
        context_h_transformed = self.lin(context_h_ori_dropout)
        context_h_transformed = context_h_transformed.view(-1, self.nhead, self.out_channels)
        
        att_score = self.att(context_h_transformed).squeeze(-1)
        att_score = F.leaky_relu(att_score, negative_slope=0.2)
        
        from torch_scatter import scatter_softmax, scatter_add
        att_weights = scatter_softmax(att_score, context_y.long(), dim=0)
        
        att_h = context_h_ori_dropout.unsqueeze(1) * att_weights.unsqueeze(-1)
        
        pooled_h = torch.zeros(num_classes, self.nhead, self.in_channels, 
                               device=context_h_input.device, dtype=context_h_input.dtype)
        
        pooled_h = scatter_add(att_h, context_y, dim=0, out=pooled_h)
        
        final_h = pooled_h.view(num_classes, self.nhead * self.in_channels)
        return final_h
    
class IdentityProjection(nn.Module):
    """
    Simple identity-preserving projection layer
    Maps from small_dim to large_dim by keeping original features + learning extra features
    """
    def __init__(self, small_dim, large_dim):
        super().__init__()
        assert large_dim >= small_dim, f"large_dim ({large_dim}) must be >= small_dim ({small_dim})"
        
        self.small_dim = small_dim
        self.large_dim = large_dim
        
        if large_dim > small_dim:
            # Project only the "extra" dimensions
            extra_dim = large_dim - small_dim
            self.extra_proj = nn.Linear(small_dim, extra_dim)
            
            # Small initialization to start close to identity behavior
            nn.init.xavier_uniform_(self.extra_proj.weight, gain=0.1)
            nn.init.zeros_(self.extra_proj.bias)
        else:
            self.extra_proj = None
    
    def forward(self, x):
        if self.extra_proj is None:
            return x  # No projection needed
        
        # Keep original dimensions + add projected dimensions
        extra_dims = self.extra_proj(x)
        return torch.cat([x, extra_dims], dim=1)
    
class UnifiedGNN(nn.Module):
    """
    Unified GNN model that consolidates GCN, LightGCN, and PureGCN variants.
    
    Args:
        model_type: 'gcn', 'lightgcn', 'puregcn'
        in_feats: Input feature dimension
        h_feats: Hidden feature dimension  
        prop_step: Number of propagation steps
        conv: Convolution type ('GCN', 'SAGE', 'GAT', 'GIN')
        multilayer: Use separate conv layers for each step
        norm: Apply layer normalization
        relu: Apply ReLU activation
        dropout: Dropout rate
        residual: Residual connection strength
        linear: Apply linear transformation after conv
        alpha: Alpha parameter for LightGCN
        exp: Use exponential alpha weights
        res: Residual connections
        supports_edge_weight: Whether model supports edge weights
        no_parameters: Use parameter-free convolutions
    """
    def __init__(self, model_type='gcn', in_feats=128, h_feats=128, prop_step=2,
                 conv='GCN', multilayer=False, norm=False, relu=False, dropout=0.2,
                 residual=1.0, linear=False, alpha=0.5, exp=False, res=False,
                 supports_edge_weight=False, no_parameters=False, input_norm=False, activation='relu'):
        super(UnifiedGNN, self).__init__()
        
        self.model_type = model_type.lower()
        self.conv_type = conv
        self.multilayer = multilayer
        self.norm = norm
        self.relu = relu
        self.prop_step = prop_step

        # Handle activation function
        if relu:  # If relu is True, use activation function
            self.activation_fn = get_activation_fn(activation)
        else:  # If relu is False, no activation (linear GNN)
            self.activation_fn = None

        self.residual = residual
        self.linear = linear
        self.res = res
        self.supports_edge_weight = supports_edge_weight
        self.no_parameters = no_parameters
        self.input_norm = input_norm
        self.dp = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.in_feats = in_feats
        self.h_feats = h_feats
        
        # Convolution layers
        self._build_conv_layers(in_feats, h_feats, dropout)
        
        # LightGCN alpha parameters
        if self.model_type == 'lightgcn':
            if exp:
                self.alphas = nn.Parameter(alpha ** torch.arange(prop_step))
            else:
                self.alphas = nn.Parameter(torch.ones(prop_step))
        
        # Normalization
        if norm:
            self.norms = nn.ModuleList()
            self.norms.append(nn.LayerNorm(in_feats))
            for _ in range(1, prop_step):
                self.norms.append(nn.LayerNorm(h_feats))
        
        # Linear transformations
        if linear:
            # All models use the same MLP structure: h_feats -> h_feats
            self.mlps = nn.ModuleList([MLP(h_feats, h_feats, h_feats, 2, dropout) for _ in range(prop_step)])

    def _build_conv_layers(self, in_feats, h_feats, dropout):
        """Build convolution layers based on configuration."""
        if self.multilayer:
            # Use separate conv layers for each propagation step
            self.convs = nn.ModuleList()
            for i in range(self.prop_step):
                input_dim = in_feats if i == 0 else h_feats
                self.convs.append(self._create_conv_layer(input_dim, h_feats, dropout))
        else:
            # Use single conv layer shared across propagation steps
            self.conv1 = self._create_conv_layer(in_feats, h_feats, dropout)
            self.conv2 = self._create_conv_layer(h_feats, h_feats, dropout)

    def _create_conv_layer(self, in_dim, out_dim, dropout):
        """Create a single convolution layer."""
        if self.conv_type == 'GCN':
            bias = not self.no_parameters
            return GCNConv(in_dim, out_dim, bias=bias)
        elif self.conv_type == 'SAGE':
            bias = not self.no_parameters
            return SAGEConv(in_dim, out_dim, 'mean', bias=bias)
        elif self.conv_type == 'GAT':
            return GATConv(in_dim, out_dim, heads=1, concat=False)
        elif self.conv_type == 'GIN':
            mlp = MLP(in_dim, out_dim, out_dim, 2, dropout, self.norm)
            return GINConv(mlp)
        else:
            raise ValueError(f"Unsupported convolution type: {self.conv_type}")

    def _apply_norm_and_activation(self, x, i):
        """Apply normalization and activation."""
        if self.norm:
            x = self.norms[i](x)
        if self.activation_fn is not None:
            x = self.activation_fn(x)
        if self.dp:
            x = self.dp(x)
        return x
    
    def _apply_conv(self, conv_layer, adj_t, x, e_feat):
        """Apply convolution layer with conditional edge weight support."""
        if self.conv_type == 'GCN' and e_feat is not None:
            # Only GCN supports edge_weight parameter
            return conv_layer(x, adj_t, edge_weight=e_feat).flatten(1)
        else:
            # SAGE, GAT, GIN don't support edge_weight, PyG handles SparseTensor automatically
            return conv_layer(x, adj_t).flatten(1)

    def forward(self, in_feat, adj_t, e_feat=None):
        """Forward pass."""
        # Input projection - LightGCN handles input transformation with conv1, others use lin
        if self.model_type == 'lightgcn':
            # Pass raw input directly to LightGCN since conv1 handles input transformation
            return self._forward_lightgcn(adj_t, in_feat, e_feat)
        else:
            return self._forward_gcn(adj_t, in_feat, e_feat)

    def _forward_lightgcn(self, adj_t, in_feat, e_feat):
        """LightGCN forward pass matching original implementation exactly."""
        alpha = F.softmax(self.alphas, dim=0)
        if self.input_norm:
            in_feat = self._apply_norm_and_activation(in_feat, 0)
        
        if self.multilayer:
            # Use separate conv layers for each step
            h = self._apply_conv(self.convs[0], adj_t, in_feat, e_feat)
            res = h * alpha[0]
            for i in range(1, self.prop_step):
                if self.linear and self.conv_type != 'GIN':
                    h = self.mlps[i](h)
                h = self._apply_norm_and_activation(h, i)
                h = self._apply_conv(self.convs[i], adj_t, h, e_feat)
                res += h * alpha[i]
            return res
        else:
            # Match original LightGCN exactly: conv1 on raw input, then conv2 repeatedly
            h = self._apply_conv(self.conv1, adj_t, in_feat, e_feat)  # Apply conv1 to raw input
            res = h * alpha[0]  # First layer gets alpha[0]
            for i in range(1, self.prop_step):
                if self.linear and self.conv_type != 'GIN':
                    h = self.mlps[i](h)
                h = self._apply_norm_and_activation(h, i)  # Apply norm/activation to previous h
                h = self._apply_conv(self.conv2, adj_t, h, e_feat)  # Apply conv2 to processed h
                res += h * alpha[i]  # Add to result with alpha[i]
            return res

    def _forward_gcn(self, adj_t, in_feat, e_feat):
        if self.input_norm:
            in_feat = self._apply_norm_and_activation(in_feat, 0)
        """Standard GCN forward pass."""
        ori = in_feat
        if self.multilayer:
            h = self._apply_conv(self.convs[0], adj_t, in_feat, e_feat)
            if self.in_feats != self.h_feats:
                ori = h
            # Use separate conv layers for each step
            for i in range(1, self.prop_step):
                if self.linear and hasattr(self, 'mlps') and self.conv_type != 'GIN':
                    h = self.mlps[i](h)
                if self.res:
                    h = h + self.residual * ori
                h = self._apply_norm_and_activation(h, i)

                # Apply convolution
                h = self._apply_conv(self.convs[i], adj_t, h, e_feat)
        else:
            h = self._apply_conv(self.conv1, adj_t, in_feat, e_feat)
            if self.in_feats != self.h_feats:
                ori = h
            # Use the same conv layers for each step
            for i in range(1, self.prop_step):
                if self.linear and hasattr(self, 'mlps') and self.conv_type != 'GIN':
                    h = self.mlps[i](h)
                if self.res:
                    h = h + self.residual * ori
                h = self._apply_norm_and_activation(h, i)

                # Apply convolution
                h = self._apply_conv(self.conv2, adj_t, h, e_feat)
        
        return h