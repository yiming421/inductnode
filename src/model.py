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


class BankOfTags(nn.Module):
    """
    Fixed random tag bank for class labels (like VQ-VAE codebook).

    Key properties:
    - Tags are fixed (never trained) random vectors
    - True orthogonality via QR when hidden_dim >= max_num_classes
    - Normalized random vectors when hidden_dim < max_num_classes
    - Random permutation mapping class_idx -> tag_idx to prevent overfitting
    - Permutation can be refreshed periodically (e.g., every epoch)

    Usage:
        bank = BankOfTags(max_num_classes=200, hidden_dim=128)
        bank.refresh_permutation(num_classes=7, seed=42)  # For dataset with 7 classes
        tags = bank.get_tags(class_indices)  # [num_samples, hidden_dim]
    """
    def __init__(self, max_num_classes, hidden_dim, seed=42):
        """
        Args:
            max_num_classes: Maximum number of classes to support
            hidden_dim: Dimension of each tag vector
            seed: Random seed for reproducibility
        """
        super().__init__()

        torch.manual_seed(seed)

        if hidden_dim >= max_num_classes:
            # True orthogonality via QR decomposition
            # Generate random matrix and orthogonalize
            # Transpose to get correct shape: QR of [hidden_dim, max_num_classes] gives Q [hidden_dim, max_num_classes]
            random_matrix = torch.randn(hidden_dim, max_num_classes)
            Q, R = torch.linalg.qr(random_matrix)
            tags = Q.t()  # Transpose to get [max_num_classes, hidden_dim] orthonormal vectors
        else:
            # Pseudo-orthogonality: normalized random vectors
            # When we have more classes than dimensions, true orthogonality is impossible
            tags = torch.randn(max_num_classes, hidden_dim)
            tags = F.normalize(tags, p=2, dim=1)  # Unit vectors

        # Register as buffer (not parameter) - won't be trained
        self.register_buffer('tags', tags)

        # Permutation mapping: class_idx -> tag_idx (refreshed periodically)
        # Initially identity mapping [0, 1, 2, ..., max_num_classes-1]
        self.register_buffer('permutation', torch.arange(max_num_classes))

    def refresh_permutation(self, num_classes, seed=None):
        """
        Randomly shuffle the class->tag mapping to prevent overfitting.

        This ensures the model can't memorize "tag[0] always means class strawberry".
        Instead, the mapping changes each time it's refreshed.

        Args:
            num_classes: Number of classes in current dataset
            seed: Random seed for reproducibility (optional)
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Random permutation of tag indices
        perm = torch.randperm(self.tags.size(0))[:num_classes]
        self.permutation[:num_classes] = perm

    def get_tags(self, class_indices):
        """
        Lookup tags by class indices through random permutation.

        Args:
            class_indices: Tensor of class indices [num_samples]

        Returns:
            tags: Tensor of tag vectors [num_samples, hidden_dim]
        """
        # Map class indices to tag indices via permutation
        tag_indices = self.permutation[class_indices]
        return self.tags[tag_indices]


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
                 relu=False, norm_affine=True, activation='relu', use_virtual_node=False):
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
        self.use_virtual_node = use_virtual_node

        # Handle activation function
        if relu:  # If relu is True, use activation function
            self.activation_fn = get_activation_fn(activation)
        else:  # If relu is False, no activation (linear GNN)
            self.activation_fn = None

        # Use separate LayerNorm instances per layer if normalization is enabled
        if self.norm:
            self.norms = nn.ModuleList([nn.LayerNorm(hidden, elementwise_affine=norm_affine) for _ in range(num_layers)])

        # Virtual node: learnable embedding added as real node in graph
        if self.use_virtual_node:
            self.virtualnode_embedding = nn.Embedding(1, hidden)
            nn.init.constant_(self.virtualnode_embedding.weight.data, 0)

    def forward(self, x, adj_t, batch=None):
        import time
        x = self.lin(x)  # Apply input projection

        # Add virtual node to graph if enabled (only for graph-level tasks with batch info)
        if self.use_virtual_node and batch is not None:
            num_nodes = x.size(0)
            num_graphs = int(batch.max().item()) + 1

            # Add virtual node embedding for each graph
            virtualnode_emb = self.virtualnode_embedding.weight.repeat(num_graphs, 1)
            x = torch.cat([virtualnode_emb, x], dim=0)

            # Add virtual edges: bidirectional connections between virtual node and all real nodes
            # Virtual node indices: 0, 1, 2, ... (num_graphs-1)
            # Real node indices: num_graphs, num_graphs+1, ..., num_graphs+num_nodes-1
            real_node_indices = torch.arange(num_graphs, num_graphs + num_nodes, device=x.device)
            # Map each real node to its virtual node (use original batch indices)
            virtual_node_indices = batch

            # Create bidirectional edges
            edge_list = []
            edge_list.append(torch.stack([virtual_node_indices, real_node_indices], dim=0))  # vn -> real
            edge_list.append(torch.stack([real_node_indices, virtual_node_indices], dim=0))  # real -> vn
            vn_edges = torch.cat(edge_list, dim=1)

            # Convert adj_t to edge_index, add virtual edges, convert back
            from torch_sparse import SparseTensor
            row, col, edge_attr = adj_t.coo()
            # Shift existing edges by num_graphs
            edge_index_shifted = torch.stack([row + num_graphs, col + num_graphs], dim=0)
            # Combine with virtual edges
            edge_index_full = torch.cat([vn_edges, edge_index_shifted], dim=1)
            # Create new SparseTensor (symmetric and coalesced like original)
            adj_t = SparseTensor(row=edge_index_full[0], col=edge_index_full[1],
                                sparse_sizes=(x.size(0), x.size(0))).to_symmetric().coalesce()

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

        # Add final residual connection after all layers
        if self.res:
            x = x + ori

        # Remove virtual node from output if present
        if self.use_virtual_node and batch is not None:
            virtualnode_out = x[:num_graphs]  # Extract virtual nodes
            x = x[num_graphs:]  # Remove virtual nodes, keep only real nodes
            return x, virtualnode_out

        return x

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2,
                 norm=False, tailact=False, norm_affine=True):
        super(MLP, self).__init__()
        self.lins = torch.nn.Sequential()
        self.num_layers = num_layers

        # Handle num_layers=0: Identity mapping
        if num_layers == 0:
            assert in_channels == out_channels, \
                f"num_layers=0 requires in_channels==out_channels, got {in_channels}!={out_channels}"
            # Empty sequential acts as identity
            pass
        # Handle num_layers=1: Single linear layer
        elif num_layers == 1:
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
            if tailact:
                if norm:
                    self.lins.append(nn.LayerNorm(out_channels, elementwise_affine=norm_affine))
                self.lins.append(nn.ReLU())
                if dropout > 0:
                    self.lins.append(nn.Dropout(dropout))
        # Handle num_layers>=2: Standard MLP with hidden layers
        else:
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

    def _init_identity_like(self):
        """
        Initialize MLP to approximate identity transformation.
        This prevents the head from disrupting well-calibrated embeddings at initialization.

        Strategy:
        - Single layer (num_layers=1): Perfect identity (W=I, b=0)
        - Multi-layer: All layers initialized very small, last layer identity-like
        """
        import torch.nn.init as init

        if self.num_layers == 0:
            return  # No layers to initialize

        # Find all linear layers
        linear_layers = [m for m in self.lins if isinstance(m, nn.Linear)]

        if len(linear_layers) == 0:
            return

        # Initialize ALL layers with very small weights
        for i, linear in enumerate(linear_layers):
            in_feat = linear.in_features
            out_feat = linear.out_features

            if i == len(linear_layers) - 1 and in_feat == out_feat:
                # Final layer with matching dims: Perfect identity
                init.eye_(linear.weight)
            else:
                # All other layers: Very small random init (gain=0.01)
                # This ensures the network is close to identity even with ReLU
                init.xavier_uniform_(linear.weight, gain=0.01)

            if linear.bias is not None:
                init.zeros_(linear.bias)

    def forward(self, x):
        if self.num_layers == 0:
            # Identity mapping - return input as-is
            return x.squeeze()
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


class GraphClassificationMLPHead(nn.Module):
    """Supervised MLP head for graph classification (multi-task or single-task)."""
    def __init__(self, in_dim, out_dim, num_layers=2, dropout=0.2, norm=False, norm_affine=True):
        super().__init__()
        self.out_dim = out_dim
        self.mlp = MLP(
            in_channels=in_dim,
            hidden_channels=in_dim,
            out_channels=out_dim,
            num_layers=num_layers,
            dropout=dropout,
            norm=norm,
            tailact=False,
            norm_affine=norm_affine
        )

    def forward(self, x):
        out = self.mlp(x)
        if out.dim() == 1:
            out = out.unsqueeze(0)
        return out

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

class RidgeRegressionPredictor(nn.Module):
    """
    Ridge Regression Predictor for node classification.

    Uses closed-form ridge regression solution: W = (X^T X + λI)^{-1} X^T Y
    """
    def __init__(self, ridge_alpha=1.0):
        super().__init__()
        self.ridge_alpha = ridge_alpha
        self._debug_preds = None

    def forward(self, data, context_h, target_h, context_y, class_h=None):
        """
        Args:
            data: Data object (for compatibility)
            context_h: Context node embeddings [n_support, dim]
            target_h: Target node embeddings [n_target, dim]
            context_y: Context labels [n_support]
            class_h: Class prototypes (unused for ridge regression)

        Returns:
            logits: [n_target, n_classes] prediction logits
            class_h: Unchanged class prototypes (for compatibility)
        """
        num_classes = context_y.max().item() + 1

        # One-hot encode labels
        support_y = F.one_hot(context_y.long(), num_classes=num_classes).float()  # [n_support, n_classes]

        # Solve ridge regression: W = (X^T X + λI)^{-1} X^T Y
        XtX = context_h.t() @ context_h  # [dim, dim]
        XtY = context_h.t() @ support_y  # [dim, n_classes]

        # Add regularization
        I = torch.eye(context_h.size(1), device=context_h.device)
        W = torch.linalg.solve(XtX + self.ridge_alpha * I, XtY)  # [dim, n_classes]

        # Predict
        logits = target_h @ W  # [n_target, n_classes]

        return logits, class_h

class PFNTransformerLayer(nn.Module):
    def __init__(self, hidden_dim, n_head=1, mlp_layers=2, dropout=0.2, norm=False,
                 separate_att=False, unsqueeze=False, norm_affine=True, norm_type='post',
                 ffn_expansion_ratio=4):
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
        # FFN layer
        self.ffn = MLP(
            in_channels=hidden_dim,
            hidden_channels=ffn_expansion_ratio * hidden_dim,
            out_channels=hidden_dim,
            num_layers=mlp_layers,
            dropout=dropout,
            norm=norm,
            tailact=False,
            norm_affine=norm_affine
        )

        self.context_norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=norm_affine)  # For context self-attention
        self.context_norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=norm_affine)  # For context FFN
        self.context_norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=norm_affine)  # For context as key/value in cross-attention
        self.tar_norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=norm_affine)      # For target cross-attention
        self.tar_norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=norm_affine)      # For target FFN
        self.separate_att = separate_att
        self.unsqueeze = unsqueeze
        self.norm_type = norm_type

    def forward(self, x_context, x_target):
        # DEBUG: Track anisotropy collapse through the layer
        debug_collapse = False
        if hasattr(self, '_debug_collapse_step') and self._debug_collapse_step:
            debug_collapse = True
            import torch.nn.functional as F

            def compute_mean_dominance(x, name):
                """Compute how much each sample points toward the mean"""
                x_squeezed = x.squeeze(1) if x.dim() == 3 else x
                mean_vec = x_squeezed.mean(dim=0, keepdim=True)
                sims = F.cosine_similarity(x_squeezed, mean_vec.expand_as(x_squeezed), dim=-1)
                mean_dom = sims.mean().item()
                norm = x_squeezed.norm(dim=-1).mean().item()
                print(f"    {name}: mean_dom={mean_dom:.4f}, avg_norm={norm:.4f}")
                return mean_dom

        if self.norm_type == 'pre':
            if debug_collapse:
                print(f"\n  [TRANSFORMER LAYER DEBUG - Pre-Norm]")
                x_target_in_dom = compute_mean_dominance(x_target, "Input x_target")

            # Pre-norm: LayerNorm before sublayers
            # Context self-attention
            x_context_norm = self.context_norm1(x_context)
            x_context_att, _ = self.self_att(x_context_norm, x_context_norm, x_context_norm)
            x_context = x_context_att + x_context

            # Context FFN
            x_context_norm = self.context_norm2(x_context)
            # Store original shape to preserve it
            orig_shape = x_context_norm.shape
            x_context_fnn = self.ffn(x_context_norm)

            # If FFN changed the shape, restore it
            if x_context_fnn.shape != orig_shape:
                x_context_fnn = x_context_fnn.view(orig_shape)

            x_context = x_context_fnn + x_context

            # Target cross/self-attention
            # In pre-norm: normalize both query (target) and key/value (context)
            x_target_norm = self.tar_norm1(x_target)

            if debug_collapse:
                after_norm_dom = compute_mean_dominance(x_target_norm, "After LayerNorm")

            context_for_att = self.context_norm3(x_context)  # Normalize context for use as key/value

            if self.separate_att:
                x_target_att, attn_weights = self.cross_att(x_target_norm, context_for_att, context_for_att)
            else:
                x_target_att, attn_weights = self.self_att(x_target_norm, context_for_att, context_for_att)

            if debug_collapse:
                att_out_dom = compute_mean_dominance(x_target_att, "After Cross-Att")
                # Check attention weight uniformity
                if attn_weights is not None:
                    attn_entropy = -(attn_weights * torch.log(attn_weights + 1e-10)).sum(dim=-1).mean().item()
                    max_entropy = torch.log(torch.tensor(attn_weights.size(-1), dtype=torch.float))
                    normalized_entropy = attn_entropy / max_entropy
                    print(f"    Attention entropy: {normalized_entropy:.4f} (1.0=uniform, 0.0=peaked)")

            x_target = x_target_att + x_target

            if debug_collapse:
                after_residual_dom = compute_mean_dominance(x_target, "After residual (att+input)")

            # Target FFN
            x_target_norm = self.tar_norm2(x_target)
            # Store original shape to preserve it
            orig_shape = x_target_norm.shape
            x_target_fnn = self.ffn(x_target_norm)

            # If FFN changed the shape, restore it
            if x_target_fnn.shape != orig_shape:
                x_target_fnn = x_target_fnn.view(orig_shape)

            if debug_collapse:
                ffn_out_dom = compute_mean_dominance(x_target_fnn, "After FFN")

            x_target = x_target_fnn + x_target

            if debug_collapse:
                final_dom = compute_mean_dominance(x_target, "Final output (ffn+input)")
                print(f"    Anisotropy increase: {x_target_in_dom:.4f} → {final_dom:.4f} (+{final_dom-x_target_in_dom:.4f})")

        else:  # post-norm (original behavior)
            # Post-norm: LayerNorm after sublayers
            # Context self-attention
            x_context_att, _ = self.self_att(x_context, x_context, x_context)
            x_context = x_context_att + x_context
            x_context = self.context_norm1(x_context)

            # Context FFN
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

        return x_context, x_target

class NodeClassificationHead(nn.Module):
    """Task-specific head for node classification."""
    def __init__(self, hidden_dim, dropout=0.2, norm=True, norm_affine=True, sim='dot', ridge_alpha=1.0, head_num_layers=2):
        super(NodeClassificationHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.sim = sim
        self.ridge_alpha = ridge_alpha

        # Input normalization layers to stabilize features from transformer
        # Separate norms for target and context since they may have different statistics
        self.target_input_norm = nn.LayerNorm(hidden_dim, elementwise_affine=norm_affine)
        self.context_input_norm = nn.LayerNorm(hidden_dim, elementwise_affine=norm_affine)

        # Initialize LayerNorms to identity-like (weight=1, bias=0) to avoid disrupting embeddings
        if norm_affine:
            nn.init.ones_(self.target_input_norm.weight)
            nn.init.zeros_(self.target_input_norm.bias)
            nn.init.ones_(self.context_input_norm.weight)
            nn.init.zeros_(self.context_input_norm.bias)

        # Task-specific projection MLP
        self.proj = MLP(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=head_num_layers,
            dropout=dropout,
            norm=norm,
            tailact=False,
            norm_affine=norm_affine
        )
        # Initialize to approximate identity to avoid disrupting embeddings
        self.proj._init_identity_like()

        # Optional MLP for similarity computation
        if sim == 'mlp':
            self.sim_mlp = MLP(
                in_channels=hidden_dim,
                hidden_channels=hidden_dim,
                out_channels=hidden_dim,
                num_layers=head_num_layers,
                dropout=dropout,
                norm=norm,
                tailact=False,
                norm_affine=norm_affine
            )
            # Initialize to approximate identity
            self.sim_mlp._init_identity_like()

    def forward(self, target_label_emb, context_label_emb, context_y, data,
                degree_normalize=False, attention_pool_module=None, mlp_module=None, normalize=False):
        """
        Args:
            target_label_emb: [num_target, hidden_dim]
            context_label_emb: [num_context, hidden_dim]
            context_y: [num_context] - labels for context nodes
            data: PyG Data object with .y and .context_sample
            degree_normalize: Whether to apply degree normalization
            attention_pool_module: Optional attention pooling module
            mlp_module: Optional MLP for prototype transformation
            normalize: Whether to normalize embeddings

        Returns:
            logits: [num_target, num_classes]
            class_emb: [num_classes, hidden_dim] - class prototypes
        """
        from .utils import process_node_features

        # Project embeddings through task-specific head (restore 9fa7dd8 behavior)
        target_label_emb = self.proj(target_label_emb)
        context_label_emb = self.proj(context_label_emb)

        # Fix: Ensure 2D shape (MLP.squeeze() can remove batch dim when batch_size=1)
        if target_label_emb.dim() == 1:
            target_label_emb = target_label_emb.unsqueeze(0)
        if context_label_emb.dim() == 1:
            context_label_emb = context_label_emb.unsqueeze(0)

        # Compute logits using similarity or ridge regression
        if self.sim == 'ridge':
            # Ridge Regression: W = (X^T X + λI)^{-1} X^T Y
            num_classes = context_y.max().item() + 1

            # Input validation
            if context_y.numel() == 0:
                raise ValueError("Context labels cannot be empty for ridge regression")
            if self.ridge_alpha <= 0:
                raise ValueError("Ridge alpha must be positive for numerical stability")
            if context_label_emb.size(0) < 2:
                raise ValueError("Ridge regression requires at least 2 support samples")

            # One-hot encode labels
            support_y = F.one_hot(context_y.long(), num_classes=num_classes).float()

            # Solve ridge regression
            XtX = context_label_emb.t() @ context_label_emb
            XtY = context_label_emb.t() @ support_y

            # Add regularization
            I = torch.eye(context_label_emb.size(1), device=context_label_emb.device)
            try:
                W = torch.linalg.solve(XtX + self.ridge_alpha * I, XtY)
            except torch.linalg.LinAlgError as e:
                raise RuntimeError(f"Ridge regression solver failed - try increasing ridge_alpha. Error: {e}")

            # Predict
            logits = target_label_emb @ W
            class_emb = None  # Not used in ridge regression
        else:
            # Compute class prototypes using shared pooling logic (for non-ridge methods)
            class_emb = process_node_features(
                context_label_emb,
                data,
                degree_normalize=degree_normalize,
                attention_pool_module=attention_pool_module,
                mlp_module=mlp_module,
                normalize=normalize
            )

            # Compute logits using similarity
            if self.sim == 'dot':
                logits = torch.matmul(target_label_emb, class_emb.t())
            elif self.sim == 'cos':
                target_label_emb = F.normalize(target_label_emb, p=2, dim=-1)
                class_emb = F.normalize(class_emb, p=2, dim=-1)
                logits = torch.matmul(target_label_emb, class_emb.t())
            elif self.sim == 'euclidean':
                distances = torch.cdist(target_label_emb, class_emb, p=2)
                logits = -distances
            elif self.sim == 'mlp':
                target_label_emb = self.sim_mlp(target_label_emb)
                class_emb = self.sim_mlp(class_emb)
                logits = torch.matmul(target_label_emb, class_emb.t())
            else:
                raise ValueError(f"Invalid similarity type: {self.sim}")

        return logits, class_emb


class LinkPredictionHead(nn.Module):
    """Task-specific head for link prediction."""
    def __init__(self, hidden_dim, dropout=0.2, norm=True, norm_affine=True, sim='dot', ridge_alpha=1.0, head_num_layers=2):
        super(LinkPredictionHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.sim = sim
        self.ridge_alpha = ridge_alpha
        self.head_num_layers = head_num_layers

        if head_num_layers > 0:
            self.linear = nn.Linear(hidden_dim, 1)
            self.proj = None
            self.sim_mlp = None
        else:
            self.linear = None
            self.proj = MLP(
                in_channels=hidden_dim,
                hidden_channels=hidden_dim,
                out_channels=hidden_dim,
                num_layers=head_num_layers,
                dropout=dropout,
                norm=norm,
                tailact=False,
                norm_affine=norm_affine
            )
            self.proj._init_identity_like()
            if sim == 'mlp':
                self.sim_mlp = MLP(
                    in_channels=hidden_dim,
                    hidden_channels=hidden_dim,
                    out_channels=hidden_dim,
                    num_layers=head_num_layers,
                    dropout=dropout,
                    norm=norm,
                    tailact=False,
                    norm_affine=norm_affine
                )
                self.sim_mlp._init_identity_like()

    def forward(self, target_label_emb, context_label_emb, context_y, class_x,
                attention_pool_module=None, mlp_module=None, normalize=False):
        """
        Args:
            target_label_emb: [num_target, hidden_dim]
            context_label_emb: [num_context, hidden_dim]
            context_y: [num_context] - labels (0: no-link, 1: link)
            class_x: [2, hidden_dim] - link prototypes
            attention_pool_module: Optional attention pooling module
            mlp_module: Optional MLP for prototype transformation
            normalize: Whether to normalize embeddings

        Returns:
            logits: [num_target] when head layers enabled, or [num_target, 2] when head is disabled
            class_emb: [2, hidden_dim] - link prototypes
        """
        if self.linear is not None:
            if target_label_emb.dim() == 1:
                target_label_emb = target_label_emb.unsqueeze(0)
            logits = self.linear(target_label_emb).squeeze(-1)
            return logits, class_x

        target_label_emb = self.proj(target_label_emb)
        context_label_emb = self.proj(context_label_emb)

        if target_label_emb.dim() == 1:
            target_label_emb = target_label_emb.unsqueeze(0)
        if context_label_emb.dim() == 1:
            context_label_emb = context_label_emb.unsqueeze(0)

        device = target_label_emb.device
        num_classes = class_x.size(0)
        actual_hidden_dim = target_label_emb.size(-1)
        class_emb = torch.zeros(num_classes, actual_hidden_dim, device=device, dtype=target_label_emb.dtype)

        for class_idx in range(num_classes):
            class_mask = (context_y == class_idx)
            if class_mask.any():
                if attention_pool_module is not None:
                    class_embeddings = context_label_emb[class_mask]
                    class_labels = torch.zeros(class_embeddings.size(0), dtype=torch.long, device=device)
                    class_emb[class_idx] = attention_pool_module(class_embeddings, class_labels, num_classes=1).squeeze(0)
                else:
                    class_emb[class_idx] = context_label_emb[class_mask].mean(dim=0)

        if mlp_module is not None:
            class_emb = mlp_module(class_emb)

        if normalize:
            class_emb = F.normalize(class_emb, p=2, dim=-1)

        if self.sim == 'ridge':
            num_classes = 2
            if context_y.numel() == 0:
                raise ValueError("Context labels cannot be empty for ridge regression")
            if self.ridge_alpha <= 0:
                raise ValueError("Ridge alpha must be positive for numerical stability")
            if context_label_emb.size(0) < 2:
                raise ValueError("Ridge regression requires at least 2 support samples")

            support_y = F.one_hot(context_y.long(), num_classes=num_classes).float()
            XtX = context_label_emb.t() @ context_label_emb
            XtY = context_label_emb.t() @ support_y
            I = torch.eye(context_label_emb.size(1), device=context_label_emb.device)
            try:
                W = torch.linalg.solve(XtX + self.ridge_alpha * I, XtY)
            except torch.linalg.LinAlgError as e:
                raise RuntimeError(f"Ridge regression solver failed - try increasing ridge_alpha. Error: {e}")

            logits = target_label_emb @ W
            class_emb = None
        elif self.sim == 'dot':
            logits = torch.matmul(target_label_emb, class_emb.t())
        elif self.sim == 'cos':
            target_label_emb = F.normalize(target_label_emb, p=2, dim=-1)
            class_emb = F.normalize(class_emb, p=2, dim=-1)
            logits = torch.matmul(target_label_emb, class_emb.t())
        elif self.sim == 'euclidean':
            distances = torch.cdist(target_label_emb, class_emb, p=2)
            logits = -distances
        elif self.sim == 'mlp':
            target_label_emb = self.sim_mlp(target_label_emb)
            class_emb = self.sim_mlp(class_emb)
            logits = torch.matmul(target_label_emb, class_emb.t())
        else:
            raise ValueError(f"Invalid similarity type: {self.sim}")

        return logits, class_emb


class GraphClassificationHead(nn.Module):
    """Task-specific head for graph classification."""
    def __init__(self, hidden_dim, dropout=0.2, norm=True, norm_affine=True, sim='dot', ridge_alpha=1.0, head_num_layers=2):
        super(GraphClassificationHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.sim = sim
        self.ridge_alpha = ridge_alpha

        # Input normalization layers to stabilize features from transformer
        # Separate norms for target and context since they may have different statistics
        self.target_input_norm = nn.LayerNorm(hidden_dim, elementwise_affine=norm_affine)
        self.context_input_norm = nn.LayerNorm(hidden_dim, elementwise_affine=norm_affine)

        # Initialize LayerNorms to identity-like (weight=1, bias=0) to avoid disrupting embeddings
        if norm_affine:
            nn.init.ones_(self.target_input_norm.weight)
            nn.init.zeros_(self.target_input_norm.bias)
            nn.init.ones_(self.context_input_norm.weight)
            nn.init.zeros_(self.context_input_norm.bias)

        # Task-specific projection MLP
        self.proj = MLP(
            in_channels=hidden_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=head_num_layers,
            dropout=dropout,
            norm=norm,
            tailact=False,
            norm_affine=norm_affine
        )
        # Initialize to approximate identity to avoid disrupting embeddings
        self.proj._init_identity_like()

        # Optional MLP for similarity computation
        if sim == 'mlp':
            self.sim_mlp = MLP(
                in_channels=hidden_dim,
                hidden_channels=hidden_dim,
                out_channels=hidden_dim,
                num_layers=head_num_layers,
                dropout=dropout,
                norm=norm,
                tailact=False,
                norm_affine=norm_affine
            )
            # Initialize to approximate identity
            self.sim_mlp._init_identity_like()

    def forward(self, target_label_emb, context_label_emb, context_y, class_x,
                attention_pool_module=None, mlp_module=None, normalize=False):
        """
        Args:
            target_label_emb: [num_target_graphs, hidden_dim] - target graph embeddings
            context_label_emb: [num_context_graphs, hidden_dim] - context graph embeddings
            context_y: [num_context_graphs] - labels for context graphs
            class_x: [num_classes, hidden_dim] - class prototypes
            attention_pool_module: Optional attention pooling module
            mlp_module: Optional MLP for prototype transformation
            normalize: Whether to normalize embeddings

        Returns:
            logits: [num_target_graphs, num_classes]
            class_emb: [num_classes, hidden_dim] - class prototypes
        """
        # TEMPORARY: Completely disable head to test
        # # Normalize inputs before projection (separate norms for target and context)
        # target_label_emb = self.target_input_norm(target_label_emb)
        # context_label_emb = self.context_input_norm(context_label_emb)

        # # Project embeddings through task-specific head
        # target_label_emb = self.proj(target_label_emb)
        # context_label_emb = self.proj(context_label_emb)

        # Fix: Ensure 2D shape (MLP.squeeze() can remove batch dim when batch_size=1)
        if target_label_emb.dim() == 1:
            target_label_emb = target_label_emb.unsqueeze(0)
        if context_label_emb.dim() == 1:
            context_label_emb = context_label_emb.unsqueeze(0)

        # Compute logits using similarity or ridge regression
        if self.sim == 'ridge':
            # Ridge Regression: W = (X^T X + λI)^{-1} X^T Y
            num_classes = context_y.max().item() + 1

            # Input validation
            if context_y.numel() == 0:
                raise ValueError("Context labels cannot be empty for ridge regression")
            if self.ridge_alpha <= 0:
                raise ValueError("Ridge alpha must be positive for numerical stability")
            if context_label_emb.size(0) < 2:
                raise ValueError("Ridge regression requires at least 2 support samples")

            # One-hot encode labels
            support_y = F.one_hot(context_y.long(), num_classes=num_classes).float()

            # Solve ridge regression
            XtX = context_label_emb.t() @ context_label_emb
            XtY = context_label_emb.t() @ support_y

            # Add regularization
            I = torch.eye(context_label_emb.size(1), device=context_label_emb.device)
            try:
                W = torch.linalg.solve(XtX + self.ridge_alpha * I, XtY)
            except torch.linalg.LinAlgError as e:
                raise RuntimeError(f"Ridge regression solver failed - try increasing ridge_alpha. Error: {e}")

            # Predict
            logits = target_label_emb @ W
            class_emb = None  # Not used in ridge regression
        else:
            # Compute class prototypes
            device = target_label_emb.device
            num_classes = context_y.max().item() + 1
            actual_hidden_dim = target_label_emb.size(-1)
            class_emb = torch.zeros(num_classes, actual_hidden_dim, device=device, dtype=target_label_emb.dtype)

            # Pool refined embeddings for each class
            for class_idx in range(num_classes):
                class_mask = (context_y == class_idx)
                if class_mask.any():
                    if attention_pool_module is not None:
                        # Use attention pooling if available
                        class_embeddings = context_label_emb[class_mask]
                        class_labels = torch.zeros(class_embeddings.size(0), dtype=torch.long, device=device)
                        class_emb[class_idx] = attention_pool_module(class_embeddings, class_labels, num_classes=1).squeeze(0)
                    else:
                        # Use mean pooling as fallback
                        class_emb[class_idx] = context_label_emb[class_mask].mean(dim=0)

            # Apply MLP if specified
            if mlp_module is not None:
                class_emb = mlp_module(class_emb)

            # Normalize if specified
            if normalize:
                class_emb = F.normalize(class_emb, p=2, dim=-1)

            # Compute logits using similarity
            if self.sim == 'dot':
                logits = torch.matmul(target_label_emb, class_emb.t())
            elif self.sim == 'cos':
                target_label_emb = F.normalize(target_label_emb, p=2, dim=-1)
                class_emb = F.normalize(class_emb, p=2, dim=-1)
                logits = torch.matmul(target_label_emb, class_emb.t())
            elif self.sim == 'euclidean':
                distances = torch.cdist(target_label_emb, class_emb, p=2)
                logits = -distances
            elif self.sim == 'mlp':
                target_label_emb = self.sim_mlp(target_label_emb)
                class_emb = self.sim_mlp(class_emb)
                logits = torch.matmul(target_label_emb, class_emb.t())
            else:
                raise ValueError(f"Invalid similarity type: {self.sim}")

        return logits, class_emb


class MPLPPredictor(nn.Module):
    """
    MPLP+ Predictor that uses randomized node labeling for structural features.
    Combines structural features (from RandomizedNodeLabeling) with node features.
    """
    def __init__(self, hidden_dim, dropout=0.2, signature_dim=64, num_hops=2, feature_combine='hadamard',
                 head_num_layers=2, prop_type='combine', edge_feat_dim=None,
                 signature_sampling='torchhd', use_subgraph=True):
        super().__init__()
        self.use_node_features = use_node_features
        self.feature_combine = feature_combine
        
        # 1. Randomized Structural Encoding
        from .node_label import RandomizedNodeLabeling
        self.node_labeling = RandomizedNodeLabeling(
            signature_dim=signature_dim,
            num_hops=num_hops,
            prop_type=prop_type,
            signature_sampling=signature_sampling,
            use_subgraph=use_subgraph
        )
        
        # Structural Feature Encoder (Original MPLP+ style: 5->5, 1 layer)
        structural_input_dim = 15 if prop_type == 'combine' else 5
        self.struct_encode = MLP(
            in_channels=structural_input_dim,
            hidden_channels=structural_input_dim,
            out_channels=structural_input_dim,
            num_layers=1,
            dropout=dropout,
            norm=True,
            tailact=True
        )
        
        # Feature Encoder for the input edge embedding (refines PFN output)
        edge_feat_dim = hidden_dim if edge_feat_dim is None else edge_feat_dim
        self.feat_encode = MLP(
            in_channels=edge_feat_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=2,
            dropout=dropout,
            norm=True
        )
        
        # Final Classifier (Struct + Feat)
        # Using MLP to allow interaction between structure and features
        self.classifier = MLP(
            in_channels=hidden_dim + structural_input_dim,
            hidden_channels=hidden_dim,
            out_channels=1,
            num_layers=head_num_layers,
            dropout=dropout,
            norm=True
        )
        
    def forward(self, x, adj_t, edges):
        """
        Args:
            x: Node features [num_nodes, hidden_dim]
            adj_t: Adjacency matrix (SparseTensor)
            edges: Target edges [2, num_edges]
        """
        # 1. Compute Node Weights (Adamic-Adar style: 1/log(d))
        # This helps the random labeling capture "weighted" overlap
        degree = adj_t.sum(dim=1)
        # Avoid log(0) or div by zero. Add 1 to degree so log(d+1) > 0
        node_weight = torch.sqrt(1.0 / (torch.log(degree + 1.0) + 1e-6))
        
        # 2. Get Randomized Structural Features
        # Ensure edges are [2, E]
        if edges.size(0) != 2:
            edges = edges.t()
            
        struct_feats = self.node_labeling(edges, adj_t, node_weight)
        struct_emb = self.struct_encode(struct_feats)
        
        # 3. Process Node Features
        if self.use_node_features:
            row, col = edges
            x_u = x[row]
            x_v = x[col]
            
            if self.feature_combine == 'hadamard':
                feat_combined = x_u * x_v
            elif self.feature_combine == 'concat':
                feat_combined = torch.cat([x_u, x_v], dim=-1)
                
            feat_emb = self.feat_encode(feat_combined)
            
            # Combine Structural and Feature Embeddings
            final_emb = torch.cat([struct_emb, feat_emb], dim=-1)
        else:
            final_emb = struct_emb
            
        # 4. Predict
        return self.classifier(final_emb).squeeze(-1)


class MPLPLinkPredictionHead(nn.Module):
    """
    Link Prediction Head using MPLP randomized structural features.
    """
    def __init__(self, hidden_dim, dropout=0.2, signature_dim=64, num_hops=2, feature_combine='hadamard',
                 head_num_layers=2, prop_type='combine', edge_feat_dim=None,
                 signature_sampling='torchhd', use_subgraph=True, use_degree='none'):
        super().__init__()
        self.feature_combine = feature_combine
        self.use_degree = use_degree
        
        # Randomized Structural Encoding
        from .node_label import RandomizedNodeLabeling
        self.node_labeling = RandomizedNodeLabeling(
            signature_dim=signature_dim,
            num_hops=num_hops,
            prop_type=prop_type,
            signature_sampling=signature_sampling,
            use_subgraph=use_subgraph
        )
        
        # Structural Feature Encoder (Original MPLP+ style: 5->5, 1 layer)
        structural_input_dim = 15 if prop_type == 'combine' else 5
        self.struct_encode = MLP(
            in_channels=structural_input_dim,
            hidden_channels=structural_input_dim,
            out_channels=structural_input_dim,
            num_layers=1,
            dropout=dropout,
            norm=True,
            tailact=True
        )
        
        # Feature Encoder for the input edge embedding (refines PFN output)
        edge_feat_dim = hidden_dim if edge_feat_dim is None else edge_feat_dim
        self.feat_encode = MLP(
            in_channels=edge_feat_dim,
            hidden_channels=hidden_dim,
            out_channels=hidden_dim,
            num_layers=2,
            dropout=dropout,
            norm=True
        )
        
        # Final Classifier (Struct + Feat)
        # Using MLP to allow interaction between structure and features
        self.classifier = MLP(
            in_channels=hidden_dim + structural_input_dim,
            hidden_channels=hidden_dim,
            out_channels=1,
            num_layers=head_num_layers,
            dropout=dropout,
            norm=True
        )

        self.node_weight_mlp = None
        if self.use_degree == 'mlp':
            self.node_weight_mlp = MLP(
                in_channels=hidden_dim + 1,
                hidden_channels=32,
                out_channels=1,
                num_layers=2,
                dropout=dropout,
                norm=True
            )

    def forward(self, edge_emb, adj_t, edges, node_emb=None):
        """
        Args:
            edge_emb: Refined edge embeddings from Transformer [batch_size, hidden_dim]
            adj_t: Adjacency matrix (SparseTensor)
            edges: Edge indices [2, batch_size]
        """
        # 1. Compute Node Weights (Inverse Log Degree)
        node_weight = None
        if self.use_degree == 'aa':
            degree = adj_t.sum(dim=1)
            node_weight = torch.sqrt(1.0 / (torch.log(degree + 1.0) + 1e-6))
        elif self.use_degree == 'ra':
            degree = adj_t.sum(dim=1)
            node_weight = torch.sqrt(1.0 / (degree + 1e-6))
        elif self.use_degree == 'mlp':
            if node_emb is None:
                raise ValueError("use_degree='mlp' requires node_emb to be provided.")
            degree = adj_t.sum(dim=1).view(-1, 1)
            node_weight_feat = torch.cat([node_emb, degree], dim=1)
            node_weight = self.node_weight_mlp(node_weight_feat).squeeze(-1) + 1.0
        
        # 2. Get Structural Features
        # Ensure edges are [2, E]
        if edges.size(0) != 2:
            edges = edges.t()
            
        struct_feats = self.node_labeling(edges, adj_t, node_weight)
        struct_emb = self.struct_encode(struct_feats)
        
        # 3. Process Edge Features (from PFN)
        feat_emb = self.feat_encode(edge_emb)
        
        # 4. Combine and Predict
        final_emb = torch.cat([struct_emb, feat_emb], dim=-1)
        return self.classifier(final_emb).squeeze(-1)


class PFNPredictorNodeCls(nn.Module):
    def __init__(self, hidden_dim, nhead=1, num_layers=2, mlp_layers=2, dropout=0.2,
                 norm=False, separate_att=False, degree=False, att=None, mlp=None, sim='dot',
                 padding='zero', norm_affine=True, normalize=False,
                 use_first_half_embedding=False, use_full_embedding=False, norm_type='post',
                 ffn_expansion_ratio=4, use_matching_network=False, matching_network_projection='linear',
                 matching_network_temperature=1.0, matching_network_learnable_temp=True,
                 # Node classification ridge regression
                 nc_sim='dot', nc_ridge_alpha=1.0,
                 # Link prediction ridge regression
                 lp_sim='dot', lp_ridge_alpha=1.0,
                 # Graph classification ridge regression
                 gc_sim='dot', gc_ridge_alpha=1.0,
                 head_num_layers=2,
                 # NEW: Option to skip token formulation
                 skip_token_formulation=False,
                 # NEW: Simple linear baseline for LP
                 lp_use_linear_predictor=False,
                 # NEW: LP Head Type
                 lp_head_type='standard',
                 mplp_signature_dim=64, mplp_num_hops=2, mplp_feature_combine='hadamard',
                 mplp_prop_type='combine',
                 mplp_signature_sampling='torchhd',
                 mplp_use_subgraph=True,
                 mplp_use_degree='none',
                 nc_head_num_layers=None, lp_head_num_layers=None,
                 lp_concat_common_neighbors=False):
        super(PFNPredictorNodeCls, self).__init__()
        self.lp_head_type = lp_head_type
        self.hidden_dim = hidden_dim
        self.d_label = hidden_dim  # Label embedding has the same dimension as node features
        self.embed_dim = hidden_dim + self.d_label  # Token dimension after concatenation
        self.sim = sim  # Legacy parameter for backward compatibility
        self.padding = padding
        self.use_matching_network = use_matching_network
        self.skip_token_formulation = skip_token_formulation
        self.lp_use_linear_predictor = lp_use_linear_predictor
        self.lp_linear_head = nn.Linear(hidden_dim, 1) if lp_use_linear_predictor else None
        self.mplp_prop_type = mplp_prop_type
        self.mplp_signature_sampling = mplp_signature_sampling
        self.mplp_use_subgraph = mplp_use_subgraph
        self.mplp_use_degree = mplp_use_degree
        # CN concat is not applied to MPLP head
        self.lp_concat_common_neighbors = lp_concat_common_neighbors and self.lp_head_type != 'mplp'

        # Separate ridge regression configurations
        # Use explicit task-specific parameters - they have proper defaults from config
        self.nc_sim = nc_sim
        self.nc_ridge_alpha = nc_ridge_alpha
        self.lp_sim = lp_sim
        self.lp_ridge_alpha = lp_ridge_alpha
        self.gc_sim = gc_sim
        self.gc_ridge_alpha = gc_ridge_alpha
        self.head_num_layers = head_num_layers
        self.nc_head_num_layers = head_num_layers if nc_head_num_layers is None else nc_head_num_layers
        self.lp_head_num_layers = head_num_layers if lp_head_num_layers is None else lp_head_num_layers

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
        # Use hidden_dim directly if skipping token formulation, otherwise use embed_dim
        transformer_dim = self.hidden_dim if self.skip_token_formulation else self.embed_dim
        self.transformer_row = nn.ModuleList([
            PFNTransformerLayer(
                hidden_dim=transformer_dim,
                n_head=nhead,
                mlp_layers=mlp_layers,
                dropout=dropout,
                norm=norm,
                separate_att=separate_att,
                unsqueeze=False,
                norm_type=norm_type,
                ffn_expansion_ratio=ffn_expansion_ratio
            ) for _ in range(num_layers)
        ])
        self.degree = degree
        self.att = att
        self.mlp_pool = mlp
        self.normalize = normalize
        self.use_first_half_embedding = use_first_half_embedding
        self.use_full_embedding = use_full_embedding

        # Validate embedding options (only one should be True)
        if sum([use_first_half_embedding, use_full_embedding]) > 1:
            raise ValueError("Only one of use_first_half_embedding or use_full_embedding can be True")

        # Initialize matching network if enabled
        if use_matching_network:
            # Determine input dim based on embedding option
            if use_full_embedding:
                mn_hidden_dim = self.embed_dim  # 2 * hidden_dim
            else:
                mn_hidden_dim = hidden_dim
            self.matching_network = MatchingNetworkPredictor(
                hidden_dim=mn_hidden_dim,
                projection_type=matching_network_projection,
                normalize=True,
                temperature=matching_network_temperature,
                dropout=dropout,
                norm=norm,
                mlp_layers=mlp_layers,
                norm_affine=norm_affine,
                learnable_temperature=matching_network_learnable_temp
            )

        # Task-specific heads
        # Determine embedding dimension for heads based on embedding option
        if use_full_embedding:
            head_hidden_dim = self.embed_dim  # 2 * hidden_dim
        else:
            head_hidden_dim = hidden_dim

        self.nc_head = NodeClassificationHead(
            hidden_dim=head_hidden_dim,
            dropout=dropout,
            norm=norm,
            norm_affine=norm_affine,
            sim=self.nc_sim,
            ridge_alpha=self.nc_ridge_alpha,
            head_num_layers=self.nc_head_num_layers
        )

        lp_head_input_dim = head_hidden_dim + (1 if self.lp_concat_common_neighbors else 0)

        if self.lp_head_type == 'mplp':
            self.lp_head = MPLPLinkPredictionHead(
                hidden_dim=head_hidden_dim,
                dropout=dropout,
                signature_dim=mplp_signature_dim,
                num_hops=mplp_num_hops,
                feature_combine=mplp_feature_combine,
                head_num_layers=self.lp_head_num_layers,
                prop_type=self.mplp_prop_type,
                signature_sampling=self.mplp_signature_sampling,
                use_subgraph=self.mplp_use_subgraph,
                use_degree=self.mplp_use_degree,
                edge_feat_dim=lp_head_input_dim
            )
        else:
            self.lp_head = LinkPredictionHead(
                hidden_dim=lp_head_input_dim,
                dropout=dropout,
                norm=norm,
                norm_affine=norm_affine,
                sim=self.lp_sim,
                ridge_alpha=self.lp_ridge_alpha,
                head_num_layers=self.lp_head_num_layers
            )

        self.gc_head = GraphClassificationHead(
            hidden_dim=head_hidden_dim,
            dropout=dropout,
            norm=norm,
            norm_affine=norm_affine,
            sim=self.gc_sim,
            ridge_alpha=self.gc_ridge_alpha,
            head_num_layers=self.head_num_layers
        )
    
    def forward(self, data, context_x, target_x, context_y, class_x, task_type='node_classification', adj_t=None, lp_edges=None, node_emb=None, lp_cn_context=None, lp_cn_target=None):
        """
        Unified forward pass supporting node classification, link prediction, and graph classification.

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

        For graph classification:
        - data: PFN data structure (prepared for graph classification)
        - context_x: Context graph embeddings
        - target_x: Target graph embeddings
        - context_y: Context graph labels
        - class_x: Class prototypes [num_classes, hidden_dim]
        """
        if task_type == 'link_prediction' and self.lp_use_linear_predictor:
            logits = self.lp_linear_head(target_x).squeeze(-1)
            return logits, class_x

        if self.skip_token_formulation:
            # NEW PATH: Skip token formulation, use GNN embeddings directly
            # Step 1: Prepare for transformer (add sequence dimension)
            context_tokens = context_x.unsqueeze(1)  # [num_context, 1, hidden_dim]
            target_tokens = target_x.unsqueeze(1)    # [num_target, 1, hidden_dim]

            # Step 2: Process through transformer layers
            for layer in self.transformer_row:
                context_tokens, target_tokens = layer(context_tokens, target_tokens)

            # Step 3: Extract refined embeddings
            context_label_emb = context_tokens.squeeze(1)  # [num_context, hidden_dim]
            target_label_emb = target_tokens.squeeze(1)    # [num_target, hidden_dim]
        else:
            # ORIGINAL PATH: Use token formulation with label concatenation
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
            for layer in self.transformer_row:
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


        # Step 6: Optional CN concat at LP head input (post-transformer)
        if task_type == 'link_prediction' and self.lp_concat_common_neighbors:
            if lp_cn_context is not None:
                if lp_cn_context.dim() == 1:
                    lp_cn_context = lp_cn_context.unsqueeze(1)
                context_label_emb = torch.cat([context_label_emb, lp_cn_context.to(context_label_emb.device)], dim=1)
            if lp_cn_target is not None:
                if lp_cn_target.dim() == 1:
                    lp_cn_target = lp_cn_target.unsqueeze(1)
                target_label_emb = torch.cat([target_label_emb, lp_cn_target.to(target_label_emb.device)], dim=1)

        # Step 7: Route through task-specific heads
        if task_type == 'node_classification':
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
        elif task_type == 'link_prediction':
            if self.lp_head_type == 'mplp':
                if adj_t is None or lp_edges is None:
                    raise ValueError("MPLP head requires adj_t and lp_edges to be passed to forward()")
                logits = self.lp_head(target_label_emb, adj_t, lp_edges, node_emb=node_emb)
                class_emb = class_x # Keep prototypes as is (not updated by MPLP)
            else:
                logits, class_emb = self.lp_head(
                    target_label_emb=target_label_emb,
                    context_label_emb=context_label_emb,
                    context_y=context_y,
                    class_x=class_x,
                    attention_pool_module=self.att,
                    mlp_module=self.mlp_pool,
                    normalize=self.normalize
                )
        elif task_type == 'graph_classification':
            logits, class_emb = self.gc_head(
                target_label_emb=target_label_emb,
                context_label_emb=context_label_emb,
                context_y=context_y,
                class_x=class_x,
                attention_pool_module=self.att,
                mlp_module=self.mlp_pool,
                normalize=self.normalize
            )
        else:
            raise ValueError(f"Unknown task_type: {task_type}. Choose 'node_classification', 'link_prediction', or 'graph_classification'.")

        # Legacy matching network support (kept for backward compatibility)
        if self.use_matching_network:
            # Use matching network: attention over support samples with one-hot labels as values
            num_classes = class_x.size(0)

            # DEBUG: Compare matching network vs mean pooling
            _debug_compare = True  # TEMPORARY DEBUG FLAG
            if _debug_compare:
                # Get true labels for target samples (from the batch being processed)
                # target_label_emb corresponds to the current batch targets
                # We need to get the true labels from outside - pass via data or infer

                print(f"\n{'='*60}")
                print("DEBUG: Comparing Matching Network vs Mean Pooling")
                print(f"{'='*60}")

                # Mean pooling prototypes
                prototypes = torch.zeros(num_classes, context_label_emb.size(-1), device=context_label_emb.device)
                for c in range(num_classes):
                    mask = context_y == c
                    if mask.any():
                        prototypes[c] = context_label_emb[mask].mean(dim=0)

                # Different similarity methods for mean pooling
                logits_mp_dot = target_label_emb @ prototypes.t()  # DOT
                target_norm = F.normalize(target_label_emb, p=2, dim=-1)
                proto_norm = F.normalize(prototypes, p=2, dim=-1)
                logits_mp_cos = target_norm @ proto_norm.t()  # COS

                # Matching network variants
                mn = self.matching_network
                Q = mn.q_proj(target_label_emb)
                K = mn.k_proj(context_label_emb)
                V = F.one_hot(context_y.long(), num_classes=num_classes).float()

                # MN without norm
                attn_raw = (Q @ K.t()) / mn.temperature
                logits_mn_raw = attn_raw @ V

                # MN with norm
                Q_norm = F.normalize(Q, p=2, dim=-1)
                K_norm = F.normalize(K, p=2, dim=-1)
                attn_norm = (Q_norm @ K_norm.t()) / mn.temperature
                logits_mn_norm = attn_norm @ V

                # Store predictions
                preds = {
                    'MP_DOT': logits_mp_dot.argmax(dim=1),
                    'MP_COS': logits_mp_cos.argmax(dim=1),
                    'MN_RAW': logits_mn_raw.argmax(dim=1),
                    'MN_NORM': logits_mn_norm.argmax(dim=1),
                }

                print(f"Embeddings: context={context_label_emb.shape}, target={target_label_emb.shape}")
                print(f"Prototype norms: min={prototypes.norm(dim=1).min():.4f}, max={prototypes.norm(dim=1).max():.4f}")
                print(f"Target norms: min={target_label_emb.norm(dim=1).min():.4f}, max={target_label_emb.norm(dim=1).max():.4f}")

                # Compute what the ACTUAL non-matching path does (with process_node_features)
                actual_class_emb = process_node_features(
                    context_label_emb, data,
                    degree_normalize=self.degree,
                    attention_pool_module=self.att,
                    mlp_module=self.mlp_pool,
                    normalize=self.normalize
                )
                logits_actual = target_label_emb @ actual_class_emb.t()  # Always dot
                preds['ACTUAL_PATH'] = logits_actual.argmax(dim=1)

                print(f"\nLogits ranges:")
                print(f"  MP_DOT: [{logits_mp_dot.min():.4f}, {logits_mp_dot.max():.4f}]")
                print(f"  MP_COS: [{logits_mp_cos.min():.4f}, {logits_mp_cos.max():.4f}]")
                print(f"  MN_RAW: [{logits_mn_raw.min():.4f}, {logits_mn_raw.max():.4f}]")
                print(f"  MN_NORM: [{logits_mn_norm.min():.4f}, {logits_mn_norm.max():.4f}]")
                print(f"  ACTUAL_PATH: [{logits_actual.min():.4f}, {logits_actual.max():.4f}]")
                print(f"  (degree={self.degree}, att={self.att is not None}, mlp={self.mlp_pool is not None}, norm={self.normalize})")

                # Store debug info for accuracy computation in engine
                self._debug_preds = preds
                self._debug_logits = {
                    'MP_DOT': logits_mp_dot,
                    'MP_COS': logits_mp_cos,
                    'MN_RAW': logits_mn_raw,
                    'MN_NORM': logits_mn_norm,
                    'ACTUAL_PATH': logits_actual,
                }
                print(f"(Accuracy will be computed in engine with true labels)")
                print(f"{'='*60}\n")

            # Override logits with matching network output (for backward compatibility)
            logits = self.matching_network(target_label_emb, context_label_emb, context_y, num_classes=num_classes)
            # class_emb from task heads is still returned

        return logits, class_emb

    def get_target_embeddings(self, context_x, target_x, context_y, class_x):
        """
        Extract Transformer-generated target embeddings without computing final predictions.
        Used for contrastive augmentation loss.

        Args:
            context_x: Context node embeddings [num_context, hidden_dim]
            target_x: Target node embeddings [num_target, hidden_dim]
            context_y: Context node labels [num_context]
            class_x: Class prototypes [num_classes, hidden_dim]

        Returns:
            target_label_emb: Transformer-processed target embeddings [num_target, hidden_dim or 2*hidden_dim]
        """
        if self.skip_token_formulation:
            # Skip token formulation path
            context_tokens = context_x.unsqueeze(1)  # [num_context, 1, hidden_dim]
            target_tokens = target_x.unsqueeze(1)    # [num_target, 1, hidden_dim]

            # Process through transformer layers
            for layer in self.transformer_row:
                context_tokens, target_tokens = layer(context_tokens, target_tokens)

            # Extract refined embeddings
            target_label_emb = target_tokens.squeeze(1)  # [num_target, hidden_dim]
        else:
            # Token formulation path
            class_x_y = class_x[context_y]  # [num_context, hidden_dim]
            context_tokens = torch.cat([context_x, class_x_y], dim=1)  # [num_context, 2*hidden_dim]

            # Create target tokens
            if self.padding == 'zero':
                padding = torch.zeros_like(target_x)  # [num_target, hidden_dim]
            elif self.padding == 'mlp':
                padding = self.pad_mlp(target_x)
            else:
                raise ValueError("Invalid padding type. Choose 'zero' or 'mlp'.")
            target_tokens = torch.cat([target_x, padding], dim=1)

            # Prepare for transformer
            context_tokens = context_tokens.unsqueeze(1)  # [num_context, 1, 2*hidden_dim]
            target_tokens = target_tokens.unsqueeze(1)    # [num_target, 1, 2*hidden_dim]

            # Process through transformer layers
            for layer in self.transformer_row:
                context_tokens, target_tokens = layer(context_tokens, target_tokens)

            # Extract refined embeddings
            target_tokens = target_tokens.squeeze(1)  # [num_target, 2*hidden_dim]

            # Choose which part of the embedding to use
            if self.use_full_embedding:
                target_label_emb = target_tokens
            elif self.use_first_half_embedding:
                target_label_emb = target_tokens[:, :self.hidden_dim]
            else:
                target_label_emb = target_tokens[:, self.hidden_dim:]

        return target_label_emb

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
        
class MatchingNetworkPredictor(nn.Module):
    """
    Matching Network with Label-Value Attention for few-shot classification.

    Query attends to support samples, using one-hot labels as values:
        logits = softmax(Q·K^T / tau) · one_hot(labels)

    Args:
        hidden_dim: Embedding dimension
        projection_type: 'linear' or 'mlp' for Q/K projections
        normalize: Whether to normalize Q/K before dot product (cosine attention)
        temperature: Initial temperature for scaling
        dropout: Dropout rate for MLP projections
        mlp_layers: Number of layers if using MLP projection
        learnable_temperature: Whether temperature is learnable
    """
    def __init__(self, hidden_dim, projection_type='linear', normalize=True,
                 temperature=1.0, dropout=0.2, norm=False, mlp_layers=2,
                 norm_affine=True, learnable_temperature=True):
        """
        Args:
            temperature: Scaling for attention (attn = scores / temp).
                        With normalized Q/K (cosine sim in [-1,1]), temp=1.0 is reasonable.
                        Lower temp = sharper attention. Default 1.0.
        """
        super(MatchingNetworkPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.normalize = normalize
        self.projection_type = projection_type

        # Q/K projections
        if projection_type == 'linear':
            self.q_proj = nn.Linear(hidden_dim, hidden_dim)
            self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        elif projection_type == 'mlp':
            self.q_proj = MLP(hidden_dim, hidden_dim, hidden_dim, mlp_layers, dropout, norm,
                              tailact=False, norm_affine=norm_affine)
            self.k_proj = MLP(hidden_dim, hidden_dim, hidden_dim, mlp_layers, dropout, norm,
                              tailact=False, norm_affine=norm_affine)
        else:
            raise ValueError(f"Unknown projection_type: {projection_type}. Use 'linear' or 'mlp'.")

        # Learnable temperature for attention scaling
        if learnable_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))

        self._init_weights()

    def _init_weights(self):
        if self.projection_type == 'linear':
            # Initialize as identity so early behavior mimics direct similarity (like mean pooling)
            nn.init.eye_(self.q_proj.weight)
            nn.init.eye_(self.k_proj.weight)
            nn.init.zeros_(self.q_proj.bias)
            nn.init.zeros_(self.k_proj.bias)

    def forward(self, target_emb, support_emb, support_labels, num_classes=None):
        """
        Args:
            target_emb: Target/query embeddings [num_target, hidden_dim]
            support_emb: Support set embeddings [num_support, hidden_dim]
            support_labels: Labels for support samples [num_support]
            num_classes: Number of classes (inferred from labels if None)

        Returns:
            logits: Classification logits [num_target, num_classes]
        """
        # Infer num_classes if not provided
        if num_classes is None:
            num_classes = support_labels.max().item() + 1

        # Project Q and K
        Q = self.q_proj(target_emb)    # [num_target, hidden_dim]
        K = self.k_proj(support_emb)   # [num_support, hidden_dim]

        # Normalize for cosine attention
        if self.normalize:
            Q = F.normalize(Q, p=2, dim=-1)
            K = F.normalize(K, p=2, dim=-1)

        # Compute attention scores and aggregate by class
        attn_scores = torch.matmul(Q, K.t()) / self.temperature  # [num_target, num_support]

        # One-hot labels as values
        V = F.one_hot(support_labels.long(), num_classes=num_classes).float()  # [num_support, num_classes]

        # Aggregate attention scores by class (sum pooling per class)
        # This gives unnormalized logits that can be passed to log_softmax
        logits = torch.matmul(attn_scores, V)  # [num_target, num_classes]

        return logits


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
    Includes LayerNorm at the end to stabilize features going into GNN
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

        # Add LayerNorm at the end to normalize output
        self.norm = nn.LayerNorm(large_dim)

    def forward(self, x):
        if self.extra_proj is None:
            out = x  # No projection needed
        else:
            # Keep original dimensions + add projected dimensions
            extra_dims = self.extra_proj(x)
            out = torch.cat([x, extra_dims], dim=1)

        # Normalize output to stabilize features going into GNN
        out = self.norm(out)
        return out
    
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


# ============================================================================
# Dynamic Encoder Wrapper
# ============================================================================

class GNNWithDE(nn.Module):
    """
    Wrapper that adds Dynamic Encoder (DE) before GNN for end-to-end feature projection.

    Architecture:
        Raw Features (N, d_original)
            ↓
        [Column Sampling] - sample n_s nodes → (n_s, d_original)
            ↓
        [Dynamic Encoder MLP] - learns projection matrix T (d_original, k)
            ↓
        [Universal Projection] - X @ T → (N, k)
            ↓
        [L2 Normalize] → (N, k)
            ↓
        [GNN] - existing GNN model → (N, hidden)

    Key Features:
    - Learns projection in end-to-end manner (no offline PCA)
    - Handles variable input dimensions across datasets
    - Random column sampling during training acts as augmentation
    - Uniformity loss prevents basis collapse
    """

    def __init__(self,
                 gnn_model,
                 de_sample_size=1024,
                 de_hidden_dim=512,
                 de_output_dim=256,
                 de_activation='prelu',
                 de_use_layernorm=True,
                 de_dropout=0.0,
                 de_norm_affine=True,
                 lambda_de=0.01,
                 update_sample_every_n_steps=1):
        """
        Initialize GNN with Dynamic Encoder wrapper.

        Args:
            gnn_model (nn.Module): Base GNN model (PureGCN_v1, GCN, UnifiedGNN, etc.)
            de_sample_size (int): Number of nodes to sample for DE (n_s)
            de_hidden_dim (int): Hidden dimension for DE MLP
            de_output_dim (int): DE projection output dim (should match GNN input_dim)
            de_activation (str): Activation function for DE
            de_use_layernorm (bool): Use LayerNorm in DE
            de_dropout (float): Dropout rate in DE (should match main model dropout)
            de_norm_affine (bool): Learnable affine in LayerNorm
            lambda_de (float): Weight for DE uniformity loss
            update_sample_every_n_steps (int): Update sample every N forward passes
        """
        super(GNNWithDE, self).__init__()

        from src.dynamic_encoder import DynamicEncoder

        self.gnn = gnn_model

        # Initialize DE immediately (no lazy init needed - it works for any input dim!)
        self.de = DynamicEncoder(
            sample_size=de_sample_size,
            hidden_dim=de_hidden_dim,
            output_dim=de_output_dim,
            activation=de_activation,
            use_layernorm=de_use_layernorm,
            dropout=de_dropout,
            norm_affine=de_norm_affine,
        )

        self.de_sample_size = de_sample_size
        self.lambda_de = lambda_de
        self.update_every_n = update_sample_every_n_steps

        # Learnable LayerNorm for projected features (preserves std≈1.0 for GNN)
        self.proj_layer_norm = nn.LayerNorm(de_output_dim)

        # Tracking
        self.step_counter = 0
        self.cached_sample = None
        self.current_projection_matrix = None  # For loss computation

        print(f"[DE] Initialized immediately: sample_size={de_sample_size}, output_dim={de_output_dim}")
        print(f"[DE] Parameters: {sum(p.numel() for p in self.de.parameters()):,}")
        print(f"[DE] Added learnable LayerNorm for projected features")

    def forward(self, x, adj_t, batch=None):
        """
        Forward pass with Dynamic Encoder projection.

        Args:
            x (torch.Tensor): Node features (N, d_original)
            adj_t: Adjacency tensor (SparseTensor or edge_index)
            batch (torch.Tensor, optional): Batch assignment for graph-level tasks

        Returns:
            h (torch.Tensor): Node embeddings (N, hidden)
        """
        # Step 1: Sample features for DE
        from src.dynamic_encoder import sample_feature_columns

        if self.training:
            # Training: Update sample periodically OR use cached
            # BUT: Always recompute if feature dimension changed (different dataset)
            feat_dim_changed = (self.cached_sample is not None and
                               self.cached_sample.size(1) != x.size(1))

            if self.step_counter % self.update_every_n == 0 or feat_dim_changed:
                sampled_features = sample_feature_columns(
                    x,
                    self.de_sample_size,
                    training=True,
                    deterministic_eval=False
                )
                self.cached_sample = sampled_features
                self.cached_projection = None  # Invalidate projection cache too
            else:
                # Reuse cached sample (reduces randomness)
                sampled_features = self.cached_sample if self.cached_sample is not None else \
                                  sample_feature_columns(x, self.de_sample_size, training=True)
            self.step_counter += 1
        else:
            # Evaluation: Deterministic sampling (first n_s nodes)
            sampled_features = sample_feature_columns(
                x,
                self.de_sample_size,
                training=False,
                deterministic_eval=True
            )

        # Step 2: Generate projection matrix via DE
        projection_matrix = self.de(sampled_features)  # (d_original, k)
        self.current_projection_matrix = projection_matrix  # Cache for loss

        # Step 3: Project features with learnable LayerNorm
        from src.dynamic_encoder import apply_dynamic_projection
        x_proj = apply_dynamic_projection(x, projection_matrix, normalize=True, layer_norm=self.proj_layer_norm)  # (N, k)

        # Step 4: Pass through GNN
        h = self.gnn(x_proj, adj_t, batch)

        return h

    def get_de_loss(self):
        """
        Compute DE uniformity loss.

        Returns:
            loss (torch.Tensor): Scalar DE loss (0 if not in training)
        """
        if not self.training:
            return torch.tensor(0.0, device=next(self.parameters()).device)

        # Use cached uniformity loss from DE
        return self.de.uniformity_loss() * self.lambda_de

    def get_de_diagnostics(self):
        """
        Get diagnostic information about DE for monitoring.

        Returns:
            dict with diagnostic metrics
        """
        if not hasattr(self, 'de') or self.de is None:
            return {}

        diagnostics = {}

        # Projection matrix statistics
        if self.current_projection_matrix is not None:
            proj = self.current_projection_matrix
            diagnostics['proj_mean'] = proj.mean().item()
            diagnostics['proj_std'] = proj.std().item()
            diagnostics['proj_min'] = proj.min().item()
            diagnostics['proj_max'] = proj.max().item()
            diagnostics['proj_norm'] = proj.norm().item()

            # Check for NaN or Inf
            diagnostics['proj_has_nan'] = torch.isnan(proj).any().item()
            diagnostics['proj_has_inf'] = torch.isinf(proj).any().item()

            # Basis vector norms (should be ~1 after normalization)
            basis_norms = proj.norm(dim=1)
            diagnostics['basis_norm_mean'] = basis_norms.mean().item()
            diagnostics['basis_norm_std'] = basis_norms.std().item()
            diagnostics['basis_norm_min'] = basis_norms.min().item()
            diagnostics['basis_norm_max'] = basis_norms.max().item()

            # Column statistics (output dimensions)
            col_norms = proj.norm(dim=0)
            diagnostics['col_norm_mean'] = col_norms.mean().item()
            diagnostics['col_norm_std'] = col_norms.std().item()

        # DE parameter statistics
        param_norms = []
        param_grads = []
        for name, param in self.de.named_parameters():
            param_norms.append(param.data.norm().item())
            if param.grad is not None:
                param_grads.append(param.grad.norm().item())

        if param_norms:
            diagnostics['de_param_norm_mean'] = sum(param_norms) / len(param_norms)
            diagnostics['de_param_norm_max'] = max(param_norms)

        if param_grads:
            diagnostics['de_grad_norm_mean'] = sum(param_grads) / len(param_grads)
            diagnostics['de_grad_norm_max'] = max(param_grads)
            diagnostics['de_grad_has_none'] = False
        else:
            diagnostics['de_grad_has_none'] = True

        # Uniformity loss components
        if self.de.cached_projection is not None:
            mean_vec = self.de.cached_projection.mean(dim=0)
            diagnostics['mean_vec_norm'] = mean_vec.norm().item()
            diagnostics['uniformity_loss'] = self.de.uniformity_loss().item()

        return diagnostics
