import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from torch_geometric.nn import GCNConv, GINConv
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_sparse import matmul, SparseTensor
from torch_sparse.matmul import spmm_add
from torch import Tensor
import math
from torch_geometric.utils import negative_sampling
from torch_scatter import scatter_add, scatter_softmax
from utils import process_node_features

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
                 relu=False, norm_affine=True):
        super().__init__()
        
        # Input projection
        self.lin = nn.Linear(input_dim, hidden) if input_dim != hidden else nn.Identity()
        
        # GCN Convolution Layer
        self.conv = PureGCNConv()
        self.num_layers = num_layers
        self.dp = dp
        self.norm = norm
        self.res = res
        self.relu = relu

        # Use separate LayerNorm instances per layer if normalization is enabled
        if self.norm:
            self.norms = nn.ModuleList([nn.LayerNorm(hidden, elementwise_affine=norm_affine) for _ in range(num_layers)])

    def forward(self, x, adj_t):
        x = self.lin(x)  # Apply input projection
        ori = x
        for i in range(self.num_layers):
            if i != 0:
                if self.res:
                    x = x + ori
                if self.norm:
                    x = self.norms[i](x)  # Apply per-layer normalization
                if self.relu:
                    x = F.relu(x)
                if self.dp > 0:
                    x = F.dropout(x, p=self.dp, training=self.training)
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
                self.lins.append(nn.LayerNorm(hidden_channels), elementwise_affine=norm_affine)
            self.lins.append(nn.ReLU())
            if dropout > 0:
                self.lins.append(nn.Dropout(dropout))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        if tailact:
            self.lins.append(nn.LayerNorm(out_channels), elementwise_affine=norm_affine)
            self.lins.append(nn.ReLU())
            self.lins.append(nn.Dropout(dropout))

    def forward(self, x):
        x = self.lins(x)
        return x.squeeze()
    
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, norm=False, relu=False, prop_step=2, dropout=0.2, 
                 multilayer=False, use_gin=False, res=False, norm_affine=True):
        super(GCN, self).__init__()
        self.lin = nn.Linear(in_feats, h_feats) if in_feats != h_feats else nn.Identity()
        self.multilayer = multilayer
        self.use_gin = use_gin
        if multilayer:
            self.convs = nn.ModuleList()
            for _ in range(prop_step):
                if use_gin:
                    self.convs.append(GINConv(MLP(h_feats, h_feats, h_feats, 2, dropout, norm, 
                                                  norm_affine)))
                self.convs.append(GCNConv(h_feats, h_feats))
        else:
            if use_gin:
                self.conv = GINConv(MLP(h_feats, h_feats, h_feats, 2, dropout, norm, norm_affine))
            else:
                self.conv = GCNConv(h_feats, h_feats)
        self.norm = norm
        self.relu = relu
        self.prop_step = prop_step
        if norm:
            self.norms = nn.ModuleList([nn.LayerNorm(h_feats, elementwise_affine=norm_affine) \
                                        for _ in range(prop_step)])
            self.dp = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.res = res

    def _apply_norm_and_activation(self, x, i):
        if self.norm:
            x = self.norms[i](x)
        if self.relu:
            x = F.relu(x)
        if self.norm:
            x = self.dp(x)
        return x
    
    def forward(self, in_feat, g):
        h = self.lin(in_feat)
        ori = h
        for i in range(self.prop_step):
            if i != 0:
                if self.res:
                    h = h + ori
                h = self._apply_norm_and_activation(h, i)
            if self.multilayer:
                h = self.convs[i](h, g)
            else:
                h = self.conv(h, g)
        return h

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
                 separate_att=False, unsqueeze=False, norm_affine=True):
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

    def forward(self, x_context, x_target):
        # x_context = self.context_norm1(x_context)
        x_context_att, _ = self.self_att(x_context, x_context, x_context)
        x_context = x_context_att + x_context
        x_context = self.context_norm1(x_context)

        # x_context = self.context_norm2(x_context)
        x_context_fnn = self.ffn(x_context)
        if self.unsqueeze:
            x_context_fnn = x_context_fnn.unsqueeze(1)
        x_context = x_context_fnn + x_context
        x_context = self.context_norm2(x_context)
        
        # x_target = self.tar_norm1(x_target)
        if self.separate_att:
            x_target_att, _ = self.cross_att(x_target, x_context, x_context)
        else:
            x_target_att, _ = self.self_att(x_target, x_context, x_context)
        x_target = x_target_att + x_target
        x_target = self.tar_norm1(x_target)

        # x_target = self.tar_norm2(x_target)
        x_target_fnn = self.ffn(x_target)
        if self.unsqueeze:
            x_target_fnn = x_target_fnn.unsqueeze(1)
        x_target = x_target_fnn + x_target
        x_target = self.tar_norm2(x_target)

        return x_context, x_target

class PFNPredictorNodeCls(nn.Module):
    def __init__(self, hidden_dim, nhead=1, num_layers=2, mlp_layers=2, dropout=0.2, 
                 norm=False, separate_att=False, degree=False, att=None, mlp=None, sim='dot', 
                 padding='zero', norm_affine=True, normalize=False):
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
                unsqueeze=True
            ) for _ in range(num_layers)
        ])
        self.degree = degree
        self.att = att
        self.mlp_pool = mlp
        self.normalize = normalize
    
    def forward(self, data, context_x, target_x, context_y, class_x):
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
        
        # Step 4: Prepare for transformer (add sequence dimension)
        context_tokens = context_tokens.unsqueeze(1)  # [num_context, 1, 2*hidden_dim]
        target_tokens = target_tokens.unsqueeze(1)    # [num_target, 1, 2*hidden_dim]
        
        # Step 5: Process through transformer layers
        for layer in self.transformer_row:
            context_tokens, target_tokens = layer(context_tokens, target_tokens)
        
        # Step 6: Extract refined label embeddings
        context_tokens = context_tokens.squeeze(1)  # [num_context, 2*hidden_dim]
        target_tokens = target_tokens.squeeze(1)    # [num_target, 2*hidden_dim]
        context_label_emb = context_tokens[:, self.hidden_dim:]  # [num_context, hidden_dim]
        target_label_emb = target_tokens[:, self.hidden_dim:]    # [num_target, hidden_dim]
        
        # Step 7: Compute new class embeddings by pooling refined label embeddings
        class_emb = process_node_features(
            context_label_emb,
            data,
            degree_normalize=self.degree,
            attention_pool_module=self.att,
            mlp_module=self.mlp_pool,
            normalize=self.normalize
        )

        data.final_class_h = class_emb

        # class_emb = F.normalize(class_emb, p=2, dim=1)
        
        # Step 8: Compute logits using similarity (dot product)
        if self.sim == 'dot':
            logits = torch.matmul(target_label_emb, class_emb.t())
        elif self.sim == 'cos':
            target_label_emb = F.normalize(target_label_emb, p=2, dim=-1)
            class_emb = F.normalize(class_emb, p=2, dim=-1)
            logits = torch.matmul(target_label_emb, class_emb.t())
        elif self.sim == 'mlp':
            target_label_emb = self.sim_mlp(target_label_emb)
            class_emb = self.mlp(class_emb)
            logits = torch.matmul(target_label_emb, class_emb.t())
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

    def forward(self, context_h_input, context_y, num_classes=None): # Added num_classes
        if num_classes is None:
            if context_y.numel() == 0: # Handle empty context_y
                # Output shape should be [0, nhead * in_channels] or handle as error
                return torch.empty(0, self.nhead * self.in_channels, device=context_h_input.device)
            num_classes = context_y.max().item() + 1
        
        if context_h_input.numel() == 0: # Handle empty input context_h
             return torch.zeros(num_classes, self.nhead * self.in_channels, device=context_h_input.device)


        context_h_ori_dropout = F.dropout(context_h_input, p=self.dp, training=self.training)
        
        context_h_transformed = self.lin(context_h_ori_dropout)
        context_h_transformed = context_h_transformed.view(-1, self.nhead, self.out_channels)
        
        att_score = self.att(context_h_transformed).squeeze(-1)
        att_score = F.leaky_relu(att_score, negative_slope=0.2)
        att_weights = scatter_softmax(att_score, context_y, dim=0) # [N_ctx, nhead]
        
        att_h = context_h_transformed * att_weights.unsqueeze(-1)

        pooled_h = torch.zeros(num_classes, self.nhead * self.out_channels, device=context_h_input.device)
        att_h = att_h.view(-1, self.nhead * self.out_channels)

        pooled_h = torch.scatter_reduce(
            pooled_h, 0, context_y.view(-1, 1).expand(-1, att_h.size(1)), att_h,
            reduce='sum', include_self=False
        )
        
        return pooled_h