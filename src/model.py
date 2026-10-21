import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree
from torch_sparse.matmul import spmm_add
from .utils import process_node_features

class PureGCNConv(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, adj_t):
        norm = torch.rsqrt_(1 + adj_t.sum(dim=-1)).reshape(-1, 1)
        x = norm * x
        x = spmm_add(adj_t, x) + x
        x = norm * x
        return x

class PureGCN_v1(nn.Module):

    def __init__(self, input_dim, num_layers=2, hidden=256, dp=0, norm=False, norm_affine=True):
        super().__init__()
        self.lin = nn.Linear(input_dim, hidden) if input_dim != hidden else nn.Identity()
        self.conv = PureGCNConv()
        self.num_layers = num_layers
        self.dp = dp
        self.norm = norm
        if self.norm:
            self.norms = nn.ModuleList([nn.LayerNorm(hidden, elementwise_affine=norm_affine) for _ in range(num_layers)])

    def forward(self, x, adj_t, batch=None):
        x = self.lin(x)
        ori = x
        for i in range(self.num_layers):
            if i != 0:
                if self.norm:
                    x = self.norms[i](x)
                if self.dp > 0:
                    x = F.dropout(x, p=self.dp, training=self.training)
            x = self.conv(x, adj_t)
        return x

class MLP(nn.Module):

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2, norm=False, tailact=False, norm_affine=True):
        super(MLP, self).__init__()
        self.lins = torch.nn.Sequential()
        self.num_layers = num_layers
        if num_layers == 0:
            assert in_channels == out_channels, f'num_layers=0 requires in_channels==out_channels, got {in_channels}!={out_channels}'
            pass
        elif num_layers == 1:
            self.lins.append(torch.nn.Linear(in_channels, out_channels))
            if tailact:
                if norm:
                    self.lins.append(nn.LayerNorm(out_channels, elementwise_affine=norm_affine))
                self.lins.append(nn.ReLU())
                if dropout > 0:
                    self.lins.append(nn.Dropout(dropout))
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
            return
        linear_layers = [m for m in self.lins if isinstance(m, nn.Linear)]
        if len(linear_layers) == 0:
            return
        for i, linear in enumerate(linear_layers):
            in_feat = linear.in_features
            out_feat = linear.out_features
            if i == len(linear_layers) - 1 and in_feat == out_feat:
                init.eye_(linear.weight)
            else:
                init.xavier_uniform_(linear.weight, gain=0.01)
            if linear.bias is not None:
                init.zeros_(linear.bias)

    def forward(self, x):
        if self.num_layers == 0:
            return x.squeeze()
        x = self.lins(x)
        return x.squeeze()

class PFNTransformerLayer(nn.Module):

    def __init__(self, hidden_dim, n_head=1, mlp_layers=2, dropout=0.2, norm=False, unsqueeze=False, norm_affine=True, norm_type='post', ffn_expansion_ratio=4):
        super(PFNTransformerLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.ffn_only = False
        self.self_att = None
        self.cross_att = None
        self.self_att = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_head, dropout=dropout)
        self.cross_att = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_head, dropout=dropout)
        self.ffn = MLP(in_channels=hidden_dim, hidden_channels=ffn_expansion_ratio * hidden_dim, out_channels=hidden_dim, num_layers=mlp_layers, dropout=dropout, norm=norm, tailact=False, norm_affine=norm_affine)
        self.context_norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=norm_affine)
        self.context_norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=norm_affine)
        self.context_norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=norm_affine)
        self.tar_norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=norm_affine)
        self.tar_norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=norm_affine)
        self.separate_att = True
        self.unsqueeze = unsqueeze
        self.norm_type = norm_type

    def forward(self, x_context, x_target):
        debug_collapse = False
        x_context_norm = self.context_norm1(x_context)
        x_context_att, _ = self.self_att(x_context_norm, x_context_norm, x_context_norm)
        x_context = x_context_att + x_context
        x_context_norm = self.context_norm2(x_context)
        orig_shape = x_context_norm.shape
        x_context_fnn = self.ffn(x_context_norm)
        if x_context_fnn.shape != orig_shape:
            x_context_fnn = x_context_fnn.view(orig_shape)
        x_context = x_context_fnn + x_context
        x_target_norm = self.tar_norm1(x_target)
        context_for_att = self.context_norm3(x_context)
        x_target_att, attn_weights = self.cross_att(x_target_norm, context_for_att, context_for_att)
        x_target = x_target_att + x_target
        x_target_norm = self.tar_norm2(x_target)
        orig_shape = x_target_norm.shape
        x_target_fnn = self.ffn(x_target_norm)
        if x_target_fnn.shape != orig_shape:
            x_target_fnn = x_target_fnn.view(orig_shape)
        x_target = x_target_fnn + x_target
        return (x_context, x_target)

class NodeClassificationHead(nn.Module):
    """Task-specific head for node classification."""

    def __init__(self, hidden_dim, dropout=0.2, norm=True, norm_affine=True, head_num_layers=2):
        super(NodeClassificationHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.sim = 'dot'
        self.target_input_norm = nn.LayerNorm(hidden_dim, elementwise_affine=norm_affine)
        self.context_input_norm = nn.LayerNorm(hidden_dim, elementwise_affine=norm_affine)
        if norm_affine:
            nn.init.ones_(self.target_input_norm.weight)
            nn.init.zeros_(self.target_input_norm.bias)
            nn.init.ones_(self.context_input_norm.weight)
            nn.init.zeros_(self.context_input_norm.bias)
        self.proj = MLP(in_channels=hidden_dim, hidden_channels=hidden_dim, out_channels=hidden_dim, num_layers=head_num_layers, dropout=dropout, norm=norm, tailact=False, norm_affine=norm_affine)
        self.proj._init_identity_like()

    def forward(self, target_label_emb, context_label_emb, context_y, data, degree_normalize=False, attention_pool_module=None, mlp_module=None, normalize=False):
        from .utils import process_node_features
        target_label_emb = self.proj(target_label_emb)
        context_label_emb = self.proj(context_label_emb)
        if target_label_emb.dim() == 1:
            target_label_emb = target_label_emb.unsqueeze(0)
        if context_label_emb.dim() == 1:
            context_label_emb = context_label_emb.unsqueeze(0)
        class_emb = process_node_features(context_label_emb, data, degree_normalize=False, attention_pool_module=None, mlp_module=None, normalize=normalize)
        logits = torch.matmul(target_label_emb, class_emb.t())
        return (logits, class_emb)

class GraphClassificationHead(nn.Module):
    """Task-specific head for graph classification."""

    def __init__(self, hidden_dim, dropout=0.2, norm=True, norm_affine=True, head_num_layers=2):
        super(GraphClassificationHead, self).__init__()
        self.hidden_dim = hidden_dim
        self.sim = 'dot'
        self.target_input_norm = nn.LayerNorm(hidden_dim, elementwise_affine=norm_affine)
        self.context_input_norm = nn.LayerNorm(hidden_dim, elementwise_affine=norm_affine)
        if norm_affine:
            nn.init.ones_(self.target_input_norm.weight)
            nn.init.zeros_(self.target_input_norm.bias)
            nn.init.ones_(self.context_input_norm.weight)
            nn.init.zeros_(self.context_input_norm.bias)
        self.proj = MLP(in_channels=hidden_dim, hidden_channels=hidden_dim, out_channels=hidden_dim, num_layers=head_num_layers, dropout=dropout, norm=norm, tailact=False, norm_affine=norm_affine)
        self.proj._init_identity_like()

    def forward(self, target_label_emb, context_label_emb, context_y, class_x, attention_pool_module=None, mlp_module=None, normalize=False):
        target_label_emb = self.proj(target_label_emb)
        context_label_emb = self.proj(context_label_emb)
        if target_label_emb.dim() == 1:
            target_label_emb = target_label_emb.unsqueeze(0)
        if context_label_emb.dim() == 1:
            context_label_emb = context_label_emb.unsqueeze(0)
        device = target_label_emb.device
        num_classes = context_y.max().item() + 1
        actual_hidden_dim = target_label_emb.size(-1)
        class_emb = torch.zeros(num_classes, actual_hidden_dim, device=device, dtype=target_label_emb.dtype)
        for class_idx in range(num_classes):
            class_mask = context_y == class_idx
            if class_mask.any():
                class_emb[class_idx] = context_label_emb[class_mask].mean(dim=0)
        if normalize:
            class_emb = F.normalize(class_emb, p=2, dim=-1)
        logits = torch.matmul(target_label_emb, class_emb.t())
        return (logits, class_emb)

class MPLPLinkPredictionHead(nn.Module):
    """
    Link Prediction Head using MPLP randomized structural features.
    """

    def __init__(self, hidden_dim, dropout=0.2, signature_dim=64, num_hops=2, feature_combine='hadamard', head_num_layers=2, prop_type='combine', edge_feat_dim=None, signature_sampling='torchhd', use_subgraph=True, use_degree='none'):
        super().__init__()
        self.feature_combine = feature_combine
        self.use_degree = use_degree
        self.use_context_calibrate = False
        from .node_label import RandomizedNodeLabeling
        self.node_labeling = RandomizedNodeLabeling(signature_dim=signature_dim, num_hops=num_hops, prop_type=prop_type, signature_sampling=signature_sampling, use_subgraph=use_subgraph)
        structural_input_dim = 15 if prop_type == 'combine' else 5
        self.struct_encode = MLP(in_channels=structural_input_dim, hidden_channels=structural_input_dim, out_channels=structural_input_dim, num_layers=1, dropout=dropout, norm=True, tailact=True)
        edge_feat_dim = hidden_dim if edge_feat_dim is None else edge_feat_dim
        self.feat_encode = MLP(in_channels=edge_feat_dim, hidden_channels=hidden_dim, out_channels=hidden_dim, num_layers=2, dropout=dropout, norm=True)
        self.classifier = MLP(in_channels=hidden_dim + structural_input_dim, hidden_channels=hidden_dim, out_channels=1, num_layers=head_num_layers, dropout=dropout, norm=True)
        self.struct_score = None
        self.feat_score = None
        self.node_weight_mlp = None
        if self.use_degree == 'mlp':
            self.node_weight_mlp = MLP(in_channels=hidden_dim + 1, hidden_channels=32, out_channels=1, num_layers=2, dropout=dropout, norm=True)
        self.last_gate_mean = None
        self.last_struct_score = None
        self.last_feat_score = None
        self.last_gate_value = None
        self.last_gate_calib_ms = None
        self.last_w_struct = None
        self.last_w_feat = None
        self.last_w_bias = None

    def score_components(self, edge_emb, adj_t, edges, node_emb=None):
        raise RuntimeError('Context calibration is not part of GILT.')

    def forward(self, edge_emb, adj_t, edges, node_emb=None):
        """
        Standard forward without context calibration (uses combined classifier).
        """
        node_weight = None
        if self.use_degree == 'aa':
            degree = adj_t.sum(dim=1)
            node_weight = torch.sqrt(1.0 / (torch.log(degree + 1.0) + 1e-06))
        elif self.use_degree == 'ra':
            degree = adj_t.sum(dim=1)
            node_weight = torch.sqrt(1.0 / (degree + 1e-06))
        elif self.use_degree == 'mlp':
            if node_emb is None:
                raise ValueError("use_degree='mlp' requires node_emb to be provided.")
            degree = adj_t.sum(dim=1).view(-1, 1)
            node_weight_feat = torch.cat([node_emb, degree], dim=1)
            node_weight = self.node_weight_mlp(node_weight_feat).squeeze(-1) + 1.0
        if edges.size(0) != 2:
            edges = edges.t()
        struct_feats = self.node_labeling(edges, adj_t, node_weight)
        struct_emb = self.struct_encode(struct_feats)
        feat_emb = self.feat_encode(edge_emb)
        self.last_gate_mean = None
        self.last_struct_score = None
        self.last_feat_score = None
        final_emb = torch.cat([struct_emb, feat_emb], dim=-1)
        return self.classifier(final_emb).squeeze(-1)

class PFNPredictorNodeCls(nn.Module):

    def __init__(self, hidden_dim, nhead=1, num_layers=2, mlp_layers=2, dropout=0.2, norm=False, norm_affine=True, normalize=False, norm_type='post', ffn_expansion_ratio=4, nc_proto_pooling_dropout=None, head_num_layers=2, mplp_signature_dim=64, mplp_num_hops=2, mplp_feature_combine='hadamard', mplp_prop_type='combine', mplp_signature_sampling='torchhd', mplp_use_subgraph=True, mplp_use_degree='none', nc_head_num_layers=None, lp_head_num_layers=None, gc_head_num_layers=None):
        super(PFNPredictorNodeCls, self).__init__()
        self.lp_head_type = 'mplp'
        self.hidden_dim = hidden_dim
        self.d_label = hidden_dim
        self.embed_dim = hidden_dim + self.d_label
        self.sim = 'dot'
        self.padding = 'zero'
        self.mplp_prop_type = mplp_prop_type
        self.mplp_signature_sampling = mplp_signature_sampling
        self.mplp_use_subgraph = mplp_use_subgraph
        self.mplp_use_degree = mplp_use_degree
        self.nc_proto_pooling = 'mean'
        self.head_num_layers = head_num_layers
        self.nc_head_num_layers = head_num_layers if nc_head_num_layers is None else nc_head_num_layers
        self.lp_head_num_layers = head_num_layers if lp_head_num_layers is None else lp_head_num_layers
        self.gc_head_num_layers = head_num_layers if gc_head_num_layers is None else gc_head_num_layers
        transformer_dim = self.embed_dim
        self.transformer_row = nn.ModuleList([PFNTransformerLayer(hidden_dim=transformer_dim, n_head=nhead, mlp_layers=mlp_layers, dropout=dropout, norm=norm, unsqueeze=False, norm_type=norm_type, ffn_expansion_ratio=ffn_expansion_ratio) for _ in range(num_layers)])
        self.normalize = normalize
        head_hidden_dim = hidden_dim
        self.nc_head = NodeClassificationHead(hidden_dim=head_hidden_dim, dropout=dropout, norm=norm, norm_affine=norm_affine, head_num_layers=self.nc_head_num_layers)
        lp_head_input_dim = head_hidden_dim + 0
        self.lp_head = MPLPLinkPredictionHead(hidden_dim=head_hidden_dim, dropout=dropout, signature_dim=mplp_signature_dim, num_hops=mplp_num_hops, feature_combine=mplp_feature_combine, head_num_layers=self.lp_head_num_layers, prop_type=self.mplp_prop_type, signature_sampling=self.mplp_signature_sampling, use_subgraph=self.mplp_use_subgraph, use_degree=self.mplp_use_degree, edge_feat_dim=lp_head_input_dim)
        self.gc_head = GraphClassificationHead(hidden_dim=head_hidden_dim, dropout=dropout, norm=norm, norm_affine=norm_affine, head_num_layers=self.gc_head_num_layers)

    def forward(self, data, context_x, target_x, context_y, class_x, task_type='node_classification', adj_t=None, lp_edges=None, node_emb=None, lp_cn_context=None, lp_cn_target=None, lp_context_edges=None):
        class_x_y = class_x[context_y]
        context_tokens = torch.cat([context_x, class_x_y], dim=1)
        padding = torch.zeros_like(target_x)
        target_tokens = torch.cat([target_x, padding], dim=1)
        context_tokens = context_tokens.unsqueeze(1)
        target_tokens = target_tokens.unsqueeze(1)
        for layer in self.transformer_row:
            context_tokens, target_tokens = layer(context_tokens, target_tokens)
        context_tokens = context_tokens.squeeze(1)
        target_tokens = target_tokens.squeeze(1)
        context_label_emb = context_tokens[:, self.hidden_dim:]
        target_label_emb = target_tokens[:, self.hidden_dim:]
        if task_type == 'node_classification':
            logits, class_emb = self.nc_head(target_label_emb=target_label_emb, context_label_emb=context_label_emb, context_y=context_y, data=data, degree_normalize=False, attention_pool_module=None, mlp_module=None, normalize=self.normalize)
        elif task_type == 'link_prediction':
            if adj_t is None or lp_edges is None:
                raise ValueError('MPLP head requires adj_t and lp_edges to be passed to forward()')

            def _check_finite(name, tensor):
                if tensor is None:
                    return
                finite = torch.isfinite(tensor)
                if finite.all():
                    return
                if finite.any():
                    t_f = tensor[finite]
                    stats = f'finite_mean={t_f.mean().item():.4f} finite_min={t_f.min().item():.4f} finite_max={t_f.max().item():.4f} num_nonfinite={(~finite).sum().item()}'
                else:
                    stats = 'no finite values'
                raise RuntimeError(f'MPLP numeric error: {name} has non-finite values. {stats}')
            logits = self.lp_head(target_label_emb, adj_t, lp_edges, node_emb=node_emb)
            self.lp_head.last_gate_calib_ms = None
            class_emb = class_x
        elif task_type == 'graph_classification':
            logits, class_emb = self.gc_head(target_label_emb=target_label_emb, context_label_emb=context_label_emb, context_y=context_y, class_x=class_x, attention_pool_module=None, mlp_module=None, normalize=self.normalize)
        else:
            raise ValueError(f"Unknown task_type: {task_type}. Choose 'node_classification', 'link_prediction', or 'graph_classification'.")
        return (logits, class_emb)

    def get_target_embeddings(self, context_x, target_x, context_y, class_x):
        class_x_y = class_x[context_y]
        context_tokens = torch.cat([context_x, class_x_y], dim=1)
        padding = torch.zeros_like(target_x)
        target_tokens = torch.cat([target_x, padding], dim=1)
        context_tokens = context_tokens.unsqueeze(1)
        target_tokens = target_tokens.unsqueeze(1)
        for layer in self.transformer_row:
            context_tokens, target_tokens = layer(context_tokens, target_tokens)
        target_tokens = target_tokens.squeeze(1)
        target_label_emb = target_tokens[:, self.hidden_dim:]
        return target_label_emb
