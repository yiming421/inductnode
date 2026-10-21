import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
import math
try:
    from torch_geometric.utils import k_hop_subgraph as pyg_k_hop_subgraph, to_edge_index
except Exception:
    pyg_k_hop_subgraph = None
    to_edge_index = None

class RandomizedNodeLabeling(torch.nn.Module):

    def __init__(self, signature_dim=64, num_hops=2, prop_type='exact', signature_sampling='gaussian', use_subgraph=False, minimum_degree_onehot=-1):
        super().__init__()
        self.signature_dim = signature_dim
        self.num_hops = num_hops
        self.prop_type = prop_type
        self.signature_sampling = signature_sampling
        self.use_subgraph = use_subgraph
        self.minimum_degree_onehot = minimum_degree_onehot

    def _subgraph(self, edges, adj_t, k=2):
        if pyg_k_hop_subgraph is None or to_edge_index is None:
            raise RuntimeError('torch_geometric is required for subgraph extraction')
        row, col = edges
        nodes = torch.cat((row, col), dim=-1)
        edge_index, _ = to_edge_index(adj_t)
        subset, new_edge_index, inv, _ = pyg_k_hop_subgraph(nodes, k, edge_index=edge_index, num_nodes=adj_t.size(0), relabel_nodes=True)
        new_adj_t = SparseTensor(row=new_edge_index[0], col=new_edge_index[1], sparse_sizes=(subset.size(0), subset.size(0)))
        new_edges = inv.view(2, -1)
        return (new_adj_t, new_edges, subset)

    def get_random_node_vectors(self, adj_t, node_weight=None):
        """Generates random unit vectors, optionally scaled by node weights."""
        num_nodes = adj_t.size(0)
        device = adj_t.device()
        embedding = torch.zeros(num_nodes, self.signature_dim, device=device)
        if self.minimum_degree_onehot is not None and self.minimum_degree_onehot > 0:
            degree = adj_t.sum(dim=1)
            hub_mask = degree >= self.minimum_degree_onehot
            hub_count = int(hub_mask.sum().item())
            min_rand_dim = 64
            if hub_count + min_rand_dim > self.signature_dim:
                raise ValueError(f'{hub_count} hub nodes selected by minimum_degree_onehot={self.minimum_degree_onehot}, but signature_dim={self.signature_dim} is too small; increase signature_dim or threshold.')
            if hub_count > 0:
                hub_eye = F.one_hot(torch.arange(hub_count, device=device), num_classes=hub_count).float()
                embedding[hub_mask, :hub_count] = hub_eye
        else:
            hub_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
            hub_count = 0
        rand_dim = self.signature_dim - hub_count
        non_hub_count = int((~hub_mask).sum().item())
        if self.signature_sampling == 'torchhd':
            try:
                import torchhd
                scale = math.sqrt(1.0 / rand_dim)
                rand_vecs = torchhd.random(non_hub_count, rand_dim, device=device)
                rand_vecs = rand_vecs * scale
            except ImportError as error:
                raise ImportError('MPLP requires torch-hd; install the pinned environment.') from error
        elif self.signature_sampling == 'onehot':
            if rand_dim < non_hub_count:
                raise ValueError(f'signature_sampling=onehot requires rand_dim >= number of non-hub nodes; got rand_dim={rand_dim}, non_hub_nodes={non_hub_count}')
            rand_vecs = F.one_hot(torch.arange(non_hub_count, device=device), num_classes=rand_dim).float()
        else:
            rand_vecs = torch.randn(non_hub_count, rand_dim, device=device)
            rand_vecs = F.normalize(rand_vecs, p=2, dim=1)
        embedding[~hub_mask, hub_count:] = rand_vecs
        if node_weight is not None:
            embedding = embedding * node_weight.unsqueeze(1)
        return embedding

    def forward(self, edges, adj_t, node_weight=None):
        """
        Args:
            edges: [2, num_edges] Tensor of source and target node indices.
            adj_t: SparseTensor adjacency matrix.
            node_weight: Optional [num_nodes] Tensor (e.g. 1/log(d)).
        """
        if self.use_subgraph:
            adj_t, edges, subset = self._subgraph(edges, adj_t, k=2)
            if node_weight is not None:
                node_weight = node_weight[subset]
        num_nodes = adj_t.size(0)
        device = adj_t.device()
        x = self.get_random_node_vectors(adj_t, node_weight)
        one_hop_x = matmul(adj_t, x)
        degree = adj_t.sum(dim=1).view(-1, 1)
        two_iter_x = matmul(adj_t, one_hop_x)
        if self.prop_type == 'combine':
            two_hop_x = two_iter_x - one_hop_x - x
        else:
            two_hop_x = two_iter_x - degree * x
        deg2 = None
        row, col = edges
        x_u_1 = one_hop_x[row]
        x_v_1 = one_hop_x[col]
        x_u_2 = two_hop_x[row]
        x_v_2 = two_hop_x[col]

        def dot(a, b):
            return (a * b).sum(dim=-1)
        if self.prop_type == 'combine':
            if deg2 is None:
                deg = degree.view(-1)
                deg2_raw = matmul(adj_t, deg.view(-1, 1)).view(-1)
                deg2 = deg2_raw - deg - 1.0
                deg2 = torch.clamp(deg2, min=0.0)
            else:
                deg = degree.view(-1)
            deg_u = deg[row]
            deg_v = deg[col]
            deg_u_2 = deg2[row]
            deg_v_2 = deg2[col]
            count_1_1 = dot(x_u_1, x_v_1)
            count_1_2 = dot(x_u_1, x_v_2)
            count_2_1 = dot(x_u_2, x_v_1)
            count_2_2 = dot(x_u_2, x_v_2)
            count_1_inf = deg_u + deg_v - 2 * count_1_1 - count_1_2 - count_2_1
            count_2_inf = deg_u_2 + deg_v_2 - 2 * count_2_2 - count_1_2 - count_2_1
            x_u_iter = two_iter_x[row]
            x_v_iter = two_iter_x[col]
            x_u = x[row]
            x_v = x[col]
            comb_count_1_2 = dot(x_u_1, x_v_iter)
            comb_count_2_1 = dot(x_u_iter, x_v_1)
            comb_count_2_2 = dot(x_u_iter - deg_u.view(-1, 1) * x_u, x_v_iter - deg_v.view(-1, 1) * x_v)
            comb_count_self_1_2 = dot(x_u_1, x_u_iter)
            comb_count_self_2_1 = dot(x_v_1, x_v_iter)
            features = torch.stack([count_1_1, count_1_2, count_2_1, count_2_2, count_1_inf, count_2_inf, comb_count_1_2, comb_count_2_1, comb_count_2_2, comb_count_self_1_2, comb_count_self_2_1, deg_u, deg_v, deg_u_2, deg_v_2], dim=1)
        else:
            l11 = dot(x_u_1, x_v_1)
            l12 = dot(x_u_1, x_v_2) + dot(x_u_2, x_v_1)
            l22 = dot(x_u_2, x_v_2)
            deg_u_1 = (x_u_1 ** 2).sum(dim=-1)
            deg_v_1 = (x_v_1 ** 2).sum(dim=-1)
            deg_u_2 = (x_u_2 ** 2).sum(dim=-1)
            deg_v_2 = (x_v_2 ** 2).sum(dim=-1)
            l1inf = deg_u_1 + deg_v_1 - 2 * l11 - l12
            l2inf = deg_u_2 + deg_v_2 - 2 * l22 - l12
            features = torch.stack([l11, l12, l22, l1inf, l2inf], dim=1)
        return features
