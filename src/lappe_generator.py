"""
Shared LapPE generation utilities used by both:
- scripts/generate_lappe.py
- main training pipeline (on-demand LP LapPE generation)
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, to_dense_adj, to_undirected


EPS = 1e-6


def _normalize_eigvecs_l2(eig_vecs: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    denom = eig_vecs.norm(p=2, dim=0, keepdim=True).clamp_min(eps)
    return eig_vecs / denom


def _postprocess_lappe(
    evals_np: np.ndarray,
    evects_np: np.ndarray,
    max_freqs: int,
    skip_zero_freq: bool = True,
    eigvec_abs: bool = False,
) -> torch.Tensor:
    num_nodes = evects_np.shape[0]
    offset = int(np.clip((np.abs(evals_np) < EPS).sum(), 0, num_nodes)) if skip_zero_freq else 0
    sorted_idx = np.argsort(evals_np)
    use_idx = sorted_idx[offset:max_freqs + offset]

    eig_vecs = torch.from_numpy(np.real(evects_np[:, use_idx])).float()
    eig_vecs = _normalize_eigvecs_l2(eig_vecs)

    if num_nodes < max_freqs + offset:
        eig_vecs = F.pad(eig_vecs, (0, max_freqs + offset - num_nodes), value=float("nan"))

    if eigvec_abs:
        eig_vecs = eig_vecs.abs()

    return eig_vecs


def compute_lappe(
    edge_index: torch.Tensor,
    num_nodes: int,
    dim: int,
    laplacian_norm: str = "sym",
    eigvec_norm: str = "L2",
    skip_zero_freq: bool = True,
    eigvec_abs: bool = False,
):
    if eigvec_norm != "L2":
        raise ValueError(f"Only eigvec_norm='L2' is supported, got: {eigvec_norm}")

    undir_edge_index = to_undirected(edge_index, num_nodes=num_nodes)

    target_dim = int(dim)
    if target_dim <= 0:
        raise ValueError("LapPE dimension must be positive.")
    if num_nodes <= 1:
        return torch.zeros((num_nodes, target_dim), dtype=torch.float32), "trivial", {
            "requested_dim": int(dim),
            "solved_k": 0,
            "num_zero_eigs_in_solved_k": 0,
            "requested_skip_zero_freq": bool(skip_zero_freq),
            "effective_skip_zero_freq": bool(skip_zero_freq),
            "fallback_applied": False,
        }
    max_k = num_nodes - 1
    if target_dim >= num_nodes:
        target_dim = max_k

    lap_edge_index, lap_edge_weight = get_laplacian(
        undir_edge_index, normalization=laplacian_norm, num_nodes=num_nodes
    )

    is_large_graph = num_nodes >= 50000

    if is_large_graph and not torch.cuda.is_available():
        raise RuntimeError(
            f"Large-graph LapPE generation requires CUDA for sparse_lobpcg "
            f"(num_nodes={num_nodes}, dim={int(dim)}). CUDA is not available."
        )

    def _solve_once(k_solve: int):
        if is_large_graph:
            method_local = "sparse_lobpcg"
            device = torch.device("cuda")
            l_sparse = torch.sparse_coo_tensor(
                lap_edge_index.to(device),
                lap_edge_weight.to(device),
                (num_nodes, num_nodes),
            ).coalesce()
            x_init = torch.randn(num_nodes, k_solve, device=device)
            evals_torch, evects_torch = torch.lobpcg(
                l_sparse, k=k_solve, X=x_init, largest=False, niter=200
            )
        else:
            method_local = "dense_eigh"
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            l_dense = to_dense_adj(
                lap_edge_index,
                edge_attr=lap_edge_weight,
                max_num_nodes=num_nodes,
            )[0].to(device)
            try:
                evals_torch, evects_torch = torch.linalg.eigh(l_dense)
            except RuntimeError:
                method_local = "dense_eigh_cpu_fallback"
                evals_torch, evects_torch = torch.linalg.eigh(l_dense.cpu())
        return evals_torch.cpu().numpy(), evects_torch.cpu().numpy(), method_local

    # Strict non-zero mode with adaptive oversolve:
    # solve extra low-frequency modes so skipping zeros can still return target_dim.
    if skip_zero_freq:
        if is_large_graph:
            initial_k = min(max_k, max(target_dim + max(16, target_dim), target_dim))
            k_cap = min(max_k, max(target_dim + 256, target_dim * 8))
        else:
            initial_k = max_k
            k_cap = max_k
    else:
        initial_k = target_dim
        k_cap = target_dim

    k_solve = max(1, int(initial_k))
    method = "unknown"
    evals_np = None
    evects_np = None
    num_zero_eigs = 0
    available_nonzero = 0

    retries = 0
    max_retries = 4
    while True:
        evals_np, evects_np, method = _solve_once(k_solve)
        num_zero_eigs = int((np.abs(evals_np) < EPS).sum())
        available_nonzero = int(max(0, int(evals_np.shape[0]) - num_zero_eigs))

        if not skip_zero_freq or available_nonzero >= target_dim:
            break
        if not is_large_graph:
            break
        if retries >= max_retries or k_solve >= k_cap or k_solve >= max_k:
            break

        next_k = min(
            max_k,
            k_cap,
            max(k_solve * 2, target_dim + num_zero_eigs + max(8, target_dim // 2)),
        )
        if next_k <= k_solve:
            break
        k_solve = int(next_k)
        retries += 1

    eig_vecs = _postprocess_lappe(
        evals_np=evals_np,
        evects_np=evects_np,
        max_freqs=target_dim,
        skip_zero_freq=skip_zero_freq,
        eigvec_abs=eigvec_abs,
    )

    fallback_applied = False
    effective_skip_zero = bool(skip_zero_freq)
    if skip_zero_freq and available_nonzero < target_dim:
        raise RuntimeError(
            "LapPE generation failed after skipping zero-frequency eigenvectors: "
            f"requested_dim={int(dim)}, obtained_dim={int(available_nonzero)}, "
            f"solved_k={int(evals_np.shape[0])}, num_zero_eigs_in_solved_k={int(num_zero_eigs)}. "
            "This graph appears to have too many zero modes for the current solver budget. "
            "Use a different PE or a solver strategy that explicitly computes enough non-zero frequencies."
        )

    meta = {
        "requested_dim": int(dim),
        "solved_k": int(evals_np.shape[0]),
        "num_zero_eigs_in_solved_k": int(num_zero_eigs),
        "requested_skip_zero_freq": bool(skip_zero_freq),
        "effective_skip_zero_freq": bool(effective_skip_zero),
        "fallback_applied": bool(fallback_applied),
    }
    return eig_vecs.cpu(), method, meta


def compute_lappe_for_data(
    data,
    dim: int,
    laplacian_norm: str = "sym",
    eigvec_norm: str = "L2",
    skip_zero_freq: bool = True,
    eigvec_abs: bool = False,
):
    if not hasattr(data, "edge_index") or data.edge_index is None:
        raise ValueError("Cannot compute LapPE: data.edge_index is missing")
    if not hasattr(data, "num_nodes") or data.num_nodes is None:
        raise ValueError("Cannot compute LapPE: data.num_nodes is missing")
    return compute_lappe(
        edge_index=data.edge_index,
        num_nodes=int(data.num_nodes),
        dim=int(dim),
        laplacian_norm=laplacian_norm,
        eigvec_norm=eigvec_norm,
        skip_zero_freq=skip_zero_freq,
        eigvec_abs=eigvec_abs,
    )


def save_lappe_tensor(
    lappe_tensor: torch.Tensor,
    dataset_name: str,
    output_root: str,
):
    dataset_dir = dataset_name.replace("-", "_")
    save_dir = Path(output_root) / dataset_dir / "pe_stats_LapPE" / "1.0"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "data.pt"
    torch.save(lappe_tensor.cpu(), save_path)
    return save_path


def generate_and_save_lappe_for_data(
    data,
    dataset_name: str,
    output_root: str,
    dim: int,
    skip_zero_freq: bool = True,
    eigvec_abs: bool = False,
):
    lappe_tensor, method, meta = compute_lappe_for_data(
        data=data,
        dim=dim,
        laplacian_norm="sym",
        eigvec_norm="L2",
        skip_zero_freq=skip_zero_freq,
        eigvec_abs=eigvec_abs,
    )
    save_path = save_lappe_tensor(lappe_tensor, dataset_name, output_root)
    return lappe_tensor, str(save_path), method, meta
