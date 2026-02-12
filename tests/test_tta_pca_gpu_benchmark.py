import os
import time

import pytest

psutil = pytest.importorskip("psutil")
torch = pytest.importorskip("torch")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value


def _bench_cpu_wall(proc, device, iters, fn):
    torch.cuda.synchronize(device)
    wall_start = time.perf_counter()
    cpu_start_times = proc.cpu_times()
    cpu_start = cpu_start_times.user + cpu_start_times.system

    out = None
    for _ in range(iters):
        out = fn()

    torch.cuda.synchronize(device)
    wall_end = time.perf_counter()
    cpu_end_times = proc.cpu_times()
    cpu_end = cpu_end_times.user + cpu_end_times.system

    wall = wall_end - wall_start
    cpu = cpu_end - cpu_start
    cpu_over_wall = (cpu / wall * 100.0) if wall > 1e-9 else 0.0
    return out, wall, cpu, cpu_over_wall


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for this benchmark test")
def test_tta_pca_gpu_cpu_profile_report():
    """
    Diagnostic benchmark for NC TTA PCA path.

    Run with:
        pytest -q -s tests/test_tta_pca_gpu_benchmark.py

    Optional env vars:
        TTA_PCA_BENCH_N=20000
        TTA_PCA_BENCH_D=1000
        TTA_PCA_BENCH_Q=128
        TTA_PCA_BENCH_ITERS=5
    """
    device = torch.device("cuda:0")
    proc = psutil.Process(os.getpid())

    n = _env_int("TTA_PCA_BENCH_N", 20000)
    d = _env_int("TTA_PCA_BENCH_D", 1000)
    q_requested = _env_int("TTA_PCA_BENCH_Q", 128)
    iters = max(1, _env_int("TTA_PCA_BENCH_ITERS", 5))
    q = max(1, min(q_requested, n, d))

    x = torch.randn(n, d, device=device, dtype=torch.float32)

    # Warmup to avoid one-time kernel/JIT effects dominating measurements.
    warm_rows = min(4000, n)
    warm_q = max(1, min(64, warm_rows, d))
    _ = torch.pca_lowrank(x[:warm_rows], q=warm_q)
    x_warm_centered = x[:warm_rows] - x[:warm_rows].mean(dim=0, keepdim=True)
    _ = torch.linalg.eigh(x_warm_centered.t().matmul(x_warm_centered))
    torch.cuda.synchronize(device)

    pca_out, pca_wall, pca_cpu, pca_ratio = _bench_cpu_wall(
        proc,
        device,
        iters,
        lambda: torch.pca_lowrank(x, q=q),
    )

    def _cov_eigh_project():
        x_centered = x - x.mean(dim=0, keepdim=True)
        cov = x_centered.t().matmul(x_centered)
        _, eigvecs = torch.linalg.eigh(cov)
        proj = x_centered.matmul(eigvecs[:, -q:])
        return proj

    cov_out, cov_wall, cov_cpu, cov_ratio = _bench_cpu_wall(
        proc,
        device,
        iters,
        _cov_eigh_project,
    )

    pca_u, pca_s, pca_v = pca_out

    print(
        f"[TTA_PCA_BENCH] shape=({n},{d}) q={q} iters={iters} device={torch.cuda.get_device_name(0)}"
    )
    print(
        f"[TTA_PCA_BENCH] pca_lowrank: wall={pca_wall:.3f}s cpu={pca_cpu:.3f}s cpu/wall={pca_ratio:.1f}%"
    )
    print(
        f"[TTA_PCA_BENCH] cov_eigh_proj: wall={cov_wall:.3f}s cpu={cov_cpu:.3f}s cpu/wall={cov_ratio:.1f}%"
    )

    # Sanity checks so the test fails if kernels are not actually running on CUDA tensors.
    assert pca_u.is_cuda
    assert pca_s.is_cuda
    assert pca_v.is_cuda
    assert cov_out.is_cuda

    assert pca_u.shape[0] == n
    assert pca_v.shape[0] == d
    assert pca_v.shape[1] == q
    assert cov_out.shape == (n, q)

    assert pca_wall > 0.0
    assert cov_wall > 0.0
