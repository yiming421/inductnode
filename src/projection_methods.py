"""
Projection Methods Module

This module contains different dimensionality reduction/projection methods:
- Random Orthogonal Projection (dense)
- Sparse Random Projection
- Orthogonal Noise Features

These are used for ablation studies comparing against PCA.
"""

import torch
import time


def generate_orthogonal_noise_features(num_nodes, target_dim, seed=42, device='cpu', dtype=torch.float32, rank=0):
    """
    Generate pure orthogonal noise features for ablation study.
    This creates random features using torch.nn.init.orthogonal_ to ensure orthogonality.

    Unlike random projection which projects existing features,
    this generates completely new features that are independent of input data.

    Args:
        num_nodes (int): Number of nodes/samples
        target_dim (int): Target dimension for features
        seed (int): Random seed for reproducibility
        device (str or torch.device): Device to create tensor on
        dtype (torch.dtype): Data type for the tensor
        rank (int): Process rank for logging

    Returns:
        torch.Tensor: Orthogonal noise features [num_nodes, target_dim]
    """
    if rank == 0:
        print(f"[Orthogonal Noise] Generating {num_nodes} x {target_dim} orthogonal noise features")

    start_time = time.time()

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Create empty tensor and initialize with orthogonal matrix
    if num_nodes >= target_dim:
        # Standard case: more samples than dimensions
        # Create [target_dim, num_nodes] and transpose to get [num_nodes, target_dim]
        orthogonal_features = torch.empty(target_dim, num_nodes, device=device, dtype=dtype)
        torch.nn.init.orthogonal_(orthogonal_features)
        orthogonal_features = orthogonal_features.t()  # Now [num_nodes, target_dim]
    else:
        # Edge case: fewer samples than dimensions
        # Create [num_nodes, target_dim] directly
        orthogonal_features = torch.empty(num_nodes, target_dim, device=device, dtype=dtype)
        torch.nn.init.orthogonal_(orthogonal_features)

    total_time = time.time() - start_time

    if rank == 0:
        print(f"[Orthogonal Noise] Generation completed in {total_time:.2f}s")
        print(f"  - Result shape: {orthogonal_features.shape}")
        print(f"  - Features are orthogonal (no input data dependency)")

    return orthogonal_features


def apply_random_orthogonal_projection(data_tensor, input_dim, target_dim, seed=42, rank=0):
    """
    Apply random orthogonal projection as an alternative to PCA.
    This normalizes the data first, then projects using a random orthogonal matrix.

    Args:
        data_tensor (torch.Tensor): Input data tensor [n_samples, n_features]
        input_dim (int): Input feature dimension
        target_dim (int): Target dimension after projection
        seed (int): Random seed for reproducibility
        rank (int): Process rank for logging

    Returns:
        torch.Tensor: Projected data [n_samples, target_dim]
    """
    if rank == 0:
        print(f"[Random Orthogonal] Projecting {data_tensor.shape} -> ({data_tensor.shape[0]}, {target_dim})")

    start_time = time.time()

    device = data_tensor.device
    dtype = data_tensor.dtype

    # Step 1: Normalize the data (center and scale) - like PCA does
    mean = data_tensor.mean(dim=0, keepdim=True)
    std = data_tensor.std(dim=0, keepdim=True)
    normalized_data = (data_tensor - mean) / (std + 1e-6)

    if rank == 0:
        print(f"[Random Orthogonal] Data normalized (centered and scaled)")

    # Step 2: Create random orthogonal projection matrix using nn.init.orthogonal_
    torch.manual_seed(seed)
    if data_tensor.is_cuda:
        torch.cuda.manual_seed(seed)

    # Initialize projection matrix
    # For input_dim x target_dim, we need to handle both cases
    if input_dim >= target_dim:
        # Standard case: projecting to lower dimension
        # Create [target_dim, input_dim] matrix and transpose to get [input_dim, target_dim]
        projection_matrix = torch.empty(target_dim, input_dim, device=device, dtype=dtype)
        torch.nn.init.orthogonal_(projection_matrix)
        projection_matrix = projection_matrix.t()  # Now [input_dim, target_dim]
    else:
        # Edge case: projecting to higher dimension
        projection_matrix = torch.empty(input_dim, target_dim, device=device, dtype=dtype)
        torch.nn.init.orthogonal_(projection_matrix)

    # Step 3: Apply projection: X @ Q
    projected_data = torch.mm(normalized_data, projection_matrix)

    total_time = time.time() - start_time

    if rank == 0:
        print(f"[Random Orthogonal] Projection completed in {total_time:.2f}s")
        print(f"  - Result shape: {projected_data.shape}")
        print(f"  - Data normalized before projection: Yes")

    return projected_data


def apply_sparse_random_projection(data_tensor, input_dim, target_dim, density=0.1, seed=42, rank=0):
    """
    Apply sparse random projection as an alternative to PCA.
    Uses a sparse random matrix with controlled density for efficient projection.

    Sparse random projection has several advantages:
    1. Better distance preservation (Johnson-Lindenstrauss lemma)
    2. Lower variance than dense random projection
    3. Computational efficiency due to sparsity
    4. Less prone to amplifying noise

    The projection matrix is constructed with entries:
    - -1 with probability density/2
    - 0 with probability 1-density
    - +1 with probability density/2
    Then normalized to unit column norm to prevent variance explosion.

    Args:
        data_tensor (torch.Tensor): Input data tensor [n_samples, n_features]
        input_dim (int): Input feature dimension
        target_dim (int): Target dimension after projection
        density (float): Density of non-zero entries (default 0.1 means 10% non-zero)
        seed (int): Random seed for reproducibility
        rank (int): Process rank for logging

    Returns:
        torch.Tensor: Projected data [n_samples, target_dim]
    """
    if rank == 0:
        print(f"[Sparse Random] Projecting {data_tensor.shape} -> ({data_tensor.shape[0]}, {target_dim}), density={density}")

    start_time = time.time()

    device = data_tensor.device
    dtype = data_tensor.dtype

    # Step 1: Normalize the data (center and scale)
    mean = data_tensor.mean(dim=0, keepdim=True)
    std = data_tensor.std(dim=0, keepdim=True)
    normalized_data = (data_tensor - mean) / (std + 1e-6)

    if rank == 0:
        print(f"[Sparse Random] Data normalized (centered and scaled)")

    # Step 2: Create sparse random projection matrix
    torch.manual_seed(seed)
    if data_tensor.is_cuda:
        torch.cuda.manual_seed(seed)

    # Generate random sparse matrix
    # Use the Achlioptas (2003) construction: {-1, 0, +1} with controlled sparsity
    # Generate random values in [0, 1]
    random_matrix = torch.rand(input_dim, target_dim, device=device, dtype=dtype)

    # Create sparse matrix with {-1, 0, +1} entries
    # - Values < density/2 → -1
    # - Values > 1-density/2 → +1
    # - Otherwise → 0
    projection_matrix = torch.zeros(input_dim, target_dim, device=device, dtype=dtype)
    projection_matrix[random_matrix < density/2] = -1.0
    projection_matrix[random_matrix > (1 - density/2)] = 1.0

    # CRITICAL: Normalize each column to unit norm (like PCA eigenvectors)
    # This prevents variance explosion and ensures fair comparison with PCA
    col_norms = torch.sqrt((projection_matrix ** 2).sum(dim=0, keepdim=True))
    projection_matrix = projection_matrix / (col_norms + 1e-10)

    # Calculate actual sparsity
    actual_density = (projection_matrix != 0).float().mean().item()

    if rank == 0:
        print(f"[Sparse Random] Projection matrix created with actual density: {actual_density:.3f}")
        print(f"[Sparse Random] Columns normalized to unit norm (prevents variance explosion)")

    # Step 3: Apply projection: X @ P
    projected_data = torch.mm(normalized_data, projection_matrix)

    total_time = time.time() - start_time

    if rank == 0:
        print(f"[Sparse Random] Projection completed in {total_time:.2f}s")
        print(f"  - Result shape: {projected_data.shape}")
        print(f"  - Non-zero entries in projection matrix: {actual_density*100:.1f}%")

    return projected_data
