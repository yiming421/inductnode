import torch
import os
from typing import List, Union


def parse_gpu_spec(gpu_spec: Union[str, int, None]) -> List[int]:
    """
    Parse GPU specification and return list of GPU IDs to use.
    
    Args:
        gpu_spec: Can be:
            - None or 'auto': Use all available GPUs
            - int: Use single GPU with that ID
            - str: Comma-separated list of GPU IDs (e.g., "0,1,2,3")
    
    Returns:
        List of GPU IDs to use
        
    Raises:
        ValueError: If GPU specification is invalid or GPUs are not available
    """
    total_gpus = torch.cuda.device_count()
    
    if total_gpus == 0:
        raise ValueError("No CUDA devices are available.")
    
    # Auto mode - use all available GPUs
    if gpu_spec is None or gpu_spec == 'auto':
        return list(range(total_gpus))
    
    # Single GPU specified as int
    if isinstance(gpu_spec, int):
        if gpu_spec < 0 or gpu_spec >= total_gpus:
            raise ValueError(f"GPU {gpu_spec} is not available. Available GPUs: 0-{total_gpus-1}")
        return [gpu_spec]
    
    # Multiple GPUs specified as comma-separated string
    if isinstance(gpu_spec, str):
        try:
            gpu_ids = []
            for gpu_str in gpu_spec.split(','):
                gpu_id = int(gpu_str.strip())
                if gpu_id < 0 or gpu_id >= total_gpus:
                    raise ValueError(f"GPU {gpu_id} is not available. Available GPUs: 0-{total_gpus-1}")
                if gpu_id in gpu_ids:
                    raise ValueError(f"GPU {gpu_id} specified multiple times")
                gpu_ids.append(gpu_id)
            
            if not gpu_ids:
                raise ValueError("No valid GPU IDs found in specification")
                
            return sorted(gpu_ids)
        except ValueError as e:
            if "invalid literal for int()" in str(e):
                raise ValueError(f"Invalid GPU specification '{gpu_spec}'. Use comma-separated integers or 'auto'.")
            raise
    
    raise ValueError(f"Invalid GPU specification type: {type(gpu_spec)}")


def setup_cuda_visible_devices(gpu_ids: List[int]) -> None:
    """
    Set CUDA_VISIBLE_DEVICES environment variable to restrict visible GPUs.
    
    Args:
        gpu_ids: List of GPU IDs to make visible
    """
    gpu_str = ','.join(map(str, gpu_ids))
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_str
    print(f"Set CUDA_VISIBLE_DEVICES={gpu_str}")


def get_effective_world_size(gpu_ids: List[int]) -> int:
    """
    Get the effective world size after setting CUDA_VISIBLE_DEVICES.
    
    Args:
        gpu_ids: List of GPU IDs being used
        
    Returns:
        Number of GPUs that will be visible to torch
    """
    return len(gpu_ids)


def validate_gpu_availability(gpu_ids: List[int]) -> None:
    """
    Validate that all specified GPUs are available and functional.
    
    Args:
        gpu_ids: List of GPU IDs to validate
        
    Raises:
        RuntimeError: If any GPU is not available or not functional
    """
    total_gpus = torch.cuda.device_count()
    
    for gpu_id in gpu_ids:
        if gpu_id >= total_gpus:
            raise RuntimeError(f"GPU {gpu_id} is not available. Total GPUs: {total_gpus}")
        
        try:
            # Test GPU functionality
            device = torch.device(f'cuda:{gpu_id}')
            torch.zeros(1, device=device)
        except Exception as e:
            raise RuntimeError(f"GPU {gpu_id} is not functional: {e}")


def print_gpu_info(gpu_ids: List[int]) -> None:
    """
    Print information about the GPUs being used.
    
    Args:
        gpu_ids: List of GPU IDs being used
    """
    print(f"Using {len(gpu_ids)} GPU(s): {gpu_ids}")
    
    for i, gpu_id in enumerate(gpu_ids):
        props = torch.cuda.get_device_properties(gpu_id)
        print(f"  GPU {gpu_id} (Rank {i}): {props.name} - {props.total_memory / 1e9:.1f} GB")
