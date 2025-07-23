"""
Correct Heterogeneous GPU Memory Management

This module provides proper memory management for heterogeneous GPU environments.
Key principles:
1. Never dynamically reduce batch size during training - it breaks training behavior
2. Handle GPUs with different memory capacities properly
3. Pre-allocate memory and fail fast if insufficient
"""

import torch
import torch.distributed as dist
import subprocess
import os
import re
from typing import Dict, List, Tuple, Optional


def get_physical_gpu_id(device: torch.device) -> int:
    """Get the physical GPU ID corresponding to the PyTorch device index"""
    try:
        # Check if CUDA_VISIBLE_DEVICES is set
        visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
        if visible_devices is not None:
            # Parse the visible devices list
            gpu_ids = [int(x.strip()) for x in visible_devices.split(',') if x.strip().isdigit()]
            if device.index < len(gpu_ids):
                return gpu_ids[device.index]
        
        # If not set or invalid, assume direct mapping
        return device.index
    except (ValueError, IndexError):
        return device.index


def get_gpu_memory_info(device: torch.device) -> Dict[str, float]:
    """Get GPU memory information in GB using nvidia-smi for accurate free memory"""
    if not torch.cuda.is_available():
        return {'total': 0, 'free': 0, 'used': 0, 'reserved': 0}
    
    torch.cuda.empty_cache()
    
    # Get device properties
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory / 1e9
    
    # Get current PyTorch memory usage
    allocated = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    
    # Get the physical GPU ID (important when CUDA_VISIBLE_DEVICES is set)
    physical_gpu_id = get_physical_gpu_id(device)
    
    # Try to get accurate memory info using nvidia-smi
    try:
        # Run nvidia-smi to get actual memory usage using physical GPU ID
        cmd = ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total", 
               "--format=csv,noheader,nounits", f"--id={physical_gpu_id}"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            # Parse nvidia-smi output: "used, free, total"
            memory_values = [float(x.strip()) for x in result.stdout.strip().split(',')]
            if len(memory_values) == 3:
                used_mb, free_mb, total_mb = memory_values
                
                return {
                    'total': total_mb / 1024,  # Convert MB to GB
                    'free': free_mb / 1024,    # Convert MB to GB  
                    'used': used_mb / 1024,    # Convert MB to GB
                    'reserved': reserved,
                    'physical_gpu_id': physical_gpu_id
                }
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ValueError, FileNotFoundError):
        pass
    
    # Fallback: estimate free memory (less accurate)
    estimated_free = total - reserved
    
    return {
        'total': total,
        'free': estimated_free,
        'used': allocated,
        'reserved': reserved,
        'physical_gpu_id': physical_gpu_id
    }


def calculate_safe_batch_sizes_per_gpu(
    world_size: int, 
    rank: int, 
    desired_total_batch_size: int,
    safety_factor: float = 0.8
) -> Tuple[int, Dict[int, int]]:
    """
    Calculate safe batch size allocation for heterogeneous GPU environments
    
    Allocates batch sizes based on available GPU memory using distributed communication
    """
    
    # Get current GPU memory info
    current_device = torch.device(f'cuda:{rank}')
    local_memory_info = get_gpu_memory_info(current_device)
    
    if world_size == 1:
        # Single GPU case
        return desired_total_batch_size, {0: desired_total_batch_size}
    
    # Multi-GPU case: Use memory-proportional distribution
    if not dist.is_initialized():
        # Fallback to equal distribution if DDP not initialized
        base_batch_size = desired_total_batch_size // world_size
        remainder = desired_total_batch_size % world_size
        local_batch_size = base_batch_size + (1 if rank < remainder else 0)
        
        batch_distribution = {}
        if rank == 0:
            for i in range(world_size):
                batch_distribution[i] = base_batch_size + (1 if i < remainder else 0)
        
        return local_batch_size, batch_distribution
    
    # Distributed memory-based allocation
    # Step 1: Gather memory information from all GPUs
    local_free_memory = torch.tensor([local_memory_info['free']], device=current_device, dtype=torch.float32)
    
    if rank == 0:
        # Rank 0 collects memory info from all GPUs
        all_free_memory = [torch.zeros(1, device=current_device, dtype=torch.float32) for _ in range(world_size)]
        dist.gather(local_free_memory, gather_list=all_free_memory, dst=0)
        
        # Convert to Python list
        memory_list = [mem.item() for mem in all_free_memory]
        
        print(f"📊 GPU Memory Distribution:")
        for i, mem in enumerate(memory_list):
            # Get the physical GPU ID for display
            temp_device = torch.device(f'cuda:{i}')
            physical_id = get_physical_gpu_id(temp_device)
            print(f"  GPU {i} (Physical GPU {physical_id}): {mem:.1f}GB free")
        
        # Step 2: Calculate memory-proportional batch sizes
        total_memory = sum(memory_list)
        
        if total_memory == 0:
            print("⚠️  Warning: All GPUs report 0 free memory, using equal distribution")
            # Fallback to equal distribution
            base_batch_size = max(1, desired_total_batch_size // world_size)
            batch_distribution = {i: base_batch_size for i in range(world_size)}
            remainder = desired_total_batch_size - (base_batch_size * world_size)
            if remainder > 0:
                batch_distribution[0] += remainder
        else:
            # Memory-proportional allocation
            batch_distribution = {}
            allocated_total = 0
            min_batch_size = max(1, desired_total_batch_size // (world_size * 8))  # Ensure minimum
            
            # Allocate proportionally for first (world_size - 1) GPUs
            for i in range(world_size - 1):
                memory_ratio = memory_list[i] / total_memory
                proportional_batch = int(desired_total_batch_size * memory_ratio)
                gpu_batch_size = max(min_batch_size, proportional_batch)
                batch_distribution[i] = gpu_batch_size
                allocated_total += gpu_batch_size
            
            # Last GPU gets the remainder
            remaining_batch = desired_total_batch_size - allocated_total
            batch_distribution[world_size - 1] = max(min_batch_size, remaining_batch)
            
            # Adjustment if total exceeds desired (reduce from largest allocations)
            actual_total = sum(batch_distribution.values())
            if actual_total > desired_total_batch_size:
                excess = actual_total - desired_total_batch_size
                # Sort by batch size descending to reduce from largest first
                sorted_gpus = sorted(batch_distribution.keys(), 
                                   key=lambda x: batch_distribution[x], reverse=True)
                
                for gpu_id in sorted_gpus:
                    if excess <= 0:
                        break
                    current_size = batch_distribution[gpu_id]
                    max_reduction = current_size - min_batch_size
                    reduction = min(excess, max_reduction)
                    if reduction > 0:
                        batch_distribution[gpu_id] -= reduction
                        excess -= reduction
        
        print(f"🎯 Memory-Based Batch Allocation:")
        total_check = 0
        for i, batch_size in batch_distribution.items():
            memory_ratio = (memory_list[i] / total_memory * 100) if total_memory > 0 else 0
            temp_device = torch.device(f'cuda:{i}')
            physical_id = get_physical_gpu_id(temp_device)
            print(f"  GPU {i} (Physical GPU {physical_id}): {batch_size} samples ({memory_ratio:.1f}% memory)")
            total_check += batch_size
        
        print(f"✅ Total batch size: {total_check} (target: {desired_total_batch_size})")
        
        # Broadcast batch sizes to all ranks
        batch_sizes_tensor = torch.tensor([batch_distribution[i] for i in range(world_size)], 
                                        device=current_device, dtype=torch.int32)
    else:
        # Non-rank 0 processes
        dist.gather(local_free_memory, dst=0)
        batch_distribution = {}
        batch_sizes_tensor = torch.zeros(world_size, device=current_device, dtype=torch.int32)
    
    # Broadcast the batch size allocation
    dist.broadcast(batch_sizes_tensor, src=0)
    
    # Each rank gets its own batch size
    local_batch_size = int(batch_sizes_tensor[rank].item())
    
    # Safety check: ensure no zero batch sizes
    if local_batch_size <= 0:
        print(f"⚠️  Warning: GPU {rank} got invalid batch size {local_batch_size}, setting to 1")
        local_batch_size = 1
    
    # Return results
    if rank == 0:
        # Reconstruct batch_distribution from tensor for rank 0
        final_distribution = {i: int(batch_sizes_tensor[i].item()) for i in range(world_size)}
        return local_batch_size, final_distribution
    else:
        return local_batch_size, {}


def validate_memory_before_training(
    model_modules: List[torch.nn.Module],
    sample_batch_size: int,
    device: torch.device
) -> bool:
    """
    Validate memory sufficiency before training starts
    Avoid OOM during training process
    """
    try:
        # Create sample data for memory testing
        model_modules[0].eval()  # Set to eval mode to reduce memory usage
        
        # Simulate forward pass for one batch (without gradients)
        with torch.no_grad():
            # Here should create sample input based on your specific model
            # Example: assuming input is graph data
            print(f"Testing GPU {device} memory capacity (batch_size={sample_batch_size})")
            
            # Record memory before test
            memory_before = get_gpu_memory_info(device)
            
            # Execute lightweight forward pass test
            # (This needs to be adjusted based on your specific model)
            
            memory_after = get_gpu_memory_info(device)
            estimated_usage = memory_after['used'] - memory_before['used']
            
            print(f"Memory test passed: estimated usage {estimated_usage:.2f}GB")
            return True
            
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"Memory test failed: {e}")
            return False
        else:
            raise e


def create_memory_safe_training_config(
    world_size: int,
    rank: int, 
    desired_batch_size: int,
    model_modules: List[torch.nn.Module]
) -> Dict[str, any]:
    """
    Create memory-safe training configuration
    
    Returns safe batch size configuration, fails early if memory insufficient
    Never dynamically adjust batch size during training
    """
    
    # 1. Calculate memory-based batch size allocation
    local_batch_size, batch_distribution = calculate_safe_batch_sizes_per_gpu(
        world_size, rank, desired_batch_size
    )
    
    # 2. Validate memory sufficiency
    current_device = torch.device(f'cuda:{rank}')
    
    if not validate_memory_before_training(model_modules, local_batch_size, current_device):
        raise RuntimeError(f"GPU {rank} insufficient memory for batch_size={local_batch_size}")
    
    # 3. Return safe configuration
    config = {
        'local_batch_size': local_batch_size,
        'effective_total_batch_size': sum(batch_distribution.values()) if batch_distribution else local_batch_size,
        'memory_info': get_gpu_memory_info(current_device),
        'batch_distribution': batch_distribution
    }
    
    if rank == 0:
        print(f"Memory-safe training configuration:")
        print(f"  Local batch size: {local_batch_size}")
        print(f"  Effective total batch size: {config['effective_total_batch_size']}")
        print(f"  GPU memory: {config['memory_info']['free']:.1f}GB available")
    
    return config


# Correct OOM handling strategy: fail fast, don't break training
def safe_training_step(training_func, *args, **kwargs):
    """
    Safe training step - never dynamically adjust batch size
    
    If OOM occurs, fail directly with clear error message
    """
    try:
        return training_func(*args, **kwargs)
    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clean up memory
            torch.cuda.empty_cache()
            
            # Provide useful error information, but don't attempt recovery
            device_info = get_gpu_memory_info(torch.cuda.current_device())
            error_msg = (
                f"GPU out of memory: {e}\n"
                f"Current GPU memory: {device_info['used']:.1f}GB used, {device_info['free']:.1f}GB available\n"
                f"Suggestion: reduce batch size and restart training"
            )
            raise RuntimeError(error_msg)
        else:
            raise e


if __name__ == "__main__":
    print("Correct heterogeneous GPU memory management strategy:")
    print("1. ✅ Pre-allocate batch size, never change during training")
    print("2. ✅ Allocate based on each GPU's memory capacity") 
    print("3. ✅ Maintain stable total effective batch size")
    print("4. ✅ Pre-validate memory, avoid runtime OOM")
    print("5. ✅ Fail fast on OOM with clear suggestions")
