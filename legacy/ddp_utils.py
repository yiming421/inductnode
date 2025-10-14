import os
import torch
import torch.distributed as dist
import signal
import sys

def setup_ddp(rank, world_size, port=12355):
    """Setup distributed training with better error handling"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    
    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print(f"[RANK {rank}] Received signal {signum}, cleaning up...")
        cleanup_ddp()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    print(f"[RANK {rank}] Setting up DDP...", flush=True)
    
    try:
        # Initialize process group with timeout
        dist.init_process_group(
            "nccl", 
            rank=rank, 
            world_size=world_size,
            timeout=torch.distributed.default_pg_timeout * 2  # Double the timeout
        )
        torch.cuda.set_device(rank)
        print(f"[RANK {rank}] Using GPU {torch.cuda.current_device()}, Device: cuda:{rank}", flush=True)
        
        # Test communication
        test_tensor = torch.tensor([rank], device=f'cuda:{rank}')
        dist.all_reduce(test_tensor)
        print(f"[RANK {rank}] DDP communication test passed", flush=True)
        
    except Exception as e:
        print(f"[RANK {rank}] DDP setup failed: {e}", flush=True)
        raise e

def cleanup_ddp():
    """Clean up distributed training with better error handling"""
    try:
        if dist.is_initialized():
            print(f"[RANK {dist.get_rank()}] Cleaning up DDP...", flush=True)
            dist.destroy_process_group()
            print(f"[RANK {dist.get_rank()}] DDP cleanup complete", flush=True)
    except Exception as e:
        print(f"Error during DDP cleanup: {e}", flush=True)
    finally:
        # Force cleanup CUDA context
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize() 