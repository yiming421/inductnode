import os
import torch
import torch.distributed as dist

def setup_ddp(rank, world_size, port=12355):
    """Setup distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    print(f"[RANK {rank}] Setting up DDP...", flush=True)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    print(f"[RANK {rank}] Using GPU {torch.cuda.current_device()}, Device: cuda:{rank}", flush=True)

def cleanup_ddp():
    """Clean up distributed training"""
    dist.destroy_process_group() 