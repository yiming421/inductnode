import time
import threading
import torch.distributed as dist
import torch


class DDPProcessMonitor:
    """Monitor DDP processes and handle hanging/failed processes"""
    
    def __init__(self, rank, world_size, timeout_minutes=30):
        self.rank = rank
        self.world_size = world_size
        self.timeout_seconds = timeout_minutes * 60
        self.last_activity = time.time()
        self.monitoring = False
        self.monitor_thread = None
        
    def update_activity(self):
        """Update the last activity timestamp"""
        self.last_activity = time.time()
        
    def start_monitoring(self):
        """Start monitoring the process"""
        if self.world_size <= 1:
            return  # No need to monitor single process
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print(f"[RANK {self.rank}] Started DDP monitoring (no timeout)")
        
    def stop_monitoring(self):
        """Stop monitoring the process"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            
    def _monitor_loop(self):
        """Main monitoring loop without timeout handling"""
        while self.monitoring:
            try:
                time.sleep(30)  # Check every 30 seconds
                
                if not self.monitoring:
                    break
                    
                # Only do periodic communication health check every 5 minutes
                elapsed = time.time() - self.last_activity
                if elapsed > 300:  # Check communication every 5 minutes
                    try:
                        if dist.is_initialized():
                            test_tensor = torch.tensor([self.rank], device=f'cuda:{self.rank}')
                            dist.all_reduce(test_tensor, async_op=False)
                            self.update_activity()
                    except Exception as e:
                        print(f"[RANK {self.rank}] Communication health check failed: {e}")
                        
            except Exception as e:
                print(f"[RANK {self.rank}] Monitor loop error: {e}")
                continue


def create_ddp_process_monitor(rank, world_size, timeout_minutes=30):
    """Create and return a DDP process monitor"""
    return DDPProcessMonitor(rank, world_size, timeout_minutes)
