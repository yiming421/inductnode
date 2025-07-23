"""
Enhanced GPU Usage Monitoring for DDP Training

This module provides comprehensive monitoring of:
1. GPU memory usage (allocated, reserved, free)
2. GPU utilization and compute efficiency  
3. DDP communication overhead
4. Data loading and duplication analysis
5. Cross-GPU synchronization costs
"""

import torch
import torch.distributed as dist
import time
import psutil
import os
from typing import Dict, List, Optional
from collections import defaultdict
import threading


class DDPGPUMonitor:
    """Comprehensive GPU and DDP monitoring"""
    
    def __init__(self, rank: int, world_size: int, device: torch.device):
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.start_time = time.time()
        
        # Monitoring data
        self.memory_history = []
        self.communication_times = []
        self.batch_times = []
        self.model_sizes = {}
        
        # Threading for continuous monitoring
        self.monitoring = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval: float = 5.0):
        """Start continuous GPU monitoring"""
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        if self.rank == 0:
            print(f"🔍 Started GPU monitoring (interval: {interval}s)")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        if self.rank == 0:
            print("⏹️  Stopped GPU monitoring")
    
    def _monitor_loop(self, interval: float):
        """Continuous monitoring loop"""
        while self.monitoring:
            self._collect_metrics()
            time.sleep(interval)
    
    def _collect_metrics(self):
        """Collect current GPU metrics"""
        if torch.cuda.is_available():
            # GPU memory info
            allocated = torch.cuda.memory_allocated(self.device) / 1e9
            reserved = torch.cuda.memory_reserved(self.device) / 1e9
            props = torch.cuda.get_device_properties(self.device)
            total = props.total_memory / 1e9
            
            # System memory
            system_mem = psutil.virtual_memory()
            
            metrics = {
                'timestamp': time.time() - self.start_time,
                'gpu_allocated': allocated,
                'gpu_reserved': reserved, 
                'gpu_total': total,
                'gpu_free': total - reserved,
                'system_used': system_mem.used / 1e9,
                'system_total': system_mem.total / 1e9,
                'rank': self.rank
            }
            
            self.memory_history.append(metrics)
    
    def get_model_memory_breakdown(self, models: List[torch.nn.Module], names: List[str]):
        """Analyze memory usage per model component"""
        breakdown = {}
        
        for model, name in zip(models, names):
            if model is not None:
                param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
                buffer_memory = sum(b.numel() * b.element_size() for b in model.buffers()) / 1e9
                
                # Check if DDP wrapped
                is_ddp = hasattr(model, 'module')
                
                breakdown[name] = {
                    'parameters_gb': param_memory,
                    'buffers_gb': buffer_memory,
                    'total_gb': param_memory + buffer_memory,
                    'is_ddp_wrapped': is_ddp,
                    'parameter_count': sum(p.numel() for p in model.parameters())
                }
        
        self.model_sizes = breakdown
        return breakdown
    
    def measure_data_loading_overhead(self, data_loader, num_batches: int = 5):
        """Measure data loading and transfer overhead"""
        if self.rank == 0:
            print(f"📊 Measuring data loading overhead ({num_batches} batches)...")
        
        times = []
        for i, batch in enumerate(data_loader):
            if i >= num_batches:
                break
                
            start_time = time.time()
            
            # Move to GPU if needed
            if hasattr(batch, 'to'):
                batch = batch.to(self.device)
            elif isinstance(batch, (list, tuple)):
                batch = [b.to(self.device) if hasattr(b, 'to') else b for b in batch]
            
            transfer_time = time.time() - start_time
            times.append(transfer_time)
        
        avg_time = sum(times) / len(times) if times else 0
        if self.rank == 0:
            print(f"  Average data transfer time: {avg_time*1000:.2f}ms per batch")
        
        return avg_time
    
    def measure_ddp_communication_overhead(self):
        """Measure DDP communication costs"""
        if self.world_size == 1:
            return None
        
        if self.rank == 0:
            print("📡 Measuring DDP communication overhead...")
        
        # Create dummy tensor for communication test
        test_tensor = torch.randn(1000, 1000, device=self.device)
        
        # Measure AllReduce time
        start_time = time.time()
        dist.all_reduce(test_tensor)
        allreduce_time = time.time() - start_time
        
        # Measure Broadcast time  
        start_time = time.time()
        dist.broadcast(test_tensor, src=0)
        broadcast_time = time.time() - start_time
        
        # Measure Gather time
        start_time = time.time()
        if self.rank == 0:
            gather_list = [torch.zeros_like(test_tensor) for _ in range(self.world_size)]
            dist.gather(test_tensor, gather_list, dst=0)
        else:
            dist.gather(test_tensor, dst=0)
        gather_time = time.time() - start_time
        
        comm_overhead = {
            'allreduce_ms': allreduce_time * 1000,
            'broadcast_ms': broadcast_time * 1000,
            'gather_ms': gather_time * 1000
        }
        
        if self.rank == 0:
            print(f"  AllReduce: {comm_overhead['allreduce_ms']:.2f}ms")
            print(f"  Broadcast: {comm_overhead['broadcast_ms']:.2f}ms") 
            print(f"  Gather: {comm_overhead['gather_ms']:.2f}ms")
        
        return comm_overhead
    
    def analyze_dataset_duplication(self, datasets: List, dataset_names: List[str]):
        """Analyze dataset memory usage and duplication"""
        if self.rank == 0:
            print("📋 Analyzing dataset memory usage...")
        
        dataset_info = {}
        total_dataset_memory = 0
        
        for dataset, name in zip(datasets, dataset_names):
            if hasattr(dataset, 'x') and hasattr(dataset.x, 'numel'):
                # Graph dataset
                node_features_mb = dataset.x.numel() * dataset.x.element_size() / 1e6
                edge_memory_mb = 0
                if hasattr(dataset, 'edge_index'):
                    edge_memory_mb = dataset.edge_index.numel() * dataset.edge_index.element_size() / 1e6
                
                total_mb = node_features_mb + edge_memory_mb
                total_dataset_memory += total_mb
                
                dataset_info[name] = {
                    'node_features_mb': node_features_mb,
                    'edge_memory_mb': edge_memory_mb,
                    'total_mb': total_mb,
                    'num_nodes': dataset.x.size(0) if hasattr(dataset, 'x') else 0,
                    'num_edges': dataset.edge_index.size(1) if hasattr(dataset, 'edge_index') else 0
                }
        
        # Calculate duplication cost
        duplication_cost_gb = (total_dataset_memory * self.world_size) / 1000
        
        if self.rank == 0:
            print(f"  Total dataset memory per GPU: {total_dataset_memory:.1f}MB")
            print(f"  Duplication cost across {self.world_size} GPUs: {duplication_cost_gb:.2f}GB")
            print(f"  Memory waste from duplication: {duplication_cost_gb - (total_dataset_memory/1000):.2f}GB")
        
        return dataset_info, duplication_cost_gb
    
    def print_memory_summary(self):
        """Print comprehensive memory usage summary"""
        if not torch.cuda.is_available():
            return
        
        current_metrics = self._get_current_metrics()
        
        print(f"\n{'='*60}")
        print(f"GPU {self.rank} Memory Summary")
        print(f"{'='*60}")
        print(f"GPU Memory:")
        print(f"  Allocated: {current_metrics['gpu_allocated']:.2f}GB")
        print(f"  Reserved:  {current_metrics['gpu_reserved']:.2f}GB") 
        print(f"  Free:      {current_metrics['gpu_free']:.2f}GB")
        print(f"  Total:     {current_metrics['gpu_total']:.2f}GB")
        print(f"  Usage:     {(current_metrics['gpu_reserved']/current_metrics['gpu_total']*100):.1f}%")
        
        print(f"\nSystem Memory:")
        print(f"  Used:      {current_metrics['system_used']:.2f}GB")
        print(f"  Total:     {current_metrics['system_total']:.2f}GB")
        print(f"  Usage:     {(current_metrics['system_used']/current_metrics['system_total']*100):.1f}%")
        
        if self.model_sizes:
            print(f"\nModel Memory Breakdown:")
            total_model_memory = 0
            for name, info in self.model_sizes.items():
                ddp_status = "DDP" if info['is_ddp_wrapped'] else "Single"
                print(f"  {name}: {info['total_gb']:.3f}GB ({info['parameter_count']:,} params) [{ddp_status}]")
                total_model_memory += info['total_gb']
            print(f"  Total Models: {total_model_memory:.3f}GB")
        
        print(f"{'='*60}")
    
    def _get_current_metrics(self):
        """Get current metrics snapshot"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1e9
            reserved = torch.cuda.memory_reserved(self.device) / 1e9
            props = torch.cuda.get_device_properties(self.device)
            total = props.total_memory / 1e9
            
            system_mem = psutil.virtual_memory()
            
            return {
                'gpu_allocated': allocated,
                'gpu_reserved': reserved,
                'gpu_total': total,
                'gpu_free': total - reserved,
                'system_used': system_mem.used / 1e9,
                'system_total': system_mem.total / 1e9
            }
        return {}
    
    def get_memory_efficiency_report(self):
        """Generate memory efficiency analysis"""
        if not self.memory_history:
            return "No monitoring data available"
        
        recent_metrics = self.memory_history[-10:]  # Last 10 measurements
        
        avg_allocated = sum(m['gpu_allocated'] for m in recent_metrics) / len(recent_metrics)
        avg_reserved = sum(m['gpu_reserved'] for m in recent_metrics) / len(recent_metrics)
        total_memory = recent_metrics[0]['gpu_total']
        
        efficiency = (avg_allocated / total_memory) * 100
        waste = ((avg_reserved - avg_allocated) / total_memory) * 100
        
        report = f"""
Memory Efficiency Report (GPU {self.rank}):
  Actual Usage: {efficiency:.1f}% ({avg_allocated:.2f}GB / {total_memory:.2f}GB)
  Reserved Waste: {waste:.1f}% ({avg_reserved - avg_allocated:.2f}GB unused)
  Memory Efficiency: {'Good' if efficiency > 70 else 'Moderate' if efficiency > 40 else 'Poor'}
  Recommendation: {'Increase batch size' if efficiency < 40 else 'Optimize memory usage' if waste > 20 else 'Good utilization'}
"""
        return report


def create_ddp_monitor(rank: int, world_size: int, device: torch.device) -> DDPGPUMonitor:
    """Factory function to create DDP monitor"""
    return DDPGPUMonitor(rank, world_size, device)


if __name__ == "__main__":
    # Test the monitoring functionality
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        monitor = DDPGPUMonitor(rank=0, world_size=1, device=device)
        monitor.print_memory_summary()
    else:
        print("CUDA not available for testing")
