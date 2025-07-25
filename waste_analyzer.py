"""
Distributed Training Waste Analyzer

This script measures the proportion of computational resources that are:
1. Truly distributed (efficient)
2. Redundantly replicated (wasted)
3. Communication overhead

Run this to quantify the inefficiencies in your current distributed setup.
"""

import torch
import torch.distributed as dist
import time
import psutil
import os
from typing import Dict, List, Tuple
import numpy as np


class DistributedWasteAnalyzer:
    """
    Analyzes computational and memory waste in distributed training.
    """
    
    def __init__(self, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.metrics = {
            'memory_usage': {},
            'computation_time': {},
            'redundancy_analysis': {},
            'efficiency_scores': {}
        }
        
    def measure_memory_usage(self, stage: str):
        """Measure current memory usage."""
        # GPU memory
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated() / 1e9  # GB
            gpu_reserved = torch.cuda.memory_reserved() / 1e9   # GB
        else:
            gpu_allocated = gpu_reserved = 0
            
        # CPU memory
        process = psutil.Process(os.getpid())
        cpu_memory = process.memory_info().rss / 1e9  # GB
        
        self.metrics['memory_usage'][stage] = {
            'rank': self.rank,
            'gpu_allocated_gb': gpu_allocated,
            'gpu_reserved_gb': gpu_reserved, 
            'cpu_memory_gb': cpu_memory,
            'timestamp': time.time()
        }
        
        if self.rank == 0:
            print(f"📊 {stage} - GPU: {gpu_allocated:.2f}GB, CPU: {cpu_memory:.2f}GB")
    
    def start_timing(self, operation: str):
        """Start timing an operation."""
        if operation not in self.metrics['computation_time']:
            self.metrics['computation_time'][operation] = {}
        
        self.metrics['computation_time'][operation]['start'] = time.time()
    
    def end_timing(self, operation: str):
        """End timing an operation."""
        if operation in self.metrics['computation_time'] and 'start' in self.metrics['computation_time'][operation]:
            duration = time.time() - self.metrics['computation_time'][operation]['start']
            self.metrics['computation_time'][operation]['duration'] = duration
            self.metrics['computation_time'][operation]['rank'] = self.rank
            
            if self.rank == 0:
                print(f"⏱️  {operation}: {duration:.3f}s")
    
    def analyze_data_loading_waste(self, dataset_names: List[str]) -> Dict:
        """
        Analyze waste in data loading phase.
        """
        total_datasets = len(dataset_names)
        datasets_per_rank = total_datasets // self.world_size if self.world_size > 1 else total_datasets
        
        current_waste = {
            'total_datasets_loaded': total_datasets * self.world_size,  # Each rank loads all
            'unique_datasets_needed': total_datasets,
            'redundancy_factor': self.world_size,
            'memory_waste_percentage': ((self.world_size - 1) / self.world_size) * 100,
            'optimal_datasets_per_rank': datasets_per_rank
        }
        
        self.metrics['redundancy_analysis']['data_loading'] = current_waste
        
        if self.rank == 0:
            print(f"\n🔍 Data Loading Waste Analysis:")
            print(f"   Current: {total_datasets} datasets × {self.world_size} ranks = {current_waste['total_datasets_loaded']} total loads")
            print(f"   Optimal: {total_datasets} datasets ÷ {self.world_size} ranks = {datasets_per_rank} loads per rank")
            print(f"   Memory waste: {current_waste['memory_waste_percentage']:.1f}%")
        
        return current_waste
    
    def analyze_computation_distribution(self, operations: List[str]) -> Dict:
        """
        Analyze which operations are truly distributed vs. replicated.
        """
        # Categorize operations
        replicated_ops = [
            'data_loading', 'graph_processing', 'node_embeddings', 
            'context_preparation', 'model_forward_full_graph'
        ]
        
        distributed_ops = [
            'edge_batch_processing', 'gradient_computation', 'loss_computation'
        ]
        
        communication_ops = [
            'gradient_synchronization', 'model_parameter_sync'
        ]
        
        analysis = {
            'replicated_operations': replicated_ops,
            'distributed_operations': distributed_ops, 
            'communication_operations': communication_ops,
            'replicated_percentage': 0,
            'distributed_percentage': 0,
            'communication_percentage': 0
        }
        
        # Calculate time percentages
        total_time = sum(
            self.metrics['computation_time'].get(op, {}).get('duration', 0) 
            for op in operations
        )
        
        if total_time > 0:
            replicated_time = sum(
                self.metrics['computation_time'].get(op, {}).get('duration', 0)
                for op in replicated_ops if op in operations
            )
            
            distributed_time = sum(
                self.metrics['computation_time'].get(op, {}).get('duration', 0)
                for op in distributed_ops if op in operations
            )
            
            communication_time = sum(
                self.metrics['computation_time'].get(op, {}).get('duration', 0)
                for op in communication_ops if op in operations
            )
            
            analysis['replicated_percentage'] = (replicated_time / total_time) * 100
            analysis['distributed_percentage'] = (distributed_time / total_time) * 100
            analysis['communication_percentage'] = (communication_time / total_time) * 100
        
        self.metrics['redundancy_analysis']['computation'] = analysis
        
        if self.rank == 0:
            print(f"\n🔍 Computation Distribution Analysis:")
            print(f"   Replicated work: {analysis['replicated_percentage']:.1f}% (wasted)")
            print(f"   Distributed work: {analysis['distributed_percentage']:.1f}% (efficient)")
            print(f"   Communication: {analysis['communication_percentage']:.1f}% (overhead)")
        
        return analysis
    
    def calculate_efficiency_scores(self) -> Dict:
        """
        Calculate overall efficiency scores.
        """
        # Memory efficiency
        data_loading = self.metrics['redundancy_analysis'].get('data_loading', {})
        memory_efficiency = 100 - data_loading.get('memory_waste_percentage', 0)
        
        # Computational efficiency  
        computation = self.metrics['redundancy_analysis'].get('computation', {})
        compute_efficiency = computation.get('distributed_percentage', 0)
        
        # Overall efficiency (weighted average)
        overall_efficiency = (memory_efficiency * 0.4 + compute_efficiency * 0.6)
        
        # Theoretical speedup vs actual speedup
        theoretical_speedup = self.world_size
        actual_speedup = compute_efficiency / 100 * self.world_size
        speedup_efficiency = (actual_speedup / theoretical_speedup) * 100
        
        scores = {
            'memory_efficiency_percent': memory_efficiency,
            'compute_efficiency_percent': compute_efficiency,
            'overall_efficiency_percent': overall_efficiency,
            'theoretical_speedup': theoretical_speedup,
            'actual_speedup': actual_speedup,
            'speedup_efficiency_percent': speedup_efficiency,
            'waste_factor': self.world_size / (actual_speedup if actual_speedup > 0 else 1)
        }
        
        self.metrics['efficiency_scores'] = scores
        
        if self.rank == 0:
            print(f"\n📈 Efficiency Scores:")
            print(f"   Memory Efficiency: {memory_efficiency:.1f}%")
            print(f"   Compute Efficiency: {compute_efficiency:.1f}%") 
            print(f"   Overall Efficiency: {overall_efficiency:.1f}%")
            print(f"   Speedup: {actual_speedup:.2f}× (vs {theoretical_speedup}× theoretical)")
            print(f"   Resource Waste Factor: {scores['waste_factor']:.2f}×")
        
        return scores
    
    def generate_optimization_recommendations(self) -> List[str]:
        """
        Generate specific recommendations to reduce waste.
        """
        print(f"[DEBUG] generate_optimization_recommendations called")
        
        recommendations = []
        
        try:
            # Memory recommendations
            print(f"[DEBUG] Getting memory waste data...")
            data_loading_metrics = self.metrics['redundancy_analysis'].get('data_loading', {})
            memory_waste = data_loading_metrics.get('memory_waste_percentage', 0)
            print(f"[DEBUG] Memory waste: {memory_waste}%")
            
            if memory_waste > 50:
                recommendations.append(
                    f"🎯 HIGH PRIORITY: Implement dataset distribution across ranks "
                    f"(will save {memory_waste:.1f}% memory)"
                )
            
            # Computation recommendations
            print(f"[DEBUG] Getting compute efficiency data...")
            efficiency_scores = self.metrics.get('efficiency_scores', {})
            compute_efficiency = efficiency_scores.get('compute_efficiency_percent', 0)
            print(f"[DEBUG] Compute efficiency: {compute_efficiency}%")
            
            if compute_efficiency < 30:
                recommendations.append(
                    f"🎯 HIGH PRIORITY: Current compute efficiency is only {compute_efficiency:.1f}%. "
                    f"Most computation is redundantly replicated across GPUs."
                )
            
            # Specific code fixes
            if memory_waste > 30:
                recommendations.append(
                    "💡 QUICK FIX: Replace data loading with: "
                    "my_datasets = train_datasets[rank::world_size]"
                )
            
            if compute_efficiency < 50:
                recommendations.append(
                    "💡 MEDIUM FIX: Use PyG's LinkNeighborLoader for memory-efficient subgraph sampling"
                )
            
            # Speedup potential
            waste_factor = efficiency_scores.get('waste_factor', 1)
            print(f"[DEBUG] Waste factor: {waste_factor}")
            
            if waste_factor > 2:
                recommendations.append(
                    f"⚡ POTENTIAL: Could achieve {waste_factor:.1f}× better resource utilization "
                    f"with proper distribution"
                )
            
            print(f"[DEBUG] Generated {len(recommendations)} recommendations successfully")
            return recommendations
            
        except Exception as e:
            print(f"[ERROR] Exception in generate_optimization_recommendations: {e}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            return []  # Return empty list on error
    
    def print_summary_report(self):
        """Print a comprehensive summary report."""
        print(f"[DEBUG] print_summary_report called on rank {self.rank}")
        
        if self.rank != 0:
            print(f"[DEBUG] Rank {self.rank} skipping report (not rank 0)")
            return
        
        print(f"[DEBUG] Rank 0 starting waste analysis report...")
        
        try:
            print("\n" + "="*80)
            print("🎯 DISTRIBUTED TRAINING WASTE ANALYSIS REPORT")
            print("="*80)
            
            # Debug: Check if metrics exist
            print(f"[DEBUG] Available metrics keys: {list(self.metrics.keys())}")
            
            # Efficiency scores
            scores = self.metrics.get('efficiency_scores', {})
            print(f"[DEBUG] Efficiency scores: {scores}")
            
            print(f"\n📊 CURRENT EFFICIENCY:")
            print(f"   Overall Efficiency: {scores.get('overall_efficiency_percent', 0):.1f}%")
            print(f"   Memory Utilization: {scores.get('memory_efficiency_percent', 0):.1f}%")
            print(f"   Compute Distribution: {scores.get('compute_efficiency_percent', 0):.1f}%")
            print(f"   Actual Speedup: {scores.get('actual_speedup', 0):.2f}× (vs {self.world_size}× possible)")
            
            # Waste breakdown
            data_loading = self.metrics['redundancy_analysis'].get('data_loading', {})
            computation = self.metrics['redundancy_analysis'].get('computation', {})
            
            print(f"[DEBUG] Data loading metrics: {data_loading}")
            print(f"[DEBUG] Computation metrics: {computation}")
            
            print(f"\n🔴 WASTE BREAKDOWN:")
            print(f"   Data Loading Redundancy: {data_loading.get('redundancy_factor', 1)}× replication")
            print(f"   Memory Waste: {data_loading.get('memory_waste_percentage', 0):.1f}%")
            print(f"   Replicated Computation: {computation.get('replicated_percentage', 0):.1f}%")
            print(f"   Communication Overhead: {computation.get('communication_percentage', 0):.1f}%")
            
            # Recommendations
            print(f"[DEBUG] Generating recommendations...")
            recommendations = self.generate_optimization_recommendations()
            print(f"[DEBUG] Generated {len(recommendations)} recommendations")
            
            if recommendations:
                print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
                for i, rec in enumerate(recommendations, 1):
                    print(f"   {i}. {rec}")
            
            print("\n" + "="*80)
            print(f"[DEBUG] Waste analysis report completed successfully")
            
        except Exception as e:
            print(f"[ERROR] Exception in print_summary_report: {e}")
            import traceback
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            raise
    
    def save_detailed_metrics(self, filepath: str):
        """Save detailed metrics to file for analysis."""
        import json
        
        if self.rank == 0:
            with open(filepath, 'w') as f:
                json.dump(self.metrics, f, indent=2, default=str)
            print(f"📁 Detailed metrics saved to: {filepath}")


def analyze_current_training_waste(rank: int, world_size: int, dataset_names: List[str]):
    """
    Main function to analyze waste in current training setup.
    
    Usage:
        # Add this to your training script
        analyzer = DistributedWasteAnalyzer(rank, world_size)
        analyze_current_training_waste(rank, world_size, train_dataset_names)
    """
    
    analyzer = DistributedWasteAnalyzer(rank, world_size)
    
    if rank == 0:
        print("🔍 Starting Distributed Training Waste Analysis...")
    
    # Measure baseline
    analyzer.measure_memory_usage("baseline")
    
    # Analyze data loading waste
    analyzer.analyze_data_loading_waste(dataset_names)
    
    # Simulate typical training operations timing
    operations = [
        'data_loading', 'graph_processing', 'node_embeddings',
        'edge_batch_processing', 'gradient_synchronization'
    ]
    
    # Mock timing for demonstration (replace with actual measurements)
    analyzer.start_timing('data_loading')
    time.sleep(0.1)  # Simulate data loading
    analyzer.end_timing('data_loading')
    
    analyzer.start_timing('graph_processing') 
    time.sleep(0.5)  # Simulate graph processing (replicated)
    analyzer.end_timing('graph_processing')
    
    analyzer.start_timing('node_embeddings')
    time.sleep(0.3)  # Simulate node embeddings (replicated)
    analyzer.end_timing('node_embeddings')
    
    analyzer.start_timing('edge_batch_processing')
    time.sleep(0.1)  # Simulate edge processing (distributed)
    analyzer.end_timing('edge_batch_processing')
    
    analyzer.start_timing('gradient_synchronization')
    time.sleep(0.05)  # Simulate gradient sync (communication)
    analyzer.end_timing('gradient_synchronization')
    
    # Analyze computation distribution
    analyzer.analyze_computation_distribution(operations)
    
    # Calculate efficiency scores
    analyzer.calculate_efficiency_scores()
    
    # Generate report
    analyzer.print_summary_report()
    
    # Save metrics
    analyzer.save_detailed_metrics(f"waste_analysis_rank_{rank}.json")
    
    return analyzer


if __name__ == "__main__":
    # Example usage
    print("🎯 Distributed Training Waste Analyzer")
    print("Run this with your actual training parameters")
    
    # Mock parameters for demonstration
    rank = 0
    world_size = 4
    dataset_names = ['ogbn-arxiv', 'CS', 'Physics', 'Computers', 'Photo']
    
    analyzer = analyze_current_training_waste(rank, world_size, dataset_names)
