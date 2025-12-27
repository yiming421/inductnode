"""
Centralized logging system for distributed training with configurable verbosity levels.
"""

import time
from typing import Dict, Any, Optional, List
from enum import Enum

class LogLevel(Enum):
    QUIET = 0      # Only critical errors and final results
    INFO = 1       # Standard training progress 
    DEBUG = 2      # Detailed debugging information
    VERBOSE = 3    # Everything including waste analysis details

class TrainingLogger:
    """
    Centralized logger for training with configurable verbosity and intervals.
    """
    
    def __init__(self, rank: int, world_size: int, log_level=LogLevel.INFO, 
                 log_interval: int = 1, eval_interval: int = 5, analysis_interval: int = 20):
        self.rank = rank
        self.world_size = world_size
        self.should_log = (rank == 0)  # Only rank 0 logs to avoid duplicate output
        
        # Handle log_level parameter - it can be LogLevel enum or string
        if isinstance(log_level, str):
            self.log_level = LogLevel[log_level.upper()]
        else:
            self.log_level = log_level
            
        self.log_interval = log_interval
        self.eval_interval = eval_interval  
        self.analysis_interval = analysis_interval
        
    def _should_print(self, level: LogLevel) -> bool:
        """Check if message should be printed based on current log level."""
        return self.should_log and level.value <= self.log_level.value
    
    def _format_time(self) -> str:
        """Format elapsed time."""
        elapsed = time.time() - self.start_time
        return f"[{elapsed:.1f}s]"
    
    def header(self, title: str, char: str = "="):
        """Print section header."""
        if self._should_print(LogLevel.INFO):
            print(f"\n{char * 60}")
            print(f"🔗 {title}")
            print(f"{char * 60}")
    
    def section(self, title: str):
        """Print subsection header."""
        if self._should_print(LogLevel.INFO):
            print(f"\n--- {title} ---")
    
    def info(self, message: str, prefix: str = "ℹ️"):
        """Standard info message."""
        if self._should_print(LogLevel.INFO):
            print(f"{prefix} {message}")
    
    def success(self, message: str):
        """Success message."""
        if self._should_print(LogLevel.INFO):
            print(f"✅ {message}")
    
    def warning(self, message: str):
        """Warning message."""
        if self._should_print(LogLevel.INFO):
            print(f"⚠️  {message}")
    
    def error(self, message: str):
        """Error message (always printed)."""
        if self.should_log:  # Always show errors if this is rank 0
            print(f"❌ {message}")
    
    def debug(self, message: str):
        """Debug message."""
        if self._should_print(LogLevel.DEBUG):
            print(f"🔧 {message}")
    
    def verbose(self, message: str):
        """Verbose message."""
        if self._should_print(LogLevel.VERBOSE):
            print(f"📝 [VERBOSE] {message}")
    
    def progress(self, message: str):
        """Progress indicator."""
        if self._should_print(LogLevel.INFO):
            print(f"⏳ {message}")
    
    # === SETUP PHASE ===
    def setup_start(self):
        """Log training setup start."""
        self.header("Inductive Link Prediction Setup")
        
    def setup_device(self, device: str, batch_sizes: Dict[str, int]):
        """Log device and batch size setup."""
        if self._should_print(LogLevel.INFO):
            print(f"🖥️  Device: {device}")
            print(f"📦 Batch sizes - Train: {batch_sizes['train']}, Test: {batch_sizes['test']}")
    
    def setup_datasets(self, train_datasets: List[str], test_datasets: List[str]):
        """Log dataset setup."""
        if self._should_print(LogLevel.DEBUG):
            print(f"📁 Training datasets: {train_datasets}")
            print(f"📁 Test datasets: {test_datasets}")
        elif self._should_print(LogLevel.INFO):
            print(f"📁 Loading {len(train_datasets)} training datasets, {len(test_datasets)} test datasets")
    
    def setup_model(self, model_info: Dict[str, Any]):
        """Log model setup."""
        if self._should_print(LogLevel.INFO):
            print(f"🤖 Model: {model_info.get('type', 'Unknown')}")
            print(f"🧠 Hidden dim: {model_info.get('hidden_dim', 'Unknown')}")
            if self._should_print(LogLevel.DEBUG):
                print(f"📊 Total parameters: {model_info.get('total_params', 'Unknown'):,}")
                print(f"💾 Model memory: {model_info.get('memory_gb', 0):.3f}GB per GPU")
    
    # === WASTE ANALYSIS ===
    def waste_analysis_start(self):
        """Start waste analysis logging."""
        if self._should_print(LogLevel.INFO):
            self.section("Distributed Training Waste Analysis")
    
    def waste_critical(self, analysis: Dict[str, Any]):
        """Log critical waste findings (always shown in INFO+)."""
        if self._should_print(LogLevel.INFO):
            memory_waste = analysis.get('memory_waste_percentage', 0)
            compute_efficiency = analysis.get('compute_efficiency_percent', 0)
            
            print(f"🔴 CRITICAL WASTE DETECTED:")
            print(f"   Memory waste: {memory_waste:.1f}%")
            print(f"   Compute efficiency: {compute_efficiency:.1f}%")
            
            if memory_waste > 50:
                print(f"   💡 URGENT: Implement distributed data loading")
    
    def waste_detailed(self, efficiency_scores: Dict[str, Any]):
        """Log detailed waste analysis."""
        if self._should_print(LogLevel.DEBUG):
            print(f"\n🎯 DETAILED EFFICIENCY ANALYSIS:")
            print(f"   Memory Efficiency: {efficiency_scores.get('memory_efficiency_percent', 0):.1f}%")
            print(f"   Compute Efficiency: {efficiency_scores.get('compute_efficiency_percent', 0):.1f}%")
            print(f"   Overall Efficiency: {efficiency_scores.get('overall_efficiency_percent', 0):.1f}%")
            print(f"   Resource Waste Factor: {efficiency_scores.get('waste_factor', 1):.2f}×")
            print(f"   Actual vs Theoretical Speedup: {efficiency_scores.get('actual_speedup', 0):.2f}× vs {efficiency_scores.get('theoretical_speedup', 1)}×")
    
    def waste_recommendations(self, recommendations: List[str], max_show: int = 3):
        """Log optimization recommendations."""
        if self._should_print(LogLevel.INFO) and recommendations:
            print(f"\n💡 TOP OPTIMIZATION RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations[:max_show], 1):
                print(f"   {i}. {rec}")
    
    # === TRAINING PHASE ===
    def training_start(self, total_epochs: int):
        """Log training start."""
        self.section(f"Training Phase ({total_epochs} epochs)")
    
    def epoch_progress(self, epoch: int, metrics: Dict[str, Any], force: bool = False):
        """Log epoch progress with configurable interval."""
        should_print = force or (epoch % self.log_interval == 0)
        
        if should_print and self._should_print(LogLevel.INFO):
            loss = metrics.get('avg_train_loss', 0)
            valid_metric = metrics.get('avg_valid_metric', 0)
            time_taken = metrics.get('time', 0)
            memory_info = metrics.get('memory_info', '')
            
            print(f"Epoch {epoch:3d}: Loss {loss:.4f}, Valid {valid_metric:.4f}, Time {time_taken:.1f}s{memory_info}")
            
            # Show additional info in DEBUG mode
            if self._should_print(LogLevel.DEBUG):
                if 'train_dataset_count' in metrics:
                    print(f"         Datasets trained: {metrics['train_dataset_count']}")
                if 'oom_count' in metrics and metrics['oom_count'] > 0:
                    print(f"         OOM events: {metrics['oom_count']}")
    
    def test_evaluation(self, epoch: int, test_results: Dict[str, float]):
        """Log test evaluation results."""
        should_print = (epoch % self.eval_interval == 0)
        
        if should_print and self._should_print(LogLevel.INFO):
            avg_metric = test_results.get('avg_test_metric', 0)
            print(f"Epoch {epoch} - Test Evaluation: {avg_metric:.4f}")
            
            if self._should_print(LogLevel.DEBUG):
                for dataset, metric in test_results.items():
                    if dataset != 'avg_test_metric':
                        print(f"         {dataset}: {metric:.4f}")
    
    def dataset_processing(self, dataset_name: str, index: int, total: int, success: bool = True):
        """Log individual dataset processing."""
        if self._should_print(LogLevel.DEBUG):
            status = "✅" if success else "❌"
            print(f"{status} Processed dataset {index+1}/{total}: {dataset_name}")
        elif index == 0 and self._should_print(LogLevel.INFO):
            print(f"⏳ Processing {total} datasets...")
    
    def context_preparation(self, dataset_name: str, context_edges: int):
        """Log context preparation."""
        if self._should_print(LogLevel.DEBUG):
            print(f"✅ Context prepared for {dataset_name}: {context_edges} edges")
    
    # === TRAINING PHASE ===
    def training_start(self, epochs: int):
        """Log training start."""
        if self._should_print(LogLevel.INFO):
            self.section("Training Phase")
            print(f"🚀 Starting Training for {epochs} epochs")
    
    def training_epoch_summary(self, summary_data: Dict):
        """Log epoch summary with structured data."""
        if self._should_print(LogLevel.INFO):
            epoch = summary_data['epoch']
            avg_loss = summary_data['avg_loss']
            avg_valid_metric = summary_data['avg_valid_metric']
            time_taken = summary_data['time']
            memory_info = summary_data.get('memory_info', '')
            train_dataset_count = summary_data.get('train_dataset_count', 0)
            oom_count = summary_data.get('oom_count', None)
            
            print(f"Epoch {epoch}: Avg Loss {avg_loss:.4f}, Avg Valid Metric {avg_valid_metric:.4f}, Time: {time_taken:.2f}s{memory_info}")
            
            if train_dataset_count == 0:
                self.warning(f"No training datasets succeeded in epoch {epoch}")
            if oom_count:
                self.warning(f"Total OOM events so far: {oom_count}")
    
    def training_new_best(self, epoch: int, metric_value: float):
        """Log new best validation metric."""
        if self._should_print(LogLevel.INFO):
            print(f"🎯 New best validation metric: {metric_value:.4f} at epoch {epoch}")
    
    def training_test_start(self, epoch: int):
        """Log start of periodic test evaluation."""
        if self._should_print(LogLevel.INFO):  # Changed from DEBUG to INFO - users need to see test starts
            print(f"📊 Starting periodic test evaluation at epoch {epoch}")
    
    def training_test_result(self, epoch: int, avg_test_metric: float):
        """Log periodic test results."""
        if self._should_print(LogLevel.INFO):
            print(f"Epoch {epoch} - Average Test Metric: {avg_test_metric:.4f}")
    
    # === TESTING PHASE ===
    def testing_start(self):
        """Log testing phase start."""
        if self._should_print(LogLevel.INFO):
            self.section("Inductive Testing Phase")
    
    def testing_datasets(self, dataset_names: List[str]):
        """Log test datasets being loaded."""
        if self._should_print(LogLevel.INFO):
            print(f"📚 Loading test datasets: {dataset_names}")
    
    def testing_dataset_result(self, dataset_name: str, metric_name: str, metric_value: float):
        """Log individual test dataset result."""
        if self._should_print(LogLevel.INFO):
            print(f"✅ Test completed for {dataset_name}: {metric_name} = {metric_value:.4f}")
    
    # === WASTE ANALYSIS ===
    def waste_pretraining_analysis(self, efficiency_scores: Dict):
        """Log pre-training waste analysis."""
        if self._should_print(LogLevel.INFO):
            print(f"\n🎯 COMPUTATIONAL WASTE ANALYSIS BEFORE TRAINING:")
            print(f"   Memory Efficiency: {efficiency_scores['memory_efficiency_percent']:.1f}%")
            print(f"   Compute Efficiency: {efficiency_scores['compute_efficiency_percent']:.1f}%")
            print(f"   Overall Efficiency: {efficiency_scores['overall_efficiency_percent']:.1f}%")
            print(f"   Resource Waste Factor: {efficiency_scores['waste_factor']:.2f}×")
            print(f"   Expected vs Actual Speedup: {efficiency_scores['actual_speedup']:.2f}× vs {efficiency_scores['theoretical_speedup']}×")
    
    def waste_recommendations(self, recommendations: List[str]):
        """Log optimization recommendations."""
        if self._should_print(LogLevel.INFO):
            print(f"\n💡 URGENT OPTIMIZATION RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
    
    def waste_training_summary(self, epoch: int, waste_analyzer):
        """Log training efficiency summary."""
        print(f"[DEBUG] waste_training_summary called for epoch {epoch}, log_level={self.log_level}")
        
        if self._should_print(LogLevel.INFO):  # Changed from DEBUG to INFO - waste analysis is important!
            print(f"[DEBUG] About to print waste analysis for epoch {epoch}")
            print(f"\n🔍 TRAINING EFFICIENCY UPDATE (Epoch {epoch}):")
            
            if hasattr(waste_analyzer, 'print_summary_report'):
                print(f"[DEBUG] Calling waste_analyzer.print_summary_report()")
                try:
                    waste_analyzer.print_summary_report()
                    print(f"[DEBUG] waste_analyzer.print_summary_report() completed")
                except Exception as e:
                    print(f"[ERROR] Exception in waste_analyzer.print_summary_report(): {e}")
                    import traceback
                    print(f"[ERROR] Traceback: {traceback.format_exc()}")
                    raise
            else:
                print(f"[ERROR] waste_analyzer does not have print_summary_report method")
        else:
            print(f"[DEBUG] Skipping waste analysis - log level check failed. Current level: {self.log_level}, should_print(INFO): {self._should_print(LogLevel.INFO)}")
    
    # === RESULTS SUMMARY ===
    def results_final_summary(self, gpu_monitor, waste_analyzer):
        """Generate comprehensive final summary."""
        if self.log_level == LogLevel.QUIET:
            return
            
        print("\n" + "="*60)
        print("🎯 Final GPU Usage Summary:")
        print("="*60)
        
        if self._should_print(LogLevel.DEBUG):
            if hasattr(gpu_monitor, 'print_memory_summary'):
                gpu_monitor.print_memory_summary()
            if hasattr(gpu_monitor, 'get_memory_efficiency_report'):
                efficiency_report = gpu_monitor.get_memory_efficiency_report()
                print(efficiency_report)
        
        print("\n" + "="*80)
        print("🔍 FINAL DISTRIBUTED TRAINING WASTE ANALYSIS")
        print("="*80)
        
        if self._should_print(LogLevel.INFO):
            if hasattr(waste_analyzer, 'print_summary_report'):
                waste_analyzer.print_summary_report()
    
    # === ERROR HANDLING (Additional) ===
    def error_oom(self, dataset_name: str, batch_size: int, oom_count=None, is_validation=False):
        """Log out-of-memory errors."""
        phase = "validation" if is_validation else "training"
        self.error(f"GPU out of memory, {phase} failed: {dataset_name}")
        if not is_validation:
            print(f"💡 Suggestion: reduce batch size from {batch_size} and restart training")
    
    def error_training(self, dataset_name: str, error_msg: str):
        """Log training errors."""
        self.warning(f"Training failed for dataset {dataset_name}: {error_msg}")
    
    def error_validation(self, dataset_name: str, error_msg: str):
        """Log validation errors."""
        self.warning(f"Validation failed for dataset {dataset_name}: {error_msg}")
    
    def error_test(self, dataset_name: str, error_msg: str):
        """Log test errors."""
        self.warning(f"Test evaluation failed for dataset {dataset_name}: {error_msg}")
        
    # === ANALYSIS PHASE ===
    def analysis_update(self, epoch: int, waste_analyzer):
        """Log periodic analysis updates."""
        should_print = (epoch % self.analysis_interval == 0 and epoch > 0)
        
        if should_print and self._should_print(LogLevel.DEBUG):
            print(f"\n🔍 EFFICIENCY UPDATE (Epoch {epoch}):")
            # Let the waste analyzer print its summary
            if hasattr(waste_analyzer, 'print_summary_report'):
                waste_analyzer.print_summary_report()
    
    # === FINAL RESULTS ===
    def training_complete(self, best_metric: float, best_epoch: int):
        """Log training completion."""
        if self._should_print(LogLevel.INFO):
            self.section("Training Complete")
            print(f"🎯 Best validation metric: {best_metric:.4f} at epoch {best_epoch}")
    
    def final_results(self, aggregated_results: Dict[str, Dict]):
        """Log final test results."""
        if self._should_print(LogLevel.INFO):
            print("\n📊 Final Inductive Test Results")
            print("=" * 60)
            
            for name, data in aggregated_results.items():
                if 'test_metric' in data and len(data['test_metric']) > 0:
                    metric_name = data.get('metric_name', 'metric')
                    avg_test = sum(data['test_metric']) / len(data['test_metric'])
                    std_test = 0
                    if len(data['test_metric']) > 1:
                        mean_val = avg_test
                        std_test = (sum((x - mean_val) ** 2 for x in data['test_metric']) / len(data['test_metric'])) ** 0.5
                    
                    print(f"{name}: {metric_name} {avg_test:.4f} ± {std_test:.4f} (n={len(data['test_metric'])})")
    
    def final_waste_analysis(self, waste_analyzer):
        """Log final waste analysis."""
        if self._should_print(LogLevel.INFO):
            print("\n" + "="*80)
            print("🔍 FINAL DISTRIBUTED TRAINING WASTE ANALYSIS")
            print("="*80)
            if hasattr(waste_analyzer, 'print_summary_report'):
                waste_analyzer.print_summary_report()
    
    # === ERROR HANDLING ===
    def oom_error(self, dataset_name: str, batch_size: int):
        """Log OOM error."""
        self.error(f"GPU out of memory for dataset {dataset_name}")
        if self._should_print(LogLevel.INFO):
            print(f"💡 Suggestion: reduce batch size from {batch_size} and restart training")
    
    def dataset_error(self, dataset_name: str, error_msg: str, with_traceback: bool = False):
        """Log dataset processing error."""
        self.error(f"Dataset {dataset_name} failed: {error_msg}")
        if with_traceback and self._should_print(LogLevel.DEBUG):
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")
    
    # === UTILITY METHODS ===
    def should_print_epoch(self, epoch: int) -> bool:
        """Check if epoch should be logged."""
        return epoch % self.log_interval == 0
    
    def should_evaluate(self, epoch: int) -> bool:
        """Check if epoch should run test evaluation."""
        return epoch % self.eval_interval == 0
    
    def should_analyze(self, epoch: int) -> bool:
        """Check if epoch should run detailed analysis."""
        return epoch % self.analysis_interval == 0 and epoch > 0
