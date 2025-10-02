#!/usr/bin/env python3
"""
Test checkpoint performance across multiple random seeds (0-99)
Usage: python test_checkpoint_seeds.py --checkpoint_path checkpoints/67+72+57=196.pt
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
import argparse
import json
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Test checkpoint performance across multiple seeds')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the checkpoint file to test')
    parser.add_argument('--start_seed', type=int, default=13,
                        help='Starting seed (default: 0)')
    parser.add_argument('--end_seed', type=int, default=99,
                        help='Ending seed (default: 99)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ID to use (default: 0)')
    parser.add_argument('--output_dir', type=str, default='seed_test_results',
                        help='Directory to save results (default: seed_test_results)')
    parser.add_argument('--base_args', type=str, default='',
                        help='Additional arguments to pass to joint_training.py')
    return parser.parse_args()

def run_single_seed_test(checkpoint_path, seed, gpu, base_args, output_dir):
    """Run joint training with a specific seed and return results."""

    print(f"🧪 Testing seed {seed}...")

    # Construct command
    cmd = [
        'python', 'scripts/joint_training.py',
        '--load_checkpoint', checkpoint_path,
        '--use_pretrained_model', 'True',
        '--seed', str(seed),
        '--gpu', gpu,
        '--runs', '5'  # Five runs per seed
    ]

    # Add base args if provided
    if base_args:
        cmd.extend(base_args.split())

    # Create output file for this seed
    output_file = os.path.join(output_dir, f'seed_{seed:03d}.log')

    try:
        # Run the command and capture output
        with open(output_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=3600  # 60 minute timeout
            )

        # Parse results from the log file
        metrics = parse_results_from_log(output_file)
        metrics['seed'] = seed
        metrics['success'] = (result.returncode == 0)

        if result.returncode == 0:
            print(f"✅ Seed {seed} completed successfully")
        else:
            print(f"❌ Seed {seed} failed with return code {result.returncode}")

        return metrics

    except subprocess.TimeoutExpired:
        print(f"⏰ Seed {seed} timed out")
        return {'seed': seed, 'success': False, 'error': 'timeout'}
    except Exception as e:
        print(f"❌ Seed {seed} failed with error: {e}")
        return {'seed': seed, 'success': False, 'error': str(e)}

def parse_results_from_log(log_file):
    """Parse performance metrics from the log file."""
    metrics = {}

    try:
        with open(log_file, 'r') as f:
            content = f.read()

        # Look for key metrics in the log
        # Example patterns to match:
        # Node Classification Test Results: 0.8234
        # Link Prediction Test Results: 0.7543
        # Graph Classification Test Results: 0.6789
        # Combined Score: 2.2566

        import re

        # Extract individual task results
        nc_match = re.search(r'Node Classification.*?Test.*?Results.*?([0-9.]+)', content, re.IGNORECASE)
        if nc_match:
            metrics['nc_test'] = float(nc_match.group(1))

        lp_match = re.search(r'Link Prediction.*?Test.*?Results.*?([0-9.]+)', content, re.IGNORECASE)
        if lp_match:
            metrics['lp_test'] = float(lp_match.group(1))

        gc_match = re.search(r'Graph Classification.*?Test.*?Results.*?([0-9.]+)', content, re.IGNORECASE)
        if gc_match:
            metrics['gc_test'] = float(gc_match.group(1))

        # Calculate combined score
        if 'nc_test' in metrics and 'lp_test' in metrics and 'gc_test' in metrics:
            metrics['combined_score'] = metrics['nc_test'] + metrics['lp_test'] + metrics['gc_test']

        # Look for final test metrics pattern
        final_test_match = re.search(r'final_test[\'\"]\s*:\s*([0-9.]+)', content)
        if final_test_match:
            metrics['final_test'] = float(final_test_match.group(1))

        # Look for individual metrics in the output
        for line in content.split('\n'):
            if 'NC Test Metric:' in line:
                try:
                    metrics['nc_test'] = float(line.split(':')[-1].strip())
                except:
                    pass
            elif 'LP Test Metric:' in line:
                try:
                    metrics['lp_test'] = float(line.split(':')[-1].strip())
                except:
                    pass
            elif 'GC Test Metric:' in line:
                try:
                    metrics['gc_test'] = float(line.split(':')[-1].strip())
                except:
                    pass

    except Exception as e:
        print(f"Warning: Could not parse results from {log_file}: {e}")

    return metrics

def calculate_statistics(results_df):
    """Calculate statistics from the results."""
    successful_results = results_df[results_df['success'] == True]

    if len(successful_results) == 0:
        print("❌ No successful runs found!")
        return {}

    stats = {}

    # Calculate statistics for each metric
    for metric in ['nc_test', 'lp_test', 'gc_test', 'combined_score']:
        if metric in successful_results.columns:
            values = successful_results[metric].dropna()
            if len(values) > 0:
                stats[metric] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'median': values.median(),
                    'count': len(values)
                }

    return stats

def save_results(results_df, stats, output_dir, checkpoint_path):
    """Save results and statistics to files."""

    # Save raw results to CSV
    results_csv = os.path.join(output_dir, 'seed_test_results.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"📊 Raw results saved to: {results_csv}")

    # Save statistics to JSON
    stats_json = os.path.join(output_dir, 'seed_test_statistics.json')
    stats_data = {
        'checkpoint_path': checkpoint_path,
        'test_date': datetime.now().isoformat(),
        'total_seeds_tested': len(results_df),
        'successful_runs': len(results_df[results_df['success'] == True]),
        'failed_runs': len(results_df[results_df['success'] == False]),
        'statistics': stats
    }

    with open(stats_json, 'w') as f:
        json.dump(stats_data, f, indent=2)
    print(f"📈 Statistics saved to: {stats_json}")

    # Create summary report
    summary_file = os.path.join(output_dir, 'summary_report.txt')
    with open(summary_file, 'w') as f:
        f.write(f"Checkpoint Performance Test Summary\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Seeds Tested: {stats_data['total_seeds_tested']}\n")
        f.write(f"Successful Runs: {stats_data['successful_runs']}\n")
        f.write(f"Failed Runs: {stats_data['failed_runs']}\n\n")

        if stats:
            f.write("Performance Statistics:\n")
            f.write("-" * 30 + "\n")
            for metric, metric_stats in stats.items():
                f.write(f"{metric.upper()}:\n")
                f.write(f"  Mean: {metric_stats['mean']:.4f}\n")
                f.write(f"  Std:  {metric_stats['std']:.4f}\n")
                f.write(f"  Min:  {metric_stats['min']:.4f}\n")
                f.write(f"  Max:  {metric_stats['max']:.4f}\n")
                f.write(f"  Median: {metric_stats['median']:.4f}\n")
                f.write(f"  Count: {metric_stats['count']}\n\n")

    print(f"📄 Summary report saved to: {summary_file}")

def main():
    args = parse_args()

    # Validate checkpoint path
    if not os.path.exists(args.checkpoint_path):
        print(f"❌ Checkpoint file not found: {args.checkpoint_path}")
        sys.exit(1)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"🚀 Starting checkpoint performance test")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Seeds: {args.start_seed} to {args.end_seed}")
    print(f"Output directory: {args.output_dir}")
    print(f"GPU: {args.gpu}")
    print()

    # Test all seeds
    results = []
    total_seeds = args.end_seed - args.start_seed + 1

    for i, seed in enumerate(range(args.start_seed, args.end_seed + 1)):
        print(f"Progress: {i+1}/{total_seeds} ({((i+1)/total_seeds)*100:.1f}%)")

        result = run_single_seed_test(
            args.checkpoint_path,
            seed,
            args.gpu,
            args.base_args,
            args.output_dir
        )
        results.append(result)

        # Print intermediate summary every 10 seeds
        if (i + 1) % 10 == 0:
            successful = sum(1 for r in results if r.get('success', False))
            print(f"📊 Intermediate summary: {successful}/{i+1} successful runs")
            print()

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Calculate statistics
    stats = calculate_statistics(results_df)

    # Save results
    save_results(results_df, stats, args.output_dir, args.checkpoint_path)

    # Print final summary
    successful = len(results_df[results_df['success'] == True])
    print(f"\n🎉 Test completed!")
    print(f"Total runs: {len(results_df)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results_df) - successful}")

    if stats and 'combined_score' in stats:
        cs = stats['combined_score']
        print(f"\nCombined Score Statistics:")
        print(f"  Mean: {cs['mean']:.4f} ± {cs['std']:.4f}")
        print(f"  Range: [{cs['min']:.4f}, {cs['max']:.4f}]")
        print(f"  Median: {cs['median']:.4f}")

        # Find and report the best performing seed
        successful_results = results_df[results_df['success'] == True]
        if 'combined_score' in successful_results.columns:
            best_idx = successful_results['combined_score'].idxmax()
            best_seed = successful_results.loc[best_idx, 'seed']
            best_score = successful_results.loc[best_idx, 'combined_score']
            print(f"\n🏆 Best performing seed: {best_seed} (score: {best_score:.4f})")

if __name__ == '__main__':
    main()