#!/usr/bin/env python3
"""
Multiple Seeds Training Script
eICU Stroke Mortality Prediction

Run the same model configuration with multiple random seeds and aggregate results
for reporting mean ± std, which is required for academic publications.

Usage:
    # Run MLP with 5 different seeds
    python run_multiple_seeds.py --model mlp --seeds 5
    
    # Run with specific seeds
    python run_multiple_seeds.py --model xgboost --seeds 42,43,44,45,46
    
    # Run all models with 5 seeds each
    python run_multiple_seeds.py --all --seeds 5

Author: Stroke Prediction Research Team
Date: 2025-11-09
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import subprocess
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def parse_seeds_arg(seeds_arg: str) -> List[int]:
    """
    Parse seeds argument.
    
    Args:
        seeds_arg: Either a number (get N seeds from pool) or comma-separated seeds
    
    Returns:
        list: List of seed values
    """
    # High-entropy academic seed pool covering different orders of magnitude
    # to avoid local correlations from pseudo-random number generators
    ACADEMIC_SEEDS_POOL = [42, 2024, 777, 9999, 12345, 54321, 10086, 31415]

    if ',' in seeds_arg:
        # Case 1: Manually specified seeds (e.g., --seeds 42,888)
        return [int(s.strip()) for s in seeds_arg.split(',')]
    else:
        # Case 2: Specify count (e.g., --seeds 5)
        n_seeds = int(seeds_arg)
        
        # If requested count <= pool size, take the first N from the pool
        if n_seeds <= len(ACADEMIC_SEEDS_POOL):
            return ACADEMIC_SEEDS_POOL[:n_seeds]
        else:
            # Fallback: if more seeds are requested than the pool has,
            # generate extra seeds sequentially
            print(f"⚠️ Warning: Requested {n_seeds} seeds, but pool only has {len(ACADEMIC_SEEDS_POOL)}. Generating extra seeds sequentially.")
            return ACADEMIC_SEEDS_POOL + list(range(42 + len(ACADEMIC_SEEDS_POOL), 42 + n_seeds))


def run_single_experiment(
    model: str,
    seed: int,
    config_path: str = None,
    verbose: bool = True,
    two_stage: bool = False
) -> Dict[str, Any]:
    """
    Run a single experiment with given seed.
    
    Args:
        model: Model type
        seed: Random seed
        config_path: Optional config path
        verbose: Print output
        two_stage: If True, use run_two_stage_train.py instead
    
    Returns:
        dict: Experiment results
    """
    # Build command
    if two_stage:
        cmd = [
            sys.executable,
            'run_two_stage_train.py',
            '--model', model,
            '--seeds', str(seed),
            '-y'
        ]
    else:
        cmd = [
            sys.executable,
            'run_unified_train.py',
            '--model', model,
            '--seed', str(seed)
        ]
    
    if config_path and not two_stage:
        cmd.extend(['--config', config_path])
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Running: {model.upper()} with seed={seed}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
            check=True,
            cwd=Path(__file__).parent
        )
        
        return {
            'seed': seed,
            'status': 'success',
            'exit_code': 0
        }
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed with seed={seed}, exit code={e.returncode}")
        return {
            'seed': seed,
            'status': 'failed',
            'exit_code': e.returncode,
            'error': str(e)
        }
    
    except Exception as e:
        print(f"❌ Error with seed={seed}: {e}")
        return {
            'seed': seed,
            'status': 'error',
            'error': str(e)
        }


def aggregate_metrics_from_experiments(
    experiments_dir: Path,
    model: str,
    seeds: List[int]
) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics from multiple experiment runs.
    
    Args:
        experiments_dir: Path to experiments directory
        model: Model type
        seeds: List of seeds used
    
    Returns:
        dict: Aggregated metrics with mean, std, min, max
    """
    all_metrics = []
    
    # Find experiment directories for this model and seeds
    for exp_dir in experiments_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        # Check if this is one of our experiments
        if model.upper() not in exp_dir.name:
            continue
        
        # Load metrics
        metrics_file = exp_dir / 'artifacts' / 'metrics_test.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                all_metrics.append(metrics)
    
    if len(all_metrics) == 0:
        print(f"⚠️  No metrics found for {model}")
        return {}
    
    # Aggregate metrics
    aggregated = {}
    
    # Get all metric names
    metric_names = set()
    for m in all_metrics:
        metric_names.update(m.keys())
    
    for metric_name in metric_names:
        # Skip non-numeric or CI metrics
        if 'ci_' in metric_name or metric_name in ['tp', 'fp', 'tn', 'fn']:
            continue
        
        values = []
        for m in all_metrics:
            if metric_name in m and isinstance(m[metric_name], (int, float)):
                values.append(m[metric_name])
        
        if len(values) > 0:
            aggregated[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values, ddof=1) if len(values) > 1 else 0.0,
                'min': np.min(values),
                'max': np.max(values),
                'n_runs': len(values),
                'values': values
            }
    
    return aggregated


def format_aggregated_results(
    aggregated: Dict[str, Dict[str, float]],
    model: str
) -> str:
    """
    Format aggregated results for display.
    
    Args:
        aggregated: Aggregated metrics
        model: Model name
    
    Returns:
        str: Formatted table
    """
    output = []
    output.append(f"\n{'='*80}")
    output.append(f"AGGREGATED RESULTS: {model.upper()}")
    output.append(f"{'='*80}\n")
    
    # Key metrics to show first
    key_metrics = ['auroc', 'auprc', 'f1', 'brier', 'ece', 
                   'precision', 'recall', 'specificity', 'balanced_accuracy']
    
    output.append(f"{'Metric':<25} {'Mean ± Std':<20} {'Min':<12} {'Max':<12} {'N':<5}")
    output.append("-" * 80)
    
    # Show key metrics first
    for metric in key_metrics:
        if metric in aggregated:
            stats = aggregated[metric]
            mean_std = f"{stats['mean']:.4f} ± {stats['std']:.4f}"
            output.append(
                f"{metric:<25} {mean_std:<20} "
                f"{stats['min']:<12.4f} {stats['max']:<12.4f} {stats['n_runs']:<5}"
            )
    
    output.append("-" * 80)
    
    # Show other metrics
    other_metrics = [m for m in sorted(aggregated.keys()) if m not in key_metrics]
    if other_metrics:
        output.append("\nOther Metrics:")
        output.append("-" * 80)
        for metric in other_metrics:
            stats = aggregated[metric]
            mean_std = f"{stats['mean']:.4f} ± {stats['std']:.4f}"
            output.append(
                f"{metric:<25} {mean_std:<20} "
                f"{stats['min']:<12.4f} {stats['max']:<12.4f} {stats['n_runs']:<5}"
            )
    
    return '\n'.join(output)


def save_aggregated_results(
    aggregated: Dict[str, Dict[str, float]],
    model: str,
    seeds: List[int],
    output_dir: Path
):
    """
    Save aggregated results to files.
    
    Args:
        aggregated: Aggregated metrics
        model: Model name
        seeds: Seeds used
        output_dir: Output directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON
    json_file = output_dir / f'aggregated_{model}_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump({
            'model': model,
            'seeds': seeds,
            'n_runs': len(seeds),
            'timestamp': timestamp,
            'metrics': aggregated
        }, f, indent=2)
    
    print(f"✓ Saved aggregated JSON: {json_file}")
    
    # Save CSV
    csv_data = []
    for metric_name, stats in aggregated.items():
        csv_data.append({
            'metric': metric_name,
            'mean': stats['mean'],
            'std': stats['std'],
            'min': stats['min'],
            'max': stats['max'],
            'n_runs': stats['n_runs']
        })
    
    df = pd.DataFrame(csv_data)
    csv_file = output_dir / f'aggregated_{model}_{timestamp}.csv'
    df.to_csv(csv_file, index=False)
    
    print(f"✓ Saved aggregated CSV: {csv_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Run multiple seeds for robust evaluation'
    )
    parser.add_argument(
        '--model',
        type=str,
        help='Model type (mlp, nn, xgboost, random_forest, transformer, dualtower, dualtower_mlp)'
    )
    parser.add_argument(
        '--seeds',
        type=str,
        default='5',
        help='Number of seeds (e.g., "5") or comma-separated seeds (e.g., "42,43,44")'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all model types'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Config file path (optional)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../../outputs/multi_seed_results',
        help='Output directory for aggregated results'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Verbose output'
    )
    parser.add_argument(
        '-y', '--yes',
        action='store_true',
        default=False,
        help='Skip confirmation prompt (required for nohup/background runs)'
    )
    parser.add_argument(
        '--two-stage',
        action='store_true',
        default=False,
        help='Use 2-stage transfer learning (combined pretrain → eICU-only fine-tune)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all and not args.model:
        parser.error("Either --model or --all must be specified")
    
    # Parse seeds
    seeds = parse_seeds_arg(args.seeds)
    print(f"\nSeeds to use: {seeds}")
    
    # Determine models to run
    if args.all:
        models = ['mlp', 'nn', 'xgboost', 'random_forest', 'transformer', 'dualtower', 'dualtower_mlp']
    else:
        models = [args.model]
    
    print(f"Models to train: {models}")
    print(f"Total experiments: {len(models) * len(seeds)}\n")
    
    # Confirm (skip if --yes or stdin is not a tty)
    if not args.yes and sys.stdin.isatty():
        response = input("Start multiple seed training? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return 0
    
    # Run experiments
    all_results = {}
    
    for model in models:
        print(f"\n{'#'*80}")
        print(f"# MODEL: {model.upper()}")
        print(f"{'#'*80}\n")
        
        model_results = []
        
        for seed in seeds:
            result = run_single_experiment(
                model=model,
                seed=seed,
                config_path=args.config,
                verbose=args.verbose,
                two_stage=args.two_stage
            )
            model_results.append(result)
        
        all_results[model] = model_results
        
        # Check success rate
        n_success = sum(1 for r in model_results if r['status'] == 'success')
        print(f"\n{model.upper()}: {n_success}/{len(seeds)} runs succeeded")
    
    # Aggregate results
    print(f"\n{'='*80}")
    print("AGGREGATING RESULTS")
    print(f"{'='*80}\n")
    
    experiments_dir = Path('../../outputs/experiments')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for model in models:
        aggregated = aggregate_metrics_from_experiments(
            experiments_dir,
            model,
            seeds
        )
        
        if aggregated:
            # Display results
            print(format_aggregated_results(aggregated, model))
            
            # Save results
            save_aggregated_results(aggregated, model, seeds, output_dir)
    
    # Save summary
    summary_file = output_dir / f'summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'models': models,
            'seeds': seeds,
            'n_runs_per_model': len(seeds),
            'timestamp': datetime.now().isoformat(),
            'results': all_results
        }, f, indent=2)
    
    print(f"\n✓ Summary saved: {summary_file}")
    print(f"\n{'='*80}")
    print("✓ MULTIPLE SEED TRAINING COMPLETED")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

