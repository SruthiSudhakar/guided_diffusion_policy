#!/usr/bin/env python3
"""
Script to compute average success rates across multiple experiment runs.
Reads eval_log.json files from all subdirectories and computes averages for each trial.
Usage:

Without multitask (original behavior):                                                                                                                                                                              
python3 compute_average_success_rates.py /path/to/base_dir                                                                                                                                                          
                                                                                                                                                                                                                    
With multitask (separate stats per task):                                                                                                                                                                           
python3 compute_average_success_rates.py /path/to/base_dir --multitask                                                                                                                                              
                                                                                                                                                                                                                    
With max runs limit:                                                                                                                                                                                                
python3 compute_average_success_rates.py /path/to/base_dir 50 --multitask                                                                                                                                           
                                                                                                                                                                                                                    
For your example directory:                                                                                                                                                                                         
python3 compute_average_success_rates.py --base_dir data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask                                                   

"""

from math import nan
from typing import Any


import json
import sys
import re
import ast
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def extract_task_name(subdir_name):
    """Extract task name from subdirectory name (first part before '_')."""
    return subdir_name.split('_')[0]


def compute_stats_for_group(subdirs, max_runs=None):
    """Compute statistics for a group of subdirectories."""
    mean_scores = []
    mean_across_trajs_scores = []
    mean_timesteps = []
    all_timesteps = []
    num_runs_included = 0

    for subdir in sorted(subdirs):
        if max_runs is not None and num_runs_included >= max_runs:
            break
        if not subdir.is_dir():
            continue
        eval_log_path = subdir / "eval_log.json"
        if not eval_log_path.exists():
            # print(f"Warning: eval_log.json not found in {subdir.name}")
            continue
        with open(eval_log_path, 'r') as f:
            data = json.load(f)

        if "train/mean_score" in data:
            num_runs_included += 1
            mean_scores.append(float(data["train/mean_score"]))
        for key, value in data.items():
            if key.startswith("train/sim_max_reward_"):
                mean_across_trajs_scores.append(float(value))

        mean_timesteps_run = []
        for key, value in data.items():
            if key.startswith("train/sim_reward_trajectory_"):
                trajectory = ast.literal_eval(value)
                if 1 in trajectory:
                    timestep = trajectory.index(1)
                    mean_timesteps_run.append(timestep)
                    all_timesteps.append(timestep)
        if len(mean_timesteps_run) > 0:
            mean_timesteps.append(np.mean(mean_timesteps_run))

    return {
        'mean_scores': mean_scores,
        'mean_across_trajs_scores': mean_across_trajs_scores,
        'mean_timesteps': mean_timesteps,
        'all_timesteps': all_timesteps,
    }

def build_results_dict(stats, base_dir=None):
    """Build results dictionary from statistics."""
    mean_scores = stats['mean_scores']
    mean_across_trajs_scores = stats['mean_across_trajs_scores']
    mean_timesteps = stats['mean_timesteps']
    all_timesteps = stats['all_timesteps']

    results = {
        "metadata": {
            "base_directory": str(base_dir) if base_dir else None,
            "total_runs": len(mean_scores),
        },
        "overall_statistics": {
            "avg_mean_score": float(np.mean(mean_scores)) if mean_scores else None,
            "std_mean_score": float(np.std(mean_scores)) if mean_scores else None,
            "min_mean_score": float(np.min(mean_scores)) if mean_scores else None,
            "max_mean_score": float(np.max(mean_scores)) if mean_scores else None,

            "avg_mean_across_trajs_score": float(np.mean(mean_across_trajs_scores)) if mean_across_trajs_scores else None,
            "std_mean_across_trajs_score": float(np.std(mean_across_trajs_scores)) if mean_across_trajs_scores else None,
            "min_mean_across_trajs_score": float(np.min(mean_across_trajs_scores)) if mean_across_trajs_scores else None,
            "max_mean_across_trajs_score": float(np.max(mean_across_trajs_scores)) if mean_across_trajs_scores else None,

            # "avg_mean_timestep_per_run": float(np.mean(mean_timesteps)) if mean_timesteps else None,
            # "std_mean_timestep_per_run": float(np.std(mean_timesteps)) if mean_timesteps else None,
            # "min_mean_timestep_per_run": float(np.min(mean_timesteps)) if mean_timesteps else None,
            # "max_mean_timestep_per_run": float(np.max(mean_timesteps)) if mean_timesteps else None,

            # "pooled_mean_timestep": float(np.mean(all_timesteps)) if all_timesteps else None,
            # "pooled_std_timestep": float(np.std(all_timesteps)) if all_timesteps else None,
            # "pooled_min_timestep": float(np.min(all_timesteps)) if all_timesteps else None,
            # "pooled_max_timestep": float(np.max(all_timesteps)) if all_timesteps else None,
        },
    }
    return results


def plot_convergence(scores, output_path):
    """Plot convergence of success rate statistics."""
    if not scores:
        return

    means = []
    stds = []
    mins = []
    maxs = []
    counts = range(1, len(scores) + 1)

    for i in counts:
        current_scores = scores[:i]
        means.append(np.mean(current_scores))
        stds.append(np.std(current_scores))
        mins.append(np.min(current_scores))
        maxs.append(np.max(current_scores))

    plt.figure(figsize=(12, 8))
    
    # Plot Mean
    plt.plot(counts, means, label='Mean', color='blue', linewidth=2)
    
    # Plot Min and Max
    plt.plot(counts, mins, label='Min', color='red', linestyle='--', alpha=0.7)
    plt.plot(counts, maxs, label='Max', color='green', linestyle='--', alpha=0.7)
    
    # Plot Std Dev
    plt.plot(counts, stds, label='Std Dev', color='orange', linestyle=':', linewidth=2)

    plt.xlabel('Number of Runs Included')
    plt.ylabel('Score')
    plt.title('Convergence of Success Rate Statistics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    try:
        plt.savefig(output_path)
        print(f"Convergence plot saved to: {output_path}")
    except Exception as e:
        print(f"Could not save plot to {output_path}: {e}")
    finally:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compute average success rates across experiment runs.')
    parser.add_argument('--base_dir', type=str, help='Base directory containing experiment runs')
    parser.add_argument('--max_runs', type=int, default=None, help='Maximum number of runs to include')
    parser.add_argument('--not_multitask', action='store_true', default=False, help='Do not compute separate statistics per task')
    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    # Collect all subdirectories
    all_subdirs = [subdir for subdir in base_dir.iterdir() if subdir.is_dir() and not subdir.name.startswith("overlay")]

    if not args.not_multitask:
        # Group subdirectories by task name
        task_groups = defaultdict(list)
        for subdir in all_subdirs:
            task_name = extract_task_name(subdir.name)
            task_groups[task_name].append(subdir)

        results = {
            "metadata": {
                "base_directory": str(base_dir),
                "multitask": True,
            },
            "per_task_statistics": {},
        }

        # Compute and print stats for each task
        for task_name in sorted(task_groups.keys()):
            subdirs = task_groups[task_name]
            stats = compute_stats_for_group(subdirs, args.max_runs)
            # print_stats(stats, label=task_name)
            results["per_task_statistics"][task_name] = build_results_dict(stats)

        # Also compute overall stats
        overall_stats = compute_stats_for_group(all_subdirs, args.max_runs)
        # print_stats(overall_stats, label="OVERALL")
        results["overall_statistics"] = build_results_dict(overall_stats)

    else:
        # Original behavior: compute stats for all subdirectories together
        stats = compute_stats_for_group(all_subdirs, args.max_runs)
        # print_stats(stats)
        results = build_results_dict(stats, base_dir)

    with open(base_dir / "average_success_rates.json", 'w') as f:
        json.dump(results, f, indent=2)

    # Plot convergence
    if not args.not_multitask:
        scores_for_plot = overall_stats['mean_scores']
    else:
        scores_for_plot = stats['mean_scores']
    
    plot_convergence(scores_for_plot, base_dir / "convergence_plot.png")

    print(f"\n{'='*81}")
    print(f"Results saved to: {base_dir / 'average_success_rates.json'}")
    print(f"{'='*81}\n")
    # Determine which stats dict to use for printing
    if "per_task_statistics" in results:
        # Multitask mode: results["overall_statistics"] contains the output of build_results_dict
        final_stats = results["overall_statistics"]["overall_statistics"]
    else:
        # Single task mode: results is the output of build_results_dict
        final_stats = results["overall_statistics"]

    print(f'Average mean score: {final_stats["avg_mean_score"]:.3f}')
    print(f'Standard deviation of mean score: {final_stats["std_mean_score"]:.3f}')
    print(f'Minimum mean score: {final_stats["min_mean_score"]:.3f}')
    print(f'Maximum mean score: {final_stats["max_mean_score"]:.3f}')
    print(f'Number of runs included: {results["overall_statistics"]["metadata"]["total_runs"]}')

if __name__ == "__main__":
    main()
