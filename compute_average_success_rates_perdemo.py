#!/usr/bin/env python3
"""
Script to compute average success rates across multiple experiment runs.
Reads eval_log.json files from all subdirectories and computes averages for each trial.
The script now computes per-demo statistics. For each task group, it tracks each demo ID (like 0_0, 10_10, etc.) across all subdirs and computes:                                                             
Per-demo stats (per_demo_statistics in JSON):                                                                                                                                                                       
- mean: average score for that demo across all subdirs                                                                                                                                                              
- std: standard deviation of scores for that demo                                                                                                                                                                   
- n: number of occurrences of that demo                                                                                                                                                                             
                                                                                                                                                                                                                    
Aggregated per-demo stats (in overall_statistics):                                                                                                                                                                  
- avg_per_demo_mean: mean of all the per-demo means (weights each demo equally)                                                                                                                                     
- std_per_demo_mean: std across the per-demo means                                                                                                                                                                  

Usage:
                                                                                                                                                                                                                      
                                                                                                                                                                                                                      
  Run it the same way:                                                                                                                                                                                                
  python3 compute_average_success_rates_perdemo.py --base_dir data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask

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


def extract_task_name(subdir_name):
    """Extract task name from subdirectory name (first part before '_')."""
    return subdir_name.split('_')[0]


def extract_demo_id(key):
    """Extract demo ID from key like 'train/sim_max_reward_0_0' -> '0_0'."""
    # Key format: train/sim_max_reward_X_Y
    prefix = "train/sim_max_reward_"
    if key.startswith(prefix):
        return key[len(prefix):]
    return None


def compute_stats_for_group(subdirs, max_runs=None):
    """Compute statistics for a group of subdirectories."""
    mean_scores = []
    mean_across_trajs_scores = []
    mean_timesteps = []
    all_timesteps = []
    num_runs_included = 0

    # Per-demo tracking: demo_id -> list of scores across subdirs
    per_demo_scores = defaultdict(list)

    for subdir in sorted(subdirs):
        if max_runs is not None and num_runs_included >= max_runs:
            break
        if not subdir.is_dir():
            continue
        eval_log_path = subdir / "eval_log.json"
        if not eval_log_path.exists():
            print(f"Warning: eval_log.json not found in {subdir.name}")
            continue
        with open(eval_log_path, 'r') as f:
            data = json.load(f)

        if "train/mean_score" in data:
            num_runs_included += 1
            mean_scores.append(float(data["train/mean_score"]))
        for key, value in data.items():
            if key.startswith("train/sim_max_reward_"):
                score = float(value)
                mean_across_trajs_scores.append(score)
                # Track per-demo scores
                demo_id = extract_demo_id(key)
                if demo_id:
                    per_demo_scores[demo_id].append(score)

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

    # Compute per-demo statistics
    per_demo_stats = {}
    for demo_id, scores in per_demo_scores.items():
        per_demo_stats[demo_id] = {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'n': len(scores),
            'any_success': int(any(s >= 1.0 for s in scores)),
        }

    return {
        'mean_scores': mean_scores,
        'mean_across_trajs_scores': mean_across_trajs_scores,
        'mean_timesteps': mean_timesteps,
        'all_timesteps': all_timesteps,
        'per_demo_stats': per_demo_stats,
    }


def build_results_dict(stats, base_dir=None):
    """Build results dictionary from statistics."""
    mean_scores = stats['mean_scores']
    mean_across_trajs_scores = stats['mean_across_trajs_scores']
    mean_timesteps = stats['mean_timesteps']
    all_timesteps = stats['all_timesteps']
    per_demo_stats = stats.get('per_demo_stats', {})

    # Compute mean of per-demo means (each demo weighted equally)
    if per_demo_stats:
        demo_means = [s['mean'] for s in per_demo_stats.values()]
        avg_per_demo_mean = float(np.mean(demo_means))
        std_per_demo_mean = float(np.std(demo_means))
        # Fraction of demos where at least one run was successful
        demo_any_success = [s['any_success'] for s in per_demo_stats.values()]
        avg_any_success_rate = float(np.mean(demo_any_success))
    else:
        avg_per_demo_mean = None
        std_per_demo_mean = None
        avg_any_success_rate = None

    results = {
        "metadata": {
            "base_directory": str(base_dir) if base_dir else None,
            "total_runs": len(mean_scores),
            "total_unique_demos": len(per_demo_stats),
        },
        "overall_statistics": {
            "avg_mean_score": float(np.mean(mean_scores)) if mean_scores else None, # Average of the per-run train/mean_score values. Each run is weighted equally
            "std_mean_score": float(np.std(mean_scores)) if mean_scores else None,
            "min_mean_score": float(np.min(mean_scores)) if mean_scores else None,
            "max_mean_score": float(np.max(mean_scores)) if mean_scores else None,

            "avg_mean_across_trajs_score": float(np.mean(mean_across_trajs_scores)) if mean_across_trajs_scores else None, # Pools ALL individual demo scores from ALL runs into one flat list, then takes the mean. Each individual evaluation is weighted equally. So if demo 0_0 appears in 10 runs and demo 10_10 appears in 2 runs, 0_0 gets 5x more weight. 
            "std_mean_across_trajs_score": float(np.std(mean_across_trajs_scores)) if mean_across_trajs_scores else None,
            "min_mean_across_trajs_score": float(np.min(mean_across_trajs_scores)) if mean_across_trajs_scores else None,
            "max_mean_across_trajs_score": float(np.max(mean_across_trajs_scores)) if mean_across_trajs_scores else None,

            # Per-demo aggregated stats (mean of per-demo means)
            "avg_per_demo_mean": avg_per_demo_mean, # For each unique demo ID (e.g. 0_0), compute its mean score across runs. Then average those per-demo means. Each demo is weighted equally regardless of how many runs it appeared in.
            "std_per_demo_mean": std_per_demo_mean,
            # Fraction of demos with at least one successful run
            "avg_any_success_rate": avg_any_success_rate, # Fraction of demos with at least one successful run
        },
        "per_demo_statistics": per_demo_stats,
    }
    return results


def main():
    parser = argparse.ArgumentParser(description='Compute average success rates across experiment runs.')
    parser.add_argument('--base_dir', type=str, help='Base directory containing experiment runs')
    parser.add_argument('--max_runs', type=int, default=None, help='Maximum number of runs to include')
    parser.add_argument('--not_multitask', action='store_true', default=False, help='Do not compute separate statistics per task')
    args = parser.parse_args()

    base_dir = Path(args.base_dir)

    # Collect all subdirectories
    all_subdirs = [subdir for subdir in base_dir.iterdir() if subdir.is_dir()]

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

    with open(base_dir / "average_success_rates_perdemo.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*81}")
    print(f"Results saved to: {base_dir / 'average_success_rates_perdemo.json'}")
    print(f"{'='*81}\n")

if __name__ == "__main__":
    main()
