#!/usr/bin/env python3
"""
Script to compute average success rates across multiple experiment runs.
Reads eval_log.json files from all subdirectories and computes averages for each trial.
"""

from math import nan
from typing import Any


import json
import sys
import re
import ast
from pathlib import Path
from collections import defaultdict
import numpy as np
import pdb
def main():
    # Base directory containing all experiment runs
    base_dir = Path(sys.argv[1])

    mean_scores = []
    mean_across_trajs_scores = []
    mean_timesteps = []
    all_timesteps = []  # For pooled mean across all trajectories
    # Iterate through all subdirectories
    num_runs_included = 0
    for subdir in sorted(base_dir.iterdir()):
        if len(sys.argv)>2 and num_runs_included >= int(sys.argv[2]): #only compute for a certain number of runs
            break
        if not subdir.is_dir():
            continue

        eval_log_path = subdir / "eval_log.json"
        if not eval_log_path.exists():
            print(f"Warning: eval_log.json not found in {subdir.name}")
            continue
        with open(eval_log_path, 'r') as f:
            data = json.load(f)

        # Extract mean score for this run
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
                    all_timesteps.append(timestep)  # Collect for pooled mean
        if len(mean_timesteps_run) > 0:
            mean_timesteps.append(np.mean(mean_timesteps_run))

    print(f"\n{'='*81}")
    print(f"SUMMARY")
    print(f"{'='*81}")
    print(f"Total runs found: {len(mean_scores)}")
    print(f"Total trajs found: {len(mean_across_trajs_scores)}")

    print(f"\n{'='*81}")
    print(f"OVERALL MEAN SCORE ACROSS RUNS STATISTICS")
    print(f"{'='*81}")
    print(f"Average mean score across all runs: {np.mean(mean_scores):.4f}")
    print(f"Std deviation: {np.std(mean_scores):.4f}")
    print(f"Min: {np.min(mean_scores):.4f}")
    print(f"Max: {np.max(mean_scores):.4f}")

    print(f"\n{'='*81}")
    print(f"OVERALL MEAN SCORE ACROSS TRAJS STATISTICS")
    print(f"{'='*81}")
    print(f"Average mean score across all runs: {np.mean(mean_across_trajs_scores):.4f}")
    print(f"Std deviation: {np.std(mean_across_trajs_scores):.4f}")
    print(f"Min: {np.min(mean_across_trajs_scores):.4f}")
    print(f"Max: {np.max(mean_across_trajs_scores):.4f}")



    print(f"\n{'='*81}")
    print(f"OVERALL MEAN TIMESTEP STATISTICS (mean of per-run means)")
    print(f"{'='*81}")
    print(f"Average timestep score across all runs: {np.mean(mean_timesteps):.4f}")
    print(f"Std deviation: {np.std(mean_timesteps):.4f}")
    print(f"Min: {np.min(mean_timesteps):.4f}")
    print(f"Max: {np.max(mean_timesteps):.4f}")

    print(f"\n{'='*81}")
    print(f"POOLED MEAN TIMESTEP STATISTICS (mean across all trajectories)")
    print(f"{'='*81}")
    print(f"Average timestep across all successful trajectories: {np.mean(all_timesteps):.4f}")
    print(f"Std deviation: {np.std(all_timesteps):.4f}")
    print(f"Min: {np.min(all_timesteps):.4f}")
    print(f"Max: {np.max(all_timesteps):.4f}")
    print(f"Total successful trajectories: {len(all_timesteps)}")


    results = {
        "metadata": {
            "base_directory": str(base_dir),
            "total_trajectories": len(mean_across_trajs_scores),
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

            "avg_mean_timestep_per_run": float(np.mean(mean_timesteps)) if mean_timesteps else None,
            "std_mean_timestep_per_run": float(np.std(mean_timesteps)) if mean_timesteps else None,
            "min_mean_timestep_per_run": float(np.min(mean_timesteps)) if mean_timesteps else None,
            "max_mean_timestep_per_run": float(np.max(mean_timesteps)) if mean_timesteps else None,

            "pooled_mean_timestep": float(np.mean(all_timesteps)) if all_timesteps else None,
            "pooled_std_timestep": float(np.std(all_timesteps)) if all_timesteps else None,
            "pooled_min_timestep": float(np.min(all_timesteps)) if all_timesteps else None,
            "pooled_max_timestep": float(np.max(all_timesteps)) if all_timesteps else None,
            "total_successful_trajectories": len(all_timesteps),
        },
    }

    with open(base_dir / "average_success_rates.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*81}")
    print(f"Results saved to: {base_dir / 'average_success_rates.json'}")
    print(f"{'='*81}\n")

if __name__ == "__main__":
    main()
