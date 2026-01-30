#!/usr/bin/env python3
"""
Script to compute average success rates across multiple experiment runs.
Reads eval_log.json files from all subdirectories and computes averages for each trial.
"""

import json
import sys
import re
import ast
from pathlib import Path
from collections import defaultdict
import numpy as np


def find_first_reward_one_timestep(trajectory):
    """
    Find the first timestep where the reward becomes 1.

    Args:
        trajectory: List or string representation of reward values

    Returns:
        int: The timestep (index) where reward first becomes 1, or None if never reached
    """
    if isinstance(trajectory, str):
        # Parse string representation of list
        trajectory = ast.literal_eval(trajectory)

    for idx, reward in enumerate(trajectory):
        if reward == 1 or reward == 1.0:
            return idx

    return None

def main():
    # Base directory containing all experiment runs
    if len(sys.argv) < 2:
        print("Usage: python3 compute_average_success_rates_old.py <base_dir> [max_runs]")
        sys.exit(1)
    base_dir = Path(sys.argv[1])

    # Dictionary to store success rates for each trial
    # Format: {trial_id: [success_rate_run1, success_rate_run2, ...]}
    trial_success_rates = defaultdict(list)

    # Dictionary to store run indices for each trial's success rates
    # Format: {trial_id: [(run_idx, success_rate), ...]}
    trial_run_indices = defaultdict(list)

    # List to store mean scores from each run
    mean_scores = []

    # Lists to store timestep data across all runs
    all_timesteps = []  # All timesteps when reward=1 is reached across all runs
    per_run_timesteps = []  # List of lists: [[run1_timesteps], [run2_timesteps], ...]
    per_run_avg_timesteps = []  # Average timestep per run

    # Counter for successful reads
    successful_runs = 0
    failed_runs = []

    # List to track run names in order
    run_names = []
    num_runs_included = 0
    # Iterate through all subdirectories
    for subdir in sorted(base_dir.iterdir()):
        if len(sys.argv)>2 and num_runs_included >= int(sys.argv[2]): #only compute for a certain number of runs
            break

        if not subdir.is_dir():
            continue

        eval_log_path = subdir / "eval_log.json"

        if not eval_log_path.exists():
            print(f"Warning: eval_log.json not found in {subdir.name}")
            failed_runs.append(subdir.name)
            continue

        try:
            with open(eval_log_path, 'r') as f:
                data = json.load(f)

            # Track this run
            run_names.append(subdir.name)
            run_idx = len(run_names) - 1

            # Extract mean score for this run
            if "train/mean_score" in data:
                num_runs_included += 1
                mean_scores.append(float(data["train/mean_score"]))

            # Analyze timesteps for this run
            run_timesteps = []
            max_reward_pattern = re.compile(r'(.+)/sim_max_reward_(\d+)_(\d+)')

            # Extract individual trial success rates
            for key, value in data.items():
                if key.startswith("train/sim_max_reward_"):
                    # Extract trial ID from key like "train/sim_max_reward_0_0"
                    trial_id = key.replace("train/sim_max_reward_", "")
                    success_rate = float(value)
                    trial_success_rates[trial_id].append(success_rate)
                    trial_run_indices[trial_id].append((run_idx, success_rate))

                    # If this trial succeeded, find the timestep
                    if success_rate == 1.0:
                        match = max_reward_pattern.match(key)
                        if match:
                            prefix = match.group(1)
                            idx1 = match.group(2)
                            idx2 = match.group(3)
                            trajectory_key = f"{prefix}/sim_reward_trajectory_{idx1}_{idx2}"

                            if trajectory_key in data:
                                trajectory = data[trajectory_key]
                                first_timestep = find_first_reward_one_timestep(trajectory)
                                if first_timestep is not None:
                                    run_timesteps.append(first_timestep)
                                    all_timesteps.append(first_timestep)

            # Store timesteps for this run
            per_run_timesteps.append(run_timesteps)
            if run_timesteps:
                per_run_avg_timesteps.append(np.mean(run_timesteps))
            else:
                per_run_avg_timesteps.append(None)

            successful_runs += 1

        except Exception as e:
            print(f"Error reading {eval_log_path}: {e}")
            failed_runs.append(subdir.name)
            continue

    # Compute and display average mean score across all runs
    if mean_scores:
        print(f"\n{'='*81}")
        print(f"OVERALL MEAN SCORE STATISTICS")
        print(f"{'='*81}")
        print(f"Average mean score across all runs: {np.mean(mean_scores):.4f}")
        print(f"Std deviation: {np.std(mean_scores):.4f}")
        print(f"Min: {np.min(mean_scores):.4f}")
        print(f"Max: {np.max(mean_scores):.4f}")

    # Compute and display timestep statistics
    if all_timesteps:
        print(f"\n{'='*81}")
        print(f"TIMESTEP STATISTICS (When Reward=1 First Reached)")
        print(f"{'='*81}")
        print(f"Total successful trajectories across all runs: {len(all_timesteps)}")
        print(f"Average timestep when reward=1 first reached: {np.mean(all_timesteps):.2f}")
        print(f"Std deviation: {np.std(all_timesteps):.2f}")
        print(f"Min timestep: {int(np.min(all_timesteps))}")
        print(f"Max timestep: {int(np.max(all_timesteps))}")

        # Show percentiles
        sorted_timesteps = sorted(all_timesteps)
        print(f"\nPercentiles:")
        print(f"  25th percentile: {int(np.percentile(sorted_timesteps, 25))}")
        print(f"  50th percentile (median): {int(np.percentile(sorted_timesteps, 50))}")
        print(f"  75th percentile: {int(np.percentile(sorted_timesteps, 75))}")

        # Show per-run average timesteps
        valid_run_avgs = [x for x in per_run_avg_timesteps if x is not None]
        if valid_run_avgs:
            print(f"\nPer-run average timesteps:")
            print(f"  Average across runs: {np.mean(valid_run_avgs):.2f}")
            print(f"  Std deviation across runs: {np.std(valid_run_avgs):.2f}")
            print(f"  Min run average: {np.min(valid_run_avgs):.2f}")
            print(f"  Max run average: {np.max(valid_run_avgs):.2f}")

    # Compute and display average success rate for each trial
    print(f"\n{'='*81}")
    print(f"AVERAGE SUCCESS RATE PER TRIAL")
    print(f"{'='*81}")
    print(f"{'Trial ID':<15} {'Avg Success Rate':<20} {'Std Dev':<15} {'Count':<10} {'Unsuccessful Runs (if std>0)'}")
    print(f"{'-'*81}")

    # Sort by trial number
    def extract_trial_num(trial_id):
        # Extract first number from "0_0" format
        return int(trial_id.split('_')[0])

    sorted_trials = sorted(trial_success_rates.keys(), key=extract_trial_num)

    trial_averages = {}
    for trial_id in sorted_trials:
        rates = trial_success_rates[trial_id]
        avg_rate = np.mean(rates)
        std_rate = np.std(rates)
        trial_averages[trial_id] = avg_rate

        # For trials with std_dev > 0, find which runs had success = 0
        unsuccessful_run_info = ""
        if std_rate > 0:
            unsuccessful_runs_list = [
                run_names[run_idx] for run_idx, success_rate in trial_run_indices[trial_id]
                if success_rate == 0.0
            ]
            if unsuccessful_runs_list:
                unsuccessful_run_info = f"Runs: {', '.join(unsuccessful_runs_list)}"

        # print(f"{trial_id:<15} {avg_rate:<20.4f} {std_rate:<15.4f} {len(rates):<10} {unsuccessful_run_info}")

    # Overall statistics across all trials
    all_averages = list(trial_averages.values())

    # Compute max success rate: percentage of trials that succeeded at least once
    trials_with_at_least_one_success = 0
    for trial_id in sorted_trials:
        rates = trial_success_rates[trial_id]
        # A trial succeeds if at least one run had success_rate > 0
        if any(rate > 0 for rate in rates):
            trials_with_at_least_one_success += 1

    max_success_rate = trials_with_at_least_one_success / len(sorted_trials) if sorted_trials else 0

    if all_averages:
        # Get the number of runs per trial (assuming all trials have same number of runs)
        num_runs = len(trial_success_rates[sorted_trials[0]]) if sorted_trials else 0

        print(f"\n{'='*81}")
        print(f"OVERALL TRIAL STATISTICS")
        print(f"{'='*81}")
        print(f"Average success rate across all trials: {np.mean(all_averages):.4f}")
        print(f"Max success rate (trial succeeds if >=1/{num_runs} total runs succeeds): {max_success_rate:.4f}")
        # print(f"  Trials with at least one success: {trials_with_at_least_one_success}/{len(sorted_trials)}")
        print(f"Std deviation across trials: {np.std(all_averages):.4f}")
        print(f"Number of trials: {len(all_averages)}")
        print(f"Min trial success rate: {np.min(all_averages):.4f}")
        print(f"Max trial success rate: {np.max(all_averages):.4f}")

    # Save results to a JSON file in the base directory
    output_file = base_dir / "average_success_rates_old.json"
    results = {
        "metadata": {
            "base_directory": str(base_dir),
            "successful_runs": run_names,
            "failed_runs": failed_runs,
            "total_successful": successful_runs,
            "total_failed": len(failed_runs)
        },
        "overall_statistics": {
            "avg_mean_score": float(np.mean(mean_scores)) if mean_scores else None,
            "std_mean_score": float(np.std(mean_scores)) if mean_scores else None,
            "min_mean_score": float(np.min(mean_scores)) if mean_scores else None,
            "max_mean_score": float(np.max(mean_scores)) if mean_scores else None,
            "num_runs": successful_runs
        },
        "timestep_statistics": {
            "total_successful_trajectories": len(all_timesteps),
            "avg_timestep_to_reward_1": float(np.mean(all_timesteps)) if all_timesteps else None,
            "std_timestep": float(np.std(all_timesteps)) if all_timesteps else None,
            "min_timestep": int(np.min(all_timesteps)) if all_timesteps else None,
            "max_timestep": int(np.max(all_timesteps)) if all_timesteps else None,
            "percentile_25": int(np.percentile(all_timesteps, 25)) if all_timesteps else None,
            "percentile_50_median": int(np.percentile(all_timesteps, 50)) if all_timesteps else None,
            "percentile_75": int(np.percentile(all_timesteps, 75)) if all_timesteps else None,
            "per_run_avg_timesteps": {
                run_names[i]: float(per_run_avg_timesteps[i]) if per_run_avg_timesteps[i] is not None else None
                for i in range(len(run_names))
            } if per_run_avg_timesteps else None,
            "per_run_statistics": {
                "avg_of_run_averages": float(np.mean([x for x in per_run_avg_timesteps if x is not None])) if any(x is not None for x in per_run_avg_timesteps) else None,
                "std_of_run_averages": float(np.std([x for x in per_run_avg_timesteps if x is not None])) if any(x is not None for x in per_run_avg_timesteps) else None,
                "min_run_average": float(np.min([x for x in per_run_avg_timesteps if x is not None])) if any(x is not None for x in per_run_avg_timesteps) else None,
                "max_run_average": float(np.max([x for x in per_run_avg_timesteps if x is not None])) if any(x is not None for x in per_run_avg_timesteps) else None
            }
        },
        "trial_averages": {
            trial_id: {
                "avg_success_rate": float(np.mean(trial_success_rates[trial_id])),
                "std_success_rate": float(np.std(trial_success_rates[trial_id])),
                "count": len(trial_success_rates[trial_id])
            }
            for trial_id in sorted_trials
        },
        "cross_trial_statistics": {
            "avg_success_rate": float(np.mean(all_averages)) if all_averages else None,
            "max_success_rate": float(max_success_rate),
            "trials_with_at_least_one_success": trials_with_at_least_one_success,
            "std_success_rate": float(np.std(all_averages)) if all_averages else None,
            "min_trial_success_rate": float(np.min(all_averages)) if all_averages else None,
            "max_trial_success_rate": float(np.max(all_averages)) if all_averages else None,
            "num_trials": len(all_averages)
        }
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*81}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*81}\n")

if __name__ == "__main__":
    main()
