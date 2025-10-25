#!/usr/bin/env python3
"""
Script to compute average success rates across multiple experiment runs.
Reads eval_log.json files from all subdirectories and computes averages for each trial.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

def main():
    # Base directory containing all experiment runs
    base_dir = Path(sys.argv[1])

    # Dictionary to store success rates for each trial
    # Format: {trial_id: [success_rate_run1, success_rate_run2, ...]}
    trial_success_rates = defaultdict(list)

    # Dictionary to store run indices for each trial's success rates
    # Format: {trial_id: [(run_idx, success_rate), ...]}
    trial_run_indices = defaultdict(list)

    # List to store mean scores from each run
    mean_scores = []

    # Counter for successful reads
    successful_runs = 0
    failed_runs = []

    # List to track run names in order
    run_names = []

    # Iterate through all subdirectories
    for subdir in sorted(base_dir.iterdir()):
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
                mean_scores.append(float(data["train/mean_score"]))

            # Extract individual trial success rates
            for key, value in data.items():
                if key.startswith("train/sim_max_reward_"):
                    # Extract trial ID from key like "train/sim_max_reward_0_0"
                    trial_id = key.replace("train/sim_max_reward_", "")
                    success_rate = float(value)
                    trial_success_rates[trial_id].append(success_rate)
                    trial_run_indices[trial_id].append((run_idx, success_rate))

            successful_runs += 1

        except Exception as e:
            print(f"Error reading {eval_log_path}: {e}")
            failed_runs.append(subdir.name)
            continue

    print(f"\n{'='*81}")
    print(f"SUMMARY")
    print(f"{'='*81}")
    print(f"Total runs found: {successful_runs}")
    if failed_runs:
        print(f"Failed runs: {len(failed_runs)}")
        for failed in failed_runs[:5]:  # Show first 5 failed runs
            print(f"  - {failed}")
        if len(failed_runs) > 5:
            print(f"  ... and {len(failed_runs) - 5} more")

    # Compute and display average mean score across all runs
    if mean_scores:
        print(f"\n{'='*81}")
        print(f"OVERALL MEAN SCORE STATISTICS")
        print(f"{'='*81}")
        print(f"Average mean score across all runs: {np.mean(mean_scores):.4f}")
        print(f"Std deviation: {np.std(mean_scores):.4f}")
        print(f"Min: {np.min(mean_scores):.4f}")
        print(f"Max: {np.max(mean_scores):.4f}")

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

        print(f"{trial_id:<15} {avg_rate:<20.4f} {std_rate:<15.4f} {len(rates):<10} {unsuccessful_run_info}")

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
        print(f"  Trials with at least one success: {trials_with_at_least_one_success}/{len(sorted_trials)}")
        print(f"Std deviation across trials: {np.std(all_averages):.4f}")
        print(f"Number of trials: {len(all_averages)}")
        print(f"Min trial success rate: {np.min(all_averages):.4f}")
        print(f"Max trial success rate: {np.max(all_averages):.4f}")

    # Save results to a JSON file in the current working directory
    output_file = Path("average_success_rates.json")
    results = {
        "overall_statistics": {
            "avg_mean_score": float(np.mean(mean_scores)) if mean_scores else None,
            "std_mean_score": float(np.std(mean_scores)) if mean_scores else None,
            "min_mean_score": float(np.min(mean_scores)) if mean_scores else None,
            "max_mean_score": float(np.max(mean_scores)) if mean_scores else None,
            "num_runs": successful_runs
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
