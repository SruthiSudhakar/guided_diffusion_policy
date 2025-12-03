#!/usr/bin/env python3
"""
Analyze variance in rewards across all eval_log.json files.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
path = sys.argv[1]
def main():
    # Base directory containing all the eval subdirectories
    base_dir = Path(path)

    # Dictionary to store rewards for each demo: demo_id -> [list of rewards]
    demo_rewards = defaultdict(list)

    # Find all eval_log.json files
    eval_logs = list(base_dir.glob("*/eval_log.json"))
    print(f"Found {len(eval_logs)} eval_log.json files")

    # Process each eval_log.json
    for eval_log_path in eval_logs:
        try:
            with open(eval_log_path, 'r') as f:
                data = json.load(f)

            # Extract all train/sim_max_reward_* keys
            for key, value in data.items():
                if key.startswith('train/sim_max_reward_'):
                    # Extract demo ID (everything after 'train/sim_max_reward_')
                    demo_id = key.replace('train/sim_max_reward_', '')
                    # Convert to float in case it's stored as string
                    try:
                        demo_rewards[demo_id].append(float(value))
                    except (ValueError, TypeError):
                        print(f"Warning: Could not convert value '{value}' for key '{key}' to float")
                        continue

        except Exception as e:
            print(f"Error processing {eval_log_path}: {e}")
            continue

    print(f"\nFound {len(demo_rewards)} unique demos")

    # Calculate variance for each demo
    demo_variances = {}
    for demo_id, rewards in demo_rewards.items():
        if len(rewards) > 1:  # Need at least 2 values to calculate variance
            demo_variances[demo_id] = {
                'variance': np.var(rewards),
                'mean': np.mean(rewards),
                'std': np.std(rewards),
                'count': len(rewards),
                'min': min(rewards),
                'max': max(rewards),
                'rewards': rewards
            }

    # Sort by variance (descending)
    sorted_demos = sorted(demo_variances.items(), key=lambda x: x[1]['variance'], reverse=True)

    # Print results
    print("\n" + "="*80)
    print("DEMOS WITH HIGHEST REWARD VARIANCE")
    print("="*80)
    print(f"{'Demo ID':<30} {'Variance':<12} {'Std Dev':<12} {'Mean':<12} {'Min-Max':<15} {'Count'}")
    print("-"*80)

    for demo_id, stats in sorted_demos[:50]:  # Top 50
        min_max_str = f"{stats['min']:.3f}-{stats['max']:.3f}"
        print(f"{demo_id:<30} {stats['variance']:<12.6f} {stats['std']:<12.6f} {stats['mean']:<12.6f} {min_max_str:<15} {stats['count']}")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    all_variances = [stats['variance'] for _, stats in sorted_demos]
    if all_variances:
        print(f"Total demos analyzed: {len(sorted_demos)}")
        print(f"Mean variance: {np.mean(all_variances):.6f}")
        print(f"Median variance: {np.median(all_variances):.6f}")
        print(f"Max variance: {np.max(all_variances):.6f}")
        print(f"Min variance: {np.min(all_variances):.6f}")

    # Optionally save full results to a file
    output_file = Path(f"{path}/reward_variance_analysis.json")
    output_txt_file = Path(f"{path}/reward_variance_analysis.txt")
    output_data = {
        demo_id: stats for demo_id, stats in sorted_demos
    }
    # Convert numpy types to Python types for JSON serialization
    for demo_id in output_data:
        for key in ['variance', 'mean', 'std', 'min', 'max']:
            output_data[demo_id][key] = float(output_data[demo_id][key])
        output_data[demo_id]['count'] = int(output_data[demo_id]['count'])
        output_data[demo_id]['rewards'] = [float(r) for r in output_data[demo_id]['rewards']]

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nFull results saved to: {output_file}")

    # Save printed output to text file
    with open(output_txt_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DEMOS WITH HIGHEST REWARD VARIANCE\n")
        f.write("="*80 + "\n")
        f.write(f"{'Demo ID':<30} {'Variance':<12} {'Std Dev':<12} {'Mean':<12} {'Min-Max':<15} {'Count'}\n")
        f.write("-"*80 + "\n")

        for demo_id, stats in sorted_demos[:50]:  # Top 50
            min_max_str = f"{stats['min']:.3f}-{stats['max']:.3f}"
            f.write(f"{demo_id:<30} {stats['variance']:<12.6f} {stats['std']:<12.6f} {stats['mean']:<12.6f} {min_max_str:<15} {stats['count']}\n")

        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY STATISTICS\n")
        f.write("="*80 + "\n")
        if all_variances:
            f.write(f"Total demos analyzed: {len(sorted_demos)}\n")
            f.write(f"Mean variance: {np.mean(all_variances):.6f}\n")
            f.write(f"Median variance: {np.median(all_variances):.6f}\n")
            f.write(f"Max variance: {np.max(all_variances):.6f}\n")
            f.write(f"Min variance: {np.min(all_variances):.6f}\n")

    print(f"Text output saved to: {output_txt_file}")

if __name__ == "__main__":
    main()
