#!/usr/bin/env python3
"""
Script to analyze eval_log.json and calculate the average timestep
when successful trajectories (sim_max_reward=1) reach a reward of 1.
"""

import json
import sys
import re
import ast
from pathlib import Path


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


def analyze_eval_log(json_path):
    """
    Analyze eval_log.json to find average timestep when reward=1 is reached
    for all successful trajectories.

    Args:
        json_path: Path to eval_log.json file
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Pattern to match reward entries
    max_reward_pattern = re.compile(r'(.+)/sim_max_reward_(\d+)_(\d+)')

    # Store timesteps when reward 1 is first reached
    first_reward_one_timesteps = []

    # Track successful and failed trajectories
    successful_count = 0
    failed_count = 0
    no_trajectory_count = 0

    # Iterate through all entries
    for key, value in data.items():
        match = max_reward_pattern.match(key)
        if match:
            prefix = match.group(1)  # e.g., "train" or "test"
            idx1 = match.group(2)
            idx2 = match.group(3)

            max_reward = float(value)

            # If max reward is 1, find the trajectory
            if max_reward == 1.0:
                trajectory_key = f"{prefix}/sim_reward_trajectory_{idx1}_{idx2}"

                if trajectory_key in data:
                    trajectory = data[trajectory_key]
                    first_timestep = find_first_reward_one_timestep(trajectory)

                    if first_timestep is not None:
                        first_reward_one_timesteps.append(first_timestep)
                        successful_count += 1
                    else:
                        # This shouldn't happen if max_reward is 1
                        print(f"Warning: {trajectory_key} has max_reward=1 but no reward=1 found in trajectory")
                        failed_count += 1
                else:
                    print(f"Warning: No trajectory found for {key}")
                    no_trajectory_count += 1

    # Calculate and display results
    if first_reward_one_timesteps:
        avg_timestep = sum(first_reward_one_timesteps) / len(first_reward_one_timesteps)
        min_timestep = min(first_reward_one_timesteps)
        max_timestep = max(first_reward_one_timesteps)

        print(f"\n{'='*60}")
        print(f"Analysis Results for: {json_path}")
        print(f"{'='*60}")
        print(f"Total successful trajectories (max_reward=1): {successful_count}")
        print(f"Trajectories with max_reward=1 but no trajectory data: {no_trajectory_count}")
        print(f"\nAverage timestep when reward=1 first reached: {avg_timestep:.2f}")
        print(f"Minimum timestep: {min_timestep}")
        print(f"Maximum timestep: {max_timestep}")
        print(f"{'='*60}\n")

        # Show distribution
        if len(first_reward_one_timesteps) > 1:
            # Calculate standard deviation
            mean = avg_timestep
            variance = sum((x - mean) ** 2 for x in first_reward_one_timesteps) / len(first_reward_one_timesteps)
            std_dev = variance ** 0.5
            print(f"Standard deviation: {std_dev:.2f}")

            # Show some percentiles
            sorted_timesteps = sorted(first_reward_one_timesteps)
            print(f"\nPercentiles:")
            print(f"  25th percentile: {sorted_timesteps[len(sorted_timesteps)//4]}")
            print(f"  50th percentile (median): {sorted_timesteps[len(sorted_timesteps)//2]}")
            print(f"  75th percentile: {sorted_timesteps[3*len(sorted_timesteps)//4]}")
            print(f"{'='*60}\n")
    else:
        print(f"\nNo successful trajectories found with max_reward=1 in {json_path}")

    return first_reward_one_timesteps


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_success_timesteps.py <path_to_eval_log.json>")
        print("\nExample:")
        print("  python analyze_success_timesteps.py data/checkpoints/.../eval_log.json")
        sys.exit(1)

    json_path = sys.argv[1]

    if not Path(json_path).exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    analyze_eval_log(json_path)


if __name__ == "__main__":
    main()
