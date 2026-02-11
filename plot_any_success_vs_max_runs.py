#!/usr/bin/env python3
"""
Runs compute_average_success_rates_perdemo.py for varying max_runs values,
collects avg_any_success_rate across all tasks, and plots the result.
"""

import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = Path("data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897")

if 'place' in sys.argv[1]:
    TASKS = [
        "PnPCabToCounter_mg_fixed_224",
        "PnPCounterToCab_mg_fixed_224",
        "PnPCounterToMicrowave_mg_fixed_224",
        "PnPMicrowaveToCounter_mg_fixed_224",
        "PnPCoffeeServeMug_mg_fixed_224",
        "PnPCounterToSink_mg_fixed_224",
        "PnPCounterToStove_mg_fixed_224",
        "PnPStoveToCounter_mg_fixed_224",
        "PnPSinkToCounter_mg_fixed_224",
    ]
elif 'fulltask' in sys.argv[1]:
    TASKS = [
        "PnPCabToCounter",
        "PnPCounterToCab",
        "PnPCounterToMicrowave",
        "PnPMicrowaveToCounter",
        "PnPCoffeeServeMug",
        "PnPCounterToSink",
        "PnPCounterToStove",
        "PnPStoveToCounter",
        "PnPSinkToCounter",
    ]
else:
    raise ValueError(f"Unknown subdir: {sys.argv[1]}")

MAX_RUNS_VALUES = list(range(1, 25))


def run_for_max_runs(max_runs, subdir):
    """Run compute script for all tasks with given max_runs, return dict of task -> avg_any_success_rate."""
    task_rates = {}
    for task in TASKS:
        task_dir = BASE_DIR / f"{subdir}_{task}"
        if not task_dir.exists():
            print(f"  Warning: {task_dir} not found, skipping")
            continue
        cmd = [
            sys.executable, "compute_average_success_rates_perdemo.py",
            "--base_dir", str(task_dir),
            "--max_runs", str(max_runs),
            "--not_multitask",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  Error running for {task} max_runs={max_runs}: {result.stderr}")
            continue

        json_path = task_dir / "average_success_rates_perdemo.json"
        with open(json_path, 'r') as f:
            data = json.load(f)

        rate = data.get("overall_statistics", {}).get("avg_any_success_rate")
        if rate is not None:
            task_rates[task] = rate
    return task_rates


def pretty_task_name(task):
    """Convert e.g. 'PnPCabToCounter_mg_fixed_224' -> 'CabToCounter'."""
    name = task.replace("PnP", "").replace("_mg_fixed_224", "")
    return name


def main():
    avg_rates = []
    std_rates = []
    valid_max_runs = []
    # per-task tracking: task -> list of rates (one per max_runs value)
    per_task_rates = {task: [] for task in TASKS}
    subdir = sys.argv[1] #"na_na_16_expert_fulltask"
    for mr in MAX_RUNS_VALUES:
        print(f"Running max_runs={mr}...")
        rates_dict = run_for_max_runs(mr, subdir)
        if rates_dict:
            all_rates = list(rates_dict.values())
            avg_rates.append(np.mean(all_rates))
            std_rates.append(np.std(all_rates))
            valid_max_runs.append(mr)
            for task in TASKS:
                per_task_rates[task].append(rates_dict.get(task))
            print(f"  avg_any_success_rate across {len(all_rates)} tasks: {np.mean(all_rates):.4f} +/- {np.std(all_rates):.4f}")

    # --- Plot 1: Aggregate (same as before) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(valid_max_runs, avg_rates, yerr=std_rates, marker='o', capsize=4,
                linewidth=2, markersize=6, color='#2196F3', ecolor='#90CAF9')
    ax.set_xlabel('Number of Runs (max_runs)', fontsize=13)
    ax.set_ylabel('avg_any_success_rate (across all tasks)', fontsize=13)
    ax.set_xticks(valid_max_runs)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=avg_rates[-1], color='gray', linestyle='--', alpha=0.5, label=f'max ({avg_rates[-1]:.3f})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{BASE_DIR}/{subdir}/any_success_rate_vs_max_runs.png", dpi=150)
    print(f"\nAggregate plot saved to any_success_rate_vs_max_runs.png")
    plt.close()

    # --- Plot 2: All tasks on one plot ---
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(TASKS)))
    for i, task in enumerate(TASKS):
        rates = per_task_rates[task]
        valid = [(mr, r) for mr, r in zip(valid_max_runs, rates) if r is not None]
        if valid:
            mrs, rs = zip(*valid)
            ax.plot(mrs, rs, marker='o', linewidth=2, markersize=5, color=colors[i],
                    label=pretty_task_name(task))
    ax.plot(valid_max_runs, avg_rates, 'k--', linewidth=3, alpha=0.7, label='Average')
    ax.set_xlabel('Number of Runs (max_runs)', fontsize=13)
    ax.set_ylabel('avg_any_success_rate', fontsize=13)
    ax.set_title('Any-Success Rate vs Number of Runs (per task)', fontsize=14)
    ax.set_xticks(valid_max_runs)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='lower right')
    plt.tight_layout()
    plt.savefig(f"{BASE_DIR}/{subdir}/any_success_rate_vs_max_runs_per_task.png", dpi=150)
    print(f"Per-task plot saved to any_success_rate_vs_max_runs_per_task.png")
    plt.close()

    # --- Plot 3: Individual subplots per task ---
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    for i, task in enumerate(TASKS):
        ax = axes_flat[i]
        rates = per_task_rates[task]
        valid = [(mr, r) for mr, r in zip(valid_max_runs, rates) if r is not None]
        if valid:
            mrs, rs = zip(*valid)
            ax.plot(mrs, rs, marker='o', linewidth=2, markersize=4, color=colors[i])
            ax.axhline(y=rs[-1], color='gray', linestyle='--', alpha=0.5)
            ax.set_title(pretty_task_name(task), fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        if i >= 6:
            ax.set_xlabel('max_runs', fontsize=11)
        if i % 3 == 0:
            ax.set_ylabel('avg_any_success_rate', fontsize=11)
    fig.suptitle('Any-Success Rate vs Number of Runs (individual tasks)', fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{BASE_DIR}/{subdir}/any_success_rate_vs_max_runs_subplots.png", dpi=150, bbox_inches='tight')
    print(f"Subplots saved to any_success_rate_vs_max_runs_subplots.png")
    plt.close()

    # Save raw data
    raw_data = {
        "max_runs": valid_max_runs,
        "avg_any_success_rate": avg_rates,
        "std_across_tasks": std_rates,
        "per_task": {pretty_task_name(t): per_task_rates[t] for t in TASKS},
    }
    with open(f"{BASE_DIR}/{subdir}/any_success_rate_vs_max_runs.json", 'w') as f:
        json.dump(raw_data, f, indent=2)
    print(f"Raw data saved to any_success_rate_vs_max_runs.json")


if __name__ == "__main__":
    main()
