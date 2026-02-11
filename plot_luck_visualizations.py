#!/usr/bin/env python3
"""
Visualizations showing the 'luck' phenomenon: more runs -> more chances to succeed.
Reads raw eval_log.json files directly for per-demo per-run data.
"""

import json
import ast
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

BASE_DIR = Path("data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897")

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


def pretty_task_name(task):
    return task.replace("PnP", "").replace("_mg_fixed_224", "")


def load_raw_data():
    """Load all per-demo per-run scores from eval_log.json files.
    Returns: dict of task -> {demo_id -> [score_run1, score_run2, ...]} (sorted by subdir name).
    """
    all_data = {}
    for task in TASKS:
        task_dir = BASE_DIR / f"feb7_na_na_16_mg_place_{task}"
        if not task_dir.exists():
            continue
        demo_scores = defaultdict(list)
        for subdir in sorted(task_dir.iterdir()):
            if not subdir.is_dir():
                continue
            eval_log = subdir / "eval_log.json"
            if not eval_log.exists():
                continue
            with open(eval_log) as f:
                data = json.load(f)
            for key, value in data.items():
                if key.startswith("train/sim_max_reward_"):
                    demo_id = key[len("train/sim_max_reward_"):]
                    demo_scores[demo_id].append(float(value))
        all_data[task] = dict(demo_scores)
    return all_data


def plot_theoretical_overlay(all_data):
    """Plot 1: Actual any-success rate vs theoretical 1-(1-p)^N curve."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, len(TASKS)))

    for i, task in enumerate(TASKS):
        ax = axes_flat[i]
        data = all_data.get(task, {})
        if not data:
            continue

        # Empirical p: overall success rate across all demos and runs
        all_scores = [s for scores in data.values() for s in scores]
        p_empirical = np.mean([s >= 1.0 for s in all_scores])

        max_n = max(len(scores) for scores in data.values())
        run_range = range(1, max_n + 1)

        # Actual any-success rate at each N
        actual_rates = []
        for n in run_range:
            any_successes = []
            for demo_id, scores in data.items():
                if len(scores) >= n:
                    any_successes.append(int(any(s >= 1.0 for s in scores[:n])))
            actual_rates.append(np.mean(any_successes) if any_successes else 0)

        # Theoretical: 1-(1-p)^N
        theoretical = [1 - (1 - p_empirical) ** n for n in run_range]

        ax.plot(list(run_range), actual_rates, 'o-', color=colors[i], linewidth=2,
                markersize=4, label='Actual', zorder=3)
        ax.plot(list(run_range), theoretical, '--', color='black', linewidth=2,
                alpha=0.6, label=f'Theory (p={p_empirical:.3f})')
        ax.set_title(pretty_task_name(task), fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower right')
        if i >= 6:
            ax.set_xlabel('Number of runs', fontsize=11)
        if i % 3 == 0:
            ax.set_ylabel('Any-success rate', fontsize=11)

    fig.suptitle('Actual vs Theoretical Any-Success Rate: 1-(1-p)$^N$\n'
                 '(If curves match, success is purely luck / independent trials)',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("viz_theoretical_overlay.png", dpi=150, bbox_inches='tight')
    print("Saved viz_theoretical_overlay.png")
    plt.close()


def plot_first_success_distribution(all_data):
    """Plot 2: Histogram of which run number gave the first success for each demo."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), sharex=False, sharey=False)
    axes_flat = axes.flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, len(TASKS)))

    # Also collect aggregate data
    all_first_successes = []

    for i, task in enumerate(TASKS):
        ax = axes_flat[i]
        data = all_data.get(task, {})
        if not data:
            continue

        first_success_runs = []
        never_succeeded = 0
        for demo_id, scores in data.items():
            success_indices = [j for j, s in enumerate(scores) if s >= 1.0]
            if success_indices:
                first_success_runs.append(success_indices[0] + 1)  # 1-indexed
                all_first_successes.append(success_indices[0] + 1)
            else:
                never_succeeded += 1

        if first_success_runs:
            max_run = max(first_success_runs)
            bins = np.arange(0.5, max_run + 1.5, 1)
            ax.hist(first_success_runs, bins=bins, color=colors[i], edgecolor='white',
                    alpha=0.8)
        total = len(data)
        succeeded = len(first_success_runs)
        ax.set_title(f'{pretty_task_name(task)}\n({succeeded}/{total} demos ever succeed)',
                     fontsize=11)
        ax.set_xlabel('Run # of first success', fontsize=10)
        ax.set_ylabel('# demos', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle('When Does Luck Strike? Distribution of First-Success Run Number',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig("viz_first_success_distribution.png", dpi=150, bbox_inches='tight')
    print("Saved viz_first_success_distribution.png")
    plt.close()

    # Aggregate histogram
    if all_first_successes:
        fig, ax = plt.subplots(figsize=(10, 5))
        max_run = max(all_first_successes)
        bins = np.arange(0.5, max_run + 1.5, 1)
        ax.hist(all_first_successes, bins=bins, color='#2196F3', edgecolor='white', alpha=0.85)
        median_val = np.median(all_first_successes)
        ax.axvline(median_val, color='red', linestyle='--', linewidth=2,
                   label=f'Median = run {median_val:.0f}')
        ax.set_xlabel('Run # of first success', fontsize=13)
        ax.set_ylabel('# demos (across all tasks)', fontsize=13)
        ax.set_title('When Does Luck Strike? (all tasks aggregated)\n'
                     'Distribution of the run number that first achieves success',
                     fontsize=14)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig("viz_first_success_aggregate.png", dpi=150)
        print("Saved viz_first_success_aggregate.png")
        plt.close()


def plot_heatmaps(all_data):
    """Plot 3: Heatmaps of success/failure per demo per run."""
    cmap = ListedColormap(['#FFCDD2', '#4CAF50'])  # red-ish = fail, green = success

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes_flat = axes.flatten()

    for i, task in enumerate(TASKS):
        ax = axes_flat[i]
        data = all_data.get(task, {})
        if not data:
            continue

        # Sort demos by their mean success rate (hardest on top)
        demo_ids = sorted(data.keys(), key=lambda d: np.mean(data[d]))
        max_runs = max(len(data[d]) for d in demo_ids)

        # Build matrix: rows=demos, cols=runs
        matrix = np.full((len(demo_ids), max_runs), np.nan)
        for row, demo_id in enumerate(demo_ids):
            for col, score in enumerate(data[demo_id]):
                matrix[row, col] = 1.0 if score >= 1.0 else 0.0

        ax.imshow(matrix, aspect='auto', cmap=cmap, interpolation='none', vmin=0, vmax=1)
        ax.set_title(pretty_task_name(task), fontsize=12)
        ax.set_xlabel('Run #', fontsize=10)
        if i % 3 == 0:
            ax.set_ylabel('Demo (sorted by difficulty)', fontsize=10)
        ax.set_yticks([])

    # Add a shared legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#4CAF50', label='Success'),
                       Patch(facecolor='#FFCDD2', label='Failure')]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=12,
               bbox_to_anchor=(0.98, 0.98))

    fig.suptitle('Success/Failure Heatmap: Each Row is a Demo, Each Column is a Run\n'
                 '(Demos sorted by difficulty — hardest at top. Sparse green = need more runs for luck)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("viz_heatmaps.png", dpi=150, bbox_inches='tight')
    print("Saved viz_heatmaps.png")
    plt.close()


def plot_marginal_gain(all_data):
    """Plot 4: Marginal gain in any-success rate from adding the Nth run."""
    # Compute aggregate any-success rate at each N
    max_n = 24
    run_range = range(1, max_n + 1)

    # Aggregate across all tasks
    aggregate_rates = []
    for n in run_range:
        task_rates = []
        for task in TASKS:
            data = all_data.get(task, {})
            if not data:
                continue
            any_successes = []
            for demo_id, scores in data.items():
                if len(scores) >= n:
                    any_successes.append(int(any(s >= 1.0 for s in scores[:n])))
            if any_successes:
                task_rates.append(np.mean(any_successes))
        aggregate_rates.append(np.mean(task_rates) if task_rates else 0)

    marginal_gains = [aggregate_rates[0]] + [aggregate_rates[j] - aggregate_rates[j-1]
                                              for j in range(1, len(aggregate_rates))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart of marginal gains
    bars = ax1.bar(list(run_range), marginal_gains, color='#FF9800', edgecolor='white', alpha=0.85)
    ax1.set_xlabel('Run #', fontsize=13)
    ax1.set_ylabel('Marginal gain in any-success rate', fontsize=13)
    ax1.set_title('Diminishing Returns:\nHow Much Does Each Additional Run Help?', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xticks(list(run_range))

    # Cumulative on the right
    ax2.fill_between(list(run_range), aggregate_rates, alpha=0.3, color='#2196F3')
    ax2.plot(list(run_range), aggregate_rates, 'o-', color='#2196F3', linewidth=2, markersize=5)
    for n_mark in [1, 3, 5, 10]:
        if n_mark <= len(aggregate_rates):
            ax2.annotate(f'{aggregate_rates[n_mark-1]:.1%}',
                         xy=(n_mark, aggregate_rates[n_mark-1]),
                         xytext=(n_mark + 1, aggregate_rates[n_mark-1] + 0.05),
                         fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'),
                         color='#333')
    ax2.set_xlabel('Number of Runs', fontsize=13)
    ax2.set_ylabel('Cumulative any-success rate', fontsize=13)
    ax2.set_title('Cumulative Any-Success Rate\n(with annotations at key run counts)', fontsize=14)
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks(list(run_range))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("viz_marginal_gain.png", dpi=150)
    print("Saved viz_marginal_gain.png")
    plt.close()


def main():
    print("Loading raw data from eval_log.json files...")
    all_data = load_raw_data()
    print(f"Loaded {len(all_data)} tasks")
    for task, demos in all_data.items():
        n_runs = [len(v) for v in demos.values()]
        print(f"  {pretty_task_name(task)}: {len(demos)} demos, {min(n_runs)}-{max(n_runs)} runs each")

    print("\n--- Plot 1: Theoretical overlay ---")
    plot_theoretical_overlay(all_data)

    print("\n--- Plot 2: First success distribution ---")
    plot_first_success_distribution(all_data)

    print("\n--- Plot 3: Heatmaps ---")
    plot_heatmaps(all_data)

    print("\n--- Plot 4: Marginal gain ---")
    plot_marginal_gain(all_data)

    print("\nAll done!")


if __name__ == "__main__":
    main()
