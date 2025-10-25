#!/usr/bin/env python3
"""
Compute MSE between generated frames and simulator rendered frames.
"""

import os, sys
import pdb
import re
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def load_image(path: str) -> np.ndarray:
    """Load an image and convert to numpy array."""
    img = Image.open(path)
    return np.array(img).astype(np.float32)


def compute_mse(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Mean Squared Error between two images."""
    # Ensure images have the same shape
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes don't match: {img1.shape} vs {img2.shape}")

    mse = np.mean((img1 - img2) ** 2)
    return float(mse)


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Structural Similarity Index (SSIM) between two images."""
    # Ensure images have the same shape
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes don't match: {img1.shape} vs {img2.shape}")

    # SSIM expects data range to be specified
    # For float images with range [0, 255], data_range should be 255
    data_range = img1.max() - img1.min()

    # If image has multiple channels, compute SSIM across channels
    if len(img1.shape) == 3:
        ssim_value = ssim(img1, img2, data_range=data_range, channel_axis=2)
    else:
        ssim_value = ssim(img1, img2, data_range=data_range)

    return float(ssim_value)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) between two images."""
    # Ensure images have the same shape
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes don't match: {img1.shape} vs {img2.shape}")

    # PSNR expects data range to be specified
    data_range = img1.max() - img1.min()

    psnr_value = psnr(img1, img2, data_range=data_range)
    return float(psnr_value)


def find_generated_frames(base_dir: Path) -> List[Tuple[int, int, int, Path]]:
    """
    Find all generated frame files.
    Returns list of (step, sample, env, path) tuples.
    """
    pattern = re.compile(rf'step_(\d+)/sample_(\d+)/{sys.argv[2]}/env_(\d+)_3cam_generated_last_frame\.jpg')
    frames = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('_3cam_generated_last_frame.jpg'):
                full_path = Path(root) / file
                rel_path = full_path.relative_to(base_dir)
                match = pattern.search(str(rel_path))
                if match:
                    step = int(match.group(1))
                    sample = int(match.group(2))
                    env = int(match.group(3))
                    frames.append((step, sample, env, full_path))

    return sorted(frames)


def get_simulator_frame_path(base_dir: Path, step: int, sample: int, env: int) -> Path:
    """Get the path to the corresponding simulator rendered frame."""
    filename = f"env_{env}_step_{step}_sample_{sample}_3view_last_frame.png"
    return base_dir / "videos" / filename


def visualize_comparisons(results: List[Dict], base_dir: Path, num_examples: int = 6):
    """
    Visualize comparison between generated and simulator frames.
    Shows best, worst, and median examples.

    Args:
        results: List of result dictionaries with MSE values
        base_dir: Base directory path
        num_examples: Number of examples to show (default 6)
    """
    if len(results) == 0:
        print("No results to visualize")
        return

    # Sort results by MSE
    sorted_results = sorted(results, key=lambda x: x['mse'])

    # Select examples: best (lowest MSE), worst (highest MSE), and some in between
    indices = []
    if len(sorted_results) >= num_examples:
        # Get best, worst, and evenly spaced examples in between
        step_size = len(sorted_results) // (num_examples - 1)
        indices = [0]  # Best
        for i in range(1, num_examples - 1):
            indices.append(min(i * step_size, len(sorted_results) - 1))
        indices.append(len(sorted_results) - 1)  # Worst
        indices = sorted(set(indices))[:num_examples]
    else:
        indices = list(range(len(sorted_results)))

    examples = [sorted_results[i] for i in indices]

    # Create visualization
    n_examples = len(examples)
    fig = plt.figure(figsize=(15, 3 * n_examples))
    gs = gridspec.GridSpec(n_examples, 4, figure=fig, wspace=0.3, hspace=0.3)

    for idx, result in enumerate(examples):
        pdb.set_trace()
        gen_path = result['generated_path']
        sim_path = result['simulator_path']
        mse = result['mse']
        ssim_val = result['ssim']
        psnr_val = result['psnr']
        step = result['step']
        sample = result['sample']
        env = result['env']

        # Load images
        gen_img = np.array(Image.open(gen_path))
        sim_img = np.array(Image.open(sim_path))

        # Compute absolute difference
        diff = np.abs(gen_img.astype(float) - sim_img.astype(float))
        diff_normalized = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

        # Plot generated frame
        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.imshow(gen_img)
        ax1.set_title(f'Generated\nStep {step}, Sample {sample}, Env {env}', fontsize=10)
        ax1.axis('off')

        # Plot simulator frame
        ax2 = fig.add_subplot(gs[idx, 1])
        ax2.imshow(sim_img)
        ax2.set_title(f'Simulator\nStep {step}, Sample {sample}, Env {env}', fontsize=10)
        ax2.axis('off')

        # Plot absolute difference
        ax3 = fig.add_subplot(gs[idx, 2])
        im = ax3.imshow(diff_normalized, cmap='hot')
        ax3.set_title(f'Abs Difference (normalized)\nMSE: {mse:.2f}', fontsize=10)
        ax3.axis('off')
        plt.colorbar(im, ax=ax3, fraction=0.046)

        # Plot metrics and difference histogram
        ax4 = fig.add_subplot(gs[idx, 3])
        ax4.hist(diff.flatten(), bins=50, color='blue', alpha=0.7)
        ax4.set_xlabel('Pixel Difference', fontsize=9)
        ax4.set_ylabel('Frequency', fontsize=9)
        ax4.set_title(f'Metrics\nSSIM: {ssim_val:.4f} | PSNR: {psnr_val:.2f} dB', fontsize=9)
        ax4.tick_params(labelsize=8)
        ax4.grid(True, alpha=0.3)

    # Add overall title
    rank_labels = []
    for i, idx_val in enumerate(indices):
        percentile = (idx_val / (len(sorted_results) - 1)) * 100 if len(sorted_results) > 1 else 0
        rank_labels.append(f"#{idx_val+1} ({percentile:.0f}th percentile)")

    fig.suptitle(f'MSE Comparison Examples\nShowing: {", ".join(rank_labels)}',
                 fontsize=14, fontweight='bold', y=0.995)

    # Save figure
    output_path = base_dir / "mse_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  - Comparison visualization saved to: {output_path}")
    plt.close()


def visualize_metric_comparisons(results: List[Dict], base_dir: Path, metric: str, num_examples: int = 6):
    """
    Visualize comparison between generated and simulator frames sorted by a specific metric.

    Args:
        results: List of result dictionaries with metric values
        base_dir: Base directory path
        metric: Which metric to use for sorting ('mse', 'ssim', 'psnr')
        num_examples: Number of examples to show (default 6)
    """
    if len(results) == 0:
        print("No results to visualize")
        return

    # Sort results by the specified metric
    if metric == 'ssim' or metric == 'psnr':
        # For SSIM and PSNR, higher is better, so reverse sort
        sorted_results = sorted(results, key=lambda x: x[metric], reverse=True)
    else:
        # For MSE, lower is better
        sorted_results = sorted(results, key=lambda x: x[metric])

    # Select examples: best, worst, and evenly spaced examples in between
    indices = []
    if len(sorted_results) >= num_examples:
        step_size = len(sorted_results) // (num_examples - 1)
        indices = [0]  # Best
        for i in range(1, num_examples - 1):
            indices.append(min(i * step_size, len(sorted_results) - 1))
        indices.append(len(sorted_results) - 1)  # Worst
        indices = sorted(set(indices))[:num_examples]
    else:
        indices = list(range(len(sorted_results)))

    examples = [sorted_results[i] for i in indices]

    # Create visualization
    n_examples = len(examples)
    fig = plt.figure(figsize=(15, 3 * n_examples))
    gs = gridspec.GridSpec(n_examples, 4, figure=fig, wspace=0.3, hspace=0.3)

    metric_name = {'mse': 'MSE', 'ssim': 'SSIM', 'psnr': 'PSNR'}[metric]

    for idx, result in enumerate(examples):
        gen_path = result['generated_path']
        sim_path = result['simulator_path']
        mse = result['mse']
        ssim_val = result['ssim']
        psnr_val = result['psnr']
        metric_val = result[metric]
        step = result['step']
        sample = result['sample']
        env = result['env']

        # Load images
        gen_img = np.array(Image.open(gen_path))
        sim_img = np.array(Image.open(sim_path))

        # Compute absolute difference
        diff = np.abs(gen_img.astype(float) - sim_img.astype(float))
        diff_normalized = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)

        # Plot generated frame
        ax1 = fig.add_subplot(gs[idx, 0])
        ax1.imshow(gen_img)
        ax1.set_title(f'Generated\nStep {step}, Sample {sample}, Env {env}', fontsize=10)
        ax1.axis('off')

        # Plot simulator frame
        ax2 = fig.add_subplot(gs[idx, 1])
        ax2.imshow(sim_img)
        ax2.set_title(f'Simulator\nStep {step}, Sample {sample}, Env {env}', fontsize=10)
        ax2.axis('off')

        # Plot absolute difference
        ax3 = fig.add_subplot(gs[idx, 2])
        im = ax3.imshow(diff_normalized, cmap='hot')
        if metric == 'mse':
            ax3.set_title(f'Abs Difference (normalized)\n{metric_name}: {metric_val:.2f}', fontsize=10)
        elif metric == 'ssim':
            ax3.set_title(f'Abs Difference (normalized)\n{metric_name}: {metric_val:.4f}', fontsize=10)
        else:  # psnr
            ax3.set_title(f'Abs Difference (normalized)\n{metric_name}: {metric_val:.2f} dB', fontsize=10)
        ax3.axis('off')
        plt.colorbar(im, ax=ax3, fraction=0.046)

        # Plot all metrics
        ax4 = fig.add_subplot(gs[idx, 3])
        ax4.axis('off')

        # Display metrics as text
        metrics_text = f'All Metrics:\n\n'
        metrics_text += f'MSE: {mse:.2f}\n'
        metrics_text += f'SSIM: {ssim_val:.4f}\n'
        metrics_text += f'PSNR: {psnr_val:.2f} dB\n\n'

        # Add histogram
        ax4_hist = fig.add_subplot(gs[idx, 3])
        ax4_hist.hist(diff.flatten(), bins=50, color='blue', alpha=0.7)
        ax4_hist.set_xlabel('Pixel Difference', fontsize=9)
        ax4_hist.set_ylabel('Frequency', fontsize=9)
        ax4_hist.set_title(f'MSE: {mse:.2f} | SSIM: {ssim_val:.4f} | PSNR: {psnr_val:.2f} dB', fontsize=8)
        ax4_hist.tick_params(labelsize=8)
        ax4_hist.grid(True, alpha=0.3)

    # Add overall title
    rank_labels = []
    for i, idx_val in enumerate(indices):
        percentile = (idx_val / (len(sorted_results) - 1)) * 100 if len(sorted_results) > 1 else 0
        rank_labels.append(f"#{idx_val+1} ({percentile:.0f}th percentile)")

    sort_direction = "Best to Worst" if metric == 'mse' else "Worst to Best"
    fig.suptitle(f'{metric_name} Comparison Examples ({sort_direction})\nShowing: {", ".join(rank_labels)}',
                 fontsize=14, fontweight='bold', y=0.995)

    # Save figure
    output_path = base_dir / f"{metric}_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  - {metric_name} comparison visualization saved to: {output_path}")
    plt.close()


def visualize_metrics_distribution(results: List[Dict], base_dir: Path):
    """
    Create a summary visualization showing the distribution of all metrics.

    Args:
        results: List of result dictionaries with metric values
        base_dir: Base directory path
    """
    if len(results) == 0:
        print("No results to visualize")
        return

    mse_values = [r['mse'] for r in results]
    ssim_values = [r['ssim'] for r in results]
    psnr_values = [r['psnr'] for r in results]

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # MSE histogram
    axes[0].hist(mse_values, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(mse_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(mse_values):.2f}')
    axes[0].axvline(np.median(mse_values), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(mse_values):.2f}')
    axes[0].set_xlabel('MSE', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('MSE Distribution\n(Lower is better)', fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # SSIM histogram
    axes[1].hist(ssim_values, bins=30, color='seagreen', alpha=0.7, edgecolor='black')
    axes[1].axvline(np.mean(ssim_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ssim_values):.4f}')
    axes[1].axvline(np.median(ssim_values), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(ssim_values):.4f}')
    axes[1].set_xlabel('SSIM', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('SSIM Distribution\n(Higher is better, range: -1 to 1)', fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # PSNR histogram
    axes[2].hist(psnr_values, bins=30, color='coral', alpha=0.7, edgecolor='black')
    axes[2].axvline(np.mean(psnr_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(psnr_values):.2f} dB')
    axes[2].axvline(np.median(psnr_values), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(psnr_values):.2f} dB')
    axes[2].set_xlabel('PSNR (dB)', fontsize=12)
    axes[2].set_ylabel('Frequency', fontsize=12)
    axes[2].set_title('PSNR Distribution\n(Higher is better)', fontsize=13, fontweight='bold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(f'Image Quality Metrics Distribution (n={len(results)} comparisons)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save figure
    output_path = base_dir / "metrics_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  - Metrics distribution saved to: {output_path}")
    plt.close()


def main():
    # Base directory
    
    base_dir = Path(sys.argv[1])

    if not base_dir.exists():
        print(f"Error: Base directory does not exist: {base_dir}")
        return

    print("Finding generated frames...")
    generated_frames = find_generated_frames(base_dir)
    print(f"Found {len(generated_frames)} generated frames")

    if len(generated_frames) == 0:
        print("No generated frames found!")
        return

    # Compute MSE for each pair
    results = []
    missing_files = []
    errors = []

    for step, sample, env, gen_path in generated_frames:
        sim_path = get_simulator_frame_path(base_dir, step, sample, env)

        if not sim_path.exists():
            missing_files.append({
                'step': step,
                'sample': sample,
                'env': env,
                'generated_path': str(gen_path),
                'simulator_path': str(sim_path)
            })
            continue

        try:
            gen_img = load_image(gen_path)
            sim_img = load_image(sim_path)
            mse = compute_mse(gen_img, sim_img)
            ssim_value = compute_ssim(gen_img, sim_img)
            psnr_value = compute_psnr(gen_img, sim_img)

            results.append({
                'step': step,
                'sample': sample,
                'env': env,
                'mse': mse,
                'ssim': ssim_value,
                'psnr': psnr_value,
                'generated_path': str(gen_path),
                'simulator_path': str(sim_path)
            })

            print(f"Step {step:2d}, Sample {sample:2d}, Env {env:2d}: MSE = {mse:.4f}, SSIM = {ssim_value:.4f}, PSNR = {psnr_value:.2f} dB")

        except Exception as e:
            errors.append({
                'step': step,
                'sample': sample,
                'env': env,
                'error': str(e),
                'generated_path': str(gen_path),
                'simulator_path': str(sim_path)
            })
            print(f"Error processing Step {step}, Sample {sample}, Env {env}: {e}")

    # Compute statistics
    if results:
        mse_values = [r['mse'] for r in results]
        ssim_values = [r['ssim'] for r in results]
        psnr_values = [r['psnr'] for r in results]

        stats = {
            'num_comparisons': len(results),
            'mse': {
                'mean': float(np.mean(mse_values)),
                'std': float(np.std(mse_values)),
                'min': float(np.min(mse_values)),
                'max': float(np.max(mse_values)),
                'median': float(np.median(mse_values))
            },
            'ssim': {
                'mean': float(np.mean(ssim_values)),
                'std': float(np.std(ssim_values)),
                'min': float(np.min(ssim_values)),
                'max': float(np.max(ssim_values)),
                'median': float(np.median(ssim_values))
            },
            'psnr': {
                'mean': float(np.mean(psnr_values)),
                'std': float(np.std(psnr_values)),
                'min': float(np.min(psnr_values)),
                'max': float(np.max(psnr_values)),
                'median': float(np.median(psnr_values))
            }
        }

        print("\n" + "="*70)
        print("STATISTICS")
        print("="*70)
        print(f"Number of comparisons: {stats['num_comparisons']}")
        print("\nMSE (Mean Squared Error) - Lower is better:")
        print(f"  Mean:   {stats['mse']['mean']:.4f}")
        print(f"  Std:    {stats['mse']['std']:.4f}")
        print(f"  Min:    {stats['mse']['min']:.4f}")
        print(f"  Max:    {stats['mse']['max']:.4f}")
        print(f"  Median: {stats['mse']['median']:.4f}")

        print("\nSSIM (Structural Similarity Index) - Higher is better (range: -1 to 1):")
        print(f"  Mean:   {stats['ssim']['mean']:.4f}")
        print(f"  Std:    {stats['ssim']['std']:.4f}")
        print(f"  Min:    {stats['ssim']['min']:.4f}")
        print(f"  Max:    {stats['ssim']['max']:.4f}")
        print(f"  Median: {stats['ssim']['median']:.4f}")

        print("\nPSNR (Peak Signal-to-Noise Ratio) - Higher is better (in dB):")
        print(f"  Mean:   {stats['psnr']['mean']:.2f} dB")
        print(f"  Std:    {stats['psnr']['std']:.2f} dB")
        print(f"  Min:    {stats['psnr']['min']:.2f} dB")
        print(f"  Max:    {stats['psnr']['max']:.2f} dB")
        print(f"  Median: {stats['psnr']['median']:.2f} dB")
    else:
        stats = None
        print("\nNo successful comparisons!")

    # Print summary of issues
    if missing_files:
        print(f"\n{len(missing_files)} simulator frames not found")

    if errors:
        print(f"\n{len(errors)} errors occurred during processing")

    # Save results to JSON
    output = {
        'statistics': stats,
        'results': results,
        'missing_files': missing_files,
        'errors': errors
    }

    output_file = base_dir / "mse_results.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Create visualizations
    if results:
        print("\nGenerating visualizations...")
        visualize_comparisons(results, base_dir, num_examples=6)
        visualize_metric_comparisons(results, base_dir, metric='ssim', num_examples=6)
        visualize_metric_comparisons(results, base_dir, metric='psnr', num_examples=6)
        visualize_metrics_distribution(results, base_dir)


if __name__ == "__main__":
    main()
