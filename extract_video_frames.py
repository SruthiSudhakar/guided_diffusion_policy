#!/usr/bin/env python3
"""
Script to extract frames from all MP4 videos in subdirectories.
For each video file, creates a folder with the same name and saves all frames inside.
Parallelized to process multiple videos concurrently.
"""

import os
import cv2
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import functools


def extract_frames_from_video(video_path, output_folder, show_progress=True):
    """
    Extract all frames from a video file and save them to the output folder.

    Args:
        video_path: Path to the video file
        output_folder: Path to the folder where frames will be saved
        show_progress: Whether to show progress bar (disable for parallel processing)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if show_progress:
        print(f"Extracting {total_frames} frames from {video_path.name}...")

    pbar = tqdm(total=total_frames, desc=f"Processing {video_path.name}", disable=not show_progress)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Save frame with zero-padded numbering
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:06d}.png")
        cv2.imwrite(frame_filename, frame)

        frame_count += 1
        pbar.update(1)

    pbar.close()
    cap.release()

    result_msg = f"Extracted {frame_count} frames from {video_path.name} to {output_folder.name}"
    if show_progress:
        print(result_msg)
    return result_msg


def process_single_video(video_path):
    """
    Process a single video file (wrapper for multiprocessing).

    Args:
        video_path: Path object pointing to the video file

    Returns:
        String message with processing result
    """
    # Create folder name by removing the .mp4 extension
    video_name_without_ext = video_path.stem
    output_folder = video_path.parent / video_name_without_ext

    return extract_frames_from_video(video_path, output_folder, show_progress=False)


def process_directory(root_dir, num_workers=None):
    """
    Process all MP4 videos in the directory and its subdirectories in parallel.

    Args:
        root_dir: Root directory to search for MP4 files
        num_workers: Number of parallel workers (default: number of CPU cores)
    """
    root_path = Path(root_dir)

    if not root_path.exists():
        print(f"Error: Directory {root_dir} does not exist")
        return

    # Find all MP4 files recursively
    mp4_files = list(root_path.rglob("*.mp4"))

    if not mp4_files:
        print(f"No MP4 files found in {root_dir}")
        return

    # Determine number of workers
    if num_workers is None:
        num_workers = min(cpu_count(), len(mp4_files))

    print(f"Found {len(mp4_files)} MP4 file(s)")
    print(f"Processing with {num_workers} parallel worker(s)")
    print("-" * 80)

    # Process videos in parallel
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_video, mp4_files),
            total=len(mp4_files),
            desc="Overall progress",
            unit="video"
        ))

    print("-" * 80)
    print("\nProcessing Summary:")
    for result in results:
        print(f"  ✓ {result}")
    print("-" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract frames from MP4 videos in parallel"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="Root directory containing MP4 files (default: data/checkpoints/...)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: auto-detect based on CPU cores)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Video Frame Extraction Script (Parallelized)")
    print("=" * 80)
    print(f"Root directory: {args.root_dir}")
    print("=" * 80)

    process_directory(args.root_dir, num_workers=args.workers)

    print("\n" + "=" * 80)
    print("Frame extraction complete!")
    print("=" * 80)
