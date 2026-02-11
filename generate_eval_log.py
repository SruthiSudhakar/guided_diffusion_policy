#!/usr/bin/env python3
import os
import json
import glob
import re
import argparse

def generate_eval_log(base_dir):
    """
    Generates eval_log.json for the given videos directory.
    Scans for demo folders, calculates trajectory length based on frame count,
    and populates keys.
    """
    if not os.path.isdir(base_dir):
        print(f"Error: Directory {base_dir} not found.")
        return

    json_path = os.path.join(base_dir, "eval_log.json")
    
    # Initialize data structure
    data = {}
    
    # Check if we should preserve existing keys or start fresh
    # We'll load if exists to preserve potential other keys, but overwrite our targets
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            print(f"Loaded existing {json_path}")
        except json.JSONDecodeError:
            print(f"Warning: {json_path} corrupted. Starting fresh.")
            data = {}
    
    # Ensure mean_score is set
    data["train/mean_score"] = "1.0"

    # Find demo directories
    demo_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("demo_")]
    
    if not demo_dirs:
        print(f"No demo directories found in {base_dir}")
        return

    updated_count = 0

    for demo_dir_name in sorted(demo_dirs):
        demo_path = os.path.join(base_dir, demo_dir_name)
        
        # Extract ID
        match = re.search(r'demo_(\d+)', demo_dir_name)
        if not match:
            continue
        demo_id = match.group(1)
        
        # Find frames
        frames = glob.glob(os.path.join(demo_path, "frame_*.png"))
        if not frames:
            print(f"Warning: No frames found in {demo_dir_name}. Skipping.")
            continue
            
        # Find max frame number
        max_frame_num = -1
        for frame_file in frames:
            basename = os.path.basename(frame_file)
            # Assuming 'frame_XXXX.png'
            num_match = re.search(r'(\d+)', basename)
            if num_match:
                num = int(num_match.group(1))
                if num > max_frame_num:
                    max_frame_num = num
        
        if max_frame_num == -1:
            print(f"Warning: Could not determine frame numbers in {demo_dir_name}. Skipping.")
            continue
            
        # "number of the last frame count, multiplies it by 2"
        n = max_frame_num * 2
        
        # "adds n-1 0s and then a 1 at the end as a list of strings"
        # We produce a list of strings: ["0.0", "0.0", ..., "1.0"] 
        # (Using float strings to match typical reward formatting)
        trajectory = str([0.0] * (n - 1) + [1])
        
        # Construct keys using {id}_{id} format
        key_suffix = f"{demo_id}_{demo_id}"
        
        # 1. Trajectory
        data[f"train/sim_reward_trajectory_{key_suffix}"] = str(trajectory) # Store as stringified list to match original file format if that's what "as a list of strings" implied in context of file editing? 
        # Wait, the user said "as a list of strings" because "current list ... is wrong". 
        # Existing file had `"[0.0, ...]"`. If I assume the User wants the JSON value to be a LIST object, I should not stringify.
        # But looking at Step 38 line 86 `str([...])` was present.
        # However, checking Step 11: `"train/sim_reward_trajectory_0_0": "[0.0, ..."`
        # The VALUE is a string.
        # If I write `data[...] = trajectory`, json.dump writes `["0.0", "1.0"]`.
        # If the consumer expects a string, `["..."]` might break it.
        # I will strictly follow "adds ... as a list of strings" literally.
        # But if existing file uses stringified lists, likely the downstream tool expects that.
        # I'll stick to stringified list of floats/strings? 
        # Actually, let's look at the user prompt again: "adds n-1 0s and then a 1 at the end as a list of strings because the current list ... is wrong"
        # Since the current list is WRONG, maybe the format `"[...]"` is exactly what's wrong?
        # Maybe they want a real list?
        # I will store it as a real list: `data[...] = trajectory`.
        
        # NOTE: Changing decision to real list based on "current list is wrong".
        data[f"train/sim_reward_trajectory_{key_suffix}"] = trajectory

        # 2. Max Reward
        data[f"train/sim_max_reward_{key_suffix}"] = "1.0"
        
        # 3. Video Path
        # Determine absolute path to video file
        # Naming convention: demo_{id}.mp4 in base_dir?
        # List dir (Step 4) showed `demo_{id}.mp4` in base_dir.
        video_filename = f"demo_{demo_id}.mp4"
        video_abs_path = os.path.abspath(os.path.join(base_dir, video_filename))
        
        if not os.path.exists(video_abs_path):
             # Try inside the demo folder? Step 4 said demo_0.mp4 is in the PARENT folder (videos/), alongside demo_0 dir.
             # Wait, Step 4: `videos/demo_0` (dir), `videos/demo_0.mp4` (file).
             # So yes, in base_dir.
             pass
        
        data[f"train/sim_video_{key_suffix}"] = video_abs_path
        
        updated_count += 1

    # Write to file
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"Successfully processed {updated_count} demos.")
    print(f"Saved eval log to: {json_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate eval_log.json for video demos.")
    parser.add_argument("--base_dir", required=True, help="Path to the directory containing demo videos subfolders.")
    
    args = parser.parse_args()
    
    generate_eval_log(args.base_dir)