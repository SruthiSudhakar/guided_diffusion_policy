import os
import json
import glob
from pathlib import Path
import pdb
import random
"""

python3 generate_visualization.py --mp PnPCoffeeServeMug_expert_ \
    --root_dir data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_mg_place_PnPCounterToCab_mg_fixed_224 
    
python3 -m http.server 8000
"""

def generate_visualization(args):
    output_file = f"{args.root_dir}/visualization.html"
    
    # Find all eval_log.json files
    search_pattern = os.path.join(args.root_dir, f"{args.mp}*", "eval_log.json")
    log_files = glob.glob(search_pattern)
    log_files = random.sample(log_files, k=min(args.num_runs, len(log_files)))

    print(f"Found {len(log_files)} log files.")

    all_demos = []
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
                
            # The directory containing the log file
            log_dir = os.path.dirname(log_file)
            
            # Extract demo IDs from keys like "train/sim_max_reward_{demo_id}"
            # We look for keys starting with "train/sim_max_reward_"
            for key, value in data.items():
                if key.startswith("train/sim_max_reward_"):
                    demo_id_suffix = key.replace("train/sim_max_reward_", "")
                    
                    # Construct related keys
                    video_key = f"train/sim_video_{demo_id_suffix}"
                    metadata_key = f"train/xobject_metadata{demo_id_suffix}"
                    
                    if video_key in data:
                        video_path_rel = data[video_key]
                        # The video path in json is relative to project root.
                        # We need to make it relative to the output HTML file.
                        # output_file is {args.root_dir}/visualization.html
                        # So we need relpath from args.root_dir to video_path_rel
                        
                        try:
                            final_video_path = os.path.relpath(video_path_rel, start=args.root_dir)
                        except ValueError:
                            # Fallback if paths are on different drives or something weird
                            final_video_path = video_path_rel

                        metadata = data.get(metadata_key, "N/A")
                        reward = float(value)
                        
                        all_demos.append({
                            'demo_id': demo_id_suffix,
                            'reward': reward,
                            'video_path': final_video_path,
                            'metadata': metadata,
                            'source_log': log_file
                        })
        except Exception as e:
            print(f"Error reading {log_file}: {e}")

    # Sort demos by ID for consistent order
    all_demos.sort(key=lambda x: x['demo_id'])
    
    # Group by Demo Group (e.g. "0" from "0_0")
    demos_by_group = {}
    for demo in all_demos:
        # Assuming format {id}_{index}, take the part before the last underscore
        # or just the first part if it looks like id_index.
        # Based on observed data "0_0", "100_16", it seems to be ID_INDEX.
        # We'll use the first part as the group ID.
        group_id = demo['demo_id'].split('_')[0]
        if group_id not in demos_by_group:
            demos_by_group[group_id] = {'success': [], 'failure': []}
        
        if demo['reward'] >= 1.0:
            demos_by_group[group_id]['success'].append(demo)
        else:
            demos_by_group[group_id]['failure'].append(demo)

    # Calculate success rates and sort groups
    group_stats = []
    for group_id, data in demos_by_group.items():
        n_success = len(data['success'])
        n_failure = len(data['failure'])
        total = n_success + n_failure
        rate = (n_success / total) * 100 if total > 0 else 0
        group_stats.append({
            'id': group_id,
            'rate': rate,
            'total': total
        })
    
    # Sort by success rate (ascending), then by ID
    def try_int(s):
        try:
            return int(s)
        except:
            return s

    group_stats.sort(key=lambda x: (x['rate'], try_int(x['id'])))
    sorted_group_ids = [g['id'] for g in group_stats]

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Evaluation Visualization</title>
    <style>
        :root {{
            --bg-color: #121212;
            --card-bg: #1e1e1e;
            --text-color: #e0e0e0;
            --accent-color: #bb86fc;
            --success-color: #03dac6;
            --failure-color: #cf6679;
            --border-color: #333;
        }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: var(--accent-color);
        }}
        .summary {{
            background-color: var(--card-bg);
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            display: flex;
            gap: 20px;
            align-items: center;
            border: 1px solid var(--border-color);
        }}
        .stat {{
            font-size: 1.2em;
        }}
        .stat-value {{
            font-weight: bold;
            color: var(--success-color);
        }}
        .demo-section {{
            margin-bottom: 40px;
            border-top: 1px solid var(--border-color);
            padding-top: 20px;
        }}
        .demo-header {{
            display: flex;
            align-items: center;
            gap: 15px;
            margin-bottom: 15px;
        }}
        .demo-title {{
            font-size: 1.5em;
            font-weight: bold;
            color: var(--text-color);
        }}
        .demo-stats {{
            font-size: 0.9em;
            color: #aaa;
        }}
        .sub-header {{
            font-size: 1.1em;
            margin: 15px 0 10px 0;
            color: #888;
            border-left: 3px solid var(--accent-color);
            padding-left: 10px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }}
        .card {{
            background-color: var(--card-bg);
            border-radius: 8px;
            overflow: hidden;
            transition: transform 0.2s;
            border: 1px solid var(--border-color);
        }}
        .card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }}
        .card-header {{
            padding: 8px 12px;
            background-color: rgba(255,255,255,0.05);
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .demo-id {{
            font-weight: bold;
            font-size: 0.85em;
        }}
        .reward-badge {{
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.75em;
            font-weight: bold;
        }}
        .success {{
            background-color: rgba(3, 218, 198, 0.2);
            color: var(--success-color);
        }}
        .failure {{
            background-color: rgba(207, 102, 121, 0.2);
            color: var(--failure-color);
        }}
        .video-container {{
            width: 100%;
            aspect-ratio: 16/9;
            background-color: #000;
        }}
        video {{
            width: 100%;
            height: 100%;
            object-fit: contain;
        }}
        .card-body {{
            padding: 8px 12px;
            font-size: 0.8em;
            color: #aaa;
        }}
        .metadata {{
            margin-top: 5px;
            font-family: monospace;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
    </style>
</head>
<body>
    <h1>Evaluation Visualization</h1>
    
    <div class="summary">
        <div class="stat">Total Videos: <span class="stat-value" style="color: var(--text-color)">{len(all_demos)}</span></div>
        <div class="stat">Total Demos: <span class="stat-value" style="color: var(--accent-color)">{len(demos_by_group)}</span></div>
    </div>
"""

    for group_id in sorted_group_ids:
        group = demos_by_group[group_id]
        successes = group['success']
        failures = group['failure']
        
        # Limit to 5 each
        shown_successes = successes[:5]
        shown_failures = failures[:5]
        
        if not shown_successes and not shown_failures:
            continue
            
        html_content += f"""
    <div class="demo-section">
        <div class="demo-header">
            <span class="demo-title">Demo {group_id}</span>
            <span class="demo-stats">
                (Success: {len(successes)}, Failure: {len(failures)})
            </span>
        </div>
"""
        
        if shown_successes:
            html_content += f"""
        <div class="sub-header">Success Examples (showing {len(shown_successes)} of {len(successes)})</div>
        <div class="grid">
"""
            for demo in shown_successes:
                html_content += create_card_html(demo, "success")
            html_content += "        </div>"

        if shown_failures:
            html_content += f"""
        <div class="sub-header">Failure Examples (showing {len(shown_failures)} of {len(failures)})</div>
        <div class="grid">
"""
            for demo in shown_failures:
                html_content += create_card_html(demo, "failure")
            html_content += "        </div>"
            
        html_content += "    </div>"

    html_content += """
</body>
</html>
"""

    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"Visualization generated at {os.path.abspath(output_file)}")

def create_card_html(demo, status_class):
    return f"""
        <div class="card">
            <div class="card-header">
                <span class="demo-id">ID: {demo['demo_id']}</span>
                <span class="reward-badge {status_class}">R: {demo['reward']}</span>
            </div>
            <div class="video-container">
                <video controls preload="none" poster="">
                    <source src="{demo['video_path']}" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
            </div>
            <div class="card-body">
                <div class="metadata" title="{demo['metadata']}">Meta: {demo['metadata']}</div>
            </div>
        </div>
    """

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--mp",
        type=str,
        default='',
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=100000,
    )
    args = parser.parse_args()
    generate_visualization(args)
