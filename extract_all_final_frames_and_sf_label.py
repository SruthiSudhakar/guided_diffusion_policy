import glob
import os
import json
import ast
from pathlib import Path
import logging
import numpy as np
from PIL import Image
import argparse
import pdb
from tqdm import tqdm
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--base_dataset_path', type=str)#, default='data/outputs/dec4/2025.12.03/22.38.27_train_diffusion_unet_clip/checkpoints/epoch_70_step_4188/dec18_PnPStoveToCounter_mg_fixed_224_na_na_16')
overlay_args = parser.parse_args()

all_dirs = sorted(glob.glob(f'{overlay_args.base_dataset_path}/*'))
job_dirs = [d for d in all_dirs if os.path.isdir(d) and os.path.exists(os.path.join(d, 'eval_log.json'))]
total_count=0
sf_data = []
# Process all job directories
for job_dir in tqdm(job_dirs):
    job_name = Path(job_dir).name
    # Load metadata for this job
    metadata_path = Path(job_dir) / 'eval_log.json'
    if not metadata_path.exists():
        logger.warning(f"Metadata file not found for {job_name}, skipping...")
        continue

    all_metadata = json.load(open(metadata_path, 'r'))
    for key,value in all_metadata.items():
        if key.startswith("train/sim_reward_trajectory_"):
            trajectory = ast.literal_eval(value)
            video_path = '/app/'+all_metadata[key.replace('train/sim_reward_trajectory_','train/sim_video_')]
            if 'data/outputs/dec4/2025.12.03/22.38.27_train_diffusion_unet_clip/checkpoints/epoch_70_step_4188/dec4_na_na_16' in video_path:
                video_path = video_path.replace('dec4_na_na_16','expert_acc')
            demo_id = int(key.split('train/sim_reward_trajectory_')[-1].split('_')[0])
            
            if 1 in trajectory:
                traj_index = int(trajectory.index(1)/2)
                image_path = video_path[:-4] + f'/frame_{traj_index:06d}.png'
                try:
                    image = np.array(Image.open(image_path))
                    sf_data.append({
                        'video_path': video_path,
                        "image_path": image_path,
                        'image': image,
                        'trajectory_index': traj_index,
                        's_or_f': 'success',
                        'demo_id': demo_id
                    })
                    total_count+=1
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    
            else:
                traj_index = int(len(trajectory)/2-20)
                image_path = video_path[:-4] + f'/frame_{traj_index:06d}.png'
                try:
                    image = np.array(Image.open(image_path))
                    sf_data.append({
                        'video_path': video_path,
                        "image_path": image_path,
                        'image': image,
                        'trajectory_index': traj_index,
                        's_or_f': 'fail', 
                        'demo_id': demo_id
                    })
                    total_count+=1
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")


# Save as numpy file
np.save(f'{overlay_args.base_dataset_path}/succeess_failure_images_and_labels.npy', sf_data)

# Save as JSON (excluding the image array for serializability if needed, or just keeping the metadata)
# We need to remove 'image' key or handle it for JSON serialization if we want to keep the JSON dump.
# The user said "actually just store a numpy file", but keeping JSON might be useful.
# However, numpy arrays are not JSON serializable.
# I will create a copy for JSON without the image data.
sf_data_json = [{k: v for k, v in item.items() if k != 'image'} for item in sf_data]

with open(f'{overlay_args.base_dataset_path}/succeess_failure_images_and_labels.json', 'w') as f:  
    json.dump(sf_data_json, f, indent=4)
