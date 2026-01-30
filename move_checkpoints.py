import os
import shutil
import json

source_base = "data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_mg_place_StoveToCounter"
dest_base = "data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_mg_place_PnPStoveToCounter"

# Ensure source exists
if not os.path.exists(source_base):
    print(f"Source directory {source_base} does not exist.")
    exit(1)

# Ensure dest exists
if not os.path.exists(dest_base):
    print(f"Destination directory {dest_base} does not exist. Creating it.")
    os.makedirs(dest_base, exist_ok=True)

# List subdirectories in source
subdirs = [d for d in os.listdir(source_base) if os.path.isdir(os.path.join(source_base, d))]

print(f"Found {len(subdirs)} subdirectories to move.")

for subdir in subdirs:
    source_path = os.path.join(source_base, subdir)
    dest_path = os.path.join(dest_base, subdir)

    print(f"Moving {source_path} to {dest_path}")
    
    # Move directory
    try:
        shutil.move(source_path, dest_path)
    except Exception as e:
        print(f"Error moving {subdir}: {e}")
        continue

    # Path to eval_log.json in the new location
    eval_log_path = os.path.join(dest_path, "eval_log.json")

    if os.path.exists(eval_log_path):
        print(f"Updating {eval_log_path}")
        try:
            with open(eval_log_path, 'r') as f:
                content = f.read()
            
            new_content = content.replace("na_na_16_mg_fulltask", "na_na_16_mg_place_PnPStoveToCounter")
            
            with open(eval_log_path, 'w') as f:
                f.write(new_content)
        except Exception as e:
            print(f"Error updating eval_log.json in {subdir}: {e}")
    else:
        print(f"eval_log.json not found in {subdir}")

print("Done.")
