import os
import shutil
import json
import sys

source = "data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask"
dest = "data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask"+sys.argv[1]

# Ensure destination exists
os.makedirs(dest, exist_ok=True)

# Get list of directories starting with PnP
dirs = sorted([d for d in os.listdir(source) if os.path.isdir(os.path.join(source, d)) and d.startswith(sys.argv[1])])

print(f"Found {len(dirs)} PnP directories in source.")
print(f"Moving {len(dirs)} directories...")

for d in dirs:
    src_path = os.path.join(source, d)
    dst_path = os.path.join(dest, d)
    
    try:
        # Move directory
        shutil.move(src_path, dst_path)
        print(f"Moved: {d}")
        
        # Update eval_log.json
        json_path = os.path.join(dst_path, "eval_log.json")
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    content = f.read()
                
                new_content = content.replace("fulltask", "fulltask_"+sys.argv[1])
                
                with open(json_path, 'w') as f:
                    f.write(new_content)
                print(f"  Updated eval_log.json in {d}")
            except Exception as e:
                print(f"  Error updating json in {d}: {e}")
        else:
            print(f"  eval_log.json not found in {d}")
            
    except Exception as e:
        print(f"Error processing {d}: {e}")

print("Done.")
