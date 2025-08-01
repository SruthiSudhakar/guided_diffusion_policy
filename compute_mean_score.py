import json, sys
from collections import defaultdict

# Load the JSON file
with open(sys.argv[1], 'r') as f:
    data = json.load(f)

# Dictionary to store max scores for each sample ID
sample_max_scores = defaultdict(float)
sample_min_scores = defaultdict(float)
sample_scores = defaultdict(list)
sample_keys = defaultdict(list)  # Store full keys to extract video paths

# Process all train/sim_max_reward entries
for key, value in data.items():
    if key.startswith('train/sim_max_reward_'):
        # Extract the sample ID (first number after sim_max_reward_)
        parts = key.split('_')
        if len(parts) >= 4:
            sample_id = int(parts[3])
            score = float(value)
            # Only collect first 4 samples per id
            if len(sample_scores[sample_id]) < int(sys.argv[2]):
                sample_scores[sample_id].append(score)
                if score == 0:
                    sample_keys[sample_id].append(key)  # Store the full key
                # Update max score for this sample ID
                sample_max_scores[sample_id] = max(sample_max_scores[sample_id], score)
                if sample_id not in sample_min_scores:
                    sample_min_scores[sample_id]=score
                else:
                    sample_min_scores[sample_id] = min(sample_min_scores[sample_id], score)

count=defaultdict(float)
for sample_id, scores in sample_scores.items():
    count[sample_id] = len(scores)

# Find IDs with at least 2 zero scores
ids_with_multiple_zeros = []
for sample_id, scores in sample_scores.items():
    zero_count = sum(1 for score in scores if score == 0.0)
    if count[sample_id] > zero_count >= 1:
        ids_with_multiple_zeros.append((sample_id, zero_count))

# Calculate mean of max scores
if sample_max_scores:
    max_scores_list = list(sample_max_scores.values())
    min_scores_list = list(sample_min_scores.values())
    mean_score = sum(max_scores_list) / len(max_scores_list)
    mean_min_score = sum(min_scores_list) / len(min_scores_list)
        
    # Output IDs with at least 2 zero scores
    if ids_with_multiple_zeros:
        print(f"\nIDs with at least 2 zero scores: {len(ids_with_multiple_zeros)}")
        for sample_id, zero_count in sorted(ids_with_multiple_zeros):
            # Extract video path from the first key
            if sample_keys[sample_id]:
                # Key format: train/sim_max_reward_<id>_<path>
                keys = sample_keys[sample_id]
                # Remove the prefix to get the video path
                video_path = [k.replace(f'train/sim_max_reward_', '') for k in keys]
            else:
                video_path = "Unknown"
            print(f"  ID {sample_id}: {zero_count} zero scores out of {len(sample_scores[sample_id])} total - Video: {video_path}")
    else:
        print("\nNo IDs found with at least 2 zero scores")
    print(f"Number of unique sample IDs: {len(sample_max_scores)}")
    print(f"Number of samples per ID: {count.values()}")
    print(f"Max scores per sample: {dict(sorted(sample_max_scores.items())[:10])}...")  # Show first 10
    print(f"Min scores per sample: {dict(sorted(sample_min_scores.items())[:10])}...")  # Show first 10
    print(f"Mean score (max over each sample): {mean_score:.4f}")
    print(f"Mean score (min over each sample): {mean_min_score:.4f}")

else:
    print("No train/sim_max_reward entries found")