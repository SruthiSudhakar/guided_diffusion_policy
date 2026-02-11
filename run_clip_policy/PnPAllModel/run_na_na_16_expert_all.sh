#!/usr/bin/env bash
set -e

declare -A MAX_STEPS_MAP=(
  ["PnPCounterToCab_expert_fixed_224"]=0.41
  ["PnPCabToCounter_expert_fixed_224"]=0.41
  # ["PnPMicrowaveToCounter_expert_fixed_224"]=0.41
  ["PnPCounterToMicrowave_expert_fixed_224"]=0.41
  ["PnPCounterToSink_expert_fixed_224"]=0.41
  ["PnPSinkToCounter_expert_fixed_224"]=0.41
  ["PnPCounterToStove_expert_fixed_224"]=0.41
  ["PnPStoveToCounter_expert_fixed_224"]=0.41
  ["PnPCoffeeServeMug_expert_fixed_224"]=0.5
)

dirs=("${!MAX_STEPS_MAP[@]}")

for i in {1..50}; do
    for dir in "${dirs[@]}"; do
        max_steps="${MAX_STEPS_MAP[$dir]}"

        prefix_dir="feb7_na_na_16_expert_place_$dir"

        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $dir | iter $i | max_steps=$max_steps"

        python final_eval_clip_policy.py \
            --checkpoint data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897.ckpt \
            --llm_path none \
            --device cuda:$1 \
            --change_test_textures \
            --list_dataset_path "$dir" \
            --start_rollout_from_state "$max_steps" \
            --prefix_dir "$prefix_dir"
    done
done
