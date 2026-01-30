#!/usr/bin/env bash
set -e

declare -A MAX_STEPS_MAP=(
  # ["PnPCounterToSink_mg_fixed_224"]=0.41
  # ["PnPCoffeeServeMug_mg_fixed_224"]=0.5
  # ["PnPCounterToStove_mg_fixed_224"]=0.41
  # ["PnPCabToCounter_mg_fixed_224"]=0.41
  # ["PnPCounterToCab_mg_fixed_224"]=0.41
  # ["PnPMicrowaveToCounter_mg_fixed_224"]=0.41
  # ["PnPCounterToMicrowave_mg_fixed_224"]=0.41
  ["PnPSinkToCounter_mg_val_kbpckt_firsthalf"]=0.41
)

dirs=("${!MAX_STEPS_MAP[@]}")

for i in {1..30}; do
    for dir in "${dirs[@]}"; do
        max_steps="${MAX_STEPS_MAP[$dir]}"

        if [ "$dir" == "PnPSinkToCounter_mg_val_kbpckt_firsthalf" ]; then
            prefix_dir="na_na_16_mg_place_PnPSinkToCounter_mg_fixed_224"
        else
            prefix_dir="na_na_16_mg_place_$dir"
        fi

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
