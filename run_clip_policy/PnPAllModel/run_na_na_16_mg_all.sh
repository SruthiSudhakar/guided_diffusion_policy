#!/usr/bin/env bash
set -e

declare -A MAX_STEPS_MAP=(
  ["PnPCounterToCab_mg_fixed_224"]=0
  ["PnPCabToCounter_mg_fixed_224"]=0
  ["PnPMicrowaveToCounter_mg_fixed_224"]=0
  ["PnPCounterToMicrowave_mg_fixed_224"]=0
  ["PnPCounterToSink_mg_fixed_224"]=0
  ["PnPSinkToCounter_mg_val_kbpckt_firsthalf"]=0
  ["PnPCounterToStove_mg_fixed_224"]=0
  ["PnPStoveToCounter_mg_fixed_224"]=0
  ["PnPCoffeeServeMug_mg_fixed_224"]=0
)

dirs=("${!MAX_STEPS_MAP[@]}")

for i in {1..50}; do
    for dir in "${dirs[@]}"; do
        max_steps="${MAX_STEPS_MAP[$dir]}"

        if [ "$dir" == "PnPSinkToCounter_mg_val_kbpckt_firsthalf" ]; then
            prefix_dir="feb7_na_na_16_mg_fulltask_PnPSinkToCounter_mg_fixed_224"
        else
            prefix_dir="feb7_na_na_16_mg_fulltask_$dir"
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
