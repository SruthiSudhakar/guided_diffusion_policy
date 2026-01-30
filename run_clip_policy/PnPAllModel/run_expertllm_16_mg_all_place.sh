#!/usr/bin/env bash
set -e

dirs=(
  "PnPCabToCounter_mg_fixed_224"
  "PnPCounterToCab_mg_fixed_224"
  "PnPCoffeeServeMug_mg_fixed_224"
  "PnPCounterToSink_mg_fixed_224"
  "PnPSinkToCounter_mg_val_kbpckt_firsthalf"
  "PnPMicrowaveToCounter_mg_fixed_224"
  "PnPCounterToMicrowave_mg_fixed_224"
  "PnPCounterToStove_mg_fixed_224"
  # "PnPStoveToCounter_mg_fixed_224"
)

for dir in "${dirs[@]}"; do
  python final_eval_clip_policy.py \
      --checkpoint data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897.ckpt \
      --llm_path data/checkpoints/llm_checkpoints/dp_llm_across_sf/PnPAll/checkpoint-8000 \
      --device cuda:$1 \
      --llm_gpu $2 \
      --change_test_textures \
      --list_dataset_path "${dir}" \
      --start_rollout_from_state 140 \
      --max_steps 200 \
      --choose_sample \
      --num_samples 5 \
      --additional_steps 1 \
      --prefix_dir jan29_expertllm_mg_place_${dir}
done