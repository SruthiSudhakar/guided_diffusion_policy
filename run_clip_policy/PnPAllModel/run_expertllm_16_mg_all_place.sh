#!/usr/bin/env bash
set -euo pipefail

dirs=(
  "PnPCabToCounter_mg_fixed_224"
  "PnPCounterToCab_mg_fixed_224"
  "PnPCoffeeServeMug_mg_fixed_224"
  "PnPCounterToSink_mg_fixed_224"
  "PnPSinkToCounter_mg_val_kbpckt_firsthalf"
  "PnPMicrowaveToCounter_mg_fixed_224"
  "PnPCounterToMicrowave_mg_fixed_224"
  "PnPCounterToStove_mg_fixed_224"
  "PnPStoveToCounter_mg_fixed_224"
)

CHECKPOINT="data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897.ckpt"
LLM_PATH="data/checkpoints/llm_checkpoints/dp_llm_across_sf/PnPAll/checkpoint-8000"

DEVICE="cuda:${1:-0}"
LLM_GPU="${2:-0}"

for iter in $(seq 1 100); do
  echo "========================================"
  echo " Iteration ${iter}/100"
  echo "========================================"

  for dir in "${dirs[@]}"; do
    echo "[iter=${iter}] Running ${dir}"

    python final_eval_clip_policy.py \
      --checkpoint "${CHECKPOINT}" \
      --llm_path "${LLM_PATH}" \
      --device "${DEVICE}" \
      --llm_gpu "${LLM_GPU}" \
      --change_test_textures \
      --list_dataset_path "${dir}" \
      --start_rollout_from_state 140 \
      --max_steps 200 \
      --choose_sample \
      --num_samples 5 \
      --additional_steps 1 \
      --prefix_dir "jan30_expertllm_mg_place_${dir}"
  done
done
