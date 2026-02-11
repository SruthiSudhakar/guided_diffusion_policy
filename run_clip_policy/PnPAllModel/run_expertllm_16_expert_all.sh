#!/usr/bin/env bash
set -euo pipefail

declare -A MAX_STEPS_MAP=(
  ["PnPCounterToCab_expert_fixed_224"]=0
  ["PnPCabToCounter_expert_fixed_224"]=0
  # ["PnPMicrowaveToCounter_expert_fixed_224"]=0
  ["PnPCounterToMicrowave_expert_fixed_224"]=0
  ["PnPCounterToSink_expert_fixed_224"]=0
  ["PnPSinkToCounter_expert_fixed_224"]=0
  ["PnPCounterToStove_expert_fixed_224"]=0
  ["PnPStoveToCounter_expert_fixed_224"]=0
  ["PnPCoffeeServeMug_expert_fixed_224"]=0
)
dirs=("${!MAX_STEPS_MAP[@]}")

CHECKPOINT="data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897.ckpt"
LLM_PATH="data/checkpoints/llm_checkpoints/dp_llm_across_sf/PnPAllExpertWeighted/checkpoint-15000"

DEVICE="cuda:${1:-0}"
LLM_GPU="${2:-0}"

for iter in $(seq 1 27); do
  echo "========================================"
  echo " Iteration ${iter}/27"
  echo "========================================"
  mapfile -t dirs < <(printf '%s\n' "${!MAX_STEPS_MAP[@]}" | shuf)

  for dir in "${dirs[@]}"; do
    echo "[iter=${iter}] Running ${dir}"

    prefix_dir="feb7_expertllm_expert_fulltask_$dir"

    python final_eval_clip_policy.py \
      --checkpoint "${CHECKPOINT}" \
      --llm_path "${LLM_PATH}" \
      --device "${DEVICE}" \
      --llm_gpu "${LLM_GPU}" \
      --change_test_textures \
      --list_dataset_path "$dir" \
      --choose_sample \
      --num_samples 5 \
      --additional_steps 1 \
      --prefix_dir "$prefix_dir"
  done
done
