#!/usr/bin/env bash
set -euo pipefail

declare -A MAX_STEPS_MAP=(
  ["PnPCounterToCab_expert_fixed_224"]=0.41
  ["PnPCabToCounter_expert_fixed_224"]=0.41
  ["PnPMicrowaveToCounter_expert_fixed_224"]=0.41
  ["PnPCounterToMicrowave_expert_fixed_224"]=0.41
  ["PnPCounterToSink_expert_fixed_224"]=0.41
  ["PnPSinkToCounter_expert_fixed_224"]=0.41
  ["PnPCounterToStove_expert_fixed_224"]=0.41
  ["PnPStoveToCounter_expert_fixed_224"]=0.41
  ["PnPCoffeeServeMug_expert_fixed_224"]=0.5
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
    max_steps="${MAX_STEPS_MAP[$dir]}"
    echo "max_steps: $max_steps"

    prefix_dir="feb7_expertllm_expert_place_$dir"

    python final_eval_clip_policy.py \
      --checkpoint "${CHECKPOINT}" \
      --llm_path "${LLM_PATH}" \
      --device "${DEVICE}" \
      --llm_gpu "${LLM_GPU}" \
      --change_test_textures \
      --list_dataset_path "$dir" \
      --start_rollout_from_state "$max_steps" \
      --choose_sample \
      --num_samples 5 \
      --additional_steps 1 \
      --prefix_dir "$prefix_dir"
  done
done
