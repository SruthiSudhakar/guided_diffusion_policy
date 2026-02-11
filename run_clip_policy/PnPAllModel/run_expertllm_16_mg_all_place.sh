#!/usr/bin/env bash
set -euo pipefail

declare -A MAX_STEPS_MAP=(
  ["PnPSinkToCounter_mg_val_kbpckt_firsthalf"]=0.41
  ["PnPCounterToCab_mg_fixed_224"]=0.41
  ["PnPCabToCounter_mg_fixed_224"]=0.41
  # ["PnPMicrowaveToCounter_mg_fixed_224"]=0.41
  ["PnPCounterToMicrowave_mg_fixed_224"]=0.41
  ["PnPCounterToSink_mg_fixed_224"]=0.41
  ["PnPCoffeeServeMug_mg_fixed_224"]=0.5
  ["PnPCounterToStove_mg_fixed_224"]=0.41
  # ["PnPStoveToCounter_mg_fixed_224"]=0.41
)
dirs=("${!MAX_STEPS_MAP[@]}")

CHECKPOINT="data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897.ckpt"
LLM_PATH="data/checkpoints/llm_checkpoints/dp_llm_across_sf/PnPAll/checkpoint-9500"

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

    if [ "$dir" == "PnPSinkToCounter_mg_val_kbpckt_firsthalf" ]; then
        prefix_dir="feb7_expertllm_mg_place_PnPSinkToCounter_mg_fixed_224"
    else
        prefix_dir="feb7_expertllm_mg_place_$dir"
    fi

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

# python final_eval_clip_policy.py \
#   --checkpoint data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897.ckpt \
#   --llm_path data/checkpoints/llm_checkpoints/dp_llm_across_sf/PnPAll/checkpoint-24500 \
#   --device cuda:4 \
#   --llm_gpu 5 \
#   --change_test_textures \
#   --list_dataset_path "PnPMicrowaveToCounter_mg_fixed_224" \
#   --start_rollout_from_state "0.41" \
#   --choose_sample \
#   --num_samples 5 \
#   --additional_steps 1 \
#   --n_envs 2 \
#   --n_test 1 \
#   --n_train 1 \
#   --prefix_dir test

