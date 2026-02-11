#!/usr/bin/env bash
set -e

declare -A MAX_STEPS_MAP=(
  ["PnPCabToCounter_mg_fixed_224"]=0.41
  ["PnPCounterToCab_mg_fixed_224"]=0.41
  ["PnPCounterToMicrowave_mg_fixed_224"]=0.41
  ["PnPMicrowaveToCounter_mg_fixed_224"]=0.41
  ["PnPCoffeeServeMug_mg_fixed_224"]=0.5
  ["PnPCounterToSink_mg_fixed_224"]=0.41
  ["PnPCounterToStove_mg_fixed_224"]=0.41
  ["PnPStoveToCounter_mg_fixed_224"]=0.41
  ["PnPSinkToCounter_mg_fixed_224"]=0.41

  # ["PnPCabToCounter"]=0.41
  # ["PnPCounterToCab"]=0.41
  # ["PnPCounterToMicrowave"]=0.41
  # ["PnPMicrowaveToCounter"]=0.41
  # ["PnPCoffeeServeMug"]=0.5
  # ["PnPCounterToSink"]=0.41
  # ["PnPCounterToStove"]=0.41
  # ["PnPStoveToCounter"]=0.41
  # ["PnPSinkToCounter"]=0.41

  # ["PnPCabToCounter_expert_fixed_224"]=0.41
  # ["PnPCounterToCab_expert_fixed_224"]=0.41
  # ["PnPCounterToMicrowave_expert_fixed_224"]=0.41
  # # ["PnPMicrowaveToCounter_expert_fixed_224"]=0.41
  # ["PnPCoffeeServeMug_expert_fixed_224"]=0.5
  # ["PnPCounterToSink_expert_fixed_224"]=0.41
  # ["PnPCounterToStove_expert_fixed_224"]=0.41
  # ["PnPStoveToCounter_expert_fixed_224"]=0.41
  # ["PnPSinkToCounter_expert_fixed_224"]=0.41

)

dirs=("${!MAX_STEPS_MAP[@]}")

for dir in "${dirs[@]}"; do
    python3 compute_average_success_rates_perdemo.py --base_dir data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/feb7_expertllm_mg_place_$dir
done
