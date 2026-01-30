#!/usr/bin/env bash
set -e

declare -A MAX_STEPS_MAP=(
  ["PnPCounterToSink_mg_fixed_224"]=0.41
  ["PnPCoffeeServeMug_mg_fixed_224"]=0.5
  ["PnPCounterToStove_mg_fixed_224"]=0.41
  ["PnPCabToCounter_mg_fixed_224"]=0.41
  ["PnPCounterToCab_mg_fixed_224"]=0.41
  ["PnPMicrowaveToCounter_mg_fixed_224"]=0.41
  ["PnPCounterToMicrowave_mg_fixed_224"]=0.41
#   ["PnPSinkToCounter_mg_val_kbpckt_firsthalf"]=0.41
)

dirs=("${!MAX_STEPS_MAP[@]}")

for dir in "${dirs[@]}"; do
    python3 compute_average_success_rates.py --base_dir data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_mg_place_$dir
done
