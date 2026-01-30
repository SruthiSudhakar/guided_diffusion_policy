#!/usr/bin/env bash
set -e

declare -A MAX_STEPS_MAP=(
  ["PnPCounterToSink"]=0.41
  ["PnPCoffeeServeMug"]=0.5
  ["PnPCounterToStove"]=0.41
  ["PnPCabToCounter"]=0.41
  ["PnPCounterToCab"]=0.41
  ["PnPMicrowaveToCounter"]=0.41
  ["PnPCounterToMicrowave"]=0.41
  ["PnPSinkToCounter"]=0.41
)

dirs=("${!MAX_STEPS_MAP[@]}")

for dir in "${dirs[@]}"; do
    rm -rf data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/na_na_16_expert_fulltask_$dir/overlay_images*/dataset_cache_val_*.pkl
done
