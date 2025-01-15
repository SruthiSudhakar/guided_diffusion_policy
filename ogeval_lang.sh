#!/bin/bash

# List of checkpoints
CHECKPOINTS=(
    '/proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.12.18/19.14.18_train_diffusion_unet_hybrid_robocasalang_PnPCounterToX/checkpoints/epoch=4500-val_loss=0.183.ckpt'
    '/proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.12.18/19.14.18_train_diffusion_unet_hybrid_robocasalang_PnPCounterToX/checkpoints/epoch=4500-val_loss=0.183.ckpt'
)
DATASETS=(
    'PnPStoveToCounter'
    'PnPCabToCounter'
)
# Number of devices (0-7)
NUM_DEVICES=8

# Function to run commands
run_command() {
    local checkpoint=$1
    local dataset=$2
    local device=$3

    echo "Running command for checkpoint: $checkpoint on device: cuda:$device"
    python ogeval.py --checkpoint "$checkpoint" \
                     --list_dataset_path "$dataset" \
                     --device "cuda:$device" \
                     --robocasa
}

# Iterate over checkpoints and assign devices
for i in "${!CHECKPOINTS[@]}"; do
    DEVICE=$((i % NUM_DEVICES))
    run_command "${CHECKPOINTS[$i]}" "${DATASETS[$i]}" "$DEVICE"
    sleep 10  # Wait for 10 seconds before the next iteration
done

# Wait for all background processes to complete
wait

echo "All commands completed."
