#!/bin/bash
# =============================================================================
# Run pi0 evaluation on RoboCasa tasks
#
# Prerequisites: start the pi0 server first with run_pi0_server.sh
#
# Usage:
#   # Evaluate on a specific task
#   bash run_pi0_eval.sh --dataset_key PnPCounterToCab_mg_fixed_224
#
#   # With custom prompt and settings
#   bash run_pi0_eval.sh \
#     --dataset_key PnPSinkToCounter_mg_fixed_textures \
#     --prompt "pick up the object from the sink and place it on the counter" \
#     --num_demos 10 \
#     --replan_steps 5 \
#     --change_textures
#
#   # Start from 40% into the demo (like your diffusion policy evals)
#   bash run_pi0_eval.sh \
#     --dataset_key PnPCounterToCab_mg_fixed_224 \
#     --start_from_state 0.4
#
# Available dataset keys (from data/dgx_data_registery.py):
#   PnPSinkToCounter_mg_fixed_textures
#   PnPCounterToSink_mg_fixed_224
#   PnPCoffeeServeMug_mg_fixed_224
#   PnPStoveToCounter_mg_fixed_224
#   PnPCounterToStove_mg_fixed_224
#   PnPCabToCounter_mg_fixed_224
#   PnPCounterToCab_mg_fixed_224
#   PnPMicrowaveToCounter_mg_fixed_224
#   PnPCounterToMicrowave_mg_fixed_224
#   ... (and expert variants, see data/dgx_data_registery.py)
# =============================================================================

set -e

export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

cd /app

python eval_pi0_robocasa.py "$@"
