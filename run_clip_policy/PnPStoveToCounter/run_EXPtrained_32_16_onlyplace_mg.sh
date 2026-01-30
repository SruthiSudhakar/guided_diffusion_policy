#!/bin/bash

# Run the evaluation command 50 times
for i in {1..100}
do
    echo "===== Running iteration $i of 50 ====="
    python final_eval_clip_policy.py \
        --checkpoint data/outputs/dec4/2025.12.03/22.38.27_train_diffusion_unet_clip/checkpoints/epoch_70_step_4188.ckpt \
        --llm_path data/checkpoints/llm_checkpoints/dp_llm_across_sf/PnPStoveToCounter/expert_trained_llm/checkpoint-10000 \
        --device cuda:$1 \
        --llm_gpu $2 \
        --change_test_textures \
        --list_dataset_path PnPStoveToCounter_mg_fixed_224 \
        --n_envs 49 \
        --n_train 48 \
        --n_test 1 \
        --choose_sample \
        --num_samples 5 \
        --additional_steps 1 \
        --start_rollout_from_state 140 \
        --max_steps 200 \
        --prefix_dir EXPtrained_32_16_mg_onlyplace
    echo "===== Completed iteration $i of 50 ====="
    echo ""
done

echo "All 50 iterations completed!"
