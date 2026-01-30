#!/bin/bash

# Run the evaluation command 50 times
for i in {1..100}
do
    echo "===== Running iteration $i of 50 ====="
    python final_eval_clip_policy.py \
        --checkpoint data/outputs/dec4/2025.12.04/00.06.44_clip_justCoffeeServeMug/checkpoints/epoch_30_step_2231.ckpt \
        --llm_path data/checkpoints/llm_checkpoints/dp_llm_across_sf/CoffeeServeMug/checkpoint-15000 \
        --device cuda:$1 \
        --llm_gpu $2 \
        --change_test_textures \
        --list_dataset_path PnPCoffeeServeMug_expert_fixed_224 \
        --n_envs 53 \
        --n_train 52 \
        --n_test 1 \
        --choose_sample \
        --num_samples 5 \
        --additional_steps 1 \
        --prefix_dir EXPtrained_32_16_expert
    echo "===== Completed iteration $i of 50 ====="
    echo ""
done

echo "All 50 iterations completed!"
