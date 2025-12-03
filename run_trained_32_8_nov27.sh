#!/bin/bash

# Run the evaluation command 50 times
for i in {1..100}
do
    echo "===== Running iteration $i of 50 ====="
    python final_eval.py --checkpoint data/checkpoints/dp_model/epoch=1100-val_loss=0.037.ckpt \
        --llm_path data/checkpoints/llm_checkpoints/dp_llm_across_sf/checkpoint-9500 \
        --device cuda:${1} \
        --change_test_textures \
        --list_dataset_path PnPSinkToCounter_mg_val_kbpckt_firsthalf \
        --n_envs 42 \
        --specific_train_exs 0,2,6,7,28,34,39,42,46,61,62,74,77,87,90,99,100,110,125,136,146,151,153,154,156,164,169,175,176,182,183,206,207,214,218,229,232,240,241,245,247 \
        --n_test 1 \
        --choose_sample \
        --num_samples 5 \
        --additional_steps 3 \
        --start_rollout_from_state 140 \
        --max_steps 200 \
        --prefix_dir nov27_trained_32_8
    echo "===== Completed iteration $i of 50 ====="
    echo ""
done

echo "All 50 iterations completed!"
