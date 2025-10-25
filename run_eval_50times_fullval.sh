#!/bin/bash

# Run the evaluation command 50 times
for i in {1..50}
do
    echo "===== Running iteration $i of 50 ====="
    python openvla_eval_new.py --checkpoint data/checkpoints/dp_model/epoch=1100-val_loss=0.037.ckpt \
        --llm_path data/checkpoints/llm_checkpoints/3view_sidebyside/checkpoint-3600 \
        --device cuda:4 \
        --robocasa \
        --change_test_textures \
        --list_dataset_path PnPSinkToCounter_mg_val_kbpckt_firsthalf \
        --n_envs 254 \
        --n_train 253 \
        --n_test 1 \
        --start_rollout_from_state 140 \
        --max_steps 200 \
        --prefix_dir oct14_max_ss140_allval

    echo "===== Completed iteration $i of 50 ====="
    echo ""
done

echo "All 50 iterations completed!"
