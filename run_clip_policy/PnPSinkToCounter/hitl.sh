for i in {1..100}
do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching iteration $i on GPU ${1}"
    python final_eval_hitl.py --checkpoint data/checkpoints/dp_model/epoch=1100-val_loss=0.037.ckpt \
        --llm_path data/checkpoints/llm_checkpoints/dp_llm_across_sf/checkpoint-9500 \
        --device cuda:0 \
        --change_test_textures \
        --list_dataset_path PnPSinkToCounter_mg_val_kbpckt_firsthalf \
        --n_envs 2 \
        --specific_train_exs 62 \
        --n_test 1 \
        --choose_sample \
        --num_samples 5 \
        --additional_steps 3 \
        --num_actions_to_execute 16 \
        --start_rollout_from_state 140 \
        --max_steps 200 \
        --prefix_dir jan9_hitl
done