for i in {1..100}
do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching iteration $i on GPU ${1}"
    python final_eval_clip_policy.py \
        --checkpoint data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897.ckpt \
        --llm_path random \
        --device cuda:$1 \
        --change_test_textures \
        --list_dataset_path "PnPStoveToCounter_mg_fixed_224" \
        --start_rollout_from_state 140 \
        --max_steps 200 \
        --choose_sample \
        --num_samples 5 \
        --additional_steps 1 \
        --prefix_dir randomllm_mg_place_PnPStoveToCounter
done