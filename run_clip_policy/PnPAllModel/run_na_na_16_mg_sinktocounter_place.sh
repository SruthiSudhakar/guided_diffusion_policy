for i in {1..100}
do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching iteration $i on GPU ${1}"
    python final_eval_clip_policy.py \
        --checkpoint data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897.ckpt \
        --llm_path none \
        --device cuda:$1 \
        --change_test_textures \
        --list_dataset_path "PnPSinkToCounter_mg_val_kbpckt_firsthalf" \
        --n_envs 52 \
        --n_train 51 \
        --n_test 1 \
        --start_rollout_from_state 140 \
        --max_steps 200 \
        --prefix_dir na_na_16_mg_place_PnPSinkToCounter
done