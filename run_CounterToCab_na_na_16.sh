for i in {1..100}
do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching iteration $i on GPU ${1}"
    python final_eval_clip_policy.py \
        --checkpoint data/outputs/dec4/2025.12.04/15.36.39_clip_justPnPCounterToCab/checkpoints/epoch_30_step_1735.ckpt \
        --llm_path none \
        --device cuda:0 \
        --change_test_textures \
        --list_dataset_path PnPCounterToCab_mg_fixed_224 \
        --n_envs 49 \
        --n_train 48 \
        --n_test 1 \
        --prefix_dir dec18_PnPCounterToCab_mg_fixed_224_na_na_16
done    