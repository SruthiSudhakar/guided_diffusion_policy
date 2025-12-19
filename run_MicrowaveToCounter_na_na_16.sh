for i in {1..100}
do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching iteration $i on GPU ${1}"
    python final_eval_clip_policy.py \
        --checkpoint data/outputs/dec4/2025.12.04/05.17.32_clip_justPnPMicrowaveToCounter/checkpoints/epoch_60_step_4025.ckpt \
        --llm_path none \
        --device cuda:0 \
        --change_test_textures \
        --list_dataset_path PnPMicrowaveToCounter_mg_fixed_224 \
        --n_envs 49 \
        --n_train 48 \
        --n_test 1 \
        --prefix_dir dec18_PnPMicrowaveToCounter_mg_fixed_224_na_na_16
done    