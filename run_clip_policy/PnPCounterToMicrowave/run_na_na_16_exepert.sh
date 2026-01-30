for i in {1..100}
do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching iteration $i on GPU ${1}"
    python final_eval_clip_policy.py \
        --checkpoint data/outputs/dec4/2025.12.04/05.20.02_clip_justPnPCounterToMicrowave/checkpoints/epoch_20_step_1931.ckpt \
        --llm_path none \
        --device cuda:$1 \
        --change_test_textures \
        --list_dataset_path PnPCounterToMicrowave_expert_fixed_224 \
        --prefix_dir run_na_na_16_expert
done    