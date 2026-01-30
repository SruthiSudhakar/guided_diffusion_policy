for i in {1..100}
do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching iteration $i on GPU ${1}"
    python final_eval_clip_policy.py \
        --checkpoint data/outputs/dec4/2025.12.03/22.38.27_train_diffusion_unet_clip/checkpoints/epoch_70_step_4188.ckpt \
        --llm_path none \
        --device cuda:7 \
        --change_test_textures \
        --list_dataset_path PnPStoveToCounter_mg_fixed_224 \
        --n_envs 49 \
        --n_train 48 \
        --n_test 1 \
        --prefix_dir dec18_PnPStoveToCounter_mg_fixed_224_na_na_16
done    