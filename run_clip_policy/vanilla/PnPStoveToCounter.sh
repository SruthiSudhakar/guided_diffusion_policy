for i in {1..100}
do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching iteration $i on GPU ${1}"
    python final_eval_clip_policy.py --checkpoint data/outputs/dec4/2025.12.03/22.38.27_train_diffusion_unet_clip/checkpoints/epoch_70_step_4188.ckpt \
        --llm_path data/checkpoints/llm_checkpoints/3view_sidebyside/checkpoint-3600 \
        --device cuda:${1} \
        --change_test_textures \
        --list_dataset_path PnPStoveToCounter_expert_fixed_224 \
        --n_envs 53 \
        --n_train 52 \
        --n_test 1 \
        --prefix_dir dec4_na_na_16
done    