for i in {1..100}
do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching iteration $i on GPU ${1}"
    python final_eval_clip_policy.py \
        --checkpoint data/outputs/dec4/2025.12.03/22.39.39_train_diffusion_unet_clip/checkpoints/epoch_30_step_2231.ckpt \
        --llm_path none \
        --device cuda:$1 \
        --change_test_textures \
        --list_dataset_path PnPCabToCounter_expert_fixed_224 \
        --specific_train_exs 5,6,7,8,9,12,13,20,44,45,48,49,51,52,5,6,7,8,9,12,13,20,44,45,48,49,51,52,5,6,7,8,9,12,13,20,44,45,48,49,51,52,5,6,7,8,9,12,13,20,44,45,48,49,51,52 \
        --n_train 56 \
        --n_test 1 \
        --prefix_dir dec4_na_na_16
done    