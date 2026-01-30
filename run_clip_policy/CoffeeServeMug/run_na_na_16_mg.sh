for i in {1..100}
do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching iteration $i on GPU ${1}"
    python final_eval_clip_policy.py \
        --checkpoint data/outputs/dec4/2025.12.04/00.06.44_clip_justCoffeeServeMug/checkpoints/epoch_30_step_2231.ckpt \
        --llm_path none \
        --device cuda:$1 \
        --change_test_textures \
        --list_dataset_path PnPCoffeeServeMug_mg_fixed_224 \
        --prefix_dir na_na_16_mg
done    