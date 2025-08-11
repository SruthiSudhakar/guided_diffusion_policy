1. Get {O_t_i,A_t_i} for each point i in training dataset
CUDA_VISIBLE_DEVICES=1 python get_demo_obs_encodings.py --config-name=get_obs_embeds_robocasa.yaml 

2. Train score_network Dm(A,O;theta)
python train_fail_detect.py --config-name=train_fd_score_network_robocasa \
    FD_save_obs_embeds_path="data/checkpoints/dp_model/human_obs/obs_embeds.pt" \
    FD_save_score_network_path="data/checkpoints/dp_model/human_obs/fd_score_network.ckpt"

3. Score some rollouts 
python score_rollouts.py \
    --checkpoint data/checkpoints/dp_model/epoch=1100-val_loss=0.037.ckpt \
    --save_score_network_path data/checkpoints/dp_model/human_obs/fd_score_network.ckpt \
    --device cuda:7 \
    --robocasa \
    --change_test_textures \
    --list_dataset_path PnPSinkToCounter_mg_val_kbpckt_firsthalf \
    --n_envs 100 \
    --n_train 99 \
    --n_test 1 \
    --start_rollout_from_state 140 \
    --max_steps 200 \
    --prefix_dir score_rollouts/mg_val_kbpckt_firsthalf

4. Construct CP band from scores
python get_band_from_scores.py --dataset_path data/checkpoints/dp_model/epoch=1100-val_loss=0.037/score_rollouts/human/PnPSinkToCounter_Human_fixed_textures_86212858_/fd_scores.pkl

5. Rollout visualization of fail-detect
python example_cp_band_eval.py \
    --checkpoint data/checkpoints/dp_model/epoch=1100-val_loss=0.037.ckpt \
    --save_score_network_path data/checkpoints/dp_model/human_obs/fd_score_network.ckpt \
    --cp_band_path data/checkpoints/dp_model/epoch=1100-val_loss=0.037/score_rollouts/mg_val_kbpckt_firsthalf/PnPSinkToCounter_mg_val_kbpckt_firsthalf_8620047_/fd_scores.pkl \
    --device cuda:7 \
    --robocasa \
    --change_test_textures \
    --list_dataset_path PnPSinkToCounter_Human_fixed_textures \
    --n_envs 50 \
    --n_train 1 \
    --n_test 49 \
    --start_rollout_from_state 140 \
    --max_steps 200 \
    --prefix_dir detect_failures/PnPSinkToCounter_Human_fixed_textures
