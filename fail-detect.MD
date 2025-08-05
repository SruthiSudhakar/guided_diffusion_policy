1. Get {O_t_i,A_t_i} for each point i in training dataset
CUDA_VISIBLE_DEVICES=1 python get_demo_obs_encodings.py --config-name=get_obs_embeds_robocasa.yaml 

2. Train score_network Dm(A,O;theta)
python train_fail_detect.py --config-name=train_fd_score_network_robocasa

3. Score some rollouts 
python score_rollouts.py \
    --checkpoint data/checkpoints/dp_model/epoch=1100-val_loss=0.037.ckpt \
    --save_score_network_path data/checkpoints/dp_model/human_obs/fd_score_network.ckpt \
    --device cuda:7 \
    --robocasa \
    --change_test_textures \
    --list_dataset_path PnPSinkToCounter_mg_val_kbpckt_firsthalf \
    --n_envs 20 \
    --n_train 19 \
    --n_test 1 \
    --start_rollout_from_state 0 \
    --max_steps 400 \
    --prefix_dir score_rollouts/mg_val_kbpckt_firsthalf

4. Construct CP band from scores
python get_band_from_scores.py --dataset_path data/checkpoints/dp_model/epoch=1100-val_loss=0.037/score_rollouts/human/PnPSinkToCounter_Human_fixed_textures_85201720_/fd_scores.pkl