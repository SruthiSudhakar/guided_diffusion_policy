python ogeval.py --checkpoint /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2025.02.15/imageonly_11.32.40_usegroupnorm/checkpoints/epoch=1100-val_loss=0.037.ckpt \
                --device cuda:0 \
                --robocasa \
                --change_test_textures \
                --list_dataset_path PnPSinkToCounter_mg_val_kbpctk_firsthalf \
                --n_envs 10 \
                --n_train 9 \
                --n_test 1 \
                --add TEST