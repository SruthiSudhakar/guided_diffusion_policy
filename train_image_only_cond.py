"""
export LD_LIBRARY_PATH=:/home/sruthi/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=osmesa 
conda activate robodiff
cd /proj/vondrick3/sruthi/robots/diffusion_policy

Usage:
Training:

accelerate launch --multi_gpu --num_machines 1 --num_processes=2 --gpu_ids=0 --main_process_port=8098 train.py \
    --config-dir=. \
    --config-name=image_train_diffusion_unet_image_pretrained_workspace.yaml \
    training.seed=42 \
    dataloader.batch_size=64 \
    val_dataloader.batch_size=64 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_15.00.33_test_imageonly_policy' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/diffusion_policy/data/robomimic/datasets/lift/ph/robomimic/datasets/lift/ph/image_abs.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/diffusion_policy/data/robomimic/datasets/lift/ph/robomimic/datasets/lift/ph/image_abs.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/diffusion_policy/data/robomimic/datasets/lift/ph/robomimic/datasets/lift/ph/image_abs.hdf5 \
    task.env_runner.max_steps=100 \
    +task.env_runner.object=redcube


accelerate launch --num_machines 1 --num_processes=1 --gpu_ids=1 --main_process_port=8098 train.py \
    --config-dir=. \
    --config-name=image_square_ph_diffusion_policy_cnn.yaml \
    training.seed=42 \
    dataloader.batch_size=64 \
    val_dataloader.batch_size=64 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_15.00.33_test_hybrid_policy' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/diffusion_policy/data/robomimic/datasets/lift/ph/robomimic/datasets/lift/ph/image_abs.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/diffusion_policy/data/robomimic/datasets/lift/ph/robomimic/datasets/lift/ph/image_abs.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/diffusion_policy/data/robomimic/datasets/lift/ph/robomimic/datasets/lift/ph/image_abs.hdf5 \
    task.env_runner.max_steps=100 \
    +task.env_runner.object=redcube


    training.resume=/proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.09.03/21.23.37_train_diffusion_unet_hybrid_15.00.33_check/checkpoints/epoch_0150_0.940.ckpt \
    training.classifier_dir=/proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.10.13/15.54.25_train_classifier_classifier_needle/checkpoints/epoch_8_validacc_0.928.ckpt \
    training.guidance_scale=9.5 \

"""