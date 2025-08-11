"""
export LD_LIBRARY_PATH=:/home/sruthi/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=osmesa 
source /proj/vondrick3/sruthi/miniconda3/bin/activate
conda activate clonejgdrobodiff 
cd /proj/vondrick3/sruthi/robots/diffusion_policy
export HYDRA_FULL_ERROR=1

using the clonejgdrobodiff bc it has robocasa and the updated version of robosuite. with changes added ontop of that to be compatible with dp
Usage:f
Training:

PnPSinkToCounter

accelerate launch --num_machines 1 --num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 --main_process_port=8080 train.py \
    --config-dir=. \
    --config-name=image_eef_policy_robocasa.yaml \
    training.seed=42 \
    dataloader.batch_size=64 \
    val_dataloader.batch_size=64 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/imageeef_${now:%H.%M.%S}' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2/demo_gentex_im128_randcams_new_images_train.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2/demo_gentex_im128_randcams_new_images_train.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2/demo_gentex_im128_randcams_new_images_train.hdf5 
    
    \
    
    training.debug=true \
    task.env_runner.max_steps=10

accelerate launch --multi-gpu --num_machines 1 --num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 --main_process_port=8086 train.py \
    --config-dir=. \
    --config-name=image_only_language_policy_robocasa.yaml \
    training.seed=42 \
    dataloader.batch_size=64 \
    val_dataloader.batch_size=64 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/imageonly_${now:%H.%M.%S}_PnPSinkToCounter_train' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2/demo_gentex_im128_randcams_new_images_train.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2/demo_gentex_im128_randcams_new_images_train.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2/demo_gentex_im128_randcams_new_images_train.hdf5 

"""

CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 python train.py --config-dir=. --config-name=train_robocasa_base_dp_clip_policy.yaml training.seed=42 task.name=‘CloseDrawer’
