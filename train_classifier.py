"""
export LD_LIBRARY_PATH=:/home/sruthi/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=osmesa 
conda activate robodiff

Usage:
Training:

accelerate launch --multi_gpu --num_machines 1 --num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 --main_process_port=8086 

python train.py \
    --config-dir=. \
    --config-name=train_classifier_config.yaml \
    training.seed=42 \
    training.device=2 \
    dataloader.batch_size=4001 \
    val_dataloader.batch_size=91 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_15.00.33_classifier' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/diffusion_policy/data/curateddata/hammer2/data.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/diffusion_policy/data/curateddata/hammer2/data.hdf5 \
    +task.env_runner.object=redcube
"""