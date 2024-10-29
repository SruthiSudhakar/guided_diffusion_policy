"""
export LD_LIBRARY_PATH=:/home/sruthi/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=osmesa 
source /proj/vondrick3/sruthi/miniconda3/bin/activate
conda activate jgdrobodiff
cd /proj/vondrick3/sruthi/robots/diffusion_policy
export HYDRA_FULL_ERROR=1


Usage:
Training:

accelerate launch --num_machines 1 --num_processes=1 --gpu_ids=4 --main_process_port=8088 train.py \
    --config-dir=. \
    --config-name=image_square_ph_classifier.yaml \
    training.seed=42 \
    training.device=3 \
    dataloader.batch_size=1024 \
    val_dataloader.batch_size=1024 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_classifier_hammer2seed6000' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/diffusion_policy/data/curateddata/hammer2_seed6000/data_all.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/diffusion_policy/data/curateddata/hammer2_seed6000/data_all.hdf5 \
    training.checkpoint_every=1 \
    optimizer.lr=0.00001 \
    +task.env_runner.object=hammer


accelerate launch --multi_gpu --num_machines 1 --num_processes=1 --gpu_ids=1 --main_process_port=8080 train.py \
    --config-dir=. \
    --config-name=image_square_ph_classifier.yaml \
    training.seed=42 \
    training.device=2 \
    dataloader.batch_size=2048 \
    val_dataloader.batch_size=2048 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_classifier_hammer_needle_greencube_mugbeige' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/diffusion_policy/data/curateddata/hammer_needle_greencube_mugbeige/combined.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/diffusion_policy/data/curateddata/hammer_needle_greencube_mugbeige/combined.hdf5 \
    training.checkpoint_every=10 \
    +task.env_runner.object=hammer

"""