"""
export LD_LIBRARY_PATH=:/home/sruthi/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=osmesa 
source /proj/vondrick3/sruthi/miniconda3/bin/activate
conda activate clonejgdrobodiff
cd /proj/vondrick3/sruthi/robots/diffusion_policy
export HYDRA_FULL_ERROR=1


Usage:
Training:

accelerate launch --multi-gpu --num_machines 1 --num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 --main_process_port=8080 train.py \
    --config-dir=. \
    --config-name=image_only_classifier.yaml \
    training.seed=42 \
    dataloader.batch_size=256 \
    val_dataloader.batch_size=256 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/imageonly_${now:%m.%d.%H.%M.%S}_big_classifier_shuffle_higherlr' \
    task.dataset_path="[\"/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-05-04-22-14-34_and_2024-05-07-07-40-21/demo_gentex_im128_randcams_new_images_train_no_kbpckt.hdf5\", \"/proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2025.02.15/imageonly_11.32.40_usegroupnorm/checkpoints/epoch=1000-val_loss=0.071/PnPSinkToCounter_mg_train_no_kbpctk_21819416/datafile.hdf5\",  \"/proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2025.02.15/imageonly_11.32.40_usegroupnorm/checkpoints/epoch=1000-val_loss=0.071/PnPSinkToCounter_mg_val_kbpctk_firsthalf_21984511/datafile.hdf5\"]" \
    training.checkpoint_every=1 \
    training.val_every=1 \
    dataloader.shuffle=True \
    val_dataloader.shuffle=True \
    optimizer.lr=1e-3 \
    training.lr_warmup_steps=250
accelerate launch --multi-gpu --num_machines 1 --num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 --main_process_port=8082 train.py \

CUDA_VISIBLE_DEVICES=0 python train.py \
accelerate launch --multi-gpu --num_machines 1 --num_processes=7 --gpu_ids=1,2,3,4,5,6,7 --main_process_port=8082 train.py \
    --config-dir=. \
    --config-name=image_only_classifier.yaml \
    training.seed=42 \
    dataloader.batch_size=256 \
    val_dataloader.batch_size=256 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/imageonly_${now:%m.%d.%H.%M.%S}_test' \
    task.dataset_path="[\"/proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2025.02.15/imageonly_11.32.40_usegroupnorm/checkpoints/epoch=1100-val_loss=0.037/PnPSinkToCounter_mg_val_kbpctk_firsthalf_322101546_midway_rollout_140/datafile.hdf5\", \"/proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2025.02.15/imageonly_11.32.40_usegroupnorm/checkpoints/epoch=1100-val_loss=0.037/PnPSinkToCounter_mg_train_no_kbpctk_322143819_midway_rollout_140/datafile.hdf5\", \"/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-05-04-22-14-34_and_2024-05-07-07-40-21/demo_gentex_im128_randcams_new_images_train_no_kbpckt.hdf5\"]" \
    training.checkpoint_every=1 \
    training.val_every=1 \
    dataloader.shuffle=True \
    val_dataloader.shuffle=True \
    optimizer.lr=5e-4 \
    training.lr_warmup_steps=250 \
    +policy.train_noisy_trajs=False

accelerate launch --multi-gpu --num_machines 1 --num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 --main_process_port=8091 train.py \
    --config-dir=. \
    --config-name=image_only_classifier.yaml \
    training.seed=42 \
    dataloader.batch_size=1024 \
    val_dataloader.batch_size=1024 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/imageonly_${now:%m.%d.%H.%M.%S}_big_classifier_preshuffled' \
    task.dataset_path="[\"/proj/vondrick3/sruthi/robots/diffusion_policy/data/big_classifier_data_combined_shuffled.hdf5\"]"
    training.checkpoint_every=1 \
    training.val_every=1 

accelerate launch --multi-gpu --num_machines 1 --num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 --main_process_port=8086 train.py \
    --config-dir=. \
    --config-name=image_only_classifier.yaml \
    training.seed=42 \
    dataloader.batch_size=256 \
    val_dataloader.batch_size=256 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/imageonly_${now:%m.%d.%H.%M.%S}_classifier_preshuffled' \
    task.dataset_path="[\"/proj/vondrick3/sruthi/robots/diffusion_policy/data/small_classifier_data_combined_shuffled.hdf5\"]" \
    training.checkpoint_every=10 \
    training.val_every=10 
"""