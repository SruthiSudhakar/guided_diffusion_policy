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
CV11 - PnPSinkToCounter224_train
CV11 - PNP TASKS 
accelerate launch --multi_gpu --num_machines 1 --num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 --main_process_port=8082 train.py \
    --config-dir=. \
    --config-name=image_language_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=64 \
    val_dataloader.batch_size=64 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/PnPX_train_multigpu' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/combined/PnPX/demo_gentex_im128_randcams_new_images_train.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/combined/PnPX/demo_gentex_im128_randcams_new_images_train.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/combined/PnPX/demo_gentex_im128_randcams_new_images_train.hdf5 \
    task.env_runner.max_steps=500 
CV14 - PnPSinkToCounterMg_train
accelerate launch --multi_gpu --num_machines 1 --num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 --main_process_port=8088 train.py \
    --config-dir=. \
    --config-name=image_language_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=64 \
    val_dataloader.batch_size=64 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_PnPSinkToCounter_mg_train_multigpu' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-05-04-22-14-34_and_2024-05-07-07-40-21/demo_gentex_im128_randcams_new_images_train.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-05-04-22-14-34_and_2024-05-07-07-40-21/demo_gentex_im128_randcams_new_images_train.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-05-04-22-14-34_and_2024-05-07-07-40-21/demo_gentex_im128_randcams_new_images_train.hdf5 

ON CV11
1. StoveToCounter
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config-dir=. \
    --config-name=image_language_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=256 \
    val_dataloader.batch_size=256 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_robocasalang_PnPStoveToCounter' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/2024-05-01/demo_gentex_im128_randcams_new_images.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/2024-05-01/demo_gentex_im128_randcams_new_images.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/2024-05-01/demo_gentex_im128_randcams_new_images.hdf5

ON CV15
2. SinkToCounter
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config-dir=. \
    --config-name=image_language_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=512 \
    val_dataloader.batch_size=512 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/PnPSinkToCounter224_mg_train' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-05-04-22-14-34_and_2024-05-07-07-40-21/demo_gentex_im128_randcams_new_images_train_224.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-05-04-22-14-34_and_2024-05-07-07-40-21/demo_gentex_im128_randcams_new_images_train_224.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-05-04-22-14-34_and_2024-05-07-07-40-21/demo_gentex_im128_randcams_new_images_train_224.hdf5 

CUDA_VISIBLE_DEVICES=2 python train.py \
    --config-dir=. \
    --config-name=image_language_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=64 \
    val_dataloader.batch_size=64 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_robocasalang_PnPSinkToCounter_trainsplit_imagenet' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2/demo_gentex_im128_randcams_new_images_train.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2/demo_gentex_im128_randcams_new_images_train.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2/demo_gentex_im128_randcams_new_images_train.hdf5 

3. MicrowaveToCounter
CUDA_VISIBLE_DEVICES=2 python train.py \
    --config-dir=. \
    --config-name=image_language_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=256 \
    val_dataloader.batch_size=256 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_robocasalang_PnPMicrowaveToCounter' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPMicrowaveToCounter/2024-04-26/demo_gentex_im128_randcams_new_images.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPMicrowaveToCounter/2024-04-26/demo_gentex_im128_randcams_new_images.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPMicrowaveToCounter/2024-04-26/demo_gentex_im128_randcams_new_images.hdf5 

4. CabToCounter
CUDA_VISIBLE_DEVICES=3 python train.py \
    --config-dir=. \
    --config-name=image_language_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=256 \
    val_dataloader.batch_size=256 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_robocasalang_PnPCabToCounter' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24/demo_gentex_im128_randcams_new_images.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24/demo_gentex_im128_randcams_new_images.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24/demo_gentex_im128_randcams_new_images.hdf5 \
    task.env_runner.max_steps=401

4. PnPXToCounter
CUDA_VISIBLE_DEVICES=4 python train.py \
    --config-dir=. \
    --config-name=image_language_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=512 \
    val_dataloader.batch_size=512 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_robocasalang_PnPXToCounter' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/combined/PnPXToCounter/demo_gentex_im128_randcams_new_images.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/combined/PnPXToCounter/demo_gentex_im128_randcams_new_images.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/combined/PnPXToCounter/demo_gentex_im128_randcams_new_images.hdf5 \
    task.env_runner.max_steps=401


5. PnPCounterToMicrowave
CUDA_VISIBLE_DEVICES=5 python train.py \
    --config-dir=. \
    --config-name=image_language_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=256 \
    val_dataloader.batch_size=256 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_robocasalang_PnPCounterToMicrowave' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/2024-04-27/demo_gentex_im128_randcams_new_images.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/2024-04-27/demo_gentex_im128_randcams_new_images.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/2024-04-27/demo_gentex_im128_randcams_new_images.hdf5 \
    task.env_runner.max_steps=500

5. PnPCounterToSink
CUDA_VISIBLE_DEVICES=6 python train.py \
    --config-dir=. \
    --config-name=image_language_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=256 \
    val_dataloader.batch_size=256 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_robocasalang_PnPCounterToSink' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/2024-04-25/demo_gentex_im128_randcams_new_images.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/2024-04-25/demo_gentex_im128_randcams_new_images.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/2024-04-25/demo_gentex_im128_randcams_new_images.hdf5 \
    task.env_runner.max_steps=500

6. PnPCounterToStove
CUDA_VISIBLE_DEVICES=7 python train.py \
    --config-dir=. \
    --config-name=image_language_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=256 \
    val_dataloader.batch_size=256 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_robocasalang_PnPCounterToStove' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToStove/2024-04-26/demo_gentex_im128_randcams_new_images.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToStove/2024-04-26/demo_gentex_im128_randcams_new_images.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToStove/2024-04-26/demo_gentex_im128_randcams_new_images.hdf5 \

ON CV 15
7. PnPCounterToCab
CUDA_VISIBLE_DEVICES=4 python train.py \
    --config-dir=. \
    --config-name=image_language_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=256 \
    val_dataloader.batch_size=256 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_robocasalang_PnPCounterToCab' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24/demo_gentex_im128_randcams_new_images.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24/demo_gentex_im128_randcams_new_images.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToCab/2024-04-24/demo_gentex_im128_randcams_new_images.hdf5 \
    task.env_runner.max_steps=300

On CV12
7. PnPCounterToX
CUDA_VISIBLE_DEVICES=0  python train.py \
    --config-dir=. \
    --config-name=image_language_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=512 \
    val_dataloader.batch_size=512 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_robocasalang_PnPCounterToX' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/combined/PnPCounterToX/demo_gentex_im128_randcams_new_images.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/combined/PnPCounterToX/demo_gentex_im128_randcams_new_images.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/combined/PnPCounterToX/demo_gentex_im128_randcams_new_images.hdf5 \
    task.env_runner.max_steps=500

7. PnPX
CUDA_VISIBLE_DEVICES=5 python train.py \
    --config-dir=. \
    --config-name=image_language_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=2048 \
    val_dataloader.batch_size=2048 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_robocasalang_PnPX_trainsplit_imagenet' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/combined/PnPX/demo_gentex_im128_randcams_new_images_train.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/combined/PnPX/demo_gentex_im128_randcams_new_images_train.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/combined/PnPX/demo_gentex_im128_randcams_new_images_train.hdf5 \
    task.env_runner.max_steps=500 
    

accelerate launch --multi_gpu --num_machines 1 --num_processes=4 --gpu_ids=4,5,6,7 --main_process_port=8086
CUDA_VISIBLE_DEVICES=5 python train.py \
    --config-dir=. \
    --config-name=image_language_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=256 \
    val_dataloader.batch_size=256 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_robocasalang_test' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToStove/2024-04-26/demo_gentex_im128_randcams_new_images_train.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToStove/2024-04-26/demo_gentex_im128_randcams_new_images_train.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToStove/2024-04-26/demo_gentex_im128_randcams_new_images_train.hdf5 \
    task.env_runner.max_steps=10 \
    training.checkpoint_every=6 \
    training.rollout_every=7 \
    training.sample_every=2 \
    training.val_every=3 \
    task.env_runner.n_envs=2 \
    task.env_runner.n_test=1 \
    task.env_runner.n_train=1 

If we have a dataset of n=64 items, and 8 GPUs and a bs=8, each *step* will go through the entire 
dataset one time as 8*8=64
"""