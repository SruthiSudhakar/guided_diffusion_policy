"""
export LD_LIBRARY_PATH=:/home/sruthi/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=osmesa 
source /proj/vondrick3/sruthi/miniconda3/bin/activate
conda activate clonejgdrobodiff 
cd /proj/vondrick3/sruthi/robots/diffusion_policy
export HYDRA_FULL_ERROR=1

using the clonejgdrobodiff bc it has robocasa and the updated version of robosuite. with changes added ontop of that to be compatible with dp
Usage:
Training:
    
1. OpenSingleDoor
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config-dir=. \
    --config-name=image_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=256 \
    val_dataloader.batch_size=256 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_robocasa_OpenSingleDoor' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenSingleDoor/2024-04-24/demo_gentex_im128_randcams.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenSingleDoor/2024-04-24/demo_gentex_im128_randcams.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenSingleDoor/2024-04-24/demo_gentex_im128_randcams.hdf5 \
    task.env_runner.max_steps=500 

2. OpenDoubleDoor
CUDA_VISIBLE_DEVICES=1 python train.py \
    --config-dir=. \
    --config-name=image_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=256 \
    val_dataloader.batch_size=256 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_robocasa_OpenDoubleDoor' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenDoubleDoor/2024-04-26/demo_gentex_im128_randcams.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenDoubleDoor/2024-04-26/demo_gentex_im128_randcams.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenDoubleDoor/2024-04-26/demo_gentex_im128_randcams.hdf5 \
    task.env_runner.max_steps=900 

3. CloseSingleDoor
CUDA_VISIBLE_DEVICES=2 python train.py \
    --config-dir=. \
    --config-name=image_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=256 \
    val_dataloader.batch_size=256 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_robocasa_CloseSingleDoor' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseSingleDoor/2024-04-24/demo_gentex_im128_randcams.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseSingleDoor/2024-04-24/demo_gentex_im128_randcams.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseSingleDoor/2024-04-24/demo_gentex_im128_randcams.hdf5 \
    task.env_runner.max_steps=400 

4. CloseDoubleDoor
CUDA_VISIBLE_DEVICES=3 python train.py \
    --config-dir=. \
    --config-name=image_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=256 \
    val_dataloader.batch_size=256 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_robocasa_CloseDoubleDoor' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseDoubleDoor/2024-04-29/demo_gentex_im128_randcams.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseDoubleDoor/2024-04-29/demo_gentex_im128_randcams.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_doors/CloseDoubleDoor/2024-04-29/demo_gentex_im128_randcams.hdf5 \
    task.env_runner.max_steps=500 

5. TurnOffMicrowave
CUDA_VISIBLE_DEVICES=4 python train.py \
    --config-dir=. \
    --config-name=image_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=256 \
    val_dataloader.batch_size=256 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_robocasa_TurnOffMicrowave' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_microwave/TurnOffMicrowave/2024-04-25/demo_gentex_im128_randcams.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_microwave/TurnOffMicrowave/2024-04-25/demo_gentex_im128_randcams.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_microwave/TurnOffMicrowave/2024-04-25/demo_gentex_im128_randcams.hdf5 \
    task.env_runner.max_steps=350 

6. TurnOnMicrowave
CUDA_VISIBLE_DEVICES=5 python train.py \
    --config-dir=. \
    --config-name=image_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=256 \
    val_dataloader.batch_size=256 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_robocasa_TurnOnMicrowave' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_microwave/TurnOnMicrowave/2024-04-25/demo_gentex_im128_randcams.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_microwave/TurnOnMicrowave/2024-04-25/demo_gentex_im128_randcams.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_microwave/TurnOnMicrowave/2024-04-25/demo_gentex_im128_randcams.hdf5 \
    task.env_runner.max_steps=300 

7. TurnOffSinkFaucet
CUDA_VISIBLE_DEVICES=6 python train.py \
    --config-dir=. \
    --config-name=image_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=256 \
    val_dataloader.batch_size=256 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_robocasa_TurnOffSinkFaucet' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnOffSinkFaucet/2024-04-25/demo_gentex_im128_randcams.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnOffSinkFaucet/2024-04-25/demo_gentex_im128_randcams.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnOffSinkFaucet/2024-04-25/demo_gentex_im128_randcams.hdf5 \
    task.env_runner.max_steps=400 

8. TurnOnSinkFaucet
CUDA_VISIBLE_DEVICES=7 python train.py \
    --config-dir=. \
    --config-name=image_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=256 \
    val_dataloader.batch_size=256 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_robocasa_TurnOnSinkFaucet' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnOnSinkFaucet/2024-04-25/demo_gentex_im128_randcams.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnOnSinkFaucet/2024-04-25/demo_gentex_im128_randcams.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_sink/TurnOnSinkFaucet/2024-04-25/demo_gentex_im128_randcams.hdf5 \
    task.env_runner.max_steps=400 

9. TurnOffStove -8
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config-dir=. \
    --config-name=image_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=256 \
    val_dataloader.batch_size=256 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_robocasa_TurnOffStove' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_stove/TurnOffStove/2024-05-02/demo_gentex_im128_randcams.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_stove/TurnOffStove/2024-05-02/demo_gentex_im128_randcams.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_stove/TurnOffStove/2024-05-02/demo_gentex_im128_randcams.hdf5 \
    task.env_runner.max_steps=400 

9. TurnOnStove -9
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config-dir=. \
    --config-name=image_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=256 \
    val_dataloader.batch_size=256 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_robocasa_TurnOnStove' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_stove/TurnOnStove/2024-05-02/demo_gentex_im128_randcams.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_stove/TurnOnStove/2024-05-02/demo_gentex_im128_randcams.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_stove/TurnOnStove/2024-05-02/demo_gentex_im128_randcams.hdf5 \
    task.env_runner.max_steps=400 

10. PnPCabToCounter -10
CUDA_VISIBLE_DEVICES=1 python train.py \
    --config-dir=. \
    --config-name=image_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=256 \
    val_dataloader.batch_size=256 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_robocasa_PnPCabToCounter' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24/demo_gentex_im128_randcams.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24/demo_gentex_im128_randcams.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24/demo_gentex_im128_randcams.hdf5 \
    task.env_runner.max_steps=500 

11. PnPCounterToSink -11
CUDA_VISIBLE_DEVICES=2 python train.py \
    --config-dir=. \
    --config-name=image_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=256 \
    val_dataloader.batch_size=256 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_robocasa_PnPCounterToSink' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/2024-04-25/demo_gentex_im128_randcams.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/2024-04-25/demo_gentex_im128_randcams.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/2024-04-25/demo_gentex_im128_randcams.hdf5 \
    task.env_runner.max_steps=700 






































"""

