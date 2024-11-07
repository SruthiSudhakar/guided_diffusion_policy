"""
export LD_LIBRARY_PATH=:/home/sruthi/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=osmesa 
source /proj/vondrick3/sruthi/miniconda3/bin/activate
conda activate jgdrobodiff
cd /proj/vondrick3/sruthi/robots/diffusion_policy
export HYDRA_FULL_ERROR=1

Usage:
Training:

accelerate launch --num_machines 1 --num_processes=1 --gpu_ids=3 --main_process_port=8074 train.py \
    --config-dir=. \
    --config-name=image_square_ph_diffusion_policy_cnn.yaml \
    training.seed=42 \
    dataloader.batch_size=2048 \
    val_dataloader.batch_size=2048 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_15.00.33_hammer_guide_subset' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/diffusion_policy/data/curateddata/hammer2/data_all.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/diffusion_policy/data/curateddata/hammer2/data_all.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/diffusion_policy/data/curateddata/hammer2/data_all.hdf5 \
    task.train_subset=51683 \
    task.val_subset=1264 \
    task.env_runner.max_steps=100 \
    training.resume=/proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.09.03/21.23.37_train_diffusion_unet_hybrid_15.00.33_check/checkpoints/epoch_0150_0.940.ckpt \
    training.classifier_dir=/proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.11.02/08.15.34_train_classifier_test_classifier_redgreencubehammerseed6000/checkpoints/epoch_0006_valid_accuracy_0.784.ckpt \
    training.guidance_scale=0.001 \
    +task.env_runner.object=hammer


accelerate launch --num_machines 1 --num_processes=1 --gpu_ids=6 --main_process_port=8086 train.py \
    --config-dir=. \
    --config-name=image_square_ph_diffusion_policy_cnn.yaml \
    training.seed=42 \
    dataloader.batch_size=1024 \
    val_dataloader.batch_size=1024 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_15.00.33_finetune_4wredcubewithguidancejgd' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/diffusion_policy/data/curateddata/hammer2/data_successful_only.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/diffusion_policy/data/curateddata/hammer2/data_successful_only.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/diffusion_policy/data/curateddata/hammer2/data_successful_only.hdf5 \
    task.env_runner.max_steps=100 \
    training.resume=/proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.09.03/21.23.37_train_diffusion_unet_hybrid_15.00.33_check/checkpoints/epoch_0150_0.940.ckpt \
    training.classifier_dir=/proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.11.06/11.06.17.18.52_train_classifier_4wredcubeseed6000/checkpoints/epoch_0004_valid_accuracy_0.905.ckpt \
    training.guidance_scale=10 \
    +task.env_runner.object=4wredcube


accelerate launch --multi_gpu --num_machines 1 --num_processes=2 --gpu_ids=2,3 --main_process_port=8082 train.py \
    --config-dir=. \
    --config-name=image_square_ph_diffusion_policy_cnn.yaml \
    training.seed=42 \
    dataloader.batch_size=4096 \
    val_dataloader.batch_size=4096 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_pickplacecan' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/diffusion_policy/data/robomimic/datasets/can/ph/image_abs.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/diffusion_policy/data/robomimic/datasets/can/ph/image_abs.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/diffusion_policy/data/robomimic/datasets/can/ph/image_abs.hdf5 \
    task.env_runner.max_steps=200 
    
    \
    +task.env_runner.object=needle


"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

if __name__ == "__main__":
    main()