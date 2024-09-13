"""
export LD_LIBRARY_PATH=:/home/sruthi/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=osmesa 
conda activate robodiff

Usage:
Training:

accelerate launch --multi_gpu --num_machines 1 --num_processes=8 --gpu_ids=0,1,2,3,4,5,6,7 --main_process_port=8086 

python train.py \
    --config-dir=. \
    --config-name=image_square_ph_diffusion_policy_cnn.yaml \
    training.seed=42 \
    training.device=2 \
    dataloader.batch_size=1 \
    val_dataloader.batch_size=1 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_15.00.33_train' \
    task.dataset.dataset_path=onex.hdf5 \
    task.dataset_path=onex.hdf5 \
    task.env_runner.dataset_path=onex.hdf5 \
    task.env_runner.max_steps=100 \
    task.env_runner.n_envs=2 \
    task.env_runner.n_train=1 \
    task.env_runner.n_train_vis=1 \
    task.env_runner.n_test=1 \
    task.env_runner.n_test_vis=1 \
    +task.env_runner.object=redcube

accelerate launch --num_machines 1 --num_processes=1 --gpu_ids=1 --main_process_port=8082 

accelerate launch --num_machines 1 --num_processes=1 --gpu_ids=5 --main_process_port=8090 train.py \
    --config-dir=. \
    --config-name=image_square_ph_diffusion_policy_cnn.yaml \
    training.seed=42 \
    dataloader.batch_size=1024 \
    val_dataloader.batch_size=1024 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_negate_failures_downweighted_0.001' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/diffusion_policy/data/curateddata/needle2/data_all.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/diffusion_policy/data/curateddata/needle2/data_all.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/diffusion_policy/data/curateddata/needle2/data_all.hdf5 \
    task.env_runner.max_steps=100 \
    +task.env_runner.object=hammer \
    +policy.negate_failure_losses=True 

/proj/vondrick3/sruthi/robots/diffusion_policy/data/curateddata/trial_hammer/data_successful_only.hdf5 \

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
