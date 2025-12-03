"""
export LD_LIBRARY_PATH=:/home/sruthi/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=osmesa 
source /proj/vondrick3/sruthi/miniconda3/bin/activate
conda activate jgdrobodiff
cd /proj/vondrick3/sruthi/robots/diffusion_policy
export HYDRA_FULL_ERROR=1

Usage:
Training:
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config-dir=. \
    --config-name=train_robocasa_base_dp_clip_policy.yaml \
    training.seed=42 \
    dataloader.batch_size=48 \
    hydra.run.dir='data/outputs/dec3/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}' \
    task.name=justPnPSinkToCounter \
    "task.dataset.human_path={PnPSinkToCounter: externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2/expert_demos_fixed_textures_224.hdf5}"
CUDA_VISIBLE_DEVICES=1 python train.py \
    --config-dir=. \
    --config-name=train_robocasa_base_dp_clip_policy.yaml \
    training.seed=42 \
    dataloader.batch_size=48 \
    hydra.run.dir='data/outputs/dec3/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}' \
    task.name=justPnPStoveToCounter \
    "task.dataset.human_path={PnPStoveToCounter: externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/2024-05-01/demo_gentex_im128_randcams_im224.hdf5}"
CUDA_VISIBLE_DEVICES=2 python train.py \
    --config-dir=. \
    --config-name=train_robocasa_base_dp_clip_policy.yaml \
    training.seed=42 \
    dataloader.batch_size=48 \
    hydra.run.dir='data/outputs/dec3/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}' \
    task.name=justPnPCabToCounter \
    "task.dataset.human_path={PnPCabToCounter: externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24/demo_gentex_im128_randcams_im224.hdf5}"
CUDA_VISIBLE_DEVICES=3 python train.py \
    --config-dir=. \
    --config-name=train_robocasa_base_dp_clip_policy.yaml \
    training.seed=42 \
    dataloader.batch_size=48 \
    hydra.run.dir='data/outputs/dec3/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}' \
    task.name=justCoffeeServeMug \
    "task.dataset.human_path={CoffeeServeMug: externals/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeServeMug/2024-05-01/demo_gentex_im128_randcams.hdf5}"
CUDA_VISIBLE_DEVICES=4 python train.py \
    --config-dir=. \
    --config-name=train_robocasa_base_dp_clip_policy.yaml \
    training.seed=42 \
    dataloader.batch_size=48 \
    hydra.run.dir='data/outputs/dec3/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}' \
    task.name=justPnPCounterToSink \
    "task.dataset.human_path={PnPCounterToSink: externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/2024-04-25/demo_gentex_im128_randcams.hdf5}"

accelerate launch --num_machines 1 --num_processes 8 --main_process_port=8082 train.py \
    --config-dir=. \
    --config-name=train_robocasa_base_dp_clip_policy.yaml \
    training.seed=42 \
    dataloader.batch_size=48 \
    hydra.run.dir='data/outputs/dec3/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}' \
    task.name=5PnPtasks \
    "task.dataset.human_path={PnPSinkToCounter: externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2/expert_demos_fixed_textures_224.hdf5, PnPStoveToCounter: externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/2024-05-01/demo_gentex_im128_randcams_im224.hdf5, PnPCabToCounter: externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCabToCounter/2024-04-24/demo_gentex_im128_randcams_im224.hdf5, CoffeeServeMug: externals/robocasa/datasets/v0.1/single_stage/kitchen_coffee/CoffeeServeMug/2024-05-01/demo_gentex_im128_randcams.hdf5, PnPCounterToSink: externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToSink/2024-04-25/demo_gentex_im128_randcams.hdf5}"

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