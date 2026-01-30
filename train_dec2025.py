"""
export LD_LIBRARY_PATH=:/home/sruthi/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=osmesa 
source /proj/vondrick3/sruthi/miniconda3/bin/activate
conda activate jgdrobodiff
cd /proj/vondrick3/sruthi/robots/diffusion_policy
export HYDRA_FULL_ERROR=1

Usage:
Training:

CoffeeServeMug, PnPCabToCounter, PnPStoveToCounter, PnPCounterToSink

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_machines 1 --num_processes 8 --gpu_ids=0,1,2,3,4,5,6,7 --main_process_port=8082 train.py \
    --config-dir=. \
    --config-name=train_robocasa_base_dp_clip_policy.yaml \
    training.seed=42 \
    dataloader.batch_size=48 \
    hydra.run.dir='data/outputs/test/${now:%Y.%m.%d}/${now:%H.%M.%S}_clip_${task_name}' \
    task.name=allPnP 

CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_machines 1 --num_processes 4 --gpu_ids=4,5,6,7 --main_process_port=8082 train.py \
    --config-dir=. \
    --config-name=train_robocasa_base_dp_clip_policy.yaml \
    training.seed=42 \
    dataloader.batch_size=48 \
    hydra.run.dir='data/outputs/dec4/${now:%Y.%m.%d}/${now:%H.%M.%S}_clip_${task_name}' \
    task.name=justPnPCounterToMicrowave \
    "task.dataset.human_path={PnPCounterToMicrowave: externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPCounterToMicrowave/2024-04-27/demo_gentex_im128_randcams_im224.hdf5}"

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