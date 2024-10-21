"""
export LD_LIBRARY_PATH=:/home/sruthi/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=osmesa 
conda activate robodiff
cd /proj/vondrick3/sruthi/robots/diffusion_policy

Usage:
Training:

accelerate launch --multi_gpu --num_machines 1 --num_processes=2 --gpu_ids=0,1 --main_process_port=8098 train.py \
    --config-dir=. \
    --config-name=image_square_ph_diffusion_policy_cnn.yaml \
    training.seed=42 \
    dataloader.batch_size=2048 \
    val_dataloader.batch_size=2048 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_15.00.33_needle_withguidance_dataall_subset' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/diffusion_policy/data/curateddata/needle2/data_all.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/diffusion_policy/data/curateddata/needle2/data_all.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/diffusion_policy/data/curateddata/needle2/data_all.hdf5 \
    task.train_subset=46888 \
    task.val_subset=1021 \
    task.env_runner.max_steps=100 \
    training.resume=/proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.09.03/21.23.37_train_diffusion_unet_hybrid_15.00.33_check/checkpoints/epoch_0150_0.940.ckpt \
    training.classifier_dir=/proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.10.13/15.54.25_train_classifier_classifier_needle/checkpoints/epoch_8_validacc_0.928.ckpt \
    training.guidance_scale=9.5 \
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