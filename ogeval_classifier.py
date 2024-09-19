"""
export LD_LIBRARY_PATH=:/home/sruthi/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=osmesa 
conda activate robodiff

Usage:
python ogeval.py --checkpoint /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.09.11/14.50.02_train_diffusion_unet_hybrid_needle_negate_0.001/checkpoints/epoch=0300-test_mean_score=0.760.ckpt \
                --output_dir /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.09.11/14.50.02_train_diffusion_unet_hybrid_needle_negate_0.001/checkpoints/epoch=0300-test_mean_score=0.760/ \
                --dataset_path /proj/vondrick3/sruthi/robots/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5 \
                --device cuda:3

"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import pdb
from omegaconf import OmegaConf,open_dict
import datetime

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-dataset_path', '--dataset_path', required=False)
@click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:0')
@click.option('-max_steps', '--max_steps', default=500)
@click.option('-n_train', '--n_train', required=True)
@click.option('-n_test', '--n_test', required=True)
@click.option('-object', '--object', default='block')
@click.option('-add', '--add', default='')
@click.option('-save', '--save', is_flag=True)
def main(checkpoint, dataset_path, output_dir, device, max_steps, object, add, n_train, n_test, save):
    current_time = datetime.datetime.now()
    output_dir+=f'{add}alift_{object}_{current_time.day}_{current_time.hour}_{current_time.minute}_{current_time.second}'
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    cfg['task']['dataset_path'] = dataset_path
    cfg['task']['dataset']['dataset_path'] = dataset_path

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()
    
    # run eval
    env_runner = hydra.utils.instantiate(
        cfg.task.env_runner,
        output_dir=output_dir)
    runner_log= env_runner.run(policy)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()