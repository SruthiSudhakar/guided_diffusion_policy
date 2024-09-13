"""
Usage:
export LD_LIBRARY_PATH=:/home/sruthi/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=osmesa 

tbd:
python eval.py --checkpoint /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.06.05/15.00.33_train_diffusion_unet_hybrid_liftph/checkpoints/epoch=0150-test_mean_score=0.980.ckpt \
               --device cuda:6
python eval.py --checkpoint /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.05.30/17.43.40_train_diffusion_unet_hybrid_listackd1/checkpoints/epoch=0100-test_mean_score=0.540.ckpt \
               --device cuda:7

running:
python eval.py --checkpoint /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.05.30/23.08.15_train_diffusion_unet_hybrid_corestackthreed1/checkpoints/epoch=0100-test_mean_score=0.980.ckpt \
               --device cuda:7
python eval.py --checkpoint /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.05.30/17.45.00_train_diffusion_unet_hybrid_lisquared1/checkpoints/epoch=0200-test_mean_score=0.000.ckpt \
               --device cuda:7
python eval.py --checkpoint /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.05.30/17.44.24_train_diffusion_unet_hybrid_licoffeed1/checkpoints/epoch=0200-test_mean_score=0.000.ckpt \
               --device cuda:3
python eval.py --checkpoint /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.05.30/17.34.20_train_diffusion_unet_hybrid_corecoffeeprepd1/checkpoints/epoch=0200-test_mean_score=0.220.ckpt \
               --device cuda:4
python eval.py --checkpoint /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.06.05/15.17.21_train_diffusion_unet_hybrid_corehammercleanupd1/checkpoints/epoch=0250-test_mean_score=0.440.ckpt \
               --device cuda:6
python eval.py --checkpoint /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.06.05/08.00.33_train_diffusion_unet_hybrid_coremugcleanupd1/checkpoints/epoch=0250-test_mean_score=0.580.ckpt \
               --device cuda:6
python eval.py --checkpoint /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.06.04/22.46.35_train_diffusion_unet_hybrid_corekitchend1/checkpoints/epoch=0100-test_mean_score=0.700.ckpt \
               --device cuda:7
python eval.py --checkpoint /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.06.04/21.46.55_train_diffusion_unet_hybrid_litpad1/checkpoints/epoch=0200-test_mean_score=0.420.ckpt \
               --device cuda:0
python eval.py --checkpoint /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.06.05/18.30.48_train_diffusion_unet_hybrid_corepickplaced0/checkpoints/epoch=0050-test_mean_score=0.005.ckpt \
               --device cuda:7


"""
CHECKPOINT_TO_DATASET = {
    "lift": (100, "/proj/vondrick3/sruthi/robots/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5", 150),
    "listack": (300, "/proj/vondrick3/sruthi/robots/mimicgen_environments/datasets/large_interpolation/stack_d1_abs.hdf5", 100),
    "stackthree": (400, "/proj/vondrick3/sruthi/robots/mimicgen_environments/datasets/core/stack_three_d1_abs.hdf5", 100),
    "square": (400, "/proj/vondrick3/sruthi/robots/mimicgen_environments/datasets/large_interpolation/square_d2_abs.hdf5", 200),
    "licoffee": (600, "/proj/vondrick3/sruthi/robots/mimicgen_environments/datasets/large_interpolation/coffee_d1_abs.hdf5", 200),
    "coffeeprep": (800, "/proj/vondrick3/sruthi/robots/mimicgen_environments/datasets/core/coffee_preparation_d1_abs.hdf5", 200),
    "hammer": (400, "/proj/vondrick3/sruthi/robots/mimicgen_environments/datasets/core/hammer_cleanup_d1_abs.hdf5", 50),
    "mug": (400, "/proj/vondrick3/sruthi/robots/mimicgen_environments/datasets/core/mug_cleanup_d1_abs.hdf5", 50),
    "kitchen": (700, "/proj/vondrick3/sruthi/robots/mimicgen_environments/datasets/core/kitchen_d1_abs.hdf5", 100),
    "tpa": (700, "/proj/vondrick3/sruthi/robots/mimicgen_environments/datasets/large_interpolation/three_piece_assembly_d1_abs.hdf5", 200),
    "pickplace": (1000, "/proj/vondrick3/sruthi/robots/mimicgen_environments/datasets/core/pick_place_d1.hdf5", 50),
}

import pdb
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
import datetime

@click.command()
@click.option('-c', '--checkpoint', required=True)
# @click.option('-dataset_path', '--dataset_path', required=False)
# @click.option('-o', '--output_dir', required=True)
@click.option('-d', '--device', default='cuda:7')
# @click.option('-max_steps', '--max_steps', default=500)
def main(checkpoint, device):
    output_dir = checkpoint[:-5]+'/test_score'
    for k,v in CHECKPOINT_TO_DATASET.items():
        if k in checkpoint:
            dataset_path = v[1]
            max_steps = v[0]
    assert dataset_path
    assert max_steps
    print(output_dir,' \n', dataset_path, ' \n', max_steps, ' \n')
    current_time = datetime.datetime.now()
    output_dir+=f'_{current_time.day}_{current_time.hour}_{current_time.minute}_{current_time.second}'
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    cfg['task']['dataset_path'] = dataset_path
    cfg['task']['env_runner']['dataset_path'] = dataset_path
    cfg['task']['dataset']['dataset_path'] = dataset_path
    cfg['task']['env_runner']['max_steps'] = max_steps

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
    runner_log = env_runner.run(policy)
    
    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value
    out_path = os.path.join(output_dir, 'eval_log.json')
    json_log['checkpoint'] = checkpoint
    json_log['dataset_path'] = dataset_path
    json_log['output_dir'] = output_dir
    json_log['max_steps'] = max_steps
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()
