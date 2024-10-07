"""
cd /proj/vondrick3/sruthi/robots/diffusion_policy
export LD_LIBRARY_PATH=:/home/sruthi/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=osmesa 
conda activate robodiff

Usage:

python ogeval.py --checkpoint /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.09.03/21.23.37_train_diffusion_unet_hybrid_15.00.33_check/checkpoints/epoch=0150-test_mean_score=0.940.ckpt \
                --output_dir /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.09.03/21.23.37_train_diffusion_unet_hybrid_15.00.33_check/checkpoints/epoch=0150-test_mean_score=0.940/ \
                --dataset_path /proj/vondrick3/sruthi/robots/diffusion_policy/data/robomimic/datasets/lift/ph/image_abs.hdf5 \
                --max_steps 100 \
                --device cuda:1 \
                --object hammer \
                --n_train 5 \
                --n_test 5 \
                --classifier_dir /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.09.16/17.26.03_train_classifier_15.00.33_classifier/checkpoints/epoch=0010-valid_accuracy=0.919 \
                --guidance_scale 950 \
                --guided_towards 1 \

                --test_start_seed 5

                --save 
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
@click.option('-classifier_dir', '--classifier_dir', required=False)
@click.option('-guidance_scale', '--guidance_scale', required=False)
@click.option('-guided_towards', '--guided_towards', required=False)
@click.option('-d', '--device', default='cuda:0')
@click.option('-max_steps', '--max_steps', default=500)
@click.option('-n_train', '--n_train', required=True)
@click.option('-n_test', '--n_test', required=True)
@click.option('-test_start_seed', '--test_start_seed', required=False)
@click.option('-object', '--object', default='block')
@click.option('-add', '--add', default='')
@click.option('-save', '--save', is_flag=True)
def main(checkpoint, dataset_path, output_dir, classifier_dir, guidance_scale, guided_towards, device, max_steps, object, add, n_train, n_test, save, test_start_seed):
    current_time = datetime.datetime.now()
    if classifier_dir:
        output_dir+=f'{add}alift_{object}_{current_time.day}_{current_time.hour}_{current_time.minute}_{current_time.second}_guided_{guided_towards}_{guidance_scale}'
    else:
        output_dir+=f'{add}alift_{object}_{current_time.day}_{current_time.hour}_{current_time.minute}_{current_time.second}'
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open (output_dir+'/save_some_deets.txt', 'w') as f: 
        deets = ['checkpoint', checkpoint, 'output_dir', output_dir, 'dataset_path', dataset_path, 'classifier_dir', classifier_dir, 'guidance_scale', guidance_scale, 'guided_towards', guided_towards, 'max_steps', max_steps, 'object',object, 'n_train', n_train, 'n_test', n_test, 'test_start_seed', test_start_seed]
        deets = [str(x) for x in deets]
        f.writelines("\n".join(deets))

    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
       
    cfg['task']['env_runner']['_target_'] = 'diffusion_policy.env_runner.robomimic_image_runner_eval.RobomimicImageRunnerEval'
    with open_dict(cfg):
        cfg['task']['env_runner']['object'] = object
        cfg['task']['env_runner']['save_stuff'] = save

    cfg['task']['dataset_path'] = dataset_path
    cfg['task']['env_runner']['dataset_path'] = dataset_path
    cfg['task']['dataset']['dataset_path'] = dataset_path
    cfg['task']['env_runner']['max_steps'] = max_steps
    cfg['task']['env_runner']['n_train'] = int(n_train)
    cfg['task']['env_runner']['n_train_vis'] = int(n_train)
    cfg['task']['env_runner']['n_test'] = int(n_test)
    cfg['task']['env_runner']['n_test_vis'] = int(n_test)
    cfg['task']['env_runner']['n_envs'] = 28
    if test_start_seed:
        cfg['task']['env_runner']['test_start_seed'] = int(test_start_seed)


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
    
    if classifier_dir:
        classifier_payload = torch.load(open(classifier_dir+'.ckpt', 'rb'), pickle_module=dill)
        classifier_cfg = classifier_payload['cfg']
        classifier_cls = hydra.utils.get_class(classifier_cfg._target_)

        classifier_workspace = classifier_cls(classifier_cfg, output_dir=classifier_dir)
        classifier_workspace: BaseWorkspace
        classifier_workspace.load_payload(classifier_payload, exclude_keys=None, include_keys=None)
        
        # get policy from workspace
        classifier_policy = classifier_workspace.model    
        classifier_policy.to(device)
        classifier_policy.eval()
        
        # run eval
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=output_dir)
        runner_log= env_runner.run(policy, classifier_policy, float(guidance_scale), float(guided_towards))
    else:
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