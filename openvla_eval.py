"""
source /proj/vondrick3/sruthi/miniconda3/bin/activate
conda activate clonejgdrobodiff
cd /proj/vondrick3/sruthi/robots/diffusion_policy
export LD_LIBRARY_PATH=:/home/sruthi/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=osmesa 
export HYDRA_FULL_ERROR=1

Usage:      

python openvla_eval.py --checkpoint /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2025.02.15/imageonly_11.32.40_usegroupnorm/checkpoints/epoch=1100-val_loss=0.037.ckpt \
                --device cuda:3 \
                --robocasa \
                --change_test_textures \
                --list_dataset_path PnPSinkToCounter_mg_val_kbpctk_firsthalf \
                --n_envs 2 \
                --n_train 1 \
                --n_test 1 \
                --add test \
                --classifier_dir /proj/vondrick3/sruthi/robots/openvla/outputs/2025.03.17/03.17.17.05.56_cv16_jgd/openvla-7b+chunk_mixture1_jgd+b800+lr-0.0005+lora-r16+dropout-0.0--image_aug--90_chkpt \
                --choose_sample \
                --debug 
                
 python openvla_eval.py --checkpoint /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2025.02.15/imageonly_11.32.40_usegroupnorm/checkpoints/epoch=1100-val_loss=0.037.ckpt \
                --device cuda:5 \
                --robocasa \
                --change_test_textures \
                --list_dataset_path PnPSinkToCounter_mg_train_no_kbpctk \
                --n_envs 256 \
                --n_train 2490 \
                --n_test 1 \
                --add midway_rollout_196 \
                --start_rollout_from_state 196 \
                --save

                --classifier_dir /proj/vondrick3/sruthi/robots/openvla/outputs/2025.03.17/03.17.17.05.56_cv16_jgd/openvla-7b+chunk_mixture1_jgd+b800+lr-0.0005+lora-r16+dropout-0.0--image_aug--90_chkpt \
                --guidance_scale 2 \
                --guided_towards 0 \
                --decode_first False \
                --grad_steps 1 \

python openvla_eval.py --checkpoint /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2025.02.15/imageonly_11.32.40_usegroupnorm/checkpoints/epoch=1100-val_loss=0.037.ckpt \
                --device cuda:5 \
                --robocasa \
                --change_test_textures \
                --list_dataset_path PnPSinkToCounter_mg_val_kbpctk_firsthalf \
                --n_envs 16 \
                --n_train 15 \
                --n_test 1 \
                --classifier_dir /proj/vondrick3/sruthi/robots/openvla/outputs/2025.03.17/03.17.17.05.56_cv16_jgd/openvla-7b+chunk_mixture1_jgd+b800+lr-0.0005+lora-r16+dropout-0.0--image_aug--90_chkpt \
                --guidance_scale 1 \
                --guided_towards 1 \
                --decode_first False \
                --adaptive_guidance linear \
                --add add_first_16

python openvla_eval.py --checkpoint /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2025.02.15/imageonly_11.32.40_usegroupnorm/checkpoints/epoch=1100-val_loss=0.037.ckpt \
                --device cuda:5 \
                --robocasa \
                --change_test_textures \
                --list_dataset_path PnPSinkToCounter_mg_val_kbpctk_secondhalf \
                --n_envs 16 \
                --n_train 256 \
                --n_test 1 \

python openvla_eval.py --checkpoint /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2025.02.15/imageonly_11.32.40_usegroupnorm/checkpoints/epoch=1100-val_loss=0.037.ckpt \
                --device cuda:0 \
                --robocasa \
                --change_test_textures \
                --list_dataset_path PnPSinkToCounter_mg_val_kbpctk_secondhalf \
                --n_envs 1 \
                --n_train 1 \
                --n_test 1 \
                --add test_sampling
"""
# 
# /proj/vondrick3/sruthi/robots/openvla/outputs/2025.03.08/03.08.09.07.32_jgd1/openvla-7b+chunk_mixture1_jgd+b80+lr-0.0005+lora-r16+dropout-0.0--image_aug--100_chkpt \
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
import yaml
import h5py
from data.dataset_registery import DATASETS
from termcolor import colored
import time
import numpy as np
import tensorflow as tf
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import ( get_action, get_image_resize_size, get_model,)
from experiments.robot.openvla_utils import (get_vla,get_vla_action,)

from torch.nn.parallel import DistributedDataParallel as DDP
from types import SimpleNamespace
@click.command()
@click.option('-checkpoint', '--checkpoint', required=True)
@click.option('-list_dataset_path', '--list_dataset_path', required=False)
@click.option('-o', '--output_dir', required=False)
@click.option('-classifier_dir', '--classifier_dir', required=False)
@click.option('-grad_steps', '--grad_steps', default=1)
@click.option('-guidance_scale', '--guidance_scale', default=0)
@click.option('-guided_towards', '--guided_towards', default=1)
@click.option('-d', '--device', default='cuda:0')
@click.option('-max_steps', '--max_steps', default=None, type=int)
@click.option('-n_train', '--n_train', default=100)
@click.option('-n_test', '--n_test', default=100)
@click.option('-n_envs', '--n_envs', default=100)
@click.option('-test_start_seed', '--test_start_seed', required=False)
@click.option('-object', '--object', default=None)
@click.option('-add', '--add', default='')
@click.option('-save', '--save', is_flag=True)
@click.option('-robocasa', '--robocasa', is_flag=True)
@click.option('-change_test_textures', '--change_test_textures', is_flag=True)
@click.option('-change_test_objects', '--change_test_objects', is_flag=True)
@click.option('-change_test_object_instances', '--change_test_object_instances', is_flag=True)
@click.option('-init_state_none', '--init_state_none', is_flag=True)
@click.option('-debug', '--debug', is_flag=True)
@click.option('-choose_sample', '--choose_sample', is_flag=True)
@click.option('-start_rollout_from_state', '--start_rollout_from_state', default=0)
@click.option('-show_classifier_scores', '--show_classifier_scores', is_flag=True)
@click.option('-adaptive_guidance', '--adaptive_guidance', default='None')
@click.option('-decode_first', '--decode_first', default=True)

def main(checkpoint, list_dataset_path, output_dir, classifier_dir, grad_steps, guidance_scale, guided_towards, device, max_steps, n_train, n_test, n_envs, test_start_seed, object, add, save, robocasa, change_test_textures, change_test_objects, change_test_object_instances, init_state_none, debug, choose_sample, start_rollout_from_state, show_classifier_scores, adaptive_guidance, decode_first):
    # Extract the value for task.dataset_path
    yaml_file = '/'.join(checkpoint.split('/')[:-2])+'/.hydra/overrides.yaml'  # Replace with your file path
    with open(yaml_file, 'r') as file:
        train_overrides = yaml.safe_load(file)
    # Convert the list into a dictionary
    parsed_data = {}
    for item in train_overrides:
        key, value = item.split('=', 1)
        parsed_data[key.strip()] = value.strip()
    
    task=''
    if list_dataset_path is None:
        list_dataset_path = [parsed_data.get('task.dataset_path')]
        print("Value of task.dataset_path:", list_dataset_path)
        for split in dataset_path.split('/'):
            if 'PnP' in split:
                task+=split
    elif 'hdf5' in list_dataset_path:
        list_dataset_path=[list_dataset_path]
        for split in dataset_path.split('/'):
            if 'PnP' in split:
                task+=split
    else:
        task+=list_dataset_path
        list_dataset_path = DATASETS[list_dataset_path]

    for dataset_path in list_dataset_path:
        output_dir = checkpoint[:-5]+'/'  # Replace with your file path
        if max_steps is None:
            max_steps=[]
            data = h5py.File(dataset_path, 'r')
            for i in data['data']:
                max_steps.append(data['data'][i]['actions'].shape[0]+50)
            print('max_steps',max(max_steps))
            data.close()    
        
        current_time = datetime.datetime.now()

        if not choose_sample and classifier_dir:
            if adaptive_guidance!='None':
                output_dir+=f'{add}_{task}_{current_time.month}{current_time.day}{current_time.hour}{current_time.minute}{current_time.second}_guided_{guided_towards}_grad_steps{grad_steps}_{guidance_scale}_{adaptive_guidance}'
            else:
                output_dir+=f'{add}_{task}_{current_time.month}{current_time.day}{current_time.hour}{current_time.minute}{current_time.second}_guided_{guided_towards}_grad_steps{grad_steps}_{guidance_scale}'
        else:
            output_dir+=f'{task}_{current_time.month}{current_time.day}{current_time.hour}{current_time.minute}{current_time.second}_{add}'
        if os.path.exists(output_dir):
            click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(colored(f'saving to: f{output_dir}', 'green'))
        
        with open (output_dir+'/save_some_deets.txt', 'w') as f: 
            deets = ['checkpoint', checkpoint, 'output_dir', output_dir, 'dataset_path', \
                dataset_path, 'classifier_dir', classifier_dir, 'grad_steps', grad_steps, 'guidance_scale', \
                guidance_scale, 'guided_towards', guided_towards, 'max_steps', max(max_steps), \
                'object',object, 'n_train', n_train, 'n_test', n_test, "n_envs", n_envs, \
                'test_start_seed', test_start_seed, 'change_test_objects', change_test_objects, \
                'change_test_textures', change_test_textures, 'change_test_object_instances', \
                change_test_object_instances, 'debug', debug, 'choose_sample',choose_sample, 'test init_state_none', init_state_none, \
                'adaptive_guidance', adaptive_guidance, 'decode_first', decode_first, 'start_rollout_from_state', start_rollout_from_state]
            deets = [str(x) for x in deets]
            f.writelines("\n".join(deets))

        # load checkpoint
        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
        cfg = payload['cfg']

        if robocasa:
            cfg['task']['env_runner']['_target_'] = 'diffusion_policy.env_runner.robocasa_robomimic_image_runner_eval.RobocasaRobomimicImageRunnerEval'
            cfg['task']['env_runner']['render_obs_key']='robot0_agentview_left_image'
            # cfg['task']['env_runner']['render_obs_key'] = 'robot0_robotview' if 'mg' in dataset_path else 'robot0_agentview_left_image'
        else:
            cfg['task']['env_runner']['_target_'] = 'diffusion_policy.env_runner.robomimic_image_runner_eval.RobomimicImageRunnerEval'
        
        with open_dict(cfg):
            cfg['task']['env_runner']['object'] = object
            cfg['task']['env_runner']['save_stuff'] = save
            if save and 'robot0_eef_pos' not in cfg['task']['env_runner']['shape_meta']['obs']:
                cfg['task']['env_runner']['shape_meta']['obs']['robot0_eef_pos']={'shape':[3]}
                cfg['task']['env_runner']['shape_meta']['obs']['robot0_eef_quat']={'shape':[4]}
                cfg['task']['env_runner']['shape_meta']['obs']['robot0_gripper_qpos']={'shape':[2]}
            cfg['task']['env_runner']['change_test_textures']= change_test_textures
            cfg['task']['env_runner']['change_test_objects']= change_test_objects
            cfg['task']['env_runner']['change_test_object_instances']= change_test_object_instances
            cfg['task']['env_runner']['debug']=debug
            cfg['task']['env_runner']['choose_sample']=choose_sample
            cfg['task']['env_runner']['start_rollout_from_state']=start_rollout_from_state
            cfg['task']['env_runner']['init_state_none']=init_state_none
            cfg['task']['env_runner']['show_classifier_scores']=show_classifier_scores
            cfg['task']['env_runner']['adaptive_guidance']=adaptive_guidance
            cfg['task']['env_runner']['decode_first']=decode_first

        cfg['task']['dataset_path'] = dataset_path
        cfg['task']['env_runner']['dataset_path'] = dataset_path
        cfg['task']['dataset']['dataset_path'] = dataset_path
        cfg['task']['env_runner']['max_steps'] = max(max_steps)
        cfg['task']['env_runner']['n_train'] = int(n_train)
        cfg['task']['env_runner']['n_train_vis'] = int(n_train)
        cfg['task']['env_runner']['n_test'] = int(n_test)
        cfg['task']['env_runner']['n_test_vis'] = int(n_test)
        cfg['task']['env_runner']['n_envs'] = int(n_envs)
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
            # classifier_payload = torch.load(open(classifier_dir+'.ckpt', 'rb'), pickle_module=dill)
            # classifier_cfg = classifier_payload['cfg']
            # classifier_cls = hydra.utils.get_class(classifier_cfg._target_)

            # classifier_workspace = classifier_cls(classifier_cfg, output_dir=classifier_dir)
            # classifier_workspace: BaseWorkspace
            # classifier_workspace.load_payload(classifier_payload, exclude_keys=None, include_keys=None)
            
            # # get policy from workspace
            # classifier_policy = classifier_workspace.model    
            # classifier_policy.to(device)
            # classifier_policy.eval()
            processor_cfg = {
                "pretrained_checkpoint": classifier_dir,
                "load_in_4bit": False,
                "load_in_8bit": False,
                "device": device,
            }
            processor_cfg=SimpleNamespace(**processor_cfg)
            classifier_processor = get_processor(processor_cfg)
            classifier_policy = get_vla(processor_cfg)
            classifier_policy.to(device)
            classifier_policy.eval()
            for name, param in classifier_policy.named_parameters():
                if "embed_tokens" in name:  # Common names for embeddings
                    print(f"Found embedding layer: {name}")
                    param.requires_grad = True
            # run eval
            env_runner = hydra.utils.instantiate(cfg.task.env_runner,output_dir=output_dir)
            if isinstance(guidance_scale, str):
                runner_log= env_runner.run(policy, classifier_processor, classifier_policy, int(grad_steps), guidance_scale, float(guided_towards))
            else:
                runner_log= env_runner.run(policy, classifier_processor, classifier_policy, int(grad_steps), float(guidance_scale), float(guided_towards))
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
                json_log[key] = str(value)
        out_path = os.path.join(output_dir, 'eval_log.json')
        json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)
        out_path = os.path.join(output_dir, 'jgddone.json')
        json.dump({'done':'JGD done'}, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()