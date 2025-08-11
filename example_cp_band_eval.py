#!/usr/bin/env python3
"""
Example script demonstrating how to use the CP Band Failure Detection Runner
for evaluating a diffusion policy with real-time failure detection.

When a failure is detected (score exceeds CP band threshold), the video will
show a red border to indicate the failure.
"""

import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from diffusion_policy.env_runner.cp_band_fail_detect_env_runner import CPBandFailDetectRunnerEval
from diffusion_policy.workspace.base_workspace import BaseWorkspace






"""
source /proj/vondrick3/sruthi/miniconda3/bin/activate
conda activate clonejgdrobodiff
cd /proj/vondrick3/sruthi/robots/diffusion_policy
export LD_LIBRARY_PATH=:/home/sruthi/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=osmesa 
export HYDRA_FULL_ERROR=1
usage:
python example_cp_band_eval.py \
    --checkpoint data/checkpoints/dp_model/epoch=1100-val_loss=0.037.ckpt \
    --save_score_network_path data/checkpoints/dp_model/human_obs/fd_score_network.ckpt \
    --cp_band_path data/checkpoints/dp_model/epoch=1100-val_loss=0.037/score_rollouts/mg_val_kbpckt_firsthalf/PnPSinkToCounter_mg_val_kbpckt_firsthalf_87172126_/fd_scores/CP_band.pkl \
    --device cuda:6 \
    --robocasa \
    --change_test_textures \
    --list_dataset_path PnPSinkToCounter_mg_val_kbpckt_firsthalf \
    --n_envs 50 \
    --n_train 1 \
    --n_test 49 \
    --prefix_dir detect_failures_state0/PnPSinkToCounter_mg_val_kbpckt_firsthalf

python example_cp_band_eval.py \
    --checkpoint data/checkpoints/dp_model/epoch=1100-val_loss=0.037.ckpt \
    --save_score_network_path data/checkpoints/dp_model/human_obs/fd_score_network.ckpt \
    --cp_band_path data/checkpoints/dp_model/epoch=1100-val_loss=0.037/score_rollouts_140/human/PnPSinkToCounter_Human_fixed_textures_862300_/fd_scores/CP_band.pkl \
    --device cuda:7 \
    --robocasa \
    --change_test_textures \
    --list_dataset_path PnPSinkToCounter_Human_fixed_textures \
    --n_envs 50 \
    --n_train 1 \
    --n_test 49 \
    --start_rollout_from_state 140 \
    --max_steps 200 \
    --prefix_dir detect_failures/PnPSinkToCounter_Human_fixed_textures


"""
import sys
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
from omegaconf import OmegaConf,open_dict
import datetime
import yaml
import h5py
from data.dgx_data_registery import DATASETS
from termcolor import colored
import numpy as np
from torch.nn.parallel import DistributedDataParallel as DDP
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pdb
import pickle

@click.command()
@click.option('-checkpoint', '--checkpoint', required=True)
@click.option('-list_dataset_path', '--list_dataset_path', required=False)
@click.option('-o', '--output_dir', required=False)
@click.option('-classifier_dir', '--classifier_dir', required=False)
@click.option('-grad_steps', '--grad_steps', default=1)
@click.option('-guidance_scale', '--guidance_scale', default='0')
@click.option('-guided_towards', '--guided_towards', default=1)
@click.option('-d', '--device', default='cuda:0')
@click.option('-max_steps', '--max_steps', default=None, type=int)
@click.option('-n_train', '--n_train', default=100)
@click.option('-n_test', '--n_test', default=100)
@click.option('-n_envs', '--n_envs', default=100)
@click.option('-test_start_seed', '--test_start_seed', required=False)
@click.option('-object', '--object', default=None)
@click.option('-add', '--add', default='')
@click.option('-prefix_dir', '--prefix_dir', default='')
@click.option('-save', '--save', is_flag=True)
@click.option('-robocasa', '--robocasa', is_flag=True)
@click.option('-change_test_textures', '--change_test_textures', is_flag=True)
@click.option('-change_test_objects', '--change_test_objects', is_flag=True)
@click.option('-change_test_object_instances', '--change_test_object_instances', is_flag=True)
@click.option('-init_state_none', '--init_state_none', is_flag=True)
@click.option('-debug', '--debug', is_flag=True)
@click.option('-choose_sample', '--choose_sample', is_flag=True)
@click.option('-num_samples', '--num_samples', default=1)
@click.option('-start_rollout_from_state', '--start_rollout_from_state', default=0)
@click.option('-show_classifier_scores', '--show_classifier_scores', is_flag=True)
@click.option('-adaptive_guidance', '--adaptive_guidance', default='None')
@click.option('-decode_first', '--decode_first', is_flag=False)
@click.option('-start_sampling', '--start_sampling', default=0)
@click.option('-end_sampling', '--end_sampling', default=27)
@click.option('-additional_steps', '--additional_steps', default=0)
@click.option('--specific_train_exs', type=str, default='', help='Comma-separated list of items.')
@click.option('-prompt_with_video', '--prompt_with_video', is_flag=True)
@click.option('-save_score_network_path', '--save_score_network_path', required=True)
@click.option('-cp_band_path', '--cp_band_path', required=True)

def main(checkpoint, list_dataset_path, output_dir, classifier_dir, grad_steps, guidance_scale, guided_towards, device, max_steps, n_train, n_test, n_envs, test_start_seed, object, add, prefix_dir, save, robocasa, change_test_textures, change_test_objects, change_test_object_instances, init_state_none, debug, choose_sample, num_samples, start_rollout_from_state, show_classifier_scores, adaptive_guidance, decode_first, start_sampling, end_sampling, additional_steps, specific_train_exs,prompt_with_video, save_score_network_path, cp_band_path):
    # Extract the value for task.dataset_path
    specific_train_exs = [x.strip() for x in specific_train_exs.split(',')] if specific_train_exs else []
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
        for split in list_dataset_path[0].split('/'):
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
            max_steps=np.array(max_steps)
            print('max_steps',int(np.percentile(max_steps,90)))
            data.close()    
            max_steps=int(np.percentile(max_steps,90))
        current_time = datetime.datetime.now()

        if not choose_sample and classifier_dir:
            if adaptive_guidance!='None':
                output_dir+=f'{prefix_dir}/{add}_{task}_{current_time.month}{current_time.day}{current_time.hour}{current_time.minute}{current_time.second}_guided_{guided_towards}_grad_steps{grad_steps}_{guidance_scale}_{adaptive_guidance}'
            else:
                output_dir+=f'{prefix_dir}/{add}_{task}_{current_time.month}{current_time.day}{current_time.hour}{current_time.minute}{current_time.second}_guided_{guided_towards}_grad_steps{grad_steps}_{guidance_scale}'
        elif choose_sample:
            output_dir+=f'{prefix_dir}/{task}_{current_time.month}{current_time.day}{current_time.hour}{current_time.minute}{current_time.second}_{add}_choose_sample_{choose_sample}_num_samples_{num_samples}_ws_{start_sampling}-{end_sampling}'
        else:
            output_dir+=f'{prefix_dir}/{task}_{current_time.month}{current_time.day}{current_time.hour}{current_time.minute}{current_time.second}_{add}'
        if os.path.exists(output_dir):
            click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(colored(f'saving to: f{output_dir}', 'green'))
        
        with open (output_dir+'/save_some_deets.txt', 'w') as f: 
            deets = ['checkpoint', checkpoint, 'save_score_network_path', save_score_network_path, 'output_dir', output_dir, 'dataset_path', \
                dataset_path, 'classifier_dir', classifier_dir, 'grad_steps', grad_steps, 'guidance_scale', \
                guidance_scale, 'guided_towards', guided_towards, 'max_steps', max_steps, \
                'object',object, 'n_train', n_train, 'n_test', n_test, "n_envs", n_envs, \
                'test_start_seed', test_start_seed, 'change_test_objects', change_test_objects, \
                'change_test_textures', change_test_textures, 'change_test_object_instances', \
                change_test_object_instances, 'debug', debug, 'choose_sample',choose_sample, 'num_samples',num_samples, 'test init_state_none', init_state_none, \
                'adaptive_guidance', adaptive_guidance, 'decode_first', decode_first, 'start_rollout_from_state', start_rollout_from_state, \
                'start sampling', start_sampling, 'end sampling', end_sampling, 'additional_steps', additional_steps, 'specific_train_exs',specific_train_exs, 'prompt_with_video',prompt_with_video,
                'cp_band_path', cp_band_path, 'save_score_network_path', save_score_network_path]
            deets = [str(x) for x in deets]
            f.writelines("\n".join(deets))

        # load checkpoint
        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
        cfg = payload['cfg']

        cfg['task']['env_runner']['_target_'] = 'diffusion_policy.env_runner.cp_band_fail_detect_env_runner.CPBandFailDetectRunnerEval'
        cfg['task']['env_runner']['render_obs_key']='robot0_agentview_left_image'
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
            cfg['task']['env_runner']['num_samples']=num_samples
            cfg['task']['env_runner']['start_rollout_from_state']=start_rollout_from_state
            cfg['task']['env_runner']['init_state_none']=init_state_none
            cfg['task']['env_runner']['show_classifier_scores']=show_classifier_scores
            cfg['task']['env_runner']['adaptive_guidance']=adaptive_guidance
            cfg['task']['env_runner']['decode_first']=decode_first
            cfg['task']['env_runner']['start_sampling']=start_sampling
            cfg['task']['env_runner']['end_sampling']=end_sampling
            cfg['task']['env_runner']['additional_steps']=additional_steps
            cfg['task']['env_runner']['specific_train_exs']=specific_train_exs
            cfg['task']['env_runner']['prompt_with_video']=prompt_with_video
            cfg['task']['env_runner']['save_score_network_path']=save_score_network_path
            cfg['task']['env_runner']['cp_band_path']=cp_band_path
            cfg['task']['env_runner']['red_border_thickness']=15,  # Thickness of red border when failure detected

            cfg['task']['dataset_path'] = dataset_path
            cfg['task']['env_runner']['dataset_path'] = dataset_path
            cfg['task']['dataset']['dataset_path'] = dataset_path
            cfg['task']['env_runner']['max_steps'] = max_steps
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
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=output_dir)
        runner_log, image_obs = env_runner.run(policy)
        # dump log to json
        json_log = dict()
        for key, value in runner_log.items():
            if isinstance(value, wandb.sdk.data_types.video.Video):
                json_log[key] = value._path
            else:
                json_log[key] = str(value)
        out_path = os.path.join(output_dir, 'computed_rollout_scores.json')
        json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

        pdb.set_trace()
        # Print results
        print("\nEvaluation Results:")
        print(f"Train mean score: {runner_log.get('train/mean_score', 'N/A')}")
        print(f"Test mean score: {runner_log.get('test/mean_score', 'N/A')}")
        print(f"Failure detection rate: {runner_log.get('failure_detection_rate', 'N/A')} = {runner_log['detected']} / {runner_log['total']} ")
        print(f"pertimestep_failure_raised_rate: {runner_log.get('pertimestep_failure_raised_rate', 'N/A')} = {runner_log['num_failure_timesteps_detected']} / {runner_log['num_total_timetseps']} ")
        
        # # Check which rollouts had failures detected
        # failure_count = 0
        # for key, value in runner_log.items():
        #     if 'failure_detected_' in key and value:
        #         failure_count += 1
        #         timestep_key = key.replace('failure_detected_', 'failure_timestep_')
        #         timestep = runner_log.get(timestep_key, 'unknown')
        #         print(f"  - {key}: Failure detected at timestep {timestep}")
        
        # print(f"Videos saved to: {output_dir}")
        # print("\nVideos with detected failures will have red borders.")

        out_path = os.path.join(output_dir, 'jgddone.json')
        json.dump({'done':'JGD done'}, open(out_path, 'w'), indent=2, sort_keys=True)
        print('done. output_dir:', output_dir)
if __name__ == '__main__':
    main()