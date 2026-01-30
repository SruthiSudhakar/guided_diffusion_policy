"""
source /proj/vondrick3/sruthi/miniconda3/bin/activate
conda activate clonejgdrobodiff
cd /proj/vondrick3/sruthi/robots/diffusion_policy
export LD_LIBRARY_PATH=:/home/sruthi/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=osmesa 
export HYDRA_FULL_ERROR=1

Usage:
docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Ports}}\t{{.Image}}"
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' container_ID
ssh -J sruthi.sudhakar@10.110.170.251 -L 5050:ip_address:5050 sruthi.sudhakar@tri-hq-ml-dgx-$1


python final_eval_clip_policy_hitl_web.py \
    --checkpoint data/outputs/dec4/2025.12.03/22.38.27_train_diffusion_unet_clip/checkpoints/epoch_70_step_4188.ckpt \
    --llm_path '' \
    --device cuda:0 \
    --change_test_textures \
    --list_dataset_path PnPStoveToCounter_mg_fixed_224 \
    --n_envs 49 \
    --n_train 48 \
    --n_test 1 \
    --choose_sample \
    --num_samples 5 \
    --additional_steps 1 \
    --num_actions_to_execute 16 \
    --start_rollout_from_state 140 \
    --max_steps 200 \
    --prefix_dir jan14_hitl


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
import pdb
from omegaconf import OmegaConf,open_dict
import datetime
import yaml
import h5py
from data.dgx_data_registery import DATASETS
from termcolor import colored
import time
import numpy as np
from PIL import Image

from torch.nn.parallel import DistributedDataParallel as DDP
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


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
@click.option('-n_train', '--n_train', default=None, type=int)
@click.option('-n_test', '--n_test', default=None, type=int)
@click.option('-n_envs', '--n_envs', default=None, type=int)
@click.option('-test_start_seed', '--test_start_seed', required=False)
@click.option('-object', '--object', default=None)
@click.option('-add', '--add', default='')
@click.option('-prefix_dir', '--prefix_dir', default='')
@click.option('-save', '--save', is_flag=True)
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
@click.option('-llm_path', '--llm_path', default='/app/data/checkpoints/llm_checkpoints/checkpoint-400', help='Path to the base LLM repository')
@click.option('-llm_gpu', '--llm_gpu', default=None, help='GPU to use for LLM')
@click.option('-num_actions_to_execute', '--num_actions_to_execute', default=None, help='num actions to execute from prediction horizon')

def main(checkpoint, list_dataset_path, output_dir, classifier_dir, grad_steps, guidance_scale, guided_towards, device, max_steps, n_train, n_test, n_envs, test_start_seed, object, add, prefix_dir, save, change_test_textures, change_test_objects, change_test_object_instances, init_state_none, debug, choose_sample, num_samples, start_rollout_from_state, show_classifier_scores, adaptive_guidance, decode_first, start_sampling, end_sampling, additional_steps, specific_train_exs,prompt_with_video, llm_path, llm_gpu, num_actions_to_execute):
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
            max_steps=int(np.percentile(max_steps,90))
            print('max_steps',max_steps)
            data.close()    
        current_time = datetime.datetime.now()
        if specific_train_exs:
            n_train = len(specific_train_exs)
        elif n_train is None:
            n_train = len(h5py.File(dataset_path, 'r')['data']) + 1
        if n_test is None:
            n_test = 1
        n_envs = n_train + n_test
        print('n_train',n_train, 'n_test',n_test,'n_envs',n_envs)
        print('n_train',n_train, 'n_test',n_test,'n_envs',n_envs)
        print('n_train',n_train, 'n_test',n_test,'n_envs',n_envs)
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
            sys.exit(f"Output path {output_dir} already exists! Exiting.")
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        print(colored(f'saving to: f{output_dir}', 'green'))
        
        with open (output_dir+'/save_some_deets.txt', 'w') as f: 
            deets = ['checkpoint', checkpoint, 'output_dir', output_dir, 'dataset_path', \
                dataset_path, 'classifier_dir', classifier_dir, 'grad_steps', grad_steps, 'guidance_scale', \
                guidance_scale, 'guided_towards', guided_towards, 'max_steps', max_steps, \
                'object',object, 'n_train', n_train, 'n_test', n_test, "n_envs", n_envs, \
                'test_start_seed', test_start_seed, 'change_test_objects', change_test_objects, \
                'change_test_textures', change_test_textures, 'change_test_object_instances', \
                change_test_object_instances, 'debug', debug, 'choose_sample',choose_sample, 'num_samples',num_samples, 'test init_state_none', init_state_none, \
                'adaptive_guidance', adaptive_guidance, 'decode_first', decode_first, 'start_rollout_from_state', start_rollout_from_state, \
                'start sampling', start_sampling, 'end sampling', end_sampling, 'additional_steps', additional_steps, 'specific_train_exs', \
                specific_train_exs, 'prompt_with_video',prompt_with_video, 'llm_path', llm_path, 'num_actions_to_execute',num_actions_to_execute, 'file','final_eval_clip_policy_hitl.py']
            deets = [str(x) for x in deets]
            f.writelines("\n".join(deets))

        # load checkpoint
        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
        cfg = payload['cfg']
            
        with open_dict(cfg):

            cfg['task']['env_runner']['_target_'] = 'diffusion_policy.env_runner.robocasa_robomimic_image_runner_eval_new_cilp_hitl_web.RobocasaRobomimicImageRunnerEvalClip'
            cfg['task']['env_runner']['render_obs_key']='robot0_agentview_left_image'

            # Copy shape_meta from task level to env_runner level if it exists at task level
            if 'shape_meta' in cfg['task'] and 'shape_meta' not in cfg['task']['env_runner']:
                cfg['task']['env_runner']['shape_meta'] = cfg['task']['shape_meta']

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
            cfg['task']['env_runner']['llm_path']=llm_path
            cfg['task']['env_runner']['llm_gpu']=llm_gpu
            cfg['task']['num_actions_to_execute']=num_actions_to_execute

            cfg['task']['dataset_path'] = dataset_path
            cfg['task']['env_runner']['dataset_path'] = dataset_path
            cfg['task']['dataset']['dataset_path'] = dataset_path
            cfg['task']['env_runner']['max_steps'] = max_steps
            cfg['task']['env_runner']['n_train'] = n_train
            cfg['task']['env_runner']['n_train_vis'] = n_train
            cfg['task']['env_runner']['n_test'] = n_test
            cfg['task']['env_runner']['n_test_vis'] = n_test
            cfg['task']['env_runner']['n_envs'] = n_envs
            if test_start_seed:
                cfg['task']['env_runner']['test_start_seed'] = int(test_start_seed)
            cfg['task']['env_runner']['clip_model_name'] = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"


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
                json_log[key] = str(value)
        out_path = os.path.join(output_dir, 'eval_log.json')
        json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)
        out_path = os.path.join(output_dir, 'jgddone.json')
        json.dump({'done':'JGD done'}, open(out_path, 'w'), indent=2, sort_keys=True)
        print('done. output_dir:', output_dir)

if __name__ == '__main__':
    main()