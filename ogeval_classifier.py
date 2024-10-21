"""
export LD_LIBRARY_PATH=:/home/sruthi/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=osmesa 
conda activate robodiff
cd /proj/vondrick3/sruthi/robots/diffusion_policy

Usage:
python ogeval_classifier.py --checkpoint /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.10.16/16.30.32_train_classifier_classifier_mugbeige2/checkpoints/epoch=0005-valid_accuracy=0.896 \
                --dataset_path /proj/vondrick3/sruthi/robots/diffusion_policy/data/curateddata/mugbeige2/data_all.hdf5 \
                --device cuda:5

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
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
import numpy as np


@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-dataset_path', '--dataset_path', required=False)
@click.option('-d', '--device', default='cuda:0')
def main(checkpoint, dataset_path, device):
    current_time = datetime.datetime.now()
    output_dir=checkpoint+f'/classify_{current_time.day}_{current_time.hour}_{current_time.minute}_{current_time.second}'
    print('output_dir: ',output_dir)
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open (output_dir+'/save_some_deets.txt', 'w') as f: 
        deets = [checkpoint, dataset_path]
        deets = [str(x) for x in deets]
        f.writelines("\n".join(deets))

    # load checkpoint
    payload = torch.load(open(checkpoint+'.ckpt', 'rb'), pickle_module=dill)
    cfg = payload['cfg']

    cfg['task']['dataset_path'] = dataset_path
    cfg['task']['dataset']['dataset_path'] = dataset_path
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    device = torch.device(device)
    workspace.model.to(device)
    
    stats = workspace.run_validation()

    # dump log to json
    json_log = dict()
    tp = np.array(stats['equals'])[np.where(np.array(stats['gt_successes'])==1)[0]].sum()
    tn = np.array(stats['equals'])[np.where(np.array(stats['gt_successes'])==0)[0]].sum()
    fp = np.array(stats['equals'])[np.where(np.array(stats['gt_successes'])==0)[0]].shape[0] - np.array(stats['equals'])[np.where(np.array(stats['gt_successes'])==0)[0]].sum()
    fn = np.array(stats['equals'])[np.where(np.array(stats['gt_successes'])==1)[0]].shape[0]- np.array(stats['equals'])[np.where(np.array(stats['gt_successes'])==1)[0]].sum()
    acc_per_class = {}

    for key, value in stats.items():
        if key=='equals' or key=='gt_objects' or key=='gt_successes':
            continue
        json_log[key] = value
    json_log['tp'] = tp
    json_log['tn'] = tn
    json_log['fp'] = fp
    json_log['fn'] = fn

    if len(stats['gt_objects']) > 0:
        for gt_object in stats['gt_objects']:
            acc_per_class[gt_object] = []
        for idx in len(range(stats['equals'])):
            acc_per_class[stats['gt_objects'][idx]].append(stats['equals'])
        for gt_object in stats['gt_objects']:
            json_log[f'acc_of_{gt_object}']= acc_per_class[gt_object].mean()

    out_path = os.path.join(output_dir, 'eval_log.json')
    json_log=dict_apply(json_log, lambda x: float(x))
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()