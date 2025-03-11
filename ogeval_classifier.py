"""
export LD_LIBRARY_PATH=:/home/sruthi/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=osmesa 
source /proj/vondrick3/sruthi/miniconda3/bin/activate
conda activate clonejgdrobodiff
cd /proj/vondrick3/sruthi/robots/diffusion_policy
export HYDRA_FULL_ERROR=1

python ../robocasa/robocasa/scripts/playback_dataset.py --dataset /proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-05-04-22-14-34_and_2024-05-07-07-40-21/demo_gentex_im128_randcams_new_images_train_no_kbpckt_first300.hdf5 --n 1 --use-actions

Usage:
python ogeval_classifier.py --checkpoint /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2025.02.20/imageonly_02.20.10.45.01_big_classifier_shuffle_higherlr/checkpoints/epoch=0009-val_loss=1.315 \
                --dataset_path "[\"/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-05-04-22-14-34_and_2024-05-07-07-40-21/demo_gentex_im128_randcams_new_images_val_kbpckt_firsthalf.hdf5\"]" \
                --device cuda:3 \
                --val \
                --train
python ogeval_classifier.py --checkpoint /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2025.02.20/imageonly_02.20.10.45.01_big_classifier_shuffle_higherlr/checkpoints/epoch=0009-val_loss=1.315 \
                --dataset_path "[\"/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-05-04-22-14-34_and_2024-05-07-07-40-21/demo_gentex_im128_randcams_new_images_train_no_kbpckt.hdf5\"]" \
                --device cuda:6 \
                --val \
                --train
python ogeval_classifier.py --checkpoint /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2025.02.20/imageonly_02.20.10.45.01_big_classifier_shuffle_higherlr/checkpoints/epoch=0009-val_loss=1.315 \
                --dataset_path "[\"/proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2025.02.15/imageonly_11.32.40_usegroupnorm/checkpoints/epoch=1000-val_loss=0.071/PnPSinkToCounter_mg_val_kbpctk_secondhalf_21984511/datafile.hdf5\"]" \
                --device cuda:7 \
                --val \
                --train

                
                
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
import ast

@click.command()
@click.option('-c', '--checkpoint', required=True)
@click.option('-dataset_path', '--dataset_path', required=False)
@click.option('-d', '--device', default='cuda:0')
@click.option('-balance_dataset', '--balance_dataset', is_flag=True)
@click.option('-train', '--train', is_flag=True)
@click.option('-val', '--val', is_flag=True)
def main(checkpoint, dataset_path, device, balance_dataset, train, val):
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
    cfg['task']['dataset_path'] = ast.literal_eval(dataset_path)
    cfg['task']['dataset']['dataset_path'] = ast.literal_eval(dataset_path)[0]
    cfg['task']['balance_dataset']=balance_dataset
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    device = torch.device(device)
    workspace.model.to(device)

    # trace = workspace.compute_hessian_trace()
    # total_params = sum(p.numel() for p in workspace.model.parameters() if p.requires_grad)
    # print('trace, total_params',trace, total_params)
    
    json_log = {}
    if train:
        stats = workspace.run_train_acc() 
        json_log['train'] ={}
        tp = np.array(stats['equals'])[np.where(np.array(stats['gt_successes'])==1)[0]].sum()
        tn = np.array(stats['equals'])[np.where(np.array(stats['gt_successes'])==0)[0]].sum()
        fp = np.array(stats['equals'])[np.where(np.array(stats['gt_successes'])==0)[0]].shape[0] - np.array(stats['equals'])[np.where(np.array(stats['gt_successes'])==0)[0]].sum()
        fn = np.array(stats['equals'])[np.where(np.array(stats['gt_successes'])==1)[0]].shape[0]- np.array(stats['equals'])[np.where(np.array(stats['gt_successes'])==1)[0]].sum()
        precision = tp/(tp+fp)
        acc_per_class = {}

        for key, value in stats.items():
            if key=='equals' or key=='gt_objects' or key=='gt_successes':
                continue
            json_log['train'][key] = value
        json_log['train']['tp'] = tp
        json_log['train']['tn'] = tn
        json_log['train']['fp'] = fp
        json_log['train']['fn'] = fn
        json_log['train']['precision'] = precision

        if len(stats['gt_objects']) > 0:
            for gt_object in stats['gt_objects']:
                acc_per_class[gt_object] = []
            for idx in len(range(stats['equals'])):
                acc_per_class[stats['gt_objects'][idx]].append(stats['equals'])
            for gt_object in stats['gt_objects']:
                json_log['train'][f'acc_of_{gt_object}']= acc_per_class[gt_object].mean()
    if val:
        stats = workspace.run_validation() 
        json_log['val'] = {}
        tp = np.array(stats['equals'])[np.where(np.array(stats['gt_successes'])==1)[0]].sum()
        tn = np.array(stats['equals'])[np.where(np.array(stats['gt_successes'])==0)[0]].sum()
        fp = np.array(stats['equals'])[np.where(np.array(stats['gt_successes'])==0)[0]].shape[0] - np.array(stats['equals'])[np.where(np.array(stats['gt_successes'])==0)[0]].sum()
        fn = np.array(stats['equals'])[np.where(np.array(stats['gt_successes'])==1)[0]].shape[0]- np.array(stats['equals'])[np.where(np.array(stats['gt_successes'])==1)[0]].sum()
        precision = tp/(tp+fp+1e-12)
        acc_per_class = {}

        for key, value in stats.items():
            if key=='equals' or key=='gt_objects' or key=='gt_successes':
                continue
            json_log['val'][key] = value
        json_log['val']['tp'] = tp
        json_log['val']['tn'] = tn
        json_log['val']['fp'] = fp
        json_log['val']['fn'] = fn
        json_log['val']['precision'] = precision

        if len(stats['gt_objects']) > 0:
            for gt_object in stats['gt_objects']:
                acc_per_class[gt_object] = []
            for idx in len(range(stats['equals'])):
                acc_per_class[stats['gt_objects'][idx]].append(stats['equals'])
            for gt_object in stats['gt_objects']:
                json_log['val'][f'acc_of_{gt_object}']= acc_per_class[gt_object].mean()

    out_path = os.path.join(output_dir, 'eval_log.json')
    json_log=dict_apply(json_log, lambda x: float(x))
    json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)
    
if __name__ == '__main__':
    main()