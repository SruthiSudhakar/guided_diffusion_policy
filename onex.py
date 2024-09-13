import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
from tqdm import tqdm
import pdb

ogdata = h5py.File('/proj/vondrick3/sruthi/robots/diffusion_policy/all_robomimic_data_combined.hdf5', 'r')
f = h5py.File('onex.hdf5', 'w')
datagrp = f.create_group('data')
datagrp.attrs['env_args'] = ogdata['data'].attrs['env_args']
index = 0
# for demo in ogdata['data']:
#     if ogdata['data'][demo]['states'].shape[0]==0:
#         print(demo,'help')
#     else:
#         demogrp = datagrp.create_group(f'demo_{index}')
#         index+=1
        
#         actionsdset = demogrp.create_dataset('actions', data = ogdata['data'][demo]['actions'])
#         statesdset = demogrp.create_dataset('states', data = ogdata['data'][demo]['states'])
    
#         obsgrp = demogrp.create_group('obs') 
#         for grp_name in ogdata['data'][demo]['obs']:
#             dset = obsgrp.create_dataset(grp_name, data = ogdata['data'][demo]['obs'][grp_name])

for demo in tqdm(ogdata['data']):
    if index >2 :
        continue
    if ogdata['data'][demo]['states'].shape[0]==0:
        print(demo,'help2')
    else:
        demogrp = datagrp.create_group(f'demo_{index}')
        index+=1
        
        actionsdset = demogrp.create_dataset('actions', data = ogdata['data'][demo]['actions'])
        statesdset = demogrp.create_dataset('states', data = ogdata['data'][demo]['states'])
    
        obsgrp = demogrp.create_group('obs') 
        for grp_name in ogdata['data'][demo]['obs']:
            dset = obsgrp.create_dataset(grp_name, data = ogdata['data'][demo]['obs'][grp_name])

datagrp.attrs['total'] = index
f.close()


'''
<KeysViewHDF5 ['actions', 'dones', 'next_obs', 'obs', 'rewards', 'states']>
<KeysViewHDF5 ['agentview_image', 'object', 'robot0_eef_pos', 'robot0_eef_quat', 
                'robot0_eef_vel_ang', 'robot0_eef_vel_lin', 'robot0_eye_in_hand_image', 
                'robot0_gripper_qpos', 'robot0_gripper_qvel', 'robot0_joint_pos', 
                'robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel']>

<KeysViewHDF5 ['actions', 'dones', 'next_obs', 'obs', 'rewards', 'states']>
<KeysViewHDF5 ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_eef_vel_ang', 
                'robot0_eef_vel_lin', 'robot0_eye_in_hand_image', 'robot0_gripper_qpos', 
                'robot0_gripper_qvel', 'robot0_joint_pos', 'robot0_joint_pos_cos', 
                'robot0_joint_pos_sin', 'robot0_joint_vel', 'sideview_image']>





'''