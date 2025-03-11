import os
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Video
import imageio
from statistics import mean
import robomimic.utils.file_utils as FileUtils
from PIL import Image, ImageDraw, ImageFont
import zarr
from diffusion_policy.common.replay_buffer import ReplayBuffer
from filelock import FileLock
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
import pdb
from tqdm import tqdm
import xml.etree.ElementTree as ET

'''FIRST split the mg dataset into train and val based on objects. these are the objects we want to take OUT OF the train set and put INTO val set: avocado,bell pepper,corn,kiwi,tangerine '''

# Open HDF5 file and write in the data_dict structure and info
base_path = '/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/mg/2024-05-04-22-14-34_and_2024-05-07-07-40-21/demo_gentex_im128_randcams_new_images.hdf5'
current_dataset=h5py.File(base_path, 'r')

savepath = base_path[:-5]+f'_train_no_kbpckt.hdf5'
f = h5py.File(savepath, 'w')
datagrp = f.create_group('data')
datagrp.attrs['ogdataset'] = base_path
datagrp.attrs['env_args'] = current_dataset['data'].attrs['env_args']
count=0
for demo in current_dataset['data']:
    for objs in json.loads(current_dataset['data'][demo].attrs['ep_meta'])['object_cfgs']:
        if objs['name']=='obj':
            object_id=objs['info']['cat']
    if object_id not in ['avocado','bell_pepper','corn','kiwi','tangerine']:
        demogrp = datagrp.create_group('demo_'+str(count))
        for k in current_dataset['data'][demo].attrs.keys():
            demogrp.attrs[k] = current_dataset['data'][demo].attrs[k]
        demogrp.attrs['og_demo_id']=demo
        actionsdset = demogrp.create_dataset('actions', data = current_dataset['data'][demo]['actions'])
        rewardsdset = demogrp.create_dataset('rewards', data = current_dataset['data'][demo]['rewards'])
        statesdset = demogrp.create_dataset('states', data = current_dataset['data'][demo]['states'])
        donesdset = demogrp.create_dataset('dones', data = current_dataset['data'][demo]['dones'])
        obsgrp = demogrp.create_group('obs') 
        for grp_name in current_dataset['data'][demo]['obs']:
            dset = obsgrp.create_dataset(grp_name, data = current_dataset['data'][demo]['obs'][grp_name])
        actiondictgrp = demogrp.create_group('action_dict') 
        for grp_name in current_dataset['data'][demo]['action_dict']:
            dset = actiondictgrp.create_dataset(grp_name, data = current_dataset['data'][demo]['action_dict'][grp_name])
        print('demo done', demo, count)
        count += 1

print('dataset done', savepath)
f.close()