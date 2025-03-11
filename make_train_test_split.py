import os, sys
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
from tqdm import tqdm


''' CREATE 1 DATASET '''
def create_dataset_split(current_dataset, object_list, split, base_path):
    savepath = base_path[:-5]+f'_{split}.hdf5'
    print(f'saving {split} to ', savepath)
    f = h5py.File(savepath, 'w')
    datagrp = f.create_group('data')
    datagrp.attrs['ogdataset'] = base_path
    datagrp.attrs['env_args'] = current_dataset['data'].attrs['env_args']
    # total=0
    # for i in paths:
    #     temp = h5py.File(i,'r')
    #     total+=temp['data'].attrs['total']
    # datagrp.attrs['total']=total

    count=0
    for demo in tqdm(current_dataset['data']):
        object_is_in_list = False
        for objs in json.loads(current_dataset['data'][demo].attrs['ep_meta'])['object_cfgs']:
            if objs['name']=='obj':
                if objs['info']['cat'] in object_list:
                    # print('OBJECT FOUND', objs['info']['cat'])
                    object_is_in_list=True
        if object_is_in_list:
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
            if 'action_dict' in current_dataset['data'][demo]:
                actiondictgrp = demogrp.create_group('action_dict') 
                for grp_name in current_dataset['data'][demo]['action_dict']:
                    dset = actiondictgrp.create_dataset(grp_name, data = current_dataset['data'][demo]['action_dict'][grp_name])
            print('demo done', demo, count)
            count += 1

    print('dataset done', savepath)
    f.close()

# Open HDF5 file and write in the data_dict structure and info
base_path = sys.argv[1]
all_objects={}
data=h5py.File(base_path, 'r')
for x in data['data']:
    try:
        for objs in json.loads(data['data'][x].attrs['ep_meta'])['object_cfgs']:
            if objs['name']=='obj':
                all_objects[objs['info']['cat']] = all_objects.get(objs['info']['cat'],0) + 1
    except:
        print('issue:', root, x)

all_objects = dict(sorted(all_objects.items(), key=lambda item: item[1]))
print(json.dumps(all_objects, indent=4))
val_objects = list(all_objects.keys())[:5]
train_objects = list(all_objects.keys())[5:]
print('val objects', val_objects)
print('train objects', train_objects)
create_dataset_split(data, train_objects, 'train', base_path)
create_dataset_split(data, val_objects, 'val', base_path)