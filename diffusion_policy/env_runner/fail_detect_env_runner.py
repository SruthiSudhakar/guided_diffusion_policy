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
import cv2
import itertools
import pdb
import json
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES, OBJ_GROUPS
import logging; logging.disable(logging.CRITICAL)
import torchvision.transforms as T

from diffusion_policy.failure_detection.UQ_baselines.CFM.net_CFM import get_unet
import diffusion_policy.failure_detection.UQ_baselines.data_loader as data_loader

def adjust_xshape(x, in_dim):
    total_dim = x.shape[1]
    # Calculate the padding needed to make total_dim a multiple of in_dim
    remain_dim = total_dim % in_dim
    if remain_dim > 0:
        pad = in_dim - remain_dim
        total_dim += pad
        x = torch.cat([x, torch.zeros(x.shape[0], pad, device=x.device)], dim=1)
    # Calculate the padding needed to make (total_dim // in_dim) a multiple of 4
    reshaped_dim = total_dim // in_dim
    if reshaped_dim % 4 != 0:
        extra_pad = (4 - (reshaped_dim % 4)) * in_dim
        x = torch.cat([x, torch.zeros(x.shape[0], extra_pad, device=x.device)], dim=1)
    return x.reshape(x.shape[0], -1, in_dim)


def logpZO_UQ(baseline_model, observation, action_pred = None, task_name = 'square'):
    observation = observation
    in_dim = 7
    observation = adjust_xshape(observation, in_dim)
    if action_pred is not None:
        action_pred = action_pred
        observation = torch.cat([observation, action_pred], dim=1)
    with torch.no_grad():
        timesteps = torch.zeros(observation.shape[0], device=observation.device)
        pred_v = baseline_model(observation, timesteps)
        observation = observation + pred_v
        logpZO = observation.reshape(len(observation), -1).pow(2).sum(dim=-1)
    return logpZO


def create_env(env_meta, shape_meta, object, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=enable_render,
        use_image_obs=enable_render, 
        object=object,
    )
    return env

def plot_and_save_images(obs, output_dir, timestep = 0, start=0):
    """
    Plots and saves images from a tensor of shape (A, 3, W, H).
    
    Parameters:
    tensor (torch.Tensor): Input tensor of shape (A, 3, W, H) with integer values between 0 and 255.
    save_dir (str): Directory where images will be saved.
    """
    # Ensure the save directory exists
    # Convert the tensor to a format suitable for plotting
    transform = T.ToPILImage()
    for key, img_tensor in obs.items():
        if len(img_tensor.shape) < 5:
            continue
        img_tensor = torch.from_numpy(img_tensor[:, 0, :, :, :])
        for i, img_tensor_ in enumerate(img_tensor):
            save_dir = os.path.join(output_dir, 'media', f'test_{start+i}')
            os.makedirs(save_dir, exist_ok=True)
            img = transform(img_tensor_)
            # Save the image
            img.save(os.path.join(save_dir, f'{key}_t={timestep}.png'))


from transformers import CLIPTokenizer, CLIPModel
import torch
# Load the CLIP model and tokenizer
clip_model_name = "openai/clip-vit-base-patch32"  # You can choose other models if desired
clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
clip_model = CLIPModel.from_pretrained(clip_model_name)

import torch
from torch.nn.functional import pairwise_distance


class FailDetectRunnerEval(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            shape_meta:dict,
            n_train=0,
            n_train_vis=0,
            train_start_idx=0,
            n_test=0,
            n_test_vis=0,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='agentview_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            object=None,
            save_stuff=False,
            change_test_textures=False,
            change_test_objects=False,
            change_test_object_instances=False,
            init_state_none=False,
            debug=False,
            choose_sample=False,
            num_samples=10,
            start_rollout_from_state=0,
            show_classifier_scores=False,
            adaptive_guidance='None',
            decode_first=True,
            take_first_n_train_samples=False,
            start_sampling=0,
            end_sampling=27,
            specific_train_exs=[],
            prompt_with_video=True,
            additional_steps=0,
            save_score_network_path='',
        ):
        super().__init__(output_dir)
        n_obs_steps=8 if save_stuff else n_obs_steps
        self.object = object
        if n_envs is None:
            n_envs = n_train + n_test

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        if self.object:
            env_meta['env_name'] = 'LiftOtherObjects'
        # disable object state observation
        env_meta['env_kwargs']['use_object_obs'] = False

        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def env_fn():
            robomimic_env = create_env(
                env_meta=env_meta, 
                shape_meta=shape_meta,
                object=self.object,
            )
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        env_model=None,
                        ep_meta=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
        
        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    shape_meta=shape_meta,
                    object=self.object,
                    enable_render=False
                )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        env_model=None,
                        ep_meta=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        clip_model.to(device)
        train_embeddings_list =[]
        batch_size = 128  # You can adjust this based on your memory capacity
        # first compute clip embeddings quickly
        if len(specific_train_exs)>0:
            with h5py.File(dataset_path, 'r') as f:
                texts_batch = []
                batch_indices = []
                for ex in specific_train_exs:
                    ep_meta = f[f'data/demo_{ex}'].attrs.get("ep_meta", None)
                    text = json.loads(ep_meta)['lang']
                    texts_batch.append(text)
                inputs = clip_tokenizer(texts_batch, padding=True, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                with torch.no_grad():
                    batch_embeddings = clip_model.get_text_features(**inputs)  # Shape: (batch_size, embedding_dim)
                    batch_embeddings = batch_embeddings.cpu().numpy()  # Move back to CPU and convert to NumPy
        
                # Append the embeddings to the list
                train_embeddings_list.extend(batch_embeddings)  # Collect all the embeddings
        else:
            with h5py.File(dataset_path, 'r') as f:
                for i in tqdm.tqdm(range(0, n_train, batch_size)):  # Process in batches
                    texts_batch = []
                    batch_indices = []
                    for j in range(batch_size):
                        idx = (i + j) % len(f['data'])
                        train_idx = train_start_idx + idx
                        ep_meta = f[f'data/demo_{train_idx}'].attrs.get("ep_meta", None)
                        text = json.loads(ep_meta)['lang']
                        texts_batch.append(text)
                        batch_indices.append(train_idx)            
                    inputs = clip_tokenizer(texts_batch, padding=True, return_tensors="pt")
                    inputs = {key: value.to(device) for key, value in inputs.items()}
                    with torch.no_grad():
                        batch_embeddings = clip_model.get_text_features(**inputs)  # Shape: (batch_size, embedding_dim)
                        batch_embeddings = batch_embeddings.cpu().numpy()  # Move back to CPU and convert to NumPy
            
                    # Append the embeddings to the list
                    train_embeddings_list.extend(batch_embeddings)  # Collect all the embeddings

        with h5py.File(dataset_path, 'r') as f:
            embedding_idx=0
            if len(specific_train_exs)>0:
                list_of_demos = range(len(specific_train_exs))
            else:
                list_of_demos = range(n_train)

            for i in tqdm.tqdm(list_of_demos):
                if len(specific_train_exs)>0:
                    train_idx = int(specific_train_exs[i])
                else:
                    train_idx = train_start_idx + (i % len(f['data']))
                enable_render = True
                init_state = f[f'data/demo_{train_idx}/states'][start_rollout_from_state]
                env_model = f[f'data/demo_{train_idx}'].attrs["model_file"]
                ep_meta = f[f'data/demo_{train_idx}'].attrs.get("ep_meta",None)
                ep_meta=ep_meta.replace('/proj/vondrick3/sruthi/robots/robocasa/robocasa/models/assets/generative_textures/','')
                language_goal_embedding = train_embeddings_list[embedding_idx]
                embedding_idx+=1

                def init_fn(env, init_state=init_state, env_model=env_model, ep_meta=ep_meta,
                    enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    env.env.env.init_state = None
                    env.env.env.env_model = None
                    env.env.env.ep_meta = None
                    env.env.env.language_goal = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'trainmedia', str(train_idx) + "_" + str(train_start_idx + i) + "_" + wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicImageWrapper)
                    env.env.env.init_state = init_state
                    env.env.env.env_model = env_model
                    env.env.env.ep_meta = ep_meta
                    env.env.env.language_goal = language_goal_embedding
                    env.env.env.env.env.hard_reset=True
                    env.env.env.reset()
                    # env.env.env.env.env.hard_reset=False

                env_seeds.append(str(train_idx) + "_" + str(train_start_idx + i))
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))

        test_embeddings_list =[]
        # first compute clip embeddings quickly
        with h5py.File(dataset_path, 'r') as f:
            for i in tqdm.tqdm(range(0, n_test, batch_size)):  # Process in batches
                texts_batch = []
                batch_indices = []
                for j in range(batch_size):
                    idx = (i + j) % len(f['data'])
                    test_idx = train_start_idx + idx
                    ep_meta = f[f'data/demo_{test_idx}'].attrs.get("ep_meta", None)
                    text = json.loads(ep_meta)['lang']
                    texts_batch.append(text)
                    batch_indices.append(test_idx)            
                inputs = clip_tokenizer(texts_batch, padding=True, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                with torch.no_grad():
                    batch_embeddings = clip_model.get_text_features(**inputs)  # Shape: (batch_size, embedding_dim)
                    batch_embeddings = batch_embeddings.cpu().numpy()  # Move back to CPU and convert to NumPy
                test_embeddings_list.extend(batch_embeddings)  # Collect all the embeddings
        #test
        with h5py.File(dataset_path, 'r') as f:
            embedding_idx=0
            for i in tqdm.tqdm(range(n_test)):
                seed = test_start_seed + i
                enable_render = i < n_test_vis
                test_idx = train_start_idx + (i % len(f['data']))
                init_state = f[f'data/demo_{test_idx}/states'][start_rollout_from_state]
                env_model = f[f'data/demo_{test_idx}'].attrs["model_file"]
                ep_meta = json.loads(f[f'data/demo_{test_idx}'].attrs.get("ep_meta",None))
                if change_test_textures:
                    if 'gen_textures' in ep_meta:
                        ep_meta['gen_textures']={}
                ep_meta = json.dumps(ep_meta)
                ep_meta=ep_meta.replace('/proj/vondrick3/sruthi/robots/robocasa/robocasa/models/assets/generative_textures/','')
                language_goal_embedding = test_embeddings_list[embedding_idx]
                embedding_idx+=1

                def init_fn(env, init_state=init_state, env_model=env_model, ep_meta=ep_meta,
                    enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    env.env.env.init_state = None
                    env.env.env.env_model = None
                    env.env.env.ep_meta = None
                    env.env.env.language_goal = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'testmedia', str(test_idx) + "_" + str(train_start_idx + i) + "_" + wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicImageWrapper)
                    if init_state_none:
                        env.env.env.init_state = None
                    else:
                        env.env.env.init_state = init_state
                    if not change_test_objects and not change_test_object_instances:
                        env.env.env.env_model = env_model
                    env.env.env.ep_meta = ep_meta
                    env.env.env.language_goal = language_goal_embedding
                    env.seed(seed)
                    env.env.env.env.env.hard_reset=True
                    env.env.env.reset()
                    # if change_test_textures:
                    #     print('set the textures so it doesnt change going forwards')
                    # env.env.env.env.env.hard_reset=True

                env_seeds.append(str(test_idx) + "_" + str(train_start_idx + i))
                env_prefixs.append('test/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        
        
        
        self.debug=debug
        if self.debug:
            env = SyncVectorEnv(env_fns)
            """ from scipy.spatial.transform import Rotation as R
            temp=env.envs[0].env.env.env.env._observables
            rot = R.from_quat(temp['robot0_base_quat'])
            R_base_to_world = rot.as_matrix()
            eef_offset_world = R_base_to_world @ temp['robot0_base_to_eef_pos']
            assert np.allclose(temp['robot0_eef_pos'] , temp['robot0_base_pos']+eef_offset_world, atol=1e-6) """
        else:
            env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)        

        if save_stuff:
            self.data_file= h5py.File(self.output_dir+'/datafile.hdf5', 'w')
            self.datagrp =  self.data_file.create_group('data')
            self.datagrp.attrs['ogdataset'] = self.output_dir
            self.datagrp.attrs['env_args'] = json.dumps(env_meta)
        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.output_dir = output_dir
        self.save_stuff = save_stuff
        self.show_classifier_scores=show_classifier_scores
        self.adaptive_guidance=adaptive_guidance
        self.decode_first=decode_first
        self.choose_sample=choose_sample
        self.num_samples=num_samples
        self.start_rollout_from_state=start_rollout_from_state
        self.start_sampling=start_sampling
        self.end_sampling=end_sampling
        self.prompt_with_video=prompt_with_video
        self.additional_steps=additional_steps
        self.render_obs_key=render_obs_key

        ## Get logpZO
        input_dim = 7
        net = get_unet(input_dim)
        ckpt = torch.load(save_score_network_path)
        net.load_state_dict(ckpt['model'])
        net.eval()

        # move net to device
        self.score_network = net

    def run(self, policy: BaseImagePolicy, classifier_processor=None, classifier=None, grad_steps=None, guidance_scale=None, guided_towards=None, modify=False):
        device = policy.device
        self.score_network.to(device)
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns) #how many to run asynchronusly at once
        n_inits = len(self.env_init_fn_dills) #how many examples to run total
        n_chunks = math.ceil(n_inits / n_envs) #total/per run = how many times to run it
        # print('N STUFF', n_envs, n_inits, n_chunks)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_logpZO = [None] * n_inits # Stores logpZO for all rollout across all steps
        all_eyeinhand_image_obs = [None] * n_inits # Stores logpZO for all rollout across all steps
        all_leftside_image_obs = [None] * n_inits # Stores logpZO for all rollout across all steps
        all_rightside_image_obs = [None] * n_inits # Stores logpZO for all rollout across all steps

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            language_goal = env.call('get_env_metadata')[this_local_slice]
            language_goal = [json.loads(x['ep_meta'])['lang'] for x in language_goal]
            done = False
            logpZO_local_slices = []
            reward_local_slices = []
            eyeinhand_local_slice = []
            rightside_local_slice = []
            leftside_local_slice = []
            modify_again = True if modify else False
            if modify or modify_again:
                assert True==False
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))

                # run policy
                with torch.no_grad():
                    action_dict, classifier_action_pred = policy.predict_action(obs_dict)
                
                # compute FD metrics
                baseline_metric = logpZO_UQ(self.score_network, action_dict['global_cond'])
                logpZO_local_slices.append(baseline_metric)

                #store imgobs_local_slices
                eyeinhand_local_slice.append(obs_dict['robot0_eye_in_hand_image'].detach().cpu())
                rightside_local_slice.append(obs_dict['robot0_agentview_right_image'].detach().cpu())
                leftside_local_slice.append(obs_dict['robot0_agentview_left_image'].detach().cpu())

                # device_transfer
                np_action_dict = dict_apply(action_dict,lambda x: x.detach().to('cpu').numpy())
                action = np_action_dict['action']

                if not np.all(np.isfinite(action)):
                    raise RuntimeError("Nan or Inf action")

                # step env
                env_action = action
                if self.abs_action:
                    pdb.set_trace()
                    print('UNDOING TRANSFORM ACTION')
                    env_action = self.undo_transform_action(action)

                add_on = np.tile([0., -0.,  0.,  0., -1.], (env_action.shape[0], env_action.shape[1], 1))
                extended_env_action = np.concatenate((env_action, add_on), axis=-1)
                obs, reward, done, info = env.step(extended_env_action)
                reward_local_slices.append(torch.from_numpy(reward))
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(extended_env_action.shape[1])
                if self.debug:
                    if env_step_index==10:
                        done=True 
               
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            
            logpZO_local_slices = torch.stack(logpZO_local_slices, dim=1) # (n_envs, max_steps // T_p)
            all_logpZO[this_global_slice] = logpZO_local_slices

            eyeinhand_local_slice = torch.stack(eyeinhand_local_slice, dim=1)
            rightside_local_slice = torch.stack(rightside_local_slice, dim=1)
            leftside_local_slice = torch.stack(leftside_local_slice, dim=1)
            all_eyeinhand_image_obs[this_global_slice] = eyeinhand_local_slice
            all_rightside_image_obs[this_global_slice] = rightside_local_slice
            all_leftside_image_obs[this_global_slice] = leftside_local_slice
            

        # clear out video buffer
        _ = env.reset()
        
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            def helper(input, i):
                return [f'{float(x.item()):.4f}' for x in input[i]]
            # max reward, log_p_cond, log_p_marginal, reward
            log_data[prefix+f'sim_max_reward_{seed}'] = [max_reward, 
                                                         # Baseline
                                                         '/'.join(map(str, helper(all_logpZO, i))),
                                                         ]
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value
        allimageobs = {
            'eyeinhand': all_eyeinhand_image_obs,
            'leftside': all_leftside_image_obs,
            'rightside': all_rightside_image_obs,
            'logdata': log_data
        }
        return log_data, allimageobs

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
