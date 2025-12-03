import os
import pickle
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import dill
import random
import time
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
import pdb
import json
import time
import logging; logging.disable(logging.CRITICAL)
import logging
import numpy as np

import subprocess
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import CLIPTokenizer, CLIPModel
import torch
# Load the CLIP model and tokenizer
# clip_model_name = "openai/clip-vit-base-patch32"  # You can choose other models if desired
# clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
# clip_model = CLIPModel.from_pretrained(clip_model_name)

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

class RobocasaRobomimicImageRunnerEval(BaseImageRunner):
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
                # Handle username variations (sruthisudhakar vs sruthi.sudhakar)
                ep_meta = ep_meta.replace('/home/sruthisudhakar/', '/app/')
                ep_meta = ep_meta.replace('/home/sruthi.sudhakar/', '/app/')
                ep_meta = ep_meta.replace('/app/guided_diffusion_policy/externals2/','/app/externals/')
                ep_meta = ep_meta.replace('/app/guided_diffusion_policy/externals2/','/app/externals/')
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
                # if change_test_objects:
                #     for obj in ep_meta['object_cfgs']:
                #         if obj['name']=='obj':
                #             temp_obj_info = obj.pop('info',None)
                #             obj.pop('cookable',None)
                #             obj.pop('microwavable',None)
                #             obj.pop('washable',None)
                #             obj.pop('freezable',None)
                #             env_name=env_meta["env_name"]
                #             obj['exclude_obj_groups']=f"{env_name}_seen"
                #             print('replacing',  temp_obj_info['cat'],'excluding these groups: ', f"{env_name}_seen")
                #             # new_object = OBJ_GROUPS[f"{env_name}_unseen"][obj['info']['cat']]
                #             # obj['info']['mjcf_path'] = new_object
                #             # obj['info']['cat'] = new_object.split('/')[-3]
                #             # print('replacing',obj['info']['cat']  ,'with',new_object)

                # if change_test_object_instances:
                #     for obj in ep_meta['object_cfgs']:
                #         if obj['name']=='obj':
                #             temp_info = obj.pop('info',None)
                #             obj['split']='B'
                #             obj['obj_groups']=temp_info['cat']
                #             # print('replacing with new instance',temp_info['cat'] )

                ep_meta = json.dumps(ep_meta)
                # Handle username variations (sruthisudhakar vs sruthi.sudhakar)
                ep_meta = ep_meta.replace('/home/sruthisudhakar/', '/app/')
                ep_meta = ep_meta.replace('/home/sruthi.sudhakar/', '/app/')
                ep_meta = ep_meta.replace('/app/guided_diffusion_policy/externals2/','/app/externals/')
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

    def sample_n_diverse_actions(self, obs_dict, policy, device, n_samples=None):
        """
        Sample N diverse actions from the policy at the current state.

        Args:
            obs_dict: Current observation dictionary (for single env)
            policy: The diffusion policy model
            device: Torch device
            n_samples: Number of diverse samples to generate (default: self.num_samples)

        Returns:
            numpy array of shape (n_samples, action_horizon, action_dim)
        """
        # Reseed for stochastic sampling
        random_seed = int((time.time() * 1000000) % (2**31 - 1))
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

        # Repeat observations for batch sampling
        reshaped_obs_dict = dict_apply(obs_dict, lambda x: x.repeat_interleave(self.num_samples, dim=0))
        action_dict = policy.predict_action(reshaped_obs_dict)[0] #action outputs are batch_sizex8x7
        actions = action_dict['action_pred'].view(-1, self.num_samples, 16, 7).detach().to('cpu').numpy()


        return actions.numpy()

    def get_video_path_for_node(self, env_idx, path_indices, seed):
        """
        Get the video path for a specific node without initializing recording.

        Args:
            env_idx: Index of the environment
            path_indices: List of action indices taken so far
            seed: Environment seed for naming

        Returns:
            video_path: Path where video will be saved
        """
        path_str = '_'.join(map(str, path_indices)) if path_indices else 'root'
        video_dir = pathlib.Path(self.output_dir).joinpath(
            f'env_{env_idx}',
            f'path_{path_str}'
        )
        video_dir.mkdir(parents=True, exist_ok=True)
        video_filename = f'video_{seed}.mp4'
        return str(video_dir / video_filename)

    def save_frames_to_video(self, frames, video_path, fps=10):
        """
        Save accumulated frames to a video file.

        Args:
            frames: List of frame arrays (numpy arrays)
            video_path: Path to save the video
            fps: Frames per second
        """
        if not frames:
            print(f"Warning: No frames to save for {video_path}")
            return

        try:
            # Validate frames
            for i, frame in enumerate(frames):
                if not isinstance(frame, np.ndarray):
                    print(f"Error: Frame {i} is not a numpy array, it's {type(frame)}")
                    return
                if frame.dtype != np.uint8:
                    print(f"Warning: Frame {i} has dtype {frame.dtype}, converting to uint8")
                    frames[i] = frame.astype(np.uint8)

            from diffusion_policy.real_world.video_recorder import VideoRecorder
            recorder = VideoRecorder.create_h264(
                fps=fps,
                codec='h264',
                input_pix_fmt='rgb24',
                crf=22,
                thread_type='FRAME',
                thread_count=1
            )
            recorder.start(video_path)
            for frame in frames:
                recorder.write_frame(frame)
            recorder.stop()
            print(f"  Saved video with {len(frames)} frames to {video_path}")
        except Exception as e:
            print(f"Warning: Could not save video to {video_path}: {e}")
            import traceback
            traceback.print_exc()

    def initialize_video_for_path(self, env, env_idx, path_indices, seed):
        """
        Initialize video recording for a specific path in the tree.

        Args:
            env: The environment (vectorized env)
            env_idx: Index of the environment
            path_indices: List of action indices taken so far (e.g., [0, 2, 1])
            seed: Environment seed for naming

        Returns:
            video_path: Path where video will be saved
        """
        # Create nested directory structure
        path_str = '_'.join(map(str, path_indices)) if path_indices else 'root'
        video_dir = pathlib.Path(self.output_dir).joinpath(
            f'env_{env_idx}',
            f'path_{path_str}'
        )
        video_dir.mkdir(parents=True, exist_ok=True)

        video_filename = f'video_{seed}.mp4'
        video_path = str(video_dir / video_filename)

        # Set video path in the environment's video recorder
        # The env structure is: MultiStepWrapper -> VideoRecordingWrapper -> RobomimicImageWrapper
        if hasattr(env, 'envs'):
            # SyncVectorEnv case - direct access
            if hasattr(env.envs[env_idx], 'env') and hasattr(env.envs[env_idx].env, 'file_path'):
                env.envs[env_idx].env.video_recoder.stop()
                env.envs[env_idx].env.file_path = video_path
                env.envs[env_idx].env.video_recoder.start()
        else:
            # AsyncVectorEnv case - use call_each
            # Create args list: video_path for target env, None for others
            args_list = [(video_path,) if i == env_idx else (None,) for i in range(env.num_envs)]
            try:
                env.call_each('set_video_path_and_start', args_list=args_list)
            except Exception as e:
                print(f"Warning: Could not set video path for AsyncVectorEnv: {e}")

        return video_path

    def execute_single_step(self, env, action, env_idx):
        """
        Execute a single action on a single environment.

        Args:
            env: The environment
            action: Action array of shape (action_horizon, action_dim)
            env_idx: Environment index

        Returns:
            obs: New observation
            reward: Reward from step
            done: Whether episode is done
            info: Additional info
        """
        # Add gripper action dimensions
        add_on = np.tile([0., -0., 0., 0., -1.], (action.shape[0], 1))
        extended_action = np.concatenate((action, add_on), axis=-1)

        # Expand to batch dimension for vector env compatibility
        extended_action = np.expand_dims(extended_action, 0)

        # Step environment
        obs, reward, done, info = env.step(extended_action)
        return obs, reward, done, info

    def compute_tree_statistics(self, all_paths):
        """
        Compute aggregate statistics from all explored paths.

        Args:
            all_paths: List of path dictionaries with keys:
                      'path_indices', 'depth', 'rewards', 'success', 'video_path'

        Returns:
            Dictionary with aggregate statistics
        """
        if not all_paths:
            return {
                'total_paths': 0,
                'success_rate': 0.0,
                'avg_depth': 0.0,
                'avg_max_reward': 0.0
            }

        total_paths = len(all_paths)
        successes = sum(1 for p in all_paths if p['success'])
        success_rate = successes / total_paths
        avg_depth = np.mean([p['depth'] for p in all_paths])
        avg_max_reward = np.mean([np.max(p['rewards']) for p in all_paths])

        # Statistics by depth
        depths = {}
        for path in all_paths:
            depth = path['depth']
            if depth not in depths:
                depths[depth] = {'count': 0, 'successes': 0, 'rewards': []}
            depths[depth]['count'] += 1
            if path['success']:
                depths[depth]['successes'] += 1
            depths[depth]['rewards'].append(np.max(path['rewards']))

        success_rate_by_depth = {
            d: depths[d]['successes'] / depths[d]['count']
            for d in depths
        }
        avg_reward_by_depth = {
            d: np.mean(depths[d]['rewards'])
            for d in depths
        }
        branches_per_depth = {
            d: depths[d]['count']
            for d in depths
        }

        return {
            'total_paths': total_paths,
            'success_rate': float(success_rate),
            'avg_depth': float(avg_depth),
            'avg_max_reward': float(avg_max_reward),
            'success_rate_by_depth': {int(k): float(v) for k, v in success_rate_by_depth.items()},
            'avg_reward_by_depth': {int(k): float(v) for k, v in avg_reward_by_depth.items()},
            'branches_per_depth': {int(k): int(v) for k, v in branches_per_depth.items()}
        }

    def save_path_results(self, env_idx, path_data, seed):
        """
        Save results for a single path.

        Args:
            env_idx: Environment index
            path_data: Dictionary with path information
            seed: Environment seed
        """
        path_str = '_'.join(map(str, path_data['path_indices'])) if path_data['path_indices'] else 'root'
        result_dir = pathlib.Path(self.output_dir).joinpath(
            f'env_{env_idx}',
            f'path_{path_str}'
        )
        result_dir.mkdir(parents=True, exist_ok=True)

        # Save path metadata
        metadata = {
            'path_indices': path_data['path_indices'],
            'depth': path_data['depth'],
            'success': bool(path_data['success']),  # Convert numpy bool_ to native Python bool
            'max_reward': float(np.max(path_data['rewards'])),
            'all_rewards': [float(r) for r in path_data['rewards']],
            'video_path': path_data['video_path'],
            'seed': seed
        }

        metadata_path = result_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def explore_action_tree_dfs(self, env, env_idx, save_env_idx, policy, device, current_obs,
                                current_depth, path_indices, seed, max_depth, accumulated_frames=None):
        """
        Recursively explore the action tree using depth-first search.

        Args:
            env: The environment (vector env with single active env)
            env_idx: Index of this environment
            policy: Diffusion policy model
            device: Torch device
            current_obs: Current observation dictionary
            current_depth: Current depth in the tree
            path_indices: List of action indices taken to reach this node
            seed: Environment seed
            accumulated_frames: List of frames accumulated from root to current node
            max_depth: Maximum depth to explore (self.max_steps)

        Returns:
            List of completed path dictionaries
        """
        # Initialize frame accumulator at root
        if accumulated_frames is None:
            accumulated_frames = []

        # Check if we've reached maximum depth
        if current_depth >= max_depth:
            print(f"  Path {path_indices}: Reached max depth {max_depth}")
            return [{
                'path_indices': path_indices.copy(),
                'depth': current_depth,
                'rewards': [],
                'success': False,
                'video_path': None,
                'termination_reason': 'max_depth',
                'frames': accumulated_frames.copy()
            }]

        # Get current state for potential restoration (both physics and wrapper state)
        current_state = env.call('get_env_state')[env_idx]

        # Save wrapper's internal buffers (obs, reward, done, etc.) before branching
        # Note: obs is a deque, reward/done/grasps are lists
        from collections import deque
        wrapper_obs = env.call('get_attr', 'obs')[env_idx].copy()  # deque.copy()
        wrapper_reward = env.call('get_attr', 'reward')[env_idx].copy()
        wrapper_done = env.call('get_attr', 'done')[env_idx].copy()
        wrapper_grasps = env.call('get_attr', 'grasps')[env_idx].copy()
        # Info is more complex (defaultdict of deques), handle separately if needed

        # Save current frame buffer state before branching
        saved_frame_count = len(accumulated_frames)

        # Sample N diverse actions
        obs_dict = dict_apply(current_obs, lambda x: torch.from_numpy(x).unsqueeze(0).to(device=device))
        actions = self.sample_n_diverse_actions(obs_dict, policy, device, n_samples=self.num_samples)
        actions = actions[0]  # Remove batch dimension (was added for single env)
        print(f"  Depth {current_depth}, Path {path_indices}: Branching into {len(actions)} actions")

        all_completed_paths = []

        # Explore each action branch
        for action_idx in range(len(actions)):
            action = actions[action_idx]
            current_path = path_indices + [action_idx]

            print(f"  Exploring action {action_idx} at depth {current_depth}, path: {current_path}")

            # Execute action and render frames
            action_to_execute = action[:8]
            obs, reward, done, info = self.execute_single_step(env, action_to_execute, env_idx)
            pdb.set_trace()
            reward = np.max(reward)
            reward = np.max(done)
            # Manually render frames after execution
            step_frames = []
            try:
                if hasattr(env, 'envs'):
                    # SyncVectorEnv - direct access through wrapper chain
                    # Structure: MultiStepWrapper -> VideoRecordingWrapper -> RobomimicImageWrapper -> robosuite env
                    frames = env.envs[env_idx].get_frames(mode='rgb_array')
                else:
                    # AsyncVectorEnv - use call_each to get frames from target env only
                    # Pass None for non-target envs to avoid unnecessary rendering
                    frames = env.call_each('get_frames',
                                         args_list=[('rgb_array',) if i == env_idx else (None,) for i in range(env.num_envs)])[env_idx]

                # Skip if frames is None (shouldn't happen for target env but check anyway)
                if frames is None:
                    print(f"Warning: get_frames returned None")
                    frame = None
                # Concatenate camera views horizontally if multiple views
                elif isinstance(frames, (list, tuple)) and len(frames) > 0:
                    # Make sure frames are numpy arrays
                    frame = np.concatenate(frames, axis=1)
                elif isinstance(frames, np.ndarray):
                    frame = frames
                else:
                    print(f"Warning: Unexpected frame type: {type(frames)}, value: {frames}")
                    frame = None

                if frame is not None and isinstance(frame, np.ndarray):
                    step_frames.append(frame)
            except Exception as e:
                print(f"Warning: Could not render frame: {e}")
                import traceback
                traceback.print_exc()

            # Add rendered frames to accumulated buffer
            accumulated_frames.extend(step_frames)
            if len(step_frames) > 0:
                print(f"  Captured {len(step_frames)} frames (total accumulated: {len(accumulated_frames)})")
            else:
                print(f"  Warning: No frames captured at this step")

            # Check if episode is complete (success or failure)
            if done:
                # Create video path for this leaf (will be used if this is a terminal node)
                video_path = self.get_video_path_for_node(save_env_idx, current_path, seed)
                # Get all rewards up to this point
                all_rewards = env.call('get_attr', 'reward')[env_idx]
                success = reward > 0.99  # Task success threshold

                print(f"  Path {current_path}: DONE at depth {current_depth+1}, success={success}, reward={reward}")
                # Save all accumulated frames to video for this terminal path
                self.save_frames_to_video(accumulated_frames, video_path, fps=self.fps)

                path_result = {
                    'path_indices': current_path,
                    'depth': current_depth + 1,
                    'rewards': all_rewards,
                    'success': success,
                    'video_path': video_path,
                    'termination_reason': 'success' if success else 'failure'
                }

                # Save this path's results
                self.save_path_results(save_env_idx, path_result, seed)
                all_completed_paths.append(path_result)

            else:
                # # Recursively explore from this new state
                # next_obs = {k: v for k, v in obs.items()}
                # Extract observations for this environment (remove batch dimension)
                next_obs = {k: v[env_idx] if isinstance(v, np.ndarray) else v for k, v in obs.items()}

                # Recursively explore deeper with accumulated frames
                child_paths = self.explore_action_tree_dfs(
                    env, env_idx, save_env_idx, policy, device, next_obs,
                    current_depth + 1, current_path, seed, max_depth,
                    accumulated_frames=accumulated_frames  # Pass frame buffer
                )
                all_completed_paths.extend(child_paths)

            # Restore state for next branch
            if action_idx < len(actions) - 1:  # Don't need to restore after last action
                # Restore physics state
                env.call_each('reset_after_hallucination', args_list=[(current_state,)])

                # Restore wrapper's internal buffers to clean state for next branch
                # set_attr is a method on the VectorEnv, expects list of values (one per env)
                env.set_attr('obs', [wrapper_obs.copy()])
                env.set_attr('reward', [wrapper_reward.copy()])
                env.set_attr('done', [wrapper_done.copy()])
                env.set_attr('grasps', [wrapper_grasps.copy()])
                # Note: info restoration is more complex (defaultdict of deques), skipping for now

                # Restore accumulated frame buffer to parent state (remove frames from this branch)
                del accumulated_frames[saved_frame_count:]

                # Re-get observations after state restoration
                current_obs = env.call('get_raw_observations')[env_idx]

        return all_completed_paths

    def generate_next_step(self, current_obs, list_actions, output_dir):
        """
        Generate predicted next frames using the world model.

        Args:
            current_states: List of environment observations, where each environment contains
                          a list of timestep observations (dicts with camera keys)
            list_actions: List of actions per environment, shape (1, 8, 7) per env

        Returns:
            List of predicted observations in the same format as current_states
        """
        os.makedirs(output_dir, exist_ok=True)
        save_data = {
            'current_states': current_obs,
            'list_actions': list_actions
        }

        save_path = os.path.join(output_dir, 'generate_next_step_inputs.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        env = self.env

        # plan for rollout - process one env at a time for DFS exploration
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)

        print(f"Starting DFS exploration for {n_inits} initial conditions")
        print(f"Using num_samples={self.num_samples} branches per node")
        print(f"Max depth={self.max_steps}")
        print("="*80)

        # Store results for all environments
        all_env_results = {}
        all_env_tree_stats = {}

        # Process each initial condition one at a time
        for init_idx in range(n_inits):
            print(f"\n{'='*80}")
            print(f"Processing environment {init_idx+1}/{n_inits}")
            print(f"{'='*80}")

            # Prepare init functions - use this init for first env, dummy for rest
            this_init_fns = [self.env_init_fn_dills[init_idx]] + [self.env_init_fn_dills[0]] * (n_envs - 1)

            # Initialize environments
            env.call_each('run_dill_function', args_list=[(x,) for x in this_init_fns])

            # Reset and get initial observation
            obs = env.reset()
            initial_obs = {k: v[0] if isinstance(v, np.ndarray) and len(v.shape) > 0 else v
                          for k, v in obs.items()}

            # Reset policy
            policy.reset()

            # Get metadata
            seed = self.env_seeds[init_idx]
            prefix = self.env_prefixs[init_idx]

            print(f"Seed: {seed}")
            print(f"Starting DFS exploration from root...")
            first_accumulated_frames = []
            try:
                if hasattr(env, 'envs'):
                    # SyncVectorEnv - direct access through wrapper chain
                    # Structure: MultiStepWrapper -> VideoRecordingWrapper -> RobomimicImageWrapper -> robosuite env
                    frames = env.envs[0].get_frames(mode='rgb_array')
                else:
                    # AsyncVectorEnv - use call_each to get frames from target env only
                    # Pass None for non-target envs to avoid unnecessary rendering
                    frames = env.call_each('get_frames',
                                         args_list=[('rgb_array',) if i == 0 else (None,) for i in range(env.num_envs)])[0]

                # Skip if frames is None (shouldn't happen for target env but check anyway)
                if frames is None:
                    print(f"Warning: get_frames returned None")
                    frame = None
                # Concatenate camera views horizontally if multiple views
                elif isinstance(frames, (list, tuple)) and len(frames) > 0:
                    # Make sure frames are numpy arrays
                    frame = np.concatenate(frames, axis=1)
                elif isinstance(frames, np.ndarray):
                    frame = frames
                else:
                    print(f"Warning: Unexpected frame type: {type(frames)}, value: {frames}")
                    frame = None

                if frame is not None and isinstance(frame, np.ndarray):
                    first_accumulated_frames.append(frame)
            except Exception as e:
                print(f"Warning: Could not render frame: {e}")
                import traceback
                traceback.print_exc()

            # Explore the action tree using DFS
            try:
                all_paths = self.explore_action_tree_dfs(
                    env=env,
                    env_idx=0,  # Always use first env in vector env
                    save_env_idx=init_idx,
                    policy=policy,
                    device=device,
                    current_obs=initial_obs,
                    current_depth=0,
                    path_indices=[],
                    seed=seed,
                    max_depth=self.max_steps,
                    accumulated_frames=first_accumulated_frames,
                )
            except Exception as e:
                print(f"ERROR during DFS exploration for env {init_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

            # Compute statistics for this environment's tree
            tree_stats = self.compute_tree_statistics(all_paths)
            tree_stats['env_idx'] = init_idx
            tree_stats['seed'] = str(seed)
            tree_stats['prefix'] = prefix
            tree_stats['paths'] = all_paths

            # Save tree statistics for this environment
            env_dir = pathlib.Path(self.output_dir) / f'env_{init_idx}'
            env_dir.mkdir(parents=True, exist_ok=True)
            stats_path = env_dir / 'tree_stats.json'

            # Save stats (excluding full paths data for the JSON, too large)
            stats_to_save = {k: v for k, v in tree_stats.items() if k != 'paths'}
            stats_to_save['num_paths'] = len(all_paths)
            stats_to_save['sample_paths'] = [
                {
                    'path_indices': p['path_indices'],
                    'depth': p['depth'],
                    'success': bool(p['success']),  # Convert numpy bool_ to native Python bool
                    'max_reward': float(np.max(p['rewards'])) if len(p['rewards']) > 0 else 0.0,
                    'video_path': p['video_path']
                }
                for p in all_paths[:100]  # Save first 100 paths as samples
            ]

            with open(stats_path, 'w') as f:
                json.dump(stats_to_save, f, indent=2)

            all_env_results[init_idx] = all_paths
            all_env_tree_stats[init_idx] = tree_stats

            print(f"\n{'='*40}")
            print(f"Completed env {init_idx} (seed {seed}):")
            print(f"  Total paths explored: {tree_stats['total_paths']}")
            print(f"  Success rate: {tree_stats['success_rate']:.2%}")
            print(f"  Average depth: {tree_stats['avg_depth']:.2f}")
            print(f"  Average max reward: {tree_stats['avg_max_reward']:.4f}")
            print(f"  Tree stats saved to: {stats_path}")
            print(f"{'='*40}\n")

        # After exploring all environments, compile aggregate statistics
        print(f"\n{'='*80}")
        print("All environments explored. Compiling aggregate statistics...")
        print(f"{'='*80}\n")

        aggregate_stats = {
            'total_envs': n_inits,
            'num_samples_per_node': self.num_samples,
            'max_depth': self.max_steps,
            'environments': {}
        }

        for init_idx, tree_stats in all_env_tree_stats.items():
            aggregate_stats['environments'][str(init_idx)] = {
                'seed': tree_stats['seed'],
                'prefix': tree_stats['prefix'],
                'total_paths': tree_stats['total_paths'],
                'success_rate': tree_stats['success_rate'],
                'avg_depth': tree_stats['avg_depth'],
                'avg_max_reward': tree_stats['avg_max_reward'],
                'success_rate_by_depth': tree_stats['success_rate_by_depth'],
                'avg_reward_by_depth': tree_stats['avg_reward_by_depth'],
                'branches_per_depth': tree_stats['branches_per_depth']
            }

        # Save aggregate statistics
        aggregate_path = pathlib.Path(self.output_dir) / 'aggregate_stats.json'
        with open(aggregate_path, 'w') as f:
            json.dump(aggregate_stats, f, indent=2)

        print(f"Aggregate statistics saved to: {aggregate_path}")

        # Compute overall metrics
        total_paths = sum(stats['total_paths'] for stats in all_env_tree_stats.values())
        overall_success_rate = np.mean([stats['success_rate'] for stats in all_env_tree_stats.values()])
        overall_avg_depth = np.mean([stats['avg_depth'] for stats in all_env_tree_stats.values()])

        print(f"\n{'='*80}")
        print("OVERALL SUMMARY:")
        print(f"  Total environments: {n_inits}")
        print(f"  Total paths explored: {total_paths}")
        print(f"  Overall success rate: {overall_success_rate:.2%}")
        print(f"  Overall average depth: {overall_avg_depth:.2f}")
        print(f"{'='*80}\n")

        # Return aggregate stats in a format compatible with existing logging
        log_data = {}
        for init_idx, tree_stats in all_env_tree_stats.items():
            prefix = tree_stats['prefix']
            seed = tree_stats['seed']
            # Find best path (highest reward)
            best_path = max(all_env_results[init_idx], key=lambda p: np.max(p['rewards']) if len(p['rewards']) > 0 else -1)
            log_data[prefix + f'sim_max_reward_{seed}'] = float(np.max(best_path['rewards'])) if len(best_path['rewards']) > 0 else 0.0
            if best_path['video_path']:
                log_data[prefix + f'sim_video_{seed}'] = wandb.Video(best_path['video_path'])

        # Add aggregate metrics
        log_data['test/mean_score'] = overall_success_rate
        log_data['test/avg_depth'] = overall_avg_depth
        log_data['test/total_paths'] = total_paths

        return log_data

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
