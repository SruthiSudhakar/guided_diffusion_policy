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
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
from lovely_numpy import lo
import pdb, json
from termcolor import colored
from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES, OBJ_GROUPS
import cv2
import time
import json


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

# def get_distinct_floorplan_and_style():
#     return None

from transformers import CLIPTokenizer, CLIPModel
import torch

# Load the CLIP model and tokenizer
clip_model_name = "openai/clip-vit-base-patch32"  # You can choose other models if desired
clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
clip_model = CLIPModel.from_pretrained(clip_model_name)

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
            start_rollout_from_state=0,
            show_classifier_scores=False,
            adaptive_guidance='None',
            decode_first=True,
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
            for i in tqdm.tqdm(range(n_train)):
                train_idx = train_start_idx + (i % len(f['data']))
                enable_render = True
                init_state = f[f'data/demo_{train_idx}/states'][start_rollout_from_state]
                env_model = f[f'data/demo_{train_idx}'].attrs["model_file"]
                ep_meta = f[f'data/demo_{train_idx}'].attrs.get("ep_meta",None)
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
                #             # pdb.set_trace()
                #             # print('replacing with new instance',temp_info['cat'] )

                ep_meta = json.dumps(ep_meta)
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
        # env = SyncVectorEnv(env_fns)
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
        self.debug=debug
        self.choose_sample=choose_sample

    def run(self, policy: BaseImagePolicy, classifier_processor=None, classifier=None, grad_steps=None, guidance_scale=None, guided_towards=None):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns) #how many to run asynchronusly at once
        n_inits = len(self.env_init_fn_dills) #how many examples to run total
        n_chunks = math.ceil(n_inits / n_envs) #total/per run = how many times to run it
        # print('N STUFF', n_envs, n_inits, n_chunks)

        # allocate data
        all_objects = [None] * n_inits
        all_video_paths = [None] * n_inits
        all_video_frames=[None]*n_inits
        all_rewards = [None] * n_inits
        all_classification_scores1_before = [None] * n_inits
        # all_classification_scores2_before = [None] * n_inits
        all_classification_scores1_after = [None] * n_inits
        # all_classification_scores2_after = [None] * n_inits

        demo_number = -1
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
            added_state = env.call('get_env_state')
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            language_goal = env.call('get_env_metadata')[this_local_slice]
            language_goal = [json.loads(x['ep_meta'])['lang'] for x in language_goal]
            done = False
            env_step_index = 0
            while not done:
                env_step_index+=1
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    if self.save_stuff:
                        del obs_dict['robot0_eef_pos']
                        del obs_dict['robot0_eef_quat']
                        del obs_dict['robot0_gripper_qpos']
                    if not self.choose_sample:
                        if self.adaptive_guidance!='None':
                            action_dict, classifier_action_pred = policy.predict_action(obs_dict, classifier_processor, classifier, grad_steps, guidance_scale, guided_towards, trajectory_step=env_step_index, adaptive_guidance=self.adaptive_guidance, max_steps=self.max_steps/self.n_action_steps, get_class_scores=self.show_classifier_scores, decode_first=self.decode_first, language_goal=language_goal)
                        elif classifier:
                            action_dict, classifier_action_pred = policy.predict_action(obs_dict, classifier_processor, classifier, grad_steps, guidance_scale, guided_towards, get_class_scores=self.show_classifier_scores, decode_first=self.decode_first, language_goal=language_goal)
                        else:
                            action_dict, classifier_action_pred = policy.predict_action(obs_dict)
                    elif self.choose_sample:
                        # resample
                        print('resample')
                        sample_many_actions=[]
                        action_logits=[]
                        pdb.set_trace()
                        for i in range(10):
                            sample_many_actions.append(policy.predict_action(obs_dict))
                        pdb.set_trace()
                        for sample in sample_many_actions:
                            action_logits.append(policy.get_class_score(sample['action_pred'],obs_dict,classifier_processor,classifier,language_goal).item())
                        pdb.set_trace()
                        action_dict=sample_many_actions[np.argmax(np.array(action_logits))]
                        pdb.set_trace()
                        
                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                #np.all(np_action_dict['action_pred'][:,1:9,:]==np_action_dict['action'])
                action = np_action_dict['action']

                if not np.all(np.isfinite(action)):
                    raise RuntimeError("Nan or Inf action")

                # step env
                env_action = action
                if self.abs_action:
                    pdb.set_trace()
                    print('UNDOING TRANSFORM ACTION')
                    env_action = self.undo_transform_action(action)

                # print('INDEX:', env_step_index)
                # if chunk_idx == n_chunks - 1:
                #     pdb.set_trace()
                # actionpath = np.load('/proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.06.05/15.00.33_train_diffusion_unet_hybrid_liftph/checkpoints/epoch=0150-test_mean_score=0.980/1000galift_hammer_8_16_53_6/actions.npy', allow_pickle=True)
                # env_action = actionpath[env_step_index,this_global_slice,:,:]
                
                add_on = np.tile([0., -0.,  0.,  0., -1.], (env_action.shape[0], env_action.shape[1], 1))
                extended_env_action = np.concatenate((env_action, add_on), axis=-1)
                obs, reward, done, info = env.step(extended_env_action)

                if classifier and self.show_classifier_scores:
                    if 'save_rollout_classification_scores_1_before' not in locals():
                        save_rollout_classification_scores_1_before = np.expand_dims(classifier_action_pred['classifier_policy_global_cond']['before'].detach().cpu().numpy(),1)
                        # save_rollout_classification_scores_2_before = np.expand_dims(classifier_action_pred['global_cond']['before'].detach().cpu().numpy(),1)
                        save_rollout_classification_scores_1_after = np.expand_dims(classifier_action_pred['classifier_policy_global_cond']['after'].detach().cpu().numpy(),1)
                        # save_rollout_classification_scores_2_after = np.expand_dims(classifier_action_pred['global_cond']['after'].detach().cpu().numpy(),1)
                    else:
                        save_rollout_classification_scores_1_before = np.hstack((save_rollout_classification_scores_1_before,np.expand_dims(classifier_action_pred['classifier_policy_global_cond']['before'].detach().cpu().numpy(),1)))
                        # save_rollout_classification_scores_2_before = np.hstack((save_rollout_classification_scores_2_before,np.expand_dims(classifier_action_pred['global_cond']['before'].detach().cpu().numpy(),1)))
                        save_rollout_classification_scores_1_after = np.hstack((save_rollout_classification_scores_1_after,np.expand_dims(classifier_action_pred['classifier_policy_global_cond']['after'].detach().cpu().numpy(),1)))
                        # save_rollout_classification_scores_2_after = np.hstack((save_rollout_classification_scores_2_after,np.expand_dims(classifier_action_pred['global_cond']['after'].detach().cpu().numpy(),1)))

                if self.save_stuff:
                    if 'save_rollout_obsdict_robot0_agentview_right_image' not in locals():
                        save_rollout_obsdict_robot0_agentview_right_image = np.expand_dims(np_obs_dict['robot0_agentview_right_image'],0)
                        save_rollout_obsdict_robot0_agentview_left_image = np.expand_dims(np_obs_dict['robot0_agentview_left_image'],0)
                        save_rollout_obsdict_eyeinhand_images = np.expand_dims(np_obs_dict['robot0_eye_in_hand_image'],0)
                        save_rollout_obsdict_robot0s = np.expand_dims(np.concatenate((np_obs_dict['robot0_eef_pos'], np_obs_dict['robot0_eef_quat'],np_obs_dict['robot0_gripper_qpos']), axis=2),0)
                        save_rollout_actions = np.expand_dims(extended_env_action,0)
                    else:
                        save_rollout_obsdict_robot0_agentview_right_image = np.vstack((save_rollout_obsdict_robot0_agentview_right_image,np.expand_dims(np_obs_dict['robot0_agentview_right_image'],0)))
                        save_rollout_obsdict_robot0_agentview_left_image = np.vstack((save_rollout_obsdict_robot0_agentview_left_image,np.expand_dims(np_obs_dict['robot0_agentview_left_image'],0)))
                        save_rollout_obsdict_eyeinhand_images = np.vstack((save_rollout_obsdict_eyeinhand_images,np.expand_dims(np_obs_dict['robot0_eye_in_hand_image'],0)))
                        save_rollout_obsdict_robot0s = np.vstack((save_rollout_obsdict_robot0s,np.expand_dims(np.concatenate((np_obs_dict['robot0_eef_pos'], np_obs_dict['robot0_eef_quat'],np_obs_dict['robot0_gripper_qpos']), axis=2),0)))
                        save_rollout_actions = np.vstack((save_rollout_actions,np.expand_dims(extended_env_action,0)))

                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(extended_env_action.shape[1])
                if self.debug:
                    done=True
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_video_frames[this_global_slice] = env.call('get_attr', 'frames')[this_local_slice]
            all_objects[this_global_slice] = env.call('get_env_metadata')[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            if classifier and self.show_classifier_scores:
                all_classification_scores1_before[this_global_slice] = save_rollout_classification_scores_1_before
                # all_classification_scores2_before[this_global_slice] = save_rollout_classification_scores_2_before
                all_classification_scores1_after[this_global_slice] = save_rollout_classification_scores_1_after
                # all_classification_scores2_after[this_global_slice] = save_rollout_classification_scores_2_after
                del save_rollout_classification_scores_1_before
                # del save_rollout_classification_scores_2_before
                del save_rollout_classification_scores_1_after
                # del save_rollout_classification_scores_2_after
            if self.save_stuff:
                print('MORE SAVING STUFF')
                #Ep meta
                for index,one_env in enumerate(all_objects[this_global_slice]):
                    demo_number+=1
                    demogrp = self.datagrp.create_group('demo_'+str(demo_number))
                    demogrp.attrs['file_path']=all_video_paths[demo_number]
                    demogrp.attrs['ep_meta'] = one_env['ep_meta']
                    demogrp.attrs['model_file'] = one_env['env_model']
                    demogrp.create_dataset('actions', data = save_rollout_actions[:,index].reshape(-1, *save_rollout_actions[:,index].shape[2:]))
                    demogrp.create_dataset('rewards', data = np.array(env.call('get_rewards')[index]))
                    demogrp.create_dataset('grasping', data = np.array(env.call('is_grasping')[index]))
                    demogrp.create_dataset('states', data = np.expand_dims(np.array(added_state[index]),0))
                    obsgrp = demogrp.create_group('obs') 
                    obsgrp.create_dataset('robot0_agentview_left_image', data = (save_rollout_obsdict_robot0_agentview_left_image[:,index].reshape(-1,*save_rollout_obsdict_robot0_agentview_left_image[:,index].shape[2:])* 255.0).astype(np.uint8).transpose(0,2,3,1))
                    obsgrp.create_dataset('robot0_agentview_right_image', data = (save_rollout_obsdict_robot0_agentview_right_image[:,index].reshape(-1,*save_rollout_obsdict_robot0_agentview_right_image[:,index].shape[2:])* 255.0).astype(np.uint8).transpose(0,2,3,1))
                    obsgrp.create_dataset('robot0_eye_in_hand_image', data = (save_rollout_obsdict_eyeinhand_images[:,index].reshape(-1,*save_rollout_obsdict_eyeinhand_images[:,index].shape[2:])* 255.0).astype(np.uint8).transpose(0,2,3,1))
                    obsgrp.create_dataset('robot0_eef_pos', data = save_rollout_obsdict_robot0s[:,index,:,:3].reshape(-1, *save_rollout_obsdict_robot0s[:,index,:,:3].shape[2:]))
                    obsgrp.create_dataset('robot0_eef_quat', data = save_rollout_obsdict_robot0s[:,index,:,3:7].reshape(-1, *save_rollout_obsdict_robot0s[:,index,:,3:7].shape[2:]))
                    obsgrp.create_dataset('robot0_gripper_qpos', data = save_rollout_obsdict_robot0s[:,index,:,7:].reshape(-1, *save_rollout_obsdict_robot0s[:,index,:,7:].shape[2:]))
                del save_rollout_obsdict_robot0_agentview_right_image
                del save_rollout_obsdict_robot0_agentview_left_image
                del save_rollout_obsdict_eyeinhand_images
                del save_rollout_obsdict_robot0s
                del save_rollout_actions
                del added_state
            #log after every 2 chunks
            if chunk_idx%2==0:
                print('started saving rollouts')
                max_rewards = collections.defaultdict(list)
                log_data = dict()
                for i in range(len(all_rewards)):
                    if all_rewards[i]==None:
                        continue
                    seed = self.env_seeds[i]
                    prefix = self.env_prefixs[i]
                    max_reward = np.max(all_rewards[i])
                    max_rewards[prefix].append(max_reward)
                    log_data[prefix+f'sim_max_reward_{seed}'] = max_reward
                    if classifier and self.show_classifier_scores:
                        log_data[prefix+'a.cpgc_before'+"_"+str(seed)] = ", ".join(f"{i}: {round(value,3)}" for i, value in enumerate(all_classification_scores1_before[i]))
                        # log_data[prefix+'a.gc_before'+"_"+str(seed)] = ", ".join(f"{i}: {round(value,3)}" for i, value in enumerate(all_classification_scores2_before[i]))
                        log_data[prefix+'a.cpgc_after'+"_"+str(seed)] = ", ".join(f"{i}: {round(value,3)}" for i, value in enumerate(all_classification_scores1_after[i]))
                        # log_data[prefix+'a.gc_after'+"_"+str(seed)] = ", ".join(f"{i}: {round(value,3)}" for i, value in enumerate(all_classification_scores2_after[i]))
                    try:
                        object_cfgs=json.loads(all_objects[i]['ep_meta'])['object_cfgs']
                        for obj in object_cfgs:
                            if obj['name']=='obj':
                                log_data[prefix+f'xobject_metadata{seed}'] = '/'.join(obj['info']['mjcf_path'].split('/')[-3:-1])
                    except:
                        print('HOLD UP could not save object info')
                    # visualize sim
                    video_path = all_video_paths[i]
                    if video_path is not None:
                        sim_video = wandb.Video(video_path)
                        log_data[prefix+f'sim_video_{seed}'] = sim_video
                # log aggregate metrics
                for prefix, value in max_rewards.items():
                    name = prefix+'mean_score'
                    value = np.mean(value)
                    log_data[name] = value
                json_log = dict()
                for key, value in log_data.items():
                    if isinstance(value, wandb.sdk.data_types.video.Video):
                        json_log[key] = value._path
                    else:
                        json_log[key] = str(value)
                out_path = os.path.join(self.output_dir, 'eval_log.json')
                json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)
                print(f'saved chunk {chunk_idx}')


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
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward
            if classifier and self.show_classifier_scores:
                log_data[prefix+'a.cpgc_before'+"_"+str(seed)] = ", ".join(f"{i}: {round(value,3)}" for i, value in enumerate(all_classification_scores1_before[i]))
                # log_data[prefix+'a.gc_before'+"_"+str(seed)] = ", ".join(f"{i}: {round(value,3)}" for i, value in enumerate(all_classification_scores2_before[i]))
                log_data[prefix+'a.cpgc_after'+"_"+str(seed)] = ", ".join(f"{i}: {round(value,3)}" for i, value in enumerate(all_classification_scores1_after[i]))
                # log_data[prefix+'a.gc_after'+"_"+str(seed)] = ", ".join(f"{i}: {round(value,3)}" for i, value in enumerate(all_classification_scores2_after[i]))
            try:
                object_cfgs=json.loads(all_objects[i]['ep_meta'])['object_cfgs']
                for obj in object_cfgs:
                    if obj['name']=='obj':
                        log_data[prefix+f'xobject_metadata{seed}'] = '/'.join(obj['info']['mjcf_path'].split('/')[-3:-1])
            except:
                print('HOLD UP could not save object info')
            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
            if classifier and self.show_classifier_scores:
                video_writer = cv2.VideoWriter(video_path[:-4]+'_cs.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (128, 128))
                for frame_idx, frame in enumerate(all_video_frames[i]):
                    text = str(frame_idx)+' : '+str(round(all_classification_scores1_before[i][frame_idx//4],3))+' -> '+ str(round(all_classification_scores1_after[i][frame_idx//4],3))
                    frame= cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.putText(frame, text, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0) , 1)
                    video_writer.write(frame)
                video_writer.release()
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        if self.save_stuff:
            self.data_file.close()

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
