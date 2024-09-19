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
import pdb

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


class RobomimicImageRunnerEval(BaseImageRunner):
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
        ):
        super().__init__(output_dir)
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

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = True
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state, 
                    enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'trainmedia', str(train_idx) + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicImageWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        
        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = True

            def init_fn(env, seed=seed, 
                enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'testmedia', str(i) + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicImageWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)

        # env = SyncVectorEnv(env_fns)


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

    def run(self, policy: BaseImagePolicy, classifier=None, guidance_scale=None, guided_towards=None):
        device = policy.device
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
            added_state = env.call('get_reset_states')

            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            done = False
            env_step_index = 0
            while not done:
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
                    new_obs_dict = {}
                    for k,v in obs_dict.items():
                        new_obs_dict[k]=v[:,-2:]
                    if classifier:
                        action_dict = policy.predict_action(new_obs_dict, classifier, guidance_scale, guided_towards)
                    else:
                        action_dict = policy.predict_action(new_obs_dict)

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
                    env_action = self.undo_transform_action(action)

                full_action = env_action#self.undo_transform_action(np_action_dict['action_pred'])

                # print('INDEX:', env_step_index)
                # if chunk_idx == n_chunks - 1:
                #     pdb.set_trace()
                # actionpath = np.load('/proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.06.05/15.00.33_train_diffusion_unet_hybrid_liftph/checkpoints/epoch=0150-test_mean_score=0.980/1000galift_hammer_8_16_53_6/actions.npy', allow_pickle=True)
                # env_action = actionpath[env_step_index,this_global_slice,:,:]
                
                obs, reward, done, info = env.step(env_action)
                # env_step_index+=1
                if self.save_stuff:
                    print('MORE SAVING STUFF')
                    if 'save_rollout_obsdict_agentview_images' not in locals():
                        save_rollout_obsdict_agentview_images = np.expand_dims(np_obs_dict['agentview_image'],0)
                        save_rollout_obsdict_eyeinhand_images = np.expand_dims(np_obs_dict['robot0_eye_in_hand_image'],0)
                        save_rollout_obsdict_robot0s = np.expand_dims(np.concatenate((np_obs_dict['robot0_eef_pos'], np_obs_dict['robot0_eef_quat'],np_obs_dict['robot0_gripper_qpos']), axis=2),0)
                        save_rollout_actions = np.expand_dims(full_action,0)
                    else:
                        save_rollout_obsdict_agentview_images = np.vstack((save_rollout_obsdict_agentview_images,np.expand_dims(np_obs_dict['agentview_image'],0)))
                        save_rollout_obsdict_eyeinhand_images = np.vstack((save_rollout_obsdict_eyeinhand_images,np.expand_dims(np_obs_dict['robot0_eye_in_hand_image'],0)))
                        save_rollout_obsdict_robot0s = np.vstack((save_rollout_obsdict_robot0s,np.expand_dims(np.concatenate((np_obs_dict['robot0_eef_pos'], np_obs_dict['robot0_eef_quat'],np_obs_dict['robot0_gripper_qpos']), axis=2),0)))
                        save_rollout_actions = np.vstack((save_rollout_actions,np.expand_dims(full_action,0)))

                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(full_action.shape[1])
            pbar.close()

            if self.save_stuff:
                print('MORE SAVING STUFF')
                if 'all_save_rollout_obsdict_agentview_images' not in locals():
                    all_save_rollout_obsdict_agentview_images = np.array(save_rollout_obsdict_agentview_images)
                    all_save_rollout_obsdict_eyeinhand_images = np.array(save_rollout_obsdict_eyeinhand_images)
                    all_save_rollout_obsdict_robot0s = np.array(save_rollout_obsdict_robot0s)
                    all_save_rollout_actions = np.array(save_rollout_actions)
                    all_save_grasping = np.array(env.call('is_grasping')) 
                    all_save_reward = np.array(env.call('get_rewards'))
                else:
                    all_save_rollout_obsdict_agentview_images = np.concatenate((all_save_rollout_obsdict_agentview_images, np.array(save_rollout_obsdict_agentview_images)), axis=1)
                    all_save_rollout_obsdict_eyeinhand_images = np.concatenate((all_save_rollout_obsdict_eyeinhand_images, np.array(save_rollout_obsdict_eyeinhand_images)), axis=1)
                    all_save_rollout_obsdict_robot0s = np.concatenate((all_save_rollout_obsdict_robot0s, np.array(save_rollout_obsdict_robot0s)), axis=1)
                    all_save_rollout_actions = np.concatenate((all_save_rollout_actions, np.array(save_rollout_actions)), axis=1)
                    all_save_grasping = np.concatenate((all_save_grasping, np.array(env.call('is_grasping'))), axis=0)
                    all_save_reward = np.concatenate((all_save_reward, np.array(env.call('get_rewards'))), axis=0)
                if 'all_added_states' in locals():
                    all_added_states = np.vstack((all_added_states, np.array(added_state)))
                elif 'added_state' in locals():
                    all_added_states = np.array(added_state)
                del save_rollout_obsdict_agentview_images
                del save_rollout_obsdict_eyeinhand_images
                del save_rollout_obsdict_robot0s
                del save_rollout_actions
                if 'added_state' in locals():
                    del added_state


            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
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

        if self.save_stuff:
            print('MORE SAVING STUFF')
            np.save(self.output_dir+'/obsdict_agentview.npy', all_save_rollout_obsdict_agentview_images)
            np.save(self.output_dir+'/obsdict_eyeinhand.npy', all_save_rollout_obsdict_eyeinhand_images)
            np.save(self.output_dir+'/obsdict_robot0s.npy', all_save_rollout_obsdict_robot0s)
            np.save(self.output_dir+'/actions.npy', all_save_rollout_actions)
            np.save(self.output_dir+'/rewards.npy', all_save_reward)
            np.save(self.output_dir+'/grasping.npy', all_save_grasping)
            if 'all_added_states' in locals():
                np.save(self.output_dir+'/startstates.npy', all_added_states)

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
