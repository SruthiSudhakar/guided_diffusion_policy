"""
export LD_LIBRARY_PATH=:/home/sruthi/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=osmesa 
source /proj/vondrick3/sruthi/miniconda3/bin/activate
conda activate clonejgdrobodiff
cd /proj/vondrick3/sruthi/robots/diffusion_policy
export HYDRA_FULL_ERROR=1

Usage:
python playback_dataset.py --dataset_path /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2025.01.13/12.45.41_train_diffusion_unet_hybrid_robocasalang_PnPSinkToCounter_trainsplit_imagenet/checkpoints/epoch=1300-val_loss=0.151/PnPSinkToCounter_None_1_15_11_45_37/datafile.hdf5 \
                --device cuda:7 \
                --classifier_checkpoint /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.09.03/21.23.37_train_diffusion_unet_hybrid_15.00.33_check/checkpoints/epoch=0150-test_mean_score=0.940.ckpt 

"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)
import robosuite
import robocasa
import time
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
from data.dataset_registery import DATASETS
import numpy as np
import imageio
from termcolor import colored

def reset_to(env, state):
    """
    Reset to a specific simulator state.

    Args:
        state (dict): current simulator state that contains one or more of:
            - states (np.ndarray): initial state of the mujoco environment
            - model (str): mujoco scene xml

    Returns:
        observation (dict): observation dictionary after setting the simulator state (only
            if "states" is in @state)
    """
    should_ret = False
    if "model" in state:
        if state.get("ep_meta", None) is not None:
            # set relevant episode information
            ep_meta = json.loads(state["ep_meta"])
        else:
            ep_meta = {}
        if hasattr(env, "set_attrs_from_ep_meta"):  # older versions had this function
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):  # newer versions
            env.set_ep_meta(ep_meta)
        # this reset is necessary.
        # while the call to env.reset_from_xml_string does call reset,
        # that is only a "soft" reset that doesn't actually reload the model.
        env.reset()
        object_configured = True
        for obj in ep_meta['object_cfgs']:
            if obj['name']=='obj':
                if obj['info'] is None:
                    object_configured=False
                    print('RELOADING XML')
        if object_configured:
            robosuite_version_id = int(robosuite.__version__.split(".")[1])
            if robosuite_version_id <= 3:
                from robosuite.utils.mjcf_utils import postprocess_model_xml

                xml = postprocess_model_xml(state["model"])
            else:
                # v1.4 and above use the class-based edit_model_xml function
                xml = env.edit_model_xml(state["model"])
            env.reset_from_xml_string(xml)
        env.sim.reset()
        # hide teleop visualization after restoring from model
        # env.sim.model.site_rgba[env.eef_site_id] = np.array([0., 0., 0., 0.])
        # env.sim.model.site_rgba[env.eef_cylinder_id] = np.array([0., 0., 0., 0.])
    elif state.get("ep_meta", None) is not None:
        ep_meta = json.loads(state["ep_meta"])
        if hasattr(env, "set_attrs_from_ep_meta"):  # older versions had this function
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):  # newer versions
            env.set_ep_meta(ep_meta)
        print('RELOADING XML')
        env.reset()
        env.sim.reset()

    if "states" in state:
        env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
        should_ret = True

    # update state as needed
    if hasattr(env, "update_sites"):
        # older versions of environment had update_sites function
        env.update_sites()
    if hasattr(env, "update_state"):
        # later versions renamed this to update_state
        env.update_state()

    # if should_ret:
    #     # only return obs if we've done a forward call - otherwise the observations will be garbage
    #     return get_observation()
    return None



def playback_trajectory_with_env(
    env,
    initial_state,
    states,
    actions=None,
    observations=None,
    render=False,
    video_writer=None,
    video_skip=5,
    camera_names=None,
    first=False,
    verbose=False,
):
    """
    Helper function to playback a single trajectory using the simulator environment.
    If @actions are not None, it will play them open-loop after loading the initial state.
    Otherwise, @states are loaded one by one.

    Args:
        env (instance of EnvBase): environment
        initial_state (dict): initial simulation state to load
        states (np.array): array of simulation states to load
        actions (np.array): if provided, play actions back open-loop instead of using @states
        render (bool): if True, render on-screen
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        camera_names (list): determines which camera(s) are used for rendering. Pass more than
            one to output a video with multiple camera views concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    write_video = True
    video_count = 0

    # load the initial state
    ## this reset call doesn't seem necessary.
    ## seems ok to remove but haven't fully tested it.
    ## removing for now
    # env.reset()

    if verbose:
        ep_meta = json.loads(initial_state["ep_meta"])
        lang = ep_meta.get("lang", None)
        if lang is not None:
            print(colored(f"Instruction: {lang}", "green"))
        print(colored("Spawning environment...", "yellow"))
    reset_to(env, initial_state)

    traj_len = actions.shape[0]

    action_playback = actions is not None
    if action_playback:
        try:
            assert states.shape[0] == actions.shape[0]
        except Exception as e:
            print('HUH assert states.shape[0] == actions.shape[0]', e)


    if render is False:
        print(colored("Running episode...", "yellow"))
    for i in range(traj_len):
        start = time.time()

        if action_playback:
            env.step(actions[i])

            nobs = classifier_policy.normalizer.normalize(
                dict_apply(observations, lambda x: x[i].reshape(-1, *x.shape[2:]))
            )
            nobs_features = classifier_policy.obs_encoder(nobs)

            classifier_prediction = classifier_policy.model(actions[i], actions[i].shape[0], local_cond=None, global_cond=nobs_features)
            output_probabilities = nn.Sigmoid()(classifier_prediction)
            print('CLASSIFIER PRED', output_probabilities)
           
            if i < traj_len - 1 and states.shape[0]==actions.shape[0]:
                # check whether the actions deterministically lead to the same recorded states
                state_playback = np.array(env.sim.get_state().flatten())
                if not np.all(np.equal(states[i + 1], state_playback)):
                    err = np.linalg.norm(states[i + 1] - state_playback)
                    if verbose or i == traj_len - 2:
                        print(
                            colored(
                                "warning: playback diverged by {} at step {}".format(
                                    err, i
                                ),
                                "yellow",
                            )
                        )
        else:
            reset_to(env, {"states": states[i]})

        # on-screen render
        if render:
            if env.viewer is None:
                env.initialize_renderer()

            # so that mujoco viewer renders
            env.viewer.update()

            max_fr = 60
            elapsed = time.time() - start
            diff = 1 / max_fr - elapsed
            if diff > 0:
                time.sleep(diff)

        # video render
        if write_video:
            if video_count % video_skip == 0:
                video_img = []
                for cam_name in camera_names:
                    im = env.sim.render(height=512, width=512, camera_name=cam_name)[
                        ::-1
                    ]
                    video_img.append(im)
                video_img = np.concatenate(
                    video_img, axis=1
                )  # concatenate horizontally
                video_writer.append_data(video_img)

            video_count += 1

        if first:
            break

    if render:
        env.viewer.close()
        env.viewer = None
def playback_trajectory_with_obs(
    traj_grp,
    video_writer,
    video_skip=5,
    image_names=None,
    first=False,
):
    """
    This function reads all "rgb" observations in the dataset trajectory and
    writes them into a video.

    Args:
        traj_grp (hdf5 file group): hdf5 group which corresponds to the dataset trajectory to playback
        video_writer (imageio writer): video writer
        video_skip (int): determines rate at which environment frames are written to video
        image_names (list): determines which image observations are used for rendering. Pass more than
            one to output a video with multiple image observations concatenated horizontally.
        first (bool): if True, only use the first frame of each episode.
    """
    assert (
        image_names is not None
    ), "error: must specify at least one image observation to use in @image_names"
    video_count = 0

    traj_len = traj_grp["obs/{}".format(image_names[0] + "_image")].shape[0]
    for i in range(traj_len):
        if video_count % video_skip == 0:
            # concatenate image obs together
            im = [traj_grp["obs/{}".format(k + "_image")][i] for k in image_names]
            frame = np.concatenate(im, axis=1)
            video_writer.append_data(frame)
        video_count += 1

        if first:
            break
def get_env_metadata_from_dataset(dataset_path, ds_format="robomimic"):
    """
    Retrieves env metadata from dataset.

    Args:
        dataset_path (str): path to dataset

    Returns:
        env_meta (dict): environment metadata. Contains 3 keys:

            :`'env_name'`: name of environment
            :`'type'`: type of environment, should be a value in EB.EnvType
            :`'env_kwargs'`: dictionary of keyword arguments to pass to environment constructor
    """
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    if ds_format == "robomimic":
        env_meta = json.loads(f["data"].attrs["env_args"])
    else:
        raise ValueError
    f.close()
    return env_meta

@click.command()
@click.option('-dataset_path', '--dataset_path', required=True)
@click.option('-classifier_checkpoint', '--classifier_checkpoint', default=None, type=str)
@click.option('-d', '--device', default='cuda:0')
@click.option('-max_steps', '--max_steps', default=None, type=int)
@click.option('-n_data', '--n_data', default=None, type=int)
@click.option('-use_obs', '--use_obs', is_flag=True)
@click.option('-verbose', '--verbose', is_flag=True)


def main(dataset_path, classifier_checkpoint, device, max_steps, n_data, use_obs, verbose):
    output_dir = dataset_path[:-5]+'/playback/'  # Replace with your file path
    current_time = datetime.datetime.now()
    if os.path.exists(output_dir):
        click.confirm(f"Output path {output_dir} already exists! Overwrite?", abort=True)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # with open (output_dir+'/save_some_deets.txt', 'w') as f: 
    #     deets = ['checkpoint', checkpoint, 'output_dir', output_dir, 'dataset_path', \
    #         dataset_path, 'classifier_dir', classifier_dir, 'guidance_scale', \
    #         guidance_scale, 'guided_towards', guided_towards, 'max_steps', max_steps, \
    #         'object',object, 'n_train', n_train, 'n_test', n_test, 'test_start_seed', \
    #         test_start_seed, 'change_test_objects', change_test_objects, \
    #         'change_test_textures', change_test_textures, 'change_test_object_instances', \
    #         change_test_object_instances]
    #     deets = [str(x) for x in deets]
    #     f.writelines("\n".join(deets))


    
    if classifier_checkpoint:
        classifier_payload = torch.load(open(classifier_checkpoint+'.ckpt', 'rb'), pickle_module=dill)
        classifier_cfg = classifier_payload['cfg']
        classifier_cls = hydra.utils.get_class(classifier_cfg._target_)

        classifier_workspace = classifier_cls(classifier_cfg, output_dir=classifier_checkpoint)
        classifier_workspace: BaseWorkspace
        classifier_workspace.load_payload(classifier_payload, exclude_keys=None, include_keys=None)
        
        # get policy from workspace
        classifier_policy = classifier_workspace.model    
        classifier_policy.to(device)
        classifier_policy.eval()
        
    env = None
    if not use_obs:
        env_meta = get_env_metadata_from_dataset(dataset_path=dataset_path)
        # if use_abs_actions:
        #     env_meta["env_kwargs"]["controller_configs"][
        #         "control_delta"
        #     ] = False  # absolute action space

        env_kwargs = env_meta["env_kwargs"]
        env_kwargs["env_name"] = env_meta["env_name"]
        env_kwargs["has_renderer"] = False
        env_kwargs["renderer"] = "mjviewer"
        env_kwargs["has_offscreen_renderer"] = True
        env_kwargs["use_camera_obs"] = False

        # env = EnvUtils.create_env_from_metadata(
        #     env_meta=env_meta,
        #     render=False, 
        #     render_offscreen=False,
        #     use_image_obs=False, 
        #     object=None,
        # )


        env = robosuite.make(**env_kwargs)
    f = h5py.File(dataset_path, "r")

    demos = list(f["data"].keys())
    inds = np.argsort([int(elem[5:]) for elem in demos])
    demos = [demos[i] for i in inds]

    # maybe reduce the number of demonstrations to playback
    if n_data is not None:
        # random.shuffle(demos)
        demos = demos[: n_data]

    # maybe dump video
    video_path = output_dir + "playback_use_actions.mp4"
    video_writer = imageio.get_writer(video_path, fps=20)

    for ind in range(1):#len(demos)):
        ep = demos[ind]
        print(colored("\nPlaying back episode: {}".format(ep), "yellow"))

        if use_obs:
            playback_trajectory_with_obs(
                traj_grp=f["data/{}".format(ep)],
                video_writer=video_writer,
                video_skip=video_skip,
                image_names=render_image_names,
                first=first,
            )
            continue

        # prepare initial state to reload from
        states = f["data/{}/states".format(ep)][()]
        initial_state = dict(states=states[0])
        initial_state["model"] = f["data/{}".format(ep)].attrs["model_file"]
        initial_state["ep_meta"] = f["data/{}".format(ep)].attrs.get("ep_meta", None)
        states = np.concatenate((states, [states[-1]] * 50))

        # supply actions if using open-loop action playback
        actions = f["data/{}/actions".format(ep)][()]
        observations = f["data/{}/obs"]

        playback_trajectory_with_env(
            env=env,
            initial_state=initial_state,
            states=states,
            actions=actions,
            observations=observations,
            render=False,
            video_writer=video_writer,
            video_skip=5,
            camera_names=["robot0_agentview_left"],
            first=False,
            verbose=verbose,
        )

    f.close()

    print(colored(f"Saved video to {video_path}", "green"))
    video_writer.close()

    if env is not None:
        env.close()



    # # dump log to json
    # json_log = dict()
    # for key, value in runner_log.items():
    #     if isinstance(value, wandb.sdk.data_types.video.Video):
    #         json_log[key] = value._path
    #     else:
    #         json_log[key] = str(value)
    # out_path = os.path.join(output_dir, 'eval_log.json')
    # json.dump(json_log, open(out_path, 'w'), indent=2, sort_keys=True)

if __name__ == '__main__':
    main()