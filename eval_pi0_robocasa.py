"""
Evaluate a pi0 policy on RoboCasa tasks via the OpenPI client-server architecture.

Usage:
  1. Start the pi0 policy server (in a separate terminal):
       cd /app/openpi && .venv/bin/python scripts/serve_policy.py --env LIBERO --port 8000

  2. Run this evaluation script:
       python eval_pi0_robocasa.py \
         --dataset_key PnPCounterToCab_mg_fixed_224 \
         --host localhost --port 8000 \
         --prompt "pick up the object from the counter and place it in the cabinet"

  3. Run with parallel environments for faster evaluation:
       python eval_pi0_robocasa.py --n_envs 4 \
         --dataset_key PnPCounterToCab_mg_fixed_224 \
         --host localhost --port 8000
"""
import sys
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import collections
import copy
import datetime
import json
import math
import multiprocessing
import pathlib
import pdb
import traceback

import cv2
import h5py
import imageio
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from termcolor import colored
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils

from data.dgx_data_registery import DATASETS


def quat2axisangle(quat):
    """Convert quaternion [x, y, z, w] to axis-angle representation.
    Adapted from robosuite transform_utils.
    """
    quat = np.array(quat, dtype=np.float64)
    # clip w component
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def create_env(env_meta, dataset_path, camera_height=256, camera_width=256,
               change_textures=False, change_objects=False):
    """Create a RoboCasa environment from dataset metadata."""
    env_meta['env_kwargs']['use_object_obs'] = False
    env_meta['env_kwargs']['renderer'] = 'mjviewer'

    # Ensure cameras render at the desired resolution
    env_meta['env_kwargs']['camera_heights'] = camera_height
    env_meta['env_kwargs']['camera_widths'] = camera_width

    if change_textures:
        env_meta['env_kwargs']['generative_textures'] = 'random'
    if change_objects:
        env_meta['env_kwargs']['obj_instance_split'] = 'B'

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta['env_name'],
        render=False,
        render_offscreen=True,
        use_image_obs=True,
    )
    return env


def get_task_description(dataset_path, demo_key="demo_0"):
    """Extract the natural language task description from the dataset."""
    with h5py.File(dataset_path, 'r') as f:
        ep_meta = f[f'data/{demo_key}'].attrs.get('ep_meta', '{}')
        if isinstance(ep_meta, bytes):
            ep_meta = ep_meta.decode('utf-8')
        meta = json.loads(ep_meta)
        lang = meta.get('lang', '')
    return lang


def preprocess_obs_for_pi0(obs, resize_size=224):
    """Convert RoboCasa observations to pi0 input format.

    RoboCasa images come upside-down from the robosuite renderer,
    so we flip them 180 degrees (same convention as LIBERO).

    Args:
        obs: dict from env.step() with image and state keys
        resize_size: target image size for pi0 (default 224)

    Returns:
        dict with pi0-compatible observation keys
    """
    # Get the main camera image (try common RoboCasa camera names)
    img_key = None
    for key in ['robot0_agentview_left_image', 'agentview_image',
                'robot0_agentview_image']:
        if key in obs:
            img_key = key
            break

    wrist_key = None
    for key in ['robot0_eye_in_hand_image', 'eye_in_hand_image']:
        if key in obs:
            wrist_key = key
            break

    if img_key is None:
        raise KeyError(
            f"No main camera image found in obs. Available keys: {list(obs.keys())}"
        )

    # Get image (RoboCasa images are already right-side up)
    img = obs[img_key]
    if img.ndim == 3 and img.shape[0] in (1, 3):
        # CHW -> HWC
        img = np.transpose(img, (1, 2, 0))
    img = image_tools.convert_to_uint8(np.ascontiguousarray(img))
    img = image_tools.resize_with_pad(img, resize_size, resize_size)

    # Wrist image
    if wrist_key is not None:
        wrist_img = obs[wrist_key]
        if wrist_img.ndim == 3 and wrist_img.shape[0] in (1, 3):
            wrist_img = np.transpose(wrist_img, (1, 2, 0))
        wrist_img = image_tools.convert_to_uint8(np.ascontiguousarray(wrist_img))
        wrist_img = image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
    else:
        # Zero image if no wrist camera available
        wrist_img = np.zeros((resize_size, resize_size, 3), dtype=np.uint8)

    # Build proprioceptive state: [eef_pos(3), eef_axisangle(3), gripper_qpos(2)]
    eef_pos = obs['robot0_eef_pos']          # (3,)
    eef_quat = obs['robot0_eef_quat']        # (4,)
    gripper_qpos = obs['robot0_gripper_qpos'] # (2,)

    eef_axisangle = quat2axisangle(eef_quat)  # (3,)
    state = np.concatenate([eef_pos, eef_axisangle, gripper_qpos]).astype(np.float32)

    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": state,
    }


def _eval_worker(worker_args):
    """Worker function that runs a subset of demos in its own env + client.

    Designed to run in a separate process via multiprocessing.Pool.
    """
    (worker_id, env_meta, dataset_path, demo_keys, num_demos_total,
     all_init_states, all_task_descriptions, all_demo_lengths,
     all_model_files, all_ep_metas, args, max_steps, output_dir,
     global_demo_indices) = worker_args

    prefix = f"[Worker {worker_id}]"

    # Each worker must init obs utils in its own process
    ObsUtils.initialize_obs_utils_with_obs_specs({
        "obs": {
            "low_dim": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
            "rgb": ["robot0_agentview_left_image", "robot0_agentview_right_image", "robot0_eye_in_hand_image"],
        }
    })

    # Create own env
    env = create_env(
        copy.deepcopy(env_meta), dataset_path,
        camera_height=args.camera_size, camera_width=args.camera_size,
        change_textures=args.change_textures,
        change_objects=args.change_objects,
    )

    # Create own client connection
    client = websocket_client_policy.WebsocketClientPolicy(
        host=args.host, port=args.port
    )
    print(colored(f"{prefix} Connected to pi0 server, {len(demo_keys)} demos assigned", "green"))

    results = {}
    successes = 0
    episodes = 0

    for local_idx, demo_key in enumerate(demo_keys):
        global_idx = global_demo_indices[local_idx]
        init_state = all_init_states[demo_key]
        task_desc = all_task_descriptions[demo_key]
        if args.prompt:
            task_desc = args.prompt

        print(colored(
            f"{prefix} [{global_idx+1}/{num_demos_total}] Demo: {demo_key} | Task: {task_desc}",
            "cyan"
        ))

        # Reset environment
        env.reset()
        reset_dict = {"states": init_state}
        if all_model_files[demo_key] is not None:
            reset_dict["model"] = all_model_files[demo_key]
        if all_ep_metas[demo_key] is not None:
            reset_dict["ep_meta"] = all_ep_metas[demo_key]
        obs = env.reset_to(reset_dict)

        action_plan = collections.deque()
        replay_images = []
        done = False
        success = False

        # Optional: skip ahead in demo
        if args.start_from_state > 0:
            with h5py.File(dataset_path, 'r') as f:
                demo_actions = f[f'data/{demo_key}/actions'][:]
            skip_steps = int(args.start_from_state * len(demo_actions))
            for t in range(min(skip_steps, len(demo_actions))):
                obs, reward, done, info = env.step(demo_actions[t])
                if done:
                    break
            if done:
                print(f"{prefix}   Demo completed during skip phase")
                continue

        # Wait for objects to settle
        action_dim = env.env.action_dim
        dummy_action = np.zeros(action_dim)
        dummy_action[-1] = -1.0
        for _ in range(args.num_wait_steps):
            obs, _, _, _ = env.step(dummy_action)

        # Main rollout loop
        ep_max_steps = 20 if args.debug else max_steps
        for t in range(ep_max_steps):
            try:
                pi0_obs = preprocess_obs_for_pi0(obs, resize_size=224)
                pi0_obs["prompt"] = task_desc

                # Save frame for replay video
                video_views = []
                for cam_key in ['robot0_agentview_left_image', 'robot0_agentview_right_image', 'robot0_eye_in_hand_image']:
                    if cam_key in obs:
                        cam_img = obs[cam_key]
                        if cam_img.ndim == 3 and cam_img.shape[0] in (1, 3):
                            cam_img = np.transpose(cam_img, (1, 2, 0))
                        cam_img = image_tools.convert_to_uint8(np.ascontiguousarray(cam_img))
                        video_views.append(cam_img)
                target_h = video_views[0].shape[0]
                resized_views = []
                for v in video_views:
                    if v.shape[0] != target_h:
                        v = cv2.resize(v, (int(v.shape[1] * target_h / v.shape[0]), target_h))
                    resized_views.append(v)
                replay_images.append(np.concatenate(resized_views, axis=1))

                # Query pi0 policy if action plan is empty
                if not action_plan:
                    result = client.infer(pi0_obs)
                    action_chunk = result["actions"]
                    n_to_take = min(args.replan_steps, len(action_chunk))
                    action_plan.extend(action_chunk[:n_to_take])

                # Execute next action
                action = action_plan.popleft()
                action = np.array(action, dtype=np.float64)

                expected_dim = action_dim
                if len(action) < expected_dim:
                    padding = np.zeros(expected_dim - len(action))
                    if expected_dim - len(action) >= 1:
                        padding[-1] = -1.0
                    action = np.concatenate([action, padding])
                elif len(action) > expected_dim:
                    action = action[:expected_dim]

                obs, reward, done, info = env.step(action)

                # Check task success (robosuite done is always False)
                if env.is_success()["task"]:
                    success = True
                    break

            except Exception as e:
                print(colored(f"{prefix}   Error at step {t}: {e}", "red"))
                traceback.print_exc()
                break

        episodes += 1
        if success:
            successes += 1

        # Save replay video
        if replay_images:
            suffix = "success" if success else "failure"
            video_path = os.path.join(output_dir, f"{demo_key}_{suffix}.mp4")
            imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)

        results[demo_key] = {
            "success": success,
            "steps": t + 1,
            "task": task_desc,
        }

        print(colored(
            f"{prefix}   {'SUCCESS' if success else 'FAILURE'} | Steps: {t+1} | "
            f"Worker running: {successes}/{episodes} ({100*successes/episodes:.1f}%)",
            "green" if success else "red"
        ))

    env.close()
    return results


def run_eval(args):
    """Run pi0 evaluation on RoboCasa."""

    # Resolve dataset path
    if args.dataset_path:
        dataset_path = args.dataset_path
    elif args.dataset_key:
        paths = DATASETS.get(args.dataset_key)
        if paths is None:
            print(f"Unknown dataset key: {args.dataset_key}")
            print(f"Available keys: {list(DATASETS.keys())}")
            sys.exit(1)
        dataset_path = paths[0]
    else:
        print("Must specify either --dataset_path or --dataset_key")
        sys.exit(1)

    # Make path absolute if relative
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join('/app', dataset_path)

    print(colored(f"Dataset: {dataset_path}", "cyan"))

    # Load env metadata and initial states
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    with h5py.File(dataset_path, 'r') as f:
        demo_keys = sorted(f['data'].keys(), key=lambda x: int(x.split('_')[1]))
        all_init_states = {}
        all_task_descriptions = {}
        all_demo_lengths = {}
        all_model_files = {}
        all_ep_metas = {}
        for dk in demo_keys:
            all_init_states[dk] = f[f'data/{dk}/states'][0]
            ep_meta = f[f'data/{dk}'].attrs.get('ep_meta', '{}')
            if isinstance(ep_meta, bytes):
                ep_meta = ep_meta.decode('utf-8')
            meta = json.loads(ep_meta)
            all_task_descriptions[dk] = meta.get('lang', args.prompt)
            all_demo_lengths[dk] = f[f'data/{dk}/actions'].shape[0]
            # Store model xml and ep_meta for exact env reconstruction
            all_ep_metas[dk] = ep_meta
            model_file = f[f'data/{dk}'].attrs.get('model_file', None)
            if isinstance(model_file, bytes):
                model_file = model_file.decode('utf-8')
            all_model_files[dk] = model_file

    # Determine number of demos to evaluate
    num_demos = len(demo_keys) if args.num_demos is None else min(args.num_demos, len(demo_keys))
    demo_keys = demo_keys[:num_demos]

    # Compute max_steps
    if args.max_steps is None:
        lengths = np.array([all_demo_lengths[dk] for dk in demo_keys])
        max_steps = int(np.percentile(lengths, 90)) + 50
    else:
        max_steps = args.max_steps
    print(colored(f"Max steps per episode: {max_steps}", "cyan"))

    # Setup output directory
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    task_name = args.dataset_key or os.path.basename(dataset_path).replace('.hdf5', '')
    output_dir = os.path.join(
        args.output_dir,
        f"pi0_{task_name}_{timestamp}"
    )
    if args.debug:
        output_dir = "debug"

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    print(colored(f"Output dir: {output_dir}", "green"))

    # Save run config
    with open(os.path.join(output_dir, 'run_config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    n_envs = args.n_envs

    # === Single-env path (no multiprocessing overhead) ===
    if n_envs <= 1:
        # Connect to pi0 policy server
        print(colored(f"Connecting to pi0 server at {args.host}:{args.port}...", "yellow"))
        client = websocket_client_policy.WebsocketClientPolicy(
            host=args.host, port=args.port
        )
        print(colored("Connected to pi0 server!", "green"))

        # Initialize observation utilities
        ObsUtils.initialize_obs_utils_with_obs_specs({
            "obs": {
                "low_dim": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
                "rgb": ["robot0_agentview_left_image", "robot0_agentview_right_image", "robot0_eye_in_hand_image"],
            }
        })

        # Create environment
        print("Creating RoboCasa environment...")
        env = create_env(
            env_meta, dataset_path,
            camera_height=args.camera_size, camera_width=args.camera_size,
            change_textures=args.change_textures,
            change_objects=args.change_objects,
        )

        # Run evaluation
        total_successes = 0
        total_episodes = 0
        results = {}

        for demo_idx, demo_key in enumerate(demo_keys):
            init_state = all_init_states[demo_key]
            task_desc = all_task_descriptions[demo_key]
            if args.prompt:
                task_desc = args.prompt

            print(colored(f"\n[{demo_idx+1}/{num_demos}] Demo: {demo_key} | Task: {task_desc}", "cyan"))

            # Reset environment with model xml and ep_meta for exact reconstruction
            env.reset()
            reset_dict = {"states": init_state}
            if all_model_files[demo_key] is not None:
                reset_dict["model"] = all_model_files[demo_key]
            if all_ep_metas[demo_key] is not None:
                reset_dict["ep_meta"] = all_ep_metas[demo_key]
            obs = env.reset_to(reset_dict)

            action_plan = collections.deque()
            replay_images = []
            done = False
            success = False

            # Optional: skip ahead in demo (start from a later state)
            if args.start_from_state > 0:
                with h5py.File(dataset_path, 'r') as f:
                    demo_actions = f[f'data/{demo_key}/actions'][:]
                skip_steps = int(args.start_from_state * len(demo_actions))
                for t in range(min(skip_steps, len(demo_actions))):
                    obs, reward, done, info = env.step(demo_actions[t])
                    if done:
                        break
                if done:
                    print(f"  Demo completed during skip phase")
                    continue

            # Wait for objects to settle
            # robomimic EnvRobosuite wraps robosuite env: action dim from inner env
            action_dim = env.env.action_dim
            dummy_action = np.zeros(action_dim)
            dummy_action[-1] = -1.0  # gripper closed
            for _ in range(args.num_wait_steps):
                obs, _, _, _ = env.step(dummy_action)

            # Main rollout loop
            if args.debug:
                max_steps = 20
            for t in range(max_steps):
                try:
                    # Preprocess observations for pi0
                    pi0_obs = preprocess_obs_for_pi0(obs, resize_size=224)
                    pi0_obs["prompt"] = task_desc

                    # Save frame for replay video (all camera views side by side)
                    video_views = []
                    for cam_key in ['robot0_agentview_left_image', 'robot0_agentview_right_image', 'robot0_eye_in_hand_image']:
                        if cam_key in obs:
                            cam_img = obs[cam_key]
                            if cam_img.ndim == 3 and cam_img.shape[0] in (1, 3):
                                cam_img = np.transpose(cam_img, (1, 2, 0))
                            cam_img = image_tools.convert_to_uint8(np.ascontiguousarray(cam_img))
                            video_views.append(cam_img)
                    # Resize all views to same height and concatenate horizontally
                    target_h = video_views[0].shape[0]
                    resized_views = []
                    for v in video_views:
                        if v.shape[0] != target_h:
                            v = cv2.resize(v, (int(v.shape[1] * target_h / v.shape[0]), target_h))
                        resized_views.append(v)
                    replay_images.append(np.concatenate(resized_views, axis=1))

                    # Query pi0 policy if action plan is empty
                    if not action_plan:
                        result = client.infer(pi0_obs)
                        action_chunk = result["actions"]
                        # Take replan_steps actions from the chunk
                        n_to_take = min(args.replan_steps, len(action_chunk))
                        action_plan.extend(action_chunk[:n_to_take])

                    # Execute next action
                    action = action_plan.popleft()
                    action = np.array(action, dtype=np.float64)

                    # pi0 LIBERO outputs 7D actions [pos(3), rot(3), gripper(1)]
                    # RoboCasa may need different action dims depending on controller config
                    if args.debug:
                        pdb.set_trace()
                    expected_dim = action_dim
                    if len(action) < expected_dim:
                        # Pad with zeros and -1 for gripper (closed)
                        padding = np.zeros(expected_dim - len(action))
                        if expected_dim - len(action) >= 1:
                            padding[-1] = -1.0
                        action = np.concatenate([action, padding])
                    elif len(action) > expected_dim:
                        action = action[:expected_dim]

                    obs, reward, done, info = env.step(action)

                    # Check task success (robosuite done is always False)
                    if env.is_success()["task"]:
                        success = True
                        break

                except Exception as e:
                    print(colored(f"  Error at step {t}: {e}", "red"))
                    traceback.print_exc()
                    break

            total_episodes += 1
            if success:
                total_successes += 1

            # Save replay video
            if replay_images:
                suffix = "success" if success else "failure"
                video_path = os.path.join(
                    output_dir, f"{demo_key}_{suffix}.mp4"
                )
                imageio.mimwrite(
                    video_path,
                    [np.asarray(x) for x in replay_images],
                    fps=10,
                )

            results[demo_key] = {
                "success": success,
                "steps": t + 1,
                "task": task_desc,
            }

            print(colored(
                f"  {'SUCCESS' if success else 'FAILURE'} | Steps: {t+1} | "
                f"Running: {total_successes}/{total_episodes} "
                f"({100*total_successes/total_episodes:.1f}%)",
                "green" if success else "red"
            ))

        env.close()

    # === Multi-env path ===
    else:
        n_envs = min(n_envs, num_demos)
        print(colored(f"Launching {n_envs} parallel environments...", "yellow"))

        # Split demo_keys into chunks (round-robin for balanced load)
        chunks = [[] for _ in range(n_envs)]
        chunk_global_indices = [[] for _ in range(n_envs)]
        for i, dk in enumerate(demo_keys):
            chunks[i % n_envs].append(dk)
            chunk_global_indices[i % n_envs].append(i)

        worker_args_list = []
        for w in range(n_envs):
            if not chunks[w]:
                continue
            worker_args_list.append((
                w,                          # worker_id
                copy.deepcopy(env_meta),    # env_meta (deepcopy so each worker has its own)
                dataset_path,
                chunks[w],                  # this worker's demo_keys
                num_demos,                  # total demos (for progress display)
                {dk: all_init_states[dk] for dk in chunks[w]},
                {dk: all_task_descriptions[dk] for dk in chunks[w]},
                {dk: all_demo_lengths[dk] for dk in chunks[w]},
                {dk: all_model_files[dk] for dk in chunks[w]},
                {dk: all_ep_metas[dk] for dk in chunks[w]},
                args,
                max_steps,
                output_dir,
                chunk_global_indices[w],    # global indices for progress display
            ))

        # Use spawn to avoid issues with MuJoCo/OpenGL in forked processes
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes=len(worker_args_list)) as pool:
            worker_results = pool.map(_eval_worker, worker_args_list)

        # Aggregate results from all workers
        results = {}
        for wr in worker_results:
            results.update(wr)

        total_successes = sum(1 for r in results.values() if r["success"])
        total_episodes = len(results)

    # Save final results
    final_results = {
        "success_rate": total_successes / max(total_episodes, 1),
        "total_successes": total_successes,
        "total_episodes": total_episodes,
        "n_envs": n_envs,
        "per_demo": results,
    }
    with open(os.path.join(output_dir, 'eval_results.json'), 'w') as f:
        json.dump(final_results, f, indent=2)

    print(colored(
        f"\n{'='*60}\n"
        f"FINAL: {total_successes}/{total_episodes} "
        f"({100*total_successes/max(total_episodes,1):.1f}%)\n"
        f"Results saved to: {output_dir}\n"
        f"{'='*60}",
        "green"
    ))

    return final_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate pi0 policy on RoboCasa tasks")

    # Dataset
    parser.add_argument('--dataset_key', type=str, default=None,
                        help='Key from data registry (e.g., PnPCounterToCab_mg_fixed_224)')
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Direct path to HDF5 dataset (overrides dataset_key)')

    # pi0 server
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=8000)

    # Task
    parser.add_argument('--prompt', type=str, default=None,
                        help='Task description override (if None, uses lang from dataset)')

    # Eval settings
    parser.add_argument('--num_demos', type=int, default=None,
                        help='Number of demos to evaluate (default: all)')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Max steps per episode (default: auto from dataset)')
    parser.add_argument('--replan_steps', type=int, default=5,
                        help='Re-query policy every N steps')
    parser.add_argument('--start_from_state', type=float, default=0.0,
                        help='Start rollout from this fraction of the demo (0-1)')
    parser.add_argument('--num_wait_steps', type=int, default=10,
                        help='Steps to wait for objects to settle after reset')
    parser.add_argument('--n_envs', type=int, default=1,
                        help='Number of parallel environments (default: 1)')

    # Environment
    parser.add_argument('--camera_size', type=int, default=256,
                        help='Camera render resolution')
    parser.add_argument('--change_textures', action='store_true',
                        help='Randomize textures')
    parser.add_argument('--change_objects', action='store_true',
                        help='Change object instances')

    # Output
    parser.add_argument('--output_dir', type=str,
                        default='/app/data/pi0_eval_results',
                        help='Output directory for results')

    parser.add_argument('--debug', action='store_true',
                        help='debug')

    args = parser.parse_args()
    run_eval(args)


if __name__ == '__main__':
    main()
