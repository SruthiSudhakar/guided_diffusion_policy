import os
import pickle
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import random
import time
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
import json
import torch.nn.functional as F
from PIL import Image
# from google import genai
import google
import ast
import concurrent.futures
import time
from typing_extensions import TypedDict, NotRequired, Annotated
import PIL
import logging; logging.disable(logging.CRITICAL)

import logging
from contextlib import contextmanager

import cv2
import numpy as np

import cv2, numpy as np, os, shutil, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import itertools
import re

import transformers
from peft import PeftModel
import qwen_vl_utils
from transformers import AutoModelForVision2Seq

import subprocess
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import CLIPTokenizer, CLIPModel
import torch
# Load the CLIP model and tokenizer
clip_model_name = "openai/clip-vit-base-patch32"  # You can choose other models if desired
clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
clip_model = CLIPModel.from_pretrained(clip_model_name)

import math
from collections import defaultdict

def elo(expected, score, k=32):
    return k * (score - expected)

def expected_score(rating_a, rating_b):
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

def update_ratings(rating_a, rating_b, winner, k=32):
    exp_a = expected_score(rating_a, rating_b)
    exp_b = 1 - exp_a

    if winner == "a":
        score_a, score_b = 1, 0
    else:  # winner == "b"
        score_a, score_b = 0, 1
    new_a = rating_a + elo(exp_a, score_a, k)
    new_b = rating_b + elo(exp_b, score_b, k)
    return new_a, new_b

def batch_update_ratings(ratings, results, k=32, use_magnitude=True, score_range=100):
    """
    ratings: dict[player] = rating
    results: list of tuples, either:
             - (player_a, player_b, winner) where winner is "a", "b", or "draw"
             - (player_a, player_b, winner, score_magnitude) for magnitude-based scoring
    use_magnitude: If True and score_magnitude provided, use it to determine degree of win
    score_range: Expected range of score magnitudes (default 100 for [-100, 100])
    """
    deltas = {p: 0 for p in ratings}

    # Compute rating change for each game based on *initial* ratings
    for result in results:
        if len(result) == 3:
            a, b, winner = result
            score_magnitude = None
        else:
            a, b, winner, score_magnitude = result

        exp_a = expected_score(ratings[a], ratings[b])
        exp_b = 1 - exp_a

        if use_magnitude and score_magnitude is not None:
            # Use magnitude to determine the score (0 to 1 scale)
            # score > 0 means b wins, score < 0 means a wins
            # Normalize from [-score_range, score_range] to [0, 1]
            normalized_score = (score_magnitude + score_range) / (2.0 * score_range)
            s_b = max(0.0, min(1.0, normalized_score))  # Clamp to [0, 1]
            s_a = 1.0 - s_b
        else:
            # Binary outcome
            if winner == "a":
                s_a, s_b = 1, 0
            elif winner == "b":
                s_a, s_b = 0, 1
            else:
                s_a, s_b = 0.5, 0.5

        deltas[a] += k * (s_a - exp_a)
        deltas[b] += k * (s_b - exp_b)

    # Apply all updates simultaneously (order-independent)
    for p in ratings:
        ratings[p] += deltas[p]

    return ratings

def get_gpu_with_lowest_memory_util():
    # Run the nvidia-smi command to get the GPU status
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Parse the result and find the GPU with the lowest memory usage
    gpu_info = result.stdout.strip().split('\n')
    
    min_util_gpu = None
    min_util = float('inf')  # Initialize with a large number
    
    for gpu in gpu_info:
        index, memory_used, memory_total = map(int, gpu.split(', '))
        memory_util = memory_used / memory_total  # Memory utilization ratio
        
        if memory_util < min_util:
            min_util = memory_util
            min_util_gpu = index

    return min_util_gpu, min_util

def add_text_to_image(input_image_path, text):
    # Read image with OpenCV (much faster than ffmpeg)
    img = cv2.imread(input_image_path)
    if img is None:
        return  # Skip if image cannot be read

    # Convert text to string if needed
    text = str(text)

    # Add text with OpenCV (purple color in BGR format)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (128, 0, 128)  # Purple in BGR
    border_color = (128, 0, 128)  # Purple border
    thickness = 2
    border_thickness = 4
    position = (50, 50)

    # Draw border (thicker text in background)
    cv2.putText(img, text, position, font, font_scale, border_color, border_thickness, cv2.LINE_AA)
    # Draw main text
    cv2.putText(img, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)

    # Write image back (much faster than ffmpeg)
    cv2.imwrite(input_image_path, img)

def images_to_video_side_by_side(images1, images2, images3, output_path='output.mp4', fps=10):
    # Ensure all images are the same size and are in the correct format
    height, width = images1[0].shape[1], images1[0].shape[2]
    
    # Assuming all images in the lists are of the same size and dimensions
    # Concatenate them side by side (width * 3)
    new_width = width * 3
    new_height = height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    # Assuming all three lists have the same length
    num_frames = len(images1)
    
    for i in range(num_frames):
        # Extract one frame from each list
        img1 = images1[i]
        img2 = images2[i]
        img3 = images3[i]
        
        # Ensure each frame is (H, W, C) by transposing from (C, H, W) if needed
        img1 = np.transpose(img1, (1, 2, 0)) if img1.shape[0] == 3 else img1
        img2 = np.transpose(img2, (1, 2, 0)) if img2.shape[0] == 3 else img2
        img3 = np.transpose(img3, (1, 2, 0)) if img3.shape[0] == 3 else img3
        
        # Check if the images have an alpha channel and remove it if necessary
        if img1.shape[-1] == 4:
            img1 = img1[..., :3]
        if img2.shape[-1] == 4:
            img2 = img2[..., :3]
        if img3.shape[-1] == 4:
            img3 = img3[..., :3]
        
        # Convert the frames to uint8 if necessary (normalize values to 0-255 range)
        img1 = (img1 * 255).astype(np.uint8) if img1.dtype == np.float32 else img1
        img2 = (img2 * 255).astype(np.uint8) if img2.dtype == np.float32 else img2
        img3 = (img3 * 255).astype(np.uint8) if img3.dtype == np.float32 else img3

        # Concatenate the images horizontally (side by side)
        concatenated_frame = np.hstack((img1, img2, img3))

        # Write the concatenated frame to the video
        video.write(cv2.cvtColor(concatenated_frame, cv2.COLOR_RGB2BGR))

    video.release()

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

def _load_and_prepare_image(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert('RGB')
    return np.array(img, dtype=np.float32)

def create_overlay_image(image1_path: str, image2_path: str) -> Image.Image:
    arr1 = _load_and_prepare_image(image1_path)
    arr2 = _load_and_prepare_image(image2_path)
    if arr1.shape != arr2.shape:
        img1 = Image.fromarray(arr1.astype(np.uint8))
        img2 = Image.fromarray(arr2.astype(np.uint8))
        if img1.size != img2.size:
            img2 = img2.resize(img1.size, Image.LANCZOS)
            arr2 = np.array(img2, dtype=np.float32)
    height, width = arr1.shape[:2]
    img1 = Image.fromarray(arr1.astype(np.uint8))
    img2 = Image.fromarray(arr2.astype(np.uint8))
    combined = Image.new('RGB', (width * 2 + 2, height))
    combined.paste(img1, (0, 0))
    combined.paste(Image.new('RGB', (2, height), (255, 255, 0)), (width, 0))
    combined.paste(img2, (width + 2, 0))
    base1, _ = os.path.splitext(image1_path)
    base2, _ = os.path.splitext(image2_path)
    output_path = f"{base1}_and_{base2.split('/')[-1]}_overlay.png"
    combined.save(output_path)
    return combined, output_path

def save_last_frame_method(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get total number of frames
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Move to last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("Could not read the last frame.")

    # Crop first 1/3 width
    h, w, _ = frame.shape
    one_third_width = w // 3
    cropped = frame#[:, :one_third_width]

    # Build output path (replace .mp4 with _last_frame.png)
    base, _ = os.path.splitext(video_path)
    output_path = f"{base}_last_frame.png"

    cv2.imwrite(output_path, cropped)
    return output_path

def get_user_input_direct_selection(all_video_paths, model, processor, n_envs, PROMPTS):
    best_indices = []
    raw_results = []

    # Validate input: list of envs, each with list of sample videos
    if not all_video_paths or not isinstance(all_video_paths[0], (list, tuple)):
        raise ValueError("all_video_paths must be a list of lists of video paths per environment")

    n_envs_local = len(all_video_paths)
    n_samples = len(all_video_paths[0]) if n_envs_local > 0 else 0
    if n_samples < 2:
        # If only one sample, trivially choose index 0 for each env
        return [0 for _ in range(n_envs_local)], [[1.0] for _ in range(n_envs_local)]

    print("\n" + "="*80)
    print("HUMAN-IN-THE-LOOP SAMPLE SELECTION")
    print("="*80)
    print(f"Number of environments: {n_envs_local}")
    print(f"Number of samples per environment: {n_samples}")
    print("\nFor each environment, you will see the last frame images from all samples.")
    print("You need to select which sample shows the most progress toward the task.")
    print("="*80 + "\n")

    # Process each environment
    for env_idx, one_env_videos in enumerate(all_video_paths):
        print(f"\n{'='*80}")
        print(f"ENVIRONMENT {env_idx + 1}/{n_envs_local}")
        print(f"{'='*80}")

        # Extract last frame from each video
        sample_image_paths = []
        for sample_idx, video_path in enumerate(one_env_videos):
            last_frame_path = save_last_frame_method(video_path)
            sample_image_paths.append(last_frame_path)
            print(f"Sample {sample_idx}:")
            print(f"  Video: {video_path}")
            print(f"  Last frame image: {last_frame_path}")

        print(f"\nPlease review the {n_samples} last frame images above.")
        print(f"Which sample (0-{n_samples-1}) shows the MOST progress toward completing the task?")

        while True:
            try:
                user_input = input(f"Enter sample index (0-{n_samples-1}): ").strip()
                selected_idx = int(user_input)
                if 0 <= selected_idx < n_samples:
                    break
                else:
                    print(f"Invalid selection. Please enter a number between 0 and {n_samples-1}:")
            except ValueError:
                print(f"Invalid input. Please enter a whole number between 0 and {n_samples-1}:")

        best_indices.append(selected_idx)
        # Create a simple "score" array with 1.0 for selected, 0.0 for others
        scores = [1.0 if i == selected_idx else 0.0 for i in range(n_samples)]
        raw_results.append(scores)

        print(f"✓ Selected sample {selected_idx} for environment {env_idx}")

    print(f"\n{'='*80}")
    print("SELECTION COMPLETE")
    print(f"{'='*80}")
    print(f"Selected indices: {best_indices}")
    print(f"{'='*80}\n")

    return best_indices, raw_results

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
            additional_steps=0,
            LLM_GPU_ID=7,
            llm_path="",
            PROMPTS={},
            wm_checkpoint_path="",
            wm_guidance=7.0,
            wm_num_sampling_steps=10,
            wm_seed=0,
            num_actions_to_execute=8,
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
        self.additional_steps=additional_steps
        self.num_actions_to_execute=num_actions_to_execute
        self.llm_path = llm_path
        if self.choose_sample:
            # Find the GPU with the lowest memory utilization
            LLM_GPU_ID, gpu_utilization = get_gpu_with_lowest_memory_util()

            if LLM_GPU_ID is not None:
                print(f"GPU with the lowest memory utilization: GPU-{LLM_GPU_ID} with {gpu_utilization * 100:.2f}% usage")
            else:
                raise Exception('cannot contain LLM ')
                print("No GPUs found.")
            
            # Initialize LLM once
            print(f"Initializing LLM on GPU {LLM_GPU_ID}...")
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            self.llm = AutoModelForVision2Seq.from_pretrained(
                self.llm_path,
                torch_dtype='bfloat16',
                device_map={"": f"cuda:{LLM_GPU_ID}"},
                trust_remote_code=True,
                # quantization_config=None,
                quantization_config=quantization_config,
            )
            self.llm = self.llm.eval()
            self.processor = transformers.AutoProcessor.from_pretrained(self.llm_path)
            task_description = "Pick and place an object from the sink to the plate on the counter"
            SYSTEM_PROMPT = f"""You are an expert roboticist tasked to compare a side-by-side of 2 images from a robot demonstration and determine which side shows more progress toward completing the task.
            The robot task is: {task_description}
            You will be given a side-by-side of 2 images from the same demonstration, and you need to identify how much closer or behind in task completion is the right image compared to the left."""

            problem = f"""Look at these two side-by-side images of a robot performing the task. \

            Left side image: Shows the robot at one point during the task. \
            Right side image: Shows the robot at another point during the task. \

            Task: Compare the two images and determine the relative progress difference. \
            - If the right image shows more progress toward task completion, respond with a positive number of how much farther (1 to 100) \
            - If the right image shows less progress toward task completion, respond with a negative number (-1 to -100) \

            The number should represent how much more or less progress the right image shows compared to the left."""

            extract_function = float

            PROMPTS = {
                "system_prompt": SYSTEM_PROMPT,
                "problem": problem,
                'extract_function': extract_function,
                'call_function': get_user_input_direct_selection
            }
            self.PROMPTS= PROMPTS

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
        chunk_step_actions={}
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
            chunk_step_actions[chunk_idx]={}
            while not done:
                env_step_index+=1
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, lambda x: torch.from_numpy(x).to(device=device))

                # Reseed RNGs before each policy prediction to get stochastic diffusion samples
                # while keeping environment conditions deterministic
                # Use time and process ID to ensure truly random seed across runs
                random_seed = int((time.time() * 1000000) % (2**31 - 1))
                torch.manual_seed(random_seed)
                np.random.seed(random_seed)
                random.seed(random_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(random_seed)

                # run policy
                with torch.no_grad():
                    if self.save_stuff:
                        del obs_dict['robot0_eef_pos']
                        del obs_dict['robot0_eef_quat']
                        del obs_dict['robot0_gripper_qpos']
                    if self.choose_sample and self.start_sampling<env_step_index<self.end_sampling:
                        reshaped_obs_dict=dict_apply(obs_dict, lambda x: x.repeat_interleave(self.num_samples, dim=0)) #each value in obs_dict is batch_sizex2x3x128x128  
                        action_dict = policy.predict_action(reshaped_obs_dict)[0] #action outputs are batch_sizex8x7
                        actions = action_dict['action_pred'].view(-1, self.num_samples, 16, 7).detach().to('cpu').numpy()
                        print('action var:', np.mean(np.var(actions,axis=1)))

                        add_on = np.tile([0., -0.,  0.,  0., -1.], (actions.shape[0], actions.shape[1], actions.shape[2], 1))
                        extended_env_action = np.concatenate((actions, add_on), axis=-1)
                        current_state = env.call('get_env_state')
                        sd_sample_obs = []
                        step2_sd_sample_obs = {}
                        for extra_step in range(self.additional_steps):  step2_sd_sample_obs[extra_step]=[]
                        current_obs = env.call('get_raw_observations')
                        # Flip images vertically (they come upside down from the renderer)
                        for idx in range(len(current_obs)):
                            for key in ['robot0_agentview_left_image', 'robot0_agentview_right_image', 'robot0_eye_in_hand_image']:
                                current_obs[idx][key] = current_obs[idx][key][::-1, :, :]  # Flip along height axis (C, H, W) format

                        pbar=tqdm.tqdm(range(self.num_samples), desc="Trying diff action samples")
                        aggregated_obs = []
                        for sample_idx in pbar:
                            pbar.set_description(f"step: {env_step_index} sampling {sample_idx}/{self.num_samples}")
                            self.generate_next_step(current_obs, [extended_env_action[i:i+1, sample_idx, :self.num_actions_to_execute] for i in range(extended_env_action.shape[0])], f'{self.output_dir}/step_{env_step_index}/sample_{sample_idx}/0')
                            results = env.call_each('hallucinate_step',args_list=[extended_env_action[i:i+1, sample_idx, :self.num_actions_to_execute] for i in range(extended_env_action.shape[0])])#,kwargs_list=[{'current_state': curr_state} for curr_state in current_state])
                            obs = [r[0] for r in results]  # temp_observations from each env
                            processed_obs = [r[1] for r in results]  # temp_processed_obs from each env
                            sd_sample_obs.append(obs)

                            for extra_step in range(self.additional_steps):
                                step2_obs_dict={'language_goal': obs_dict['language_goal'].cpu().numpy()}
                                for key in reshaped_obs_dict.keys():
                                    if key=='language_goal':
                                        continue
                                    step2_obs_dict[key] = np.stack([np.stack([one_env_obs_step[key] for one_env_obs_step in one_env_obs[-2:]]) for one_env_obs in processed_obs])
                                step2_obs_dict = dict_apply(step2_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
                                step2_actions = policy.predict_action(step2_obs_dict)[0]['action_pred'].detach().to('cpu').numpy()
                                add_on = np.tile([0., -0.,  0.,  0., -1.], (step2_actions.shape[0], step2_actions.shape[1], 1))
                                step2_extended_env_action = np.concatenate((step2_actions, add_on), axis=-1)
                                self.generate_next_step(obs,[step2_extended_env_action[i:i+1, :self.num_actions_to_execute] for i in range(step2_extended_env_action.shape[0])], f'{self.output_dir}/step_{env_step_index}/sample_{sample_idx}/{extra_step+1}')
                                results = env.call_each('hallucinate_step',args_list=[step2_extended_env_action[i:i+1, :self.num_actions_to_execute] for i in range(step2_extended_env_action.shape[0])])
                                obs = [r[0] for r in results]  # temp_observations from each env
                                processed_obs = [r[1] for r in results]  # temp_processed_obs from each env
                                step2_sd_sample_obs[extra_step].append(obs)
                            env.call_each('reset_after_hallucination',args_list=[(curr_state,) for curr_state in current_state])

                        print('saving videos')
                        os.makedirs(f'{self.output_dir}/videos',exist_ok=True)
                        videos = [[] for _ in range(env.num_envs)]
                        for sample_idx, one_of_n_samples in enumerate(sd_sample_obs):
                            for env_idx, one_env in enumerate(one_of_n_samples):
                                # for view in ['robot0_agentview_right_image']:#,'robot0_agentview_left_image','robot0_eye_in_hand_image']:
                                list1_of_images = [img['robot0_agentview_right_image'] for img in one_env]
                                for _,v in step2_sd_sample_obs.items():
                                    list1_of_images+=[img['robot0_agentview_right_image'] for img in v[sample_idx][env_idx]]
                                list2_of_images = [img['robot0_agentview_left_image'] for img in one_env]
                                for _,v in step2_sd_sample_obs.items():
                                    list2_of_images+=[img['robot0_agentview_left_image'] for img in v[sample_idx][env_idx]]
                                list3_of_images = [img['robot0_eye_in_hand_image'] for img in one_env]
                                for _,v in step2_sd_sample_obs.items():
                                    list3_of_images+=[img['robot0_eye_in_hand_image'] for img in v[sample_idx][env_idx]]
                                images_to_video_side_by_side(list1_of_images, list2_of_images, list3_of_images, output_path=f'{self.output_dir}/videos/env_{env_idx}_step_{env_step_index}_sample_{sample_idx}_3view.mp4')
                                videos[env_idx].append(f'{self.output_dir}/videos/env_{env_idx}_step_{env_step_index}_sample_{sample_idx}_3view.mp4')                        
                        print('Getting user input for sample selection')
                        # flattened_videos_list = list(itertools.chain.from_iterable(videos))
                        best_action_indices, raw_results = self.PROMPTS['call_function'](videos, self.llm, self.processor,n_envs,self.PROMPTS)
                        print('best actions idx:', best_action_indices)
                        # print('sorted orders (best to worst):', sorted_orders)
                        chunk_step_actions[chunk_idx][env_step_index]={
                            'best_action_indices':[str(x) for x in best_action_indices], 
                            'raw_results': [[str(subitem) for subitem in item] for item in raw_results]
                            # 'sorted_orders': [[str(idx) for idx in order] for order in sorted_orders]
                        }
                        json.dump(chunk_step_actions,open(f'{self.output_dir}/sampled_indices.json','w'),indent=4)
                        actions=actions[np.arange(actions.shape[0]), best_action_indices, :self.num_actions_to_execute, :]
                        action_dict={'action':torch.tensor(actions).to(device)}
                    else:
                        print(f'step: {env_step_index}')
                        action_dict, classifier_action_pred = policy.predict_action(obs_dict)
                        """
                        reshaped_obs_dict=dict_apply(obs_dict, lambda x: x.repeat_interleave(100, dim=0)) 
                        temp=policy.predict_action(reshaped_obs_dict)[0]['action']
                        temp=temp.view(-1,100,8,7)
                        for idx in range(6): print('idx:',idx,temp[idx].var(dim=0).mean())
                        """
                # device_transfer
                np_action_dict = dict_apply(action_dict,lambda x: x.detach().to('cpu').numpy())

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
                # actionpath = np.load('/proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.06.05/15.00.33_train_diffusion_unet_hybrid_liftph/checkpoints/epoch=0150-test_mean_score=0.980/1000galift_hammer_8_16_53_6/actions.npy', allow_pickle=True)
                # env_action = actionpath[env_step_index,this_global_slice,:,:]
                
                add_on = np.tile([0., -0.,  0.,  0., -1.], (env_action.shape[0], env_action.shape[1], 1))
                extended_env_action = np.concatenate((env_action, add_on), axis=-1)
                # start=time.time()
                obs, reward, done, info = env.step(extended_env_action)
                # end=time.time()
                # print(colored(f'env step time: {end - start}','green'))

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
                keys_to_save=['obj_to_robot0_eef_pos', 'obj_to_robot0_eef_quat', 'container_to_robot0_eef_pos', 'container_to_robot0_eef_quat', 'obj_pos','obj_quat','container_pos','container_quat']
                # keys_to_save=['0:3', '3:7', '7:10', '10:14', '14:17','17:21','21:24','24:28']
                
                if self.save_stuff:
                    if 'save_rollout_obsdict_robot0_agentview_right_image' not in locals():
                        save_rollout_obsdict_robot0_agentview_right_image = np.expand_dims(np_obs_dict['robot0_agentview_right_image'],0)
                        save_rollout_obsdict_robot0_agentview_left_image = np.expand_dims(np_obs_dict['robot0_agentview_left_image'],0)
                        save_rollout_obsdict_eyeinhand_images = np.expand_dims(np_obs_dict['robot0_eye_in_hand_image'],0)
                        save_rollout_obsdict_robot0s = np.expand_dims(np.concatenate((np_obs_dict['robot0_eef_pos'], np_obs_dict['robot0_eef_quat'],np_obs_dict['robot0_gripper_qpos']), axis=2),0)
                        save_rollout_actions = np.expand_dims(extended_env_action,0)
                        get_raw_obs=env.call('get_raw_observations')[this_local_slice]
                        concated_current_obs=np.concatenate([np.vstack([x[key] for x in get_raw_obs]) for key in keys_to_save],axis=1)
                        save_rollout_raw_obs = np.expand_dims(concated_current_obs,0)
                    else:
                        save_rollout_obsdict_robot0_agentview_right_image = np.vstack((save_rollout_obsdict_robot0_agentview_right_image,np.expand_dims(np_obs_dict['robot0_agentview_right_image'],0)))
                        save_rollout_obsdict_robot0_agentview_left_image = np.vstack((save_rollout_obsdict_robot0_agentview_left_image,np.expand_dims(np_obs_dict['robot0_agentview_left_image'],0)))
                        save_rollout_obsdict_eyeinhand_images = np.vstack((save_rollout_obsdict_eyeinhand_images,np.expand_dims(np_obs_dict['robot0_eye_in_hand_image'],0)))
                        save_rollout_obsdict_robot0s = np.vstack((save_rollout_obsdict_robot0s,np.expand_dims(np.concatenate((np_obs_dict['robot0_eef_pos'], np_obs_dict['robot0_eef_quat'],np_obs_dict['robot0_gripper_qpos']), axis=2),0)))
                        save_rollout_actions = np.vstack((save_rollout_actions,np.expand_dims(extended_env_action,0)))
                        get_raw_obs=env.call('get_raw_observations')[this_local_slice]
                        concated_current_obs=np.concatenate([np.vstack([x[key] for x in get_raw_obs]) for key in keys_to_save],axis=1)
                        save_rollout_raw_obs = np.vstack((save_rollout_raw_obs,np.expand_dims(concated_current_obs,0)))

                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(extended_env_action.shape[1])
                if self.debug:
                    if env_step_index==10:
                        done=True     
                # if chunk_idx+1<n_chunks:
                #     done=True
               
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
                    demogrp.create_dataset('raw_obs', data = save_rollout_raw_obs[:,index])
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
