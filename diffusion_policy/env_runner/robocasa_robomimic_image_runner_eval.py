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
import json
import torch.nn.functional as F
from PIL import Image
from google import genai
import google
import ast
import concurrent.futures
import time
from typing_extensions import TypedDict, NotRequired, Annotated
import PIL
import logging; logging.disable(logging.CRITICAL)
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

import logging
from contextlib import contextmanager

import cv2
import numpy as np

import cv2, numpy as np, os, shutil, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

def images_to_video(images, output_path='output.mp4', fps=10):
    # Assume all images are (3, H, W)
    height, width = images[0].shape[1], images[0].shape[2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img in images:
        frame = np.transpose(img, (1, 2, 0))  # CHW -> HWC
        if frame.shape[-1] == 4:
            frame = frame[..., :3]
        frame = (frame * 255).astype(np.uint8) if frame.dtype == np.float32 else frame
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video.release()


class Action(TypedDict):
    action: int
    scene_description: str
    completion_percentage: int

class Output(TypedDict):
    actions: list[Action]
    best_action: Action
    error: NotRequired[str]        # optional field for graceful failures
client = genai.Client(api_key="AIzaSyD6MKO5Hn1ryZ4mqqnLCGNvsERNXcS5pI8")
# model = genai.GenerativeModel("models/gemini-2.0-flash")

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
SYSTEM_PROMPT_CRITIC = """You are a helpful video analyzer."""
USER_PROMPT_CRITIC="This video shows a robot trying to place an object on a plate near the sink.\n\nWatch what happens AFTER the robot picks up the object:\n- TOWARDS: Robot successfully moves the object towards the plate (task succeeds)\n- AWAY: Robot fails and moves the object away from the plate (task fails)\n\nImportant: Judge based on whether the robot completes the task successfully or not.\n\nYour response MUST be:\nDirection: [TOWARDS/AWAY]\nConfidence: [High/Medium/Low]\nReasoning: [Brief explanation]"
MODEL_PATH = "/proj/vondrick3/sruthi/robots/sruthi_cosmos_reason1/models--nvidia--Cosmos-Reason1-7B/snapshots/1674a723286fd4207ddd80bdeebf63902a6676ee"
print('THE MODEL PATH IS', MODEL_PATH)
TEMPRATURE = 0.3
LLM_GPU_ID=7
# Initialize LLM once
print(f"Initializing LLM on GPU {LLM_GPU_ID}...")
pdb.set_trace()
llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 1, "video": 1},
    enforce_eager=True,
    device=f'cuda:{LLM_GPU_ID}',
    max_num_seqs=100,  # Allow batch processing
)

sampling_params = SamplingParams(
    n=1,
    temperature=TEMPRATURE,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.05,
    max_tokens=4096,
)

# Initialize processor once
processor = AutoProcessor.from_pretrained(MODEL_PATH)

def preprocess_video(video_info, processor):
    """Preprocess a single video for batch processing"""
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_CRITIC},
        {"role": "user", "content": [
                {"type": "text", "text": USER_PROMPT_CRITIC},
                {
                    "type": "video",
                    "video": video_info,
                    "fps": 1,
                },
            ]
        },
    ]
    
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    
    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }
    
    return {
        'llm_inputs': llm_inputs,
        'video_path': video_info,
    }
def process_batch(llm, video_batch, processor, sampling_params):
    """Process a batch of videos using the LLM"""
    # Preprocess all videos in parallel
    with ThreadPoolExecutor(max_workers=mp.cpu_count()) as executor:
        preprocessed = list(executor.map(
            lambda v: preprocess_video(v, processor),
            video_batch
        ))
    
    # Extract LLM inputs
    llm_inputs_list = [item['llm_inputs'] for item in preprocessed]
    
    # Batch inference
    outputs = llm.generate(llm_inputs_list, sampling_params)
    
    # Collect results
    results = {}
    for i, output in enumerate(outputs):
        generated_text = [o.text for o in output.outputs]
        video_path = preprocessed[i]['video_path']
        
        results[video_path] = generated_text
    
    return results

video_paths_list=['data/outputs/2025.02.15/imageonly_11.32.40_usegroupnorm/checkpoints/epoch=1100-val_loss=0.037/jul22_vanilla/PnPSinkToCounter_mg_val_kbpctk_firsthalf_722192245_mr140_9_2_42/trainmedia/2_10_9p6a2h08.mp4',
                  'data/outputs/2025.02.15/imageonly_11.32.40_usegroupnorm/checkpoints/epoch=1100-val_loss=0.037/jul22_vanilla/PnPSinkToCounter_mg_val_kbpctk_firsthalf_722192245_mr140_9_2_42/trainmedia/2_14_3h10g3fd.mp4']
batch_results = process_batch(llm, video_paths_list, processor, sampling_params)
print(batch_results)
# Load the CLIP model and tokenizer
clip_model_name = "openai/clip-vit-base-patch32"  # You can choose other models if desired
clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
clip_model = CLIPModel.from_pretrained(clip_model_name)
def get_gemini_response(env_idx, view, history_prompt, prompt_list):
    # Start chat and send message
    retries = 5
    for attempt in range(retries):
        try:
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt_list,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': Output,
                },
            )
            break  # If the request is successful, exit the loop
        except Exception as e:
            if attempt < retries - 1:
                wait_time = 2 ** attempt      # Exponential backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Operation failed.")
                raise e
    return env_idx, view, response, prompt_list
def get_vlm_rank(all_video_paths, object_ids, step_idx):
    

    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Use tqdm to show the progress bar
        futures = []
        # Submit the tasks for execution
        for env_idx, object_name in enumerate(object_ids):
            n_samples=str(len(all_video_paths))
            task_description = f'Pick the {object_name} from the sink and place it on the plate that is next to the sink.'
            history_prompt=history_prompt.replace('INSERT_TASK_DESC', task_description)
            history_prompt=history_prompt.replace('N_SAMPLES', n_samples)
            prompt1=prompt1.replace('N_SAMPLES', n_samples)
            for view,samples in all_video_paths[env_idx].items():
                prompt_list=[history_prompt,prompt1]
                for sample_idx in range(len(samples)):
                    prompt_list.append(f'Action {sample_idx+1} - Video: ')
                    prompt_list.append(samples[sample_idx])
                    myfile = client.files.upload(file=samples[sample_idx])
                prompt_list.append(prompt2)
                futures.append(executor.submit(get_gemini_response, env_idx, view, history_prompt, prompt_list))
        # Wait for all futures to complete before moving on
        concurrent.futures.wait(futures)
        outputs = {}
        # Collect results as they complete
        for idx in range(len(futures)):
            env_idx, view, response, prompt_list = futures[idx].result()
            # for img_idx,img in enumerate(prompt_list): 
            #     if type(img)==PIL.Image.Image:
            #         img.save(f'test166_{idx}_{img_idx}.jpg') 
        for future in concurrent.futures.as_completed(futures):
            env_idx, view, response, prompt_list = future.result()
            """
            idx=3
            env_idx, view, response, prompt_list = futures[idx].result()
            for idx,img in enumerate(prompt_list): img.save(f'temp_{idx}.jpg') if type(img)==PIL.Image.Image else print('hi') 
            """
            if env_idx not in outputs:
                outputs[env_idx]={}
            try:
                outputs[env_idx][view]={
                    'response':json.loads(response.text),
                }
            except:
                print('outputs not formatted correctly')
                pdb.set_trace()
        for env_idx,_ in outputs.items():
            try:
                pdb.set_trace()
                arc0=[x['completion_percentage'] for x in outputs[env_idx]['robot0_eye_in_hand_image']['response']['actions']]
                arc1=[x['completion_percentage'] for x in outputs[env_idx]['robot0_agentview_left_image']['response']['actions']]
                arc2=[x['completion_percentage'] for x in outputs[env_idx]['robot0_agentview_right_image']['response']['actions']]
                avg_results=[(a + b + c) / 3 for a, b, c in zip(arc0, arc1, arc2)]
                outputs[env_idx]['avg_list']= avg_results
                outputs[env_idx]['best_idx']= avg_results.index(max(avg_results))
                assert outputs[env_idx]['best_idx'] < 10
            except:
                print('some issue. could not get avg list and best idx. or best_idx>=10')
                pdb.set_trace()
    return outputs

def get_gemini_value(current_obs, all_samples_obs, object_ids, step_idx):
    history_prompt=f"You are an expert roboticist tasked with evaluating the progress \
    of a robot performing a task. The task is INSERT_TASK_DESC. We are evaluating N_SAMPLES \
    potential actions that the robot could take from its current state. Each action results in \
    a different outcome, and we have rendered images of these possible outcomes. For each action, \
    output a task completion percentage (from 0 to 100, where 100 means the task is fully completed). \
    The higher the percentage, the more progress the action has made toward placing the object into the sink. \
    Here is the initial robot scene:"

    prompt1="Now here are the N_SAMPLES images of the outcomes from the proposed actions:"

    prompt2="Return a JSON object that matches the given schema and contains:\n \
        • a list of <action, scene_description, completion_percentage>\n \
        • a best_action object.\n Do **not** wrap the JSON in markdown."
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Use tqdm to show the progress bar
        futures = []
        # Submit the tasks for execution
        for env_idx, object_name in enumerate(object_ids):
            n_samples=str(len(all_samples_obs))
            task_description = f'Pick the {object_name} from the sink and place it on the plate that is next to the sink.'
            history_prompt=history_prompt.replace('INSERT_TASK_DESC', task_description)
            history_prompt=history_prompt.replace('N_SAMPLES', n_samples)
            prompt1=prompt1.replace('N_SAMPLES', n_samples)
            for view in ['robot0_agentview_right_image','robot0_agentview_left_image','robot0_eye_in_hand_image']:
                image1 = Image.fromarray(np.flipud(current_obs[env_idx][view]))
                prompt_list=[history_prompt,image1,prompt1]
                for sample_idx in range(len(all_samples_obs)):
                    prompt_list.append(f'Action {sample_idx+1} - Image: ')
                    prompt_list.append(Image.fromarray((all_samples_obs[sample_idx][env_idx][view]*255).transpose(1,2,0).astype('uint8')))
                prompt_list.append(prompt2)
                futures.append(executor.submit(get_gemini_response, env_idx, view, history_prompt, prompt_list))
        # Wait for all futures to complete before moving on
        concurrent.futures.wait(futures)
        outputs = {}
        # Collect results as they complete
        for idx in range(len(futures)):
            env_idx, view, response, prompt_list = futures[idx].result()
            # for img_idx,img in enumerate(prompt_list): 
            #     if type(img)==PIL.Image.Image:
            #         img.save(f'test166_{idx}_{img_idx}.jpg') 
        for future in concurrent.futures.as_completed(futures):
            env_idx, view, response, prompt_list = future.result()
            """
            idx=3
            env_idx, view, response, prompt_list = futures[idx].result()
            for idx,img in enumerate(prompt_list): img.save(f'temp_{idx}.jpg') if type(img)==PIL.Image.Image else print('hi') 
            """
            if env_idx not in outputs:
                outputs[env_idx]={}
            try:
                outputs[env_idx][view]={
                    'response':json.loads(response.text),
                }
            except:
                print('outputs not formatted correctly')
                pdb.set_trace()
        for env_idx,_ in outputs.items():
            try:
                arc0=[x['completion_percentage'] for x in outputs[env_idx]['robot0_eye_in_hand_image']['response']['actions']]
                arc1=[x['completion_percentage'] for x in outputs[env_idx]['robot0_agentview_left_image']['response']['actions']]
                arc2=[x['completion_percentage'] for x in outputs[env_idx]['robot0_agentview_right_image']['response']['actions']]
                avg_results=[(a + b + c) / 3 for a, b, c in zip(arc0, arc1, arc2)]
                outputs[env_idx]['avg_list']= avg_results
                outputs[env_idx]['best_idx']= avg_results.index(max(avg_results))
                print(outputs[env_idx]['avg_list'],outputs[env_idx]['best_idx'])
                if sum(outputs[env_idx]['avg_list'])>10:
                    pdb.set_trace()
                    print('hey')
            except:
                print('could not get avg list and best idx')
                pdb.set_trace()
    return outputs

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
        self.specific_train_exs=specific_train_exs
        self.prompt_with_video=prompt_with_video
        self.additional_steps=additional_steps

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

                # run policy
                with torch.no_grad():
                    if self.save_stuff:
                        del obs_dict['robot0_eef_pos']
                        del obs_dict['robot0_eef_quat']
                        del obs_dict['robot0_gripper_qpos']
                    if self.choose_sample and self.start_sampling<env_step_index<self.end_sampling:
                        sample_number=50
                        reshaped_obs_dict=dict_apply(obs_dict, lambda x: x.repeat_interleave(sample_number, dim=0)) #each value in obs_dict is batch_sizex2x3x128x128  
                        action_dict = policy.predict_action(reshaped_obs_dict)[0] #action outputs are batch_sizex8x7
                        oversampled_actions = action_dict['action_pred'].view(-1, sample_number, 16, 7).detach().to('cpu').numpy()

                        #choosing based on variance
                        mean = np.mean(oversampled_actions, axis=(1),keepdims=True)  # Shape: (batch_size, 100)
                        stds = np.abs(oversampled_actions-mean)
                        interval = (stds.shape[1] // self.num_samples) - 1
                        top_n_indices = np.argsort(np.sum(stds,axis=(2,3)), axis=1)[:, ::interval][:, :self.num_samples]
                        # top_n_indices = np.argsort(np.sum(stds,axis=(2,3)), axis=1)[:, ::interval][:, :self.num_samples]
                        batch_indices = np.arange(oversampled_actions.shape[0])[:, None]
                        actions = oversampled_actions[batch_indices, top_n_indices]
                        print('var before:', np.sum(np.var(oversampled_actions,axis=1)),'var after:', np.sum(np.var(actions,axis=1)))
                        add_on = np.tile([0., -0.,  0.,  0., -1.], (actions.shape[0], actions.shape[1], actions.shape[2], 1))
                        extended_env_action = np.concatenate((actions, add_on), axis=-1)
                        current_state = env.call('get_env_state')
                        sd_sample_obs = []
                        step2_sd_sample_obs = {}
                        for extra_step in range(self.additional_steps):  step2_sd_sample_obs[extra_step]=[]
                        all_samples_obs = []
                        reset_obs=[]
                        current_obs=env.call('get_raw_observations')
                        pbar=tqdm.tqdm(range(self.num_samples), desc="Trying diff action samples")
                        aggregated_obs = []
                        for sample_idx in pbar:
                            pbar.set_description(f"step: {env_step_index} sampling {sample_idx}/{self.num_samples}")
                            obs = env.call_each('hallucinate_step',args_list=[extended_env_action[i:i+1, sample_idx, :8] for i in range(extended_env_action.shape[0])])#,kwargs_list=[{'current_state': curr_state} for curr_state in current_state])
                            sd_sample_obs.append(obs)

                            for extra_step in range(self.additional_steps):
                                step2_obs_dict={'language_goal': obs_dict['language_goal'].cpu().numpy()}
                                for key in reshaped_obs_dict.keys():
                                    if key=='language_goal':
                                        continue
                                    step2_obs_dict[key] = np.stack([np.stack([cv2.resize(one_env_obs_step[key].transpose(1,2,0), (128,128), interpolation=cv2.INTER_AREA).transpose(2,0,1) for one_env_obs_step in one_env_obs[-2:]]) for one_env_obs in obs])
                                step2_obs_dict = dict_apply(step2_obs_dict, lambda x: torch.from_numpy(x).to(device=device))
                                step2_actions = policy.predict_action(step2_obs_dict)[0]['action_pred'].detach().to('cpu').numpy()
                                add_on = np.tile([0., -0.,  0.,  0., -1.], (step2_actions.shape[0], step2_actions.shape[1], 1))
                                step2_extended_env_action = np.concatenate((step2_actions, add_on), axis=-1)
                                obs = env.call_each('hallucinate_step',args_list=[step2_extended_env_action[i:i+1, :8] for i in range(step2_extended_env_action.shape[0])])
                                step2_sd_sample_obs[extra_step].append(obs)
                            env.call_each('reset_after_hallucination',args_list=[(curr_state,) for curr_state in current_state])

                        print('saving videos')
                        os.makedirs(f'{self.output_dir}/videos',exist_ok=True)
                        videos = [{'robot0_agentview_right_image': [],'robot0_agentview_left_image': [],'robot0_eye_in_hand_image': []} for _ in range(env.num_envs)]
                        for sample_idx, one_of_n_samples in enumerate(sd_sample_obs):
                            for env_idx, one_env in enumerate(one_of_n_samples):
                                for view in ['robot0_agentview_right_image','robot0_agentview_left_image','robot0_eye_in_hand_image']:
                                    list_of_images = [img[view] for img in one_env]
                                    for _,v in step2_sd_sample_obs.items():
                                        list_of_images+=[img[view] for img in v[sample_idx][env_idx]]
                                    images_to_video(list_of_images,output_path=f'{self.output_dir}/videos/env_{env_idx}_step_{env_step_index}_sample_{sample_idx}_view_{view}.mp4')
                                    videos[env_idx][view].append(f'{self.output_dir}/videos/env_{env_idx}_step_{env_step_index}_sample_{sample_idx}_view_{view}.mp4')                        
                        print('querying COSMOS-REASON1')
                        best_samples_idx = get_vlm_rank(videos,[x.split('pick the ')[1].split(' from')[0] for x in language_goal], env_step_index)
                        print('best actions idx:', [v['best_idx'] for k,v in best_samples_idx.items()])
                        chunk_step_actions[chunk_idx][env_step_index]=[v['best_idx'] for k,v in best_samples_idx.items()]
                        json.dump(chunk_step_actions,open(f'{self.output_dir}/sampled_indices.json','w'),indent=4)
                        actions=actions[np.arange(actions.shape[0]), [v['best_idx'] for k,v in best_samples_idx.items()], :8, :]
                        action_dict={'action':torch.tensor(actions).to(device)}

                        actions=actions[:, 0, :8, :]
                        action_dict={'action':torch.tensor(actions).to(device)}
                    else:
                        print(f'step: {env_step_index}')
                        action_dict, classifier_action_pred = policy.predict_action(obs_dict)
                        # pdb.set_trace()
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
                #     pdb.set_trace()
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
