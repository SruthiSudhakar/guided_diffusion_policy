from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import cv2, numpy as np, os, shutil, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import re, json, pdb
from datetime import datetime
import sys
SYSTEM_PROMPT_CRITIC = """You are a helpful video analyzer."""
#TOWARDS AWAY
USER_PROMPT_CRITIC="""This video shows a robot trying to place an object on a plate near the sink.
Watch what happens AFTER the robot picks up the object:
- TOWARDS: Robot successfully moves the object towards the plate (task succeeds)
- AWAY: Robot fails and moves the object away from the plate (task fails)

Important: Judge based on whether the robot completes the task successfully or not.
Your response MUST be:
Direction: [TOWARDS/AWAY]
Confidence: [High/Medium/Low]
Reasoning: [Brief explanation]
"""

#ROBOT ARM
USER_PROMPT_CRITIC = """You are analyzing a video of a robot arm performing a movement task. The robot arm will move in one of two directions:- LEFT: Movement towards the left side of the frame- RIGHT: Movement towards the right side of the frameCarefully observe the entire video sequence and track the robot arm's movement trajectory.Your response should be formatted as:Direction: [LEFT/RIGHT]Reasoning: [Brief explanation of what you observed]"""

#PLATE POS
USER_PROMPT_CRITIC="Analyze this kitchen video and determine if there is a plate on the LEFT or RIGHT side of the sink.\n\nTo determine the position:\n1. Locate the sink in the video\n2. Look for a plate near the sink\n3. Determine if the plate is on the LEFT or RIGHT side of the sink\n\nYour response MUST be:\nPosition: [LEFT/RIGHT]\nConfidence: [High/Medium/Low]\nReasoning: [Brief explanation of where you see the plate relative to the sink]"

#ROBOT ARM
USER_PROMPT_CRITIC = """You are analyzing a video of a robot arm performing a movement task. The robot arm will move in one of two directions:- GREEN: Movement towards the green dot on the frame- RED: Movement towards the red dot on the frame. Carefully observe the entire video sequence and track the robot arm's movement trajectory. Your response should be formatted as:Direction: [GREEN/RED]Reasoning: [Brief explanation of what you observed]"""


MODEL_PATH = "models--nvidia--Cosmos-Reason1-7B/snapshots/1674a723286fd4207ddd80bdeebf63902a6676ee"
print('THE MODEL PATH IS', MODEL_PATH)
TEMPRATURE = 0.3
LLM_GPU_ID=sys.argv[1]
N_QUERIES = 5  # Number of times to query the VLM for each video
# Initialize LLM once
print(f"Initializing LLM on GPU {LLM_GPU_ID}...")

llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 1, "video": 1},
    enforce_eager=True,
    device=f'cuda:{LLM_GPU_ID}',
    max_num_seqs=100,  # Allow batch processing
)

sampling_params = SamplingParams(
    n=N_QUERIES,  # Generate N responses per prompt
    temperature=TEMPRATURE,
    top_k=50,
    top_p=0.95,
    repetition_penalty=1.05,
    max_tokens=4096,
)

# Initialize processor once
processor = AutoProcessor.from_pretrained(MODEL_PATH)

def crop_middle_video(input_path):
    """Extract the middle video from a 3-video side-by-side stack"""
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate dimensions for middle third
    third_width = width // 3
    start_x = third_width
    end_x = 2 * third_width
    
    # Create temporary output file
    temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    temp_output_path = temp_output.name
    temp_output.close()
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (third_width, height))
    
    # Process frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Crop middle third
        cropped_frame = frame[:, start_x:end_x]
        
        # Add circles to the cropped frame
        # Calculate circle positions and size
        circle_radius = min(third_width, height) // 20  # 5% of smaller dimension
        circle_thickness = -1  # Filled circle
        
        # Red circle on top left
        red_center = (circle_radius + 10, circle_radius + 10)
        cv2.circle(cropped_frame, red_center, circle_radius, (0, 0, 255), circle_thickness)
        
        # Green circle on top right
        green_center = (third_width - circle_radius - 10, circle_radius + 10)
        cv2.circle(cropped_frame, green_center, circle_radius, (0, 255, 0), circle_thickness)
        
        out.write(cropped_frame)
    
    # Release resources
    cap.release()
    out.release()
    
    return temp_output_path

def preprocess_video(video_info, processor):
    """Preprocess a single video for batch processing"""
    
    # Crop the middle video from the 3-video stack
    cropped_video_path = crop_middle_video(video_info)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_CRITIC},
        {"role": "user", "content": [
                {"type": "text", "text": USER_PROMPT_CRITIC},
                {
                    "type": "video",
                    "video": cropped_video_path,
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
        'temp_video_path': cropped_video_path,
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
        video_path = preprocessed[i]['video_path']
        
        # Process all N responses for this video
        all_responses = []
        binary_answers = []
        
        for response in output.outputs:
            generated_text = response.text
            all_responses.append(generated_text)
            
            # Extract binary answer from each response
            match = re.search(r'Direction:\s*\[(TOWARDS|AWAY)\]', generated_text, re.IGNORECASE)
            if not match:
                match = re.search(r'Direction:\s*(TOWARDS|AWAY)', generated_text, re.IGNORECASE)
            
            if match:
                answer = match.group(1).lower()
                binary_answers.append(answer)
        
        # Determine final answer by taking max (TOWARDS > AWAY)
        final_answer = None
        if binary_answers:
            towards_count = binary_answers.count('towards')
            away_count = binary_answers.count('away')
            final_answer = 'towards' if towards_count >= away_count else 'away'
            
        results[video_path] = {
            'gentext': all_responses,
            'individual_answers': binary_answers,
            'towards_count': binary_answers.count('towards') if binary_answers else 0,
            'away_count': binary_answers.count('away') if binary_answers else 0,
            'binary': final_answer,
            'n_queries': len(all_responses)
        }

    # # Clean up temporary cropped videos
    # for item in preprocessed:
    #     import time
    #     time.sleep()
    #     if 'temp_video_path' in item and os.path.exists(item['temp_video_path']):
    #         os.remove(item['temp_video_path'])

    return results
video_paths_list = ['data/checkpoints/dp_model/epoch=1100-val_loss=0.037/mg253_guided_specificexs/PnPSinkToCounter_mg_val_kbpckt_firsthalf_81182649__choose_sample_True_num_samples_10_ws_0-10/videos/env_0_step_1_sample_0_3view.mp4',
                    'data/checkpoints/dp_model/epoch=1100-val_loss=0.037/mg253_guided_specificexs/PnPSinkToCounter_mg_val_kbpckt_firsthalf_81182649__choose_sample_True_num_samples_10_ws_0-10/videos/env_0_step_1_sample_1_3view.mp4']
batch_results = process_batch(llm, video_paths_list, processor, sampling_params)
pdb.set_trace()
path_save= datetime.now().strftime("%H%M%S%f") + '.json'
print(f'PATH_SAVE: {path_save}')
with open(path_save, "w") as f:
    json.dump(batch_results, f, indent=4, sort_keys=True)