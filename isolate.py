from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import cv2, numpy as np, os, shutil, subprocess, tempfile
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

SYSTEM_PROMPT_CRITIC = """You are a helpful video analyzer."""
USER_PROMPT_CRITIC="This video shows a robot trying to place an object on a plate near the sink.\n\nWatch what happens AFTER the robot picks up the object:\n- TOWARDS: Robot successfully moves the object towards the plate (task succeeds)\n- AWAY: Robot fails and moves the object away from the plate (task fails)\n\nImportant: Judge based on whether the robot completes the task successfully or not.\n\nYour response MUST be:\nDirection: [TOWARDS/AWAY]\nConfidence: [High/Medium/Low]\nReasoning: [Brief explanation]"
MODEL_PATH = 'nvidia/Cosmos-Reason1-7B' #"/proj/vondrick3/sruthi/robots/sruthi_cosmos_reason1/models--nvidia--Cosmos-Reason1-7B/snapshots/1674a723286fd4207ddd80bdeebf63902a6676ee"
print('THE MODEL PATH IS', MODEL_PATH)
TEMPRATURE = 0.3
LLM_GPU_ID=0
# Initialize LLM once
print(f"Initializing LLM on GPU {LLM_GPU_ID}...")

llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 1, "video": 1},
    enforce_eager=True,
    device=f'cuda:{LLM_GPU_ID}',
    max_num_seqs=10,  # Allow batch processing
    gpu_memory_utilization=0.6,
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

video_paths_list=['data/checkpoints/dp_model/epoch=1100-val_loss=0.037/expert_demo_test/PnPSinkToCounter_Human_725165220_/testmedia/0_0_ge9ylge2.mp4',
                  'data/checkpoints/dp_model/epoch=1100-val_loss=0.037/expert_demo_test/PnPSinkToCounter_Human_725165220_/trainmedia/47_47_m34tkaxs.mp4']
batch_results = process_batch(llm, video_paths_list, processor, sampling_params)
print(batch_results)