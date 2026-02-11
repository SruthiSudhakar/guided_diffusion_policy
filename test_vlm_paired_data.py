"""
Test VLM exactly as it runs in the evaluation pipeline.

Usage:
CUDA_VISIBLE_DEVICES=7 python3 test_vlm_paired_data.py \
    --model_path /workspace/hf_trl/trl/outputs/feb5/PnPAll_20260205_193405/checkpoint-20000 \
    --task_token "[MICROWAVE_TO_COUNTER]" \
    --image1 data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/feb2_expertllm_mg_place_PnPMicrowaveToCounter_mg_fixed_224/PnPMicrowaveToCounter_mg_fixed_224_2310941__choose_sample_True_num_samples_5_ws_0-27/videos/env_17_step_7_sample_0_last_frame.png \
    --image2 data/outputs/jan19/2026.01.19/20.04.49_clip_allPnP/checkpoints/epoch_120_step_40897/feb2_expertllm_mg_place_PnPMicrowaveToCounter_mg_fixed_224/PnPMicrowaveToCounter_mg_fixed_224_2310941__choose_sample_True_num_samples_5_ws_0-27/videos/env_17_step_7_sample_1_last_frame.png \
"""

import os
import gc

import cv2
import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoModelForImageTextToText, AutoProcessor
import qwen_vl_utils
from datetime import datetime
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
    output_path = f"test_images_MTC/overlay_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    combined.save(output_path)
    return combined, output_path


def save_last_frame_method(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Could not read the last frame.")
    base, _ = os.path.splitext(video_path)
    output_path = f"{base}_last_frame.png"
    cv2.imwrite(output_path, frame)
    return output_path


def test_vlm_standalone(model_path, image_path_1, image_path_2, task_token):
    """
    Test the VLM exactly as it is used in the evaluation pipeline.
    Reproduces: model loading, image processing, prompt construction, and inference.

    Usage:
        python robocasa_robomimic_image_runner_eval_clip.py \
            --model_path <path_to_model> \
            --image1 <path_to_image1.png> \
            --image2 <path_to_image2.png> \
            --task_token "[STOVE_TO_COUNTER]"
    """
    print("=" * 60)
    print("VLM STANDALONE TEST (pipeline-identical path)")
    print("=" * 60)

    # --- Step 1: Load model exactly as in __init__ ---
    print(f"\n[1/5] Loading model from {model_path} (8-bit quantized, AutoModelForVision2Seq)...")
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0
    )
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype='bfloat16',
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
    )
    model = model.eval()
    processor = transformers.AutoProcessor.from_pretrained(model_path)
    print("Model loaded on cuda")

    # --- Step 2: Build overlay image exactly as in create_overlay_image ---
    print(f"\n[2/5] Creating overlay image...")
    print(f"   Left:  {image_path_1}")
    print(f"   Right: {image_path_2}")
    overlay_image, overlay_path = create_overlay_image(image_path_1, image_path_2)
    print(f"   Overlay saved to: {overlay_path}")
    print(f"   Overlay size: {overlay_image.size}, mode: {overlay_image.mode}")

    # --- Step 3: Build prompts exactly as in __init__ ---
    SYSTEM_PROMPT = "Compare robot task progress. Respond with a number: positive if right image shows more progress, negative if less."
    problem = f"""Task: {task_token}
Which image shows more task progress? Respond with a number from -100 to 100."""

    print(f"\n[3/5] Prompts (exactly as pipeline constructs them):")
    print(f"   SYSTEM_PROMPT: {repr(SYSTEM_PROMPT)}")
    print(f"   problem:       {repr(problem)}")

    # --- Step 4: Build conversation exactly as in get_qwen_relative_rank_batchify ---
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": overlay_image},
                {"type": "text", "text": problem},
            ],
        },
    ]

    print(f"\n[4/5] Running inference (max_new_tokens=5, do_sample=False)...")
    batch_texts = [
        processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    ]
    batch_image_inputs, _ = zip(
        *[qwen_vl_utils.process_vision_info(conversation)]
    )
    batch_image_inputs = list(batch_image_inputs)

    inputs = processor(
        text=batch_texts,
        images=batch_image_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    print(f"   Input IDs shape: {inputs.input_ids.shape}")
    if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
        print(f"   Pixel values shape: {inputs.pixel_values.shape}")
        print(f"   Pixel values range: [{inputs.pixel_values.min().item():.3f}, {inputs.pixel_values.max().item():.3f}]")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, max_new_tokens=5, do_sample=False, return_dict_in_generate=False
        )
    generated_ids_trimmed = generated_ids[:, inputs.input_ids.shape[1]:]
    batch_outputs = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    raw_output = batch_outputs[0]

    # --- Step 5: Parse exactly as pipeline does ---
    try:
        parsed_value = float(raw_output)
    except (ValueError, TypeError):
        parsed_value = None

    print(f"\n[5/5] Results:")
    print(f"   Raw output:    {repr(raw_output)}")
    print(f"   Parsed value:  {parsed_value}")
    print("=" * 60)

    # Also print the full chat template for debugging
    print(f"\n[DEBUG] Full tokenized prompt (first 200 chars):")
    print(f"   {batch_texts[0][:200]}...")
    print(f"\n[DEBUG] Full tokenized prompt (last 200 chars):")
    print(f"   ...{batch_texts[0][-200:]}")

    # Cleanup
    del inputs, generated_ids, generated_ids_trimmed, batch_image_inputs, batch_texts
    torch.cuda.empty_cache()
    gc.collect()
    print('overlay_path', overlay_path)

    return raw_output, parsed_value


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Test VLM exactly as it runs in the evaluation pipeline"
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to VLM checkpoint")
    parser.add_argument("--image1", type=str, required=True, help="Path to left image (or video to extract last frame)")
    parser.add_argument("--image2", type=str, required=True, help="Path to right image (or video to extract last frame)")
    parser.add_argument("--task_token", type=str, default="[STOVE_TO_COUNTER]",
                        help="Task token, e.g. [STOVE_TO_COUNTER], [COUNTER_TO_CAB], etc.")
    args = parser.parse_args()

    # If inputs are videos, extract last frame (same as pipeline does)
    img1_path = args.image1
    img2_path = args.image2
    if img1_path.endswith(".mp4"):
        print(f"Extracting last frame from video: {img1_path}")
        img1_path = save_last_frame_method(img1_path)
        print(f"  -> {img1_path}")
    if img2_path.endswith(".mp4"):
        print(f"Extracting last frame from video: {img2_path}")
        img2_path = save_last_frame_method(img2_path)
        print(f"  -> {img2_path}")

    test_vlm_standalone(args.model_path, img1_path, img2_path, args.task_token)