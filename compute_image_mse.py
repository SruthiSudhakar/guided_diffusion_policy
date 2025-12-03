import numpy as np
from PIL import Image
import sys

def compute_mse(image_path):
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        
        height, width, channels = img_array.shape
        mid_point = width // 2
        
        left_half = img_array[:, :mid_point-10, :]
        right_half = img_array[:, mid_point:mid_point*2-10, :] # Ensure equal size if width is odd
        
        # Check if dimensions match (they should if we slice correctly)
        if left_half.shape != right_half.shape:
            print(f"Error: Halves have different shapes: {left_half.shape} vs {right_half.shape}")
            return

        mse = np.mean((left_half - right_half) ** 2)
        print(f"Image: {image_path}")
        print(f"Dimensions: {width}x{height}")
        print(f"MSE between left and right half: {mse}")
        
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    image_path = "data/checkpoints/dp_model/epoch=1100-val_loss=0.037/nov21_generate_dp_data_all_demos/overlay_images/svsf_PnPSinkToCounter_mg_val_kbpckt_firsthalf_112121187__trainmedia_176_28_54byizr9_frame_000006_PnPSinkToCounter_mg_val_kbpckt_firsthalf_1121213054__trainmedia_176_28_rwanyrx7_frame_000006.png"
    compute_mse(image_path)
