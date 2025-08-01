import subprocess
import os

import subprocess
import os

def add_text_to_video_with_ffmpeg(input_video_path, text):
    # Temporary output path
    temp_video_path = input_video_path + '.temp.mp4'
    
    # FFmpeg command to add text to each frame
    command = [
        'ffmpeg',
        '-i', input_video_path,  # Input video file
        '-vf', f"drawtext=text='{text}':x=50:y=50:fontsize=24:fontcolor=#800080:borderw=1:bordercolor=#800080",  # Purple text and border
        '-c:a', 'copy',  # Copy audio without re-encoding
        temp_video_path  # Output video file
    ]
    
    # Run the command and suppress logs
    with open(os.devnull, 'w') as devnull:
        subprocess.run(command, stdout=devnull, stderr=devnull, check=True)
    
    # Rename temporary file to original file after processing
    os.rename(temp_video_path, input_video_path)

# Example usage:
input_video = 'temp.mp4'
add_text_to_video_with_ffmpeg(input_video, 'towards')
