#!/bin/bash
# --- Install system dependencies for pygame ---

apt-get update && apt-get install -y \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-mixer-dev \
    libsdl2-ttf-dev \
    libportmidi-dev \
    libfreetype6-dev \
    libjpeg-dev \
    libpng-dev \
    libsmpeg-dev \
    libavformat-dev \
    libswscale-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libgles2-mesa-dev \
    libopenal-dev \
    libsndfile1 \
    python3-dev \
    build-essential

# --- Upgrade pip and setuptools ---
pip install --upgrade pip setuptools wheel


# Create externals2 folder and clone robocasa
cd externals/robocasa
pip install -e .

cd externals/robomimic
pip install -e .

cd externals/robosuite
pip install -e .

git clone https://github.com/SruthiSudhakar/diffusers.git
cd diffusers
git checkout my-edits
pip install -e .

# --- Now install your Python deps ---
# pip install -r docker_requirements_from_pred.txt

pip install -U vllm==0.8.5
pip install qwen_vl_utils==0.0.11
pip uninstall gym -y
cd externals/gym 
pip install -e .
pip uninstall numpy -y
pip install --force-reinstall --no-cache-dir numpy==1.23.3
pip install open_clip
pip install timm
pip install peft
pip install h5py
pip uninstall numpy -y
pip uninstall numpy -y
pip install --force-reinstall --no-cache-dir numpy==1.23.3
pip install protobuf==3.20.3

# curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
# source ~/.bashrc
# nvm install node
# npm install -g @anthropic-ai/claude-code
# claude
export TOKENIZERS_PARALLELISM=false

# # Set environment variables for MuJoCo/OpenGL rendering
export PYOPENGL_PLATFORM=osmesa
export MUJOCO_GL=osmesa
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

export PYTHONPATH=/app/cosmos-predict2:$PYTHONPATH

pip uninstall gym -y
cd externals/gym 
pip install -e .
cd ../../
pip install bitsandbytes
git config --global --add safe.directory /app
pip install open_clip_torch
pip install flask
