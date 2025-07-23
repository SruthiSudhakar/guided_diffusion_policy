# Use an NVIDIA CUDA base image with CUDA 12.1 and Ubuntu 20.04
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu20.04

# Set working directory
WORKDIR /app

# Set non-interactive environment to avoid prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install a broad set of common system packages
RUN apt-get update && apt-get install -y \
    # Build tools and essentials
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    # Python build dependencies
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    libbz2-dev \
    # Graphics and OpenCV dependencies
    libgl1-mesa-glx \
    libgl1-mesa-dev \
    libglew-dev \
    libglfw3-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    # Image and video processing
    libpng-dev \
    libjpeg-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    ffmpeg \
    # Linear algebra and numerical libraries
    libatlas-base-dev \
    liblapack-dev \
    gfortran \
    # Miscellaneous utilities
    libsqlite3-dev \
    vim \
    nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Download and install Python 3.9.21 from source
RUN wget https://www.python.org/ftp/python/3.9.21/Python-3.9.21.tar.xz \
    && tar -xf Python-3.9.21.tar.xz \
    && cd Python-3.9.21 \
    && ./configure --enable-optimizations \
    && make -j$(nproc) \
    && make altinstall \
    && cd .. \
    && rm -rf Python-3.9.21 Python-3.9.21.tar.xz \
    && ln -s /usr/local/bin/python3.9 /usr/local/bin/python \
    && ln -s /usr/local/bin/pip3.9 /usr/local/bin/pip

# Verify Python version
RUN python --version

# # Set up a non-root user with host UID/GID
# ARG USER_ID  # Replace with your UID
# ARG GROUP_ID  # Replace with your GID
# RUN groupadd -g ${GROUP_ID} mygroup && \
#     useradd -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash myuser

# Copy the model training folder and dataset folder
# COPY dp_exp_1_resnet/ ./dp_exp_1_resnet/
# COPY datasets/ ./datasets/

# Install Python dependencies (modify as needed)
RUN pip install --upgrade pip
# RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
# RUN pip install numpy pandas scikit-learn matplotlib

# Optional: If you have a requirements.txt, uncomment and adjust the path
COPY requirements.txt .
RUN pip install --no-deps -r requirements.txt

# # Change ownership of /app to the new user
# RUN chown -R myuser:mygroup /app

# # Switch to the non-root user
# USER myuser

# Set environment variables for the training command
ENV PYTHONPATH=.
ENV CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# ENV CUDA_VISIBLE_DEVICES=2,3
ENV HYDRA_FULL_ERROR=1
ENV PYOPENGL_PLATFORM=osmesa
ENV MUJOCO_GL=osmesa
ENV MESA_LOADER_DRIVER_OVERRIDE=osmesa
ENV LIBGL_ALWAYS_SOFTWARE=1
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu
ENV DISPLAY=""
ENV MESA_SHADER_CACHE_DIR=/local/vondrick/sruthi/mujoco_env/mesa_shader_cache

# Set the working directory to the model training folder
WORKDIR /app/video_model

# Command to run the training script
# CMD ["python", "main.py", "--base=configs/basile_svd_finetune.yaml", "--name=ft1", "--seed=24", "--num_nodes=1", "--wandb=0", "lightning.trainer.devices=0,1,2,3,4,5,6,7", "model.params.use_ema=False"]
# CMD ["python", "main.py", "--base=configs/basile_svd_finetune.yaml", "--name=ft1", "--seed=24", "--num_nodes=1", "--wandb=0", "lightning.trainer.devices=0,1", "model.params.use_ema=False"]