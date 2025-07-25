#!/bin/bash

# Docker build script for guided_diffusion_policy with dpsvd conda environment

# Set variables
IMAGE_NAME="guided_diffusion_policy_dockerimage"
TAG="2"

echo "Building Docker image: ${IMAGE_NAME}:${TAG}"

# Build the Docker image
docker build -f Dockerfile . -t ${IMAGE_NAME}:${TAG}

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Docker image built successfully!"
    echo ""
    echo "To run the container with GPU support:"
    echo "docker run --gpus all -it --rm -v \$(pwd):/app ${IMAGE_NAME}:${TAG}"
    echo ""
    echo "To run with specific GPUs (e.g., GPU 0 and 1):"
    echo "docker run --gpus '\"device=0,1\"' -it --rm -v \$(pwd):/app ${IMAGE_NAME}:${TAG}"
    echo ""
    echo "To run with X11 forwarding for GUI applications:"
    echo "docker run --gpus all -it --rm -v \$(pwd):/app -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=\$DISPLAY ${IMAGE_NAME}:${TAG}"
else
    echo "Docker build failed!"
    exit 1
fi