# docker run --gpus all -it \
#   --shm-size=82G \
#   -v /home/sruthi.sudhakar/guided_diffusion_policy:/app \
# guided_diffusion_policy_dockerimage:2


cd externals/robocasa
pip install -e .
# clone gym, edit the setup.py, install gym from source 
pip install vllm==0.8.5
pip install qwen_vl_utils==0.0.11
pip uninstall gym -y
cd ../gym 
pip install -e .
pip uninstall numpy -y
pip install --force-reinstall --no-cache-dir numpy==1.23.3
pip install lovely_tensors

# pip uninstall numpy
# pip uninstall numpy
# pip install --force-reinstall --no-cache-dir numpy==1.23.3
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install node
npm install -g @anthropic-ai/claude-code
claude
export TOKENIZERS_PARALLELISM=false
pip install scikit-learn
