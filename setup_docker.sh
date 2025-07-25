# docker run --gpus all -it \
#   --shm-size=82G \
#   -v /home/sruthi.sudhakar/guided_diffusion_policy:/app \
# guided_diffusion_policy_dockerimage:2


cd externals/robocasa
pip install -e .
# clone gym, edit the setup.py, install gym from source 
pip uninstall gym
cd ../gym 
pip install -e .
pip show numpy
# pip uninstall numpy
# pip uninstall numpy
# pip install --force-reinstall --no-cache-dir numpy==1.23.3