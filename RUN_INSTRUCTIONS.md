cd guided_diffusion_policy 
docker run --privileged  --gpus all -it \
  --shm-size=82G \
  -v /home/sruthi.sudhakar/guided_diffusion_policy:/app \
  -v /home/sruthi.sudhakar/cosmos-reason1/models--nvidia--Cosmos-Reason1-7B:/app/models--nvidia--Cosmos-Reason1-7B \
guided_diffusion_policy_dockerimage:2
./setup_docker.sh
pip install left #peft?

check openvla_eval.py for run command