# Installation
1. create a new conda environment
then install these torch libraries:
```
pip install torch==1.12.1+cu116
pip install torchaudio==0.12.1+cu116
pip install torchvision==0.13.1+cu116
```

2. run pip install -r /proj/vondrick3/sruthi/robots/diffusion_policy/jgdreq.txt

3. clone and install robomimic repo by running:
```
git clone https://github.com/SruthiSudhakar/robomimic
pip install -e .
``` 

4. clone and install robosuite repo by running:
```
git clone https://github.com/SruthiSudhakar/robosuite
pip install -e .
```

5. clone and install robocasa repo by running:
```
git clone https://github.com/SruthiSudhakar/robocasa
pip install -e .
```

# Setuping macros
```
export LD_LIBRARY_PATH=:/home/sruthi/.mujoco/mujoco210/bin:/usr/lib/nvidia
export MUJOCO_GL=osmesa 
export HYDRA_FULL_ERROR=1
```

# Training

All training can be done from the train_robocasa.py script
Here is an example of training a policy on the OpenSingleDoor task

```
CUDA_VISIBLE_DEVICES=0 python train.py \
    --config-dir=. \
    --config-name=image_robocasa_gdp.yaml \
    training.seed=42 \
    dataloader.batch_size=256 \
    val_dataloader.batch_size=256 \
    hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_robocasa_OpenSingleDoor' \
    task.dataset.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenSingleDoor/2024-04-24/demo_gentex_im128_randcams.hdf5 \
    task.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenSingleDoor/2024-04-24/demo_gentex_im128_randcams.hdf5 \
    task.env_runner.dataset_path=/proj/vondrick3/sruthi/robots/robocasa/datasets/v0.1/single_stage/kitchen_doors/OpenSingleDoor/2024-04-24/demo_gentex_im128_randcams.hdf5 \
    task.env_runner.max_steps=500 
    ```

You can also use accelerate to do multi-gpu training by chainging the first line of that command above ^ to this:
```
accelerate launch --num_machines 1 --num_processes=1 --gpu_ids=7 --main_process_port=8074
```
# Testing

All testing is done from the ogeval.py script

Here is an example of testing a certain checkpoint. note that the script autoamtically finds the dataset that was used to test on, but you can change the script to test on a differetn dataset. 

```
python ogeval.py 
    --checkpoint /proj/vondrick3/sruthi/robots/diffusion_policy/data/outputs/2024.11.30/09.52.37_train_diffusion_unet_hybrid_robocasa_coffeeservemug/checkpoints/epoch=1000-test_mean_score=0.000.ckpt \
    --device cuda:6 \
    --robocasa
```