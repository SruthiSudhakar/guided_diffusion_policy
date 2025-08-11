# train
"""

#change out the task/path in the yaml file and then run
CUDA_VISIBLE_DEVICES=4,5,6,7 HYDRA_FULL_ERROR=1 accelerate launch --num_processes 4 --gpu_ids 4,5,6,7 --main_process_port 8086 train.py \
    --config-dir=. \
    --config-name=train_robocasa_base_dp_clip_policy.yaml \
    training.seed=42 \
    task.name='PnPSinkToCounter' \
    dataloader.batch_size=48 \
    task.dataset.human_path=externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2/expert_demos_fixed_textures_224.hdf5 \
    'task.dataset.tasks={PnPSinkToCounter: null}'


CUDA_VISIBLE_DEVICES=0,2,3 HYDRA_FULL_ERROR=1 accelerate launch --num_processes 3 --gpu_ids 0,2,3 --main_process_port 8086 train.py \
    --config-dir=. \
    --config-name=train_robocasa_base_dp_clip_policy.yaml \
    training.seed=42 \
    task.name='CloseDrawer' \
    dataloader.batch_size=48 \
    task.dataset.human_path=externals/robocasa/datasets/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/demo_gentex_im128_randcams_im224.hdf5 \
    'task.dataset.tasks={CloseDrawer: null}'

CUDA_VISIBLE_DEVICES=5 HYDRA_FULL_ERROR=1 python train.py \
    --config-dir=. \
    --config-name=train_robocasa_base_dp_clip_policy.yaml \
    training.seed=42 \
    task.name='6tasks' \
    dataloader.batch_size=48 

                                                                                                                                               
CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1 python train.py \
    --config-dir=. \
    --config-name=train_robocasa_base_dp_clip_policy.yaml \
    training.seed=42 \
    task.name='3task' \
    dataloader.batch_size=48 \
    'task.dataset.human_path={CloseDrawer: externals/robocasa/datasets/v0.1/single_stage/kitchen_drawer/CloseDrawer/2024-04-30/demo_gentex_im128_randcams_im224.hdf5 \
                              PnPSinkToCounter: externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPSinkToCounter/2024-04-26_2/expert_demos_fixed_textures_224.hdf5 \
                              PnPStoveToCounter: externals/robocasa/datasets/v0.1/single_stage/kitchen_pnp/PnPStoveToCounter/2024-05-01/demo_gentex_im128_randcams_im224.hdf5}' \
    'task.dataset.tasks={CloseDrawer: null, PnPSinkToCounter: null, PnPStoveToCounter: null}'

"""

#Evalu

"""

CUDA_VISIBLE_DEVICES=1 python evaluate.py \
    --config-name=eval_robocasa_base_policy \
    task_name='CloseDrawer' \
    

"""