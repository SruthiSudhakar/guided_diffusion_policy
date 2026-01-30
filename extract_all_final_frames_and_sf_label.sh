#!/bin/bash
# python extract_all_final_frames_and_sf_label.py --base_dataset_path data/checkpoints/dp_model/epoch=1100-val_loss=0.037/nov21_generate_dp_data_all_demos
# python extract_all_final_frames_and_sf_label.py --base_dataset_path data/outputs/dec4/2025.12.03/22.38.27_train_diffusion_unet_clip/checkpoints/epoch_70_step_4188/expert_acc

python extract_video_frames.py --root_dir data/outputs/dec4/2025.12.03/22.38.27_train_diffusion_unet_clip/checkpoints/epoch_70_step_4188/expert_acc
python extract_all_final_frames_and_sf_label.py --base_dataset_path data/outputs/dec4/2025.12.03/22.38.27_train_diffusion_unet_clip/checkpoints/epoch_70_step_4188/expert_acc

python extract_video_frames.py --root_dir data/outputs/dec4/2025.12.03/22.38.27_train_diffusion_unet_clip/checkpoints/epoch_70_step_4188/dec22_na_na_16_onlyplace
python extract_all_final_frames_and_sf_label.py --base_dataset_path data/outputs/dec4/2025.12.03/22.38.27_train_diffusion_unet_clip/checkpoints/epoch_70_step_4188/dec22_na_na_16_onlyplace

python extract_video_frames.py --root_dir data/outputs/dec4/2025.12.03/22.39.39_train_diffusion_unet_clip/checkpoints/epoch_30_step_2231/dec18_PnPCabToCounter_mg_fixed_224_na_na_16
python extract_all_final_frames_and_sf_label.py --base_dataset_path data/outputs/dec4/2025.12.03/22.39.39_train_diffusion_unet_clip/checkpoints/epoch_30_step_2231/dec18_PnPCabToCounter_mg_fixed_224_na_na_16

# python extract_video_frames.py --root_dir data/outputs/dec4/2025.12.04/00.01.35_clip_justPnPCounterToSink/checkpoints/epoch_20_step_1952/dec18_PnPCounterToSink_mg_fixed_224_na_na_16
# python extract_all_final_frames_and_sf_label.py --base_dataset_path data/outputs/dec4/2025.12.04/00.01.35_clip_justPnPCounterToSink/checkpoints/epoch_20_step_1952/dec18_PnPCounterToSink_mg_fixed_224_na_na_16

# python extract_video_frames.py --root_dir data/outputs/dec4/2025.12.04/00.06.44_clip_justCoffeeServeMug/checkpoints/epoch_30_step_2231/dec4_na_na_16
# python extract_all_final_frames_and_sf_label.py --base_dataset_path data/outputs/dec4/2025.12.04/00.06.44_clip_justCoffeeServeMug/checkpoints/epoch_30_step_2231/dec4_na_na_16

# python extract_video_frames.py --root_dir data/outputs/dec4/2025.12.04/05.15.03_clip_justPnPCounterToStove/checkpoints/epoch_30_step_2076/dec18_PnPCounterToStove_mg_fixed_224_na_na_16
# python extract_all_final_frames_and_sf_label.py --base_dataset_path data/outputs/dec4/2025.12.04/05.15.03_clip_justPnPCounterToStove/checkpoints/epoch_30_step_2076/dec18_PnPCounterToStove_mg_fixed_224_na_na_16

# python extract_video_frames.py --root_dir data/outputs/dec4/2025.12.04/05.15.03_clip_justPnPCounterToStove/checkpoints/epoch_30_step_2076/dec4_na_na_16
# python extract_all_final_frames_and_sf_label.py --base_dataset_path data/outputs/dec4/2025.12.04/05.15.03_clip_justPnPCounterToStove/checkpoints/epoch_30_step_2076/dec4_na_na_16

# python extract_video_frames.py --root_dir data/outputs/dec4/2025.12.04/05.17.32_clip_justPnPMicrowaveToCounter/checkpoints/epoch_60_step_4025/dec18_PnPMicrowaveToCounter_mg_fixed_224_na_na_16
# python extract_all_final_frames_and_sf_label.py --base_dataset_path data/outputs/dec4/2025.12.04/05.17.32_clip_justPnPMicrowaveToCounter/checkpoints/epoch_60_step_4025/dec18_PnPMicrowaveToCounter_mg_fixed_224_na_na_16

# python extract_video_frames.py --root_dir data/outputs/dec4/2025.12.04/05.20.02_clip_justPnPCounterToMicrowave/checkpoints/epoch_20_step_1931/dec18_PnPCounterToMicrowave_mg_fixed_224_na_na_16
# python extract_all_final_frames_and_sf_label.py --base_dataset_path data/outputs/dec4/2025.12.04/05.20.02_clip_justPnPCounterToMicrowave/checkpoints/epoch_20_step_1931/dec18_PnPCounterToMicrowave_mg_fixed_224_na_na_16

# python extract_video_frames.py --root_dir data/outputs/dec4/2025.12.04/15.36.39_clip_justPnPCounterToCab/checkpoints/epoch_30_step_1735/dec18_PnPCounterToCab_mg_fixed_224_na_na_16
# python extract_all_final_frames_and_sf_label.py --base_dataset_path data/outputs/dec4/2025.12.04/15.36.39_clip_justPnPCounterToCab/checkpoints/epoch_30_step_1735/dec18_PnPCounterToCab_mg_fixed_224_na_na_16
