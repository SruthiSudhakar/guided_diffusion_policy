from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

import pdb


class DiffusionUnetImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: MultiImageObsEncoder,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon #16
        self.obs_feature_dim = obs_feature_dim #1033
        self.action_dim = action_dim #10
        self.n_action_steps = n_action_steps #8
        self.n_obs_steps = n_obs_steps #2
        self.obs_as_global_cond = obs_as_global_cond #true
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None, classifier_policy=None,
            classifier_policy_global_cond=None,
            guidance_scale=None, guided_towards=None,
            trajectory_step=None, adaptive_guidance=None,
            max_steps=None, get_class_scores=False,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator, requires_grad=True
        )
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)
        classifier_pred = {'classifier_policy_global_cond': {}, 'global_cond': {}}

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)


            # if classifier_policy:
            #     # 2.5 compute classifier graident
            #     timesteps = torch.zeros(trajectory.shape[0], device=trajectory.device).long()+t
            #     labels=torch.zeros(trajectory.shape[0]).unsqueeze(dim=-1).to(trajectory.device)+guided_towards
            #     #compute the gradient which points in the direction of steepest ascent
            #     guidance_gradient, cpoutput= classifier_policy.compute_classifier_gradient( trajectory,  global_cond=classifier_policy_global_cond, timesteps=timesteps, label=labels)
            #     current_guidance_scale=float(guidance_scale)
            #     if guidance_scale == 'variable':
            #         #round_to_nearest_power_of_10
            #         log10_guidance_scale = math.log10(abs((model_output.mean() / guidance_gradient.mean())))
            #         nearest_power = round(log10_guidance_scale)
            #         current_guidance_scale = 10 ** (nearest_power-1)
            #     elif adaptive_guidance=='linear':
            #         current_guidance_scale*=trajectory_step/max_steps
            #     elif adaptive_guidance=='keypoints':
            #         guidance_scales_list= [0]*100
            #         guidance_scales_list[10:27]=[current_guidance_scale] * 17
            #         guidance_scales_list[32:]=[current_guidance_scale] * (100-32)
            #         current_guidance_scale = guidance_scales_list[trajectory_step]
            #         # if current_guidance_scale>0:
            #         #     pdb.set_trace()
            #             # print('ts: ', trajectory_step, 'cgs', current_guidance_scale, 'model_output', model_output.mean(), 'guidance_gradient', guidance_gradient.mean() * float(current_guidance_scale))
            #     model_output += float(current_guidance_scale) * guidance_gradient
            if classifier_policy:
                # 2.5 compute classifier graident
                guidance_gradient, cpoutput= get_gla_score( trajectory, global_cond=classifier_policy_global_cond, timesteps=timesteps, label=labels)
                current_guidance_scale=float(guidance_scale)
                if guidance_scale == 'variable':
                    #round_to_nearest_power_of_10
                    log10_guidance_scale = math.log10(abs((model_output.mean() / guidance_gradient.mean())))
                    nearest_power = round(log10_guidance_scale)
                    current_guidance_scale = 10 ** (nearest_power-1)
                elif adaptive_guidance=='linear':
                    current_guidance_scale*=trajectory_step/max_steps
                elif adaptive_guidance=='keypoints':
                    guidance_scales_list= [0]*100
                    guidance_scales_list[10:27]=[current_guidance_scale] * 17
                    guidance_scales_list[32:]=[current_guidance_scale] * (100-32)
                    current_guidance_scale = guidance_scales_list[trajectory_step]
                    # if current_guidance_scale>0:
                    #     pdb.set_trace()
                        # print('ts: ', trajectory_step, 'cgs', current_guidance_scale, 'model_output', model_output.mean(), 'guidance_gradient', guidance_gradient.mean() * float(current_guidance_scale))
                model_output += float(current_guidance_scale) * guidance_gradient
            
            if t==0 and classifier_policy and get_class_scores:
                classifier_pred['classifier_policy_global_cond']['before']= nn.Sigmoid()(cpoutput)[:,0]
                # classifier_pred['global_cond']['before']= nn.Sigmoid()(classifier_policy.model(trajectory, timesteps, local_cond=None, global_cond=global_cond))[:,0]
                # pdb.set_trace()

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
            if t==0 and classifier_policy and get_class_scores:
                # pdb.set_trace()
                classifier_pred['classifier_policy_global_cond']['after']= nn.Sigmoid()(classifier_policy.model(trajectory, timesteps, local_cond=None, global_cond=classifier_policy_global_cond))[:,0]

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory, classifier_pred


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], classifier_policy=None, guidance_scale=None, guided_towards=None, trajectory_step=None, adaptive_guidance=None, max_steps=None, get_class_scores=False) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,-To:,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            classifier_policy_global_cond=None
            if classifier_policy:
                classifier_policy_nobs_features = classifier_policy.obs_encoder(this_nobs)
                classifier_policy_global_cond = classifier_policy_nobs_features.reshape(B, -1)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,-To:,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample, classifier_pred = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            classifier_policy=classifier_policy,
            classifier_policy_global_cond=classifier_policy_global_cond,
            guidance_scale=guidance_scale,
            guided_towards=guided_towards,
            trajectory_step=trajectory_step,
            adaptive_guidance=adaptive_guidance,
            max_steps=max_steps,
            get_class_scores=get_class_scores,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result, classifier_pred

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def forward(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,-self.n_obs_steps:,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
                
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        return pred, target, loss_mask

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,-self.n_obs_steps:,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
                
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        mse_loss = F.mse_loss(pred, target, reduction='none')
        loss = mse_loss * loss_mask.type(mse_loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = reduce(loss, 'b ... -> b', 'mean')
        loss = loss.mean()
        return loss
    def get_vla_score(vla, processor, base_vla_name, obs, task_label, unnorm_key, center_crop=False):
        """Generates an action with the VLA policy."""
        image = resize_image(obs["full_image"])
        image = Image.fromarray(image)
        image = image.convert("RGB")

        # (If trained with image augmentations) Center crop image and then resize back up to original size.
        # IMPORTANT: Let's say crop scale == 0.9. To get the new height and width (post-crop), multiply
        #            the original height and width by sqrt(0.9) -- not 0.9!
        if center_crop:
            batch_size = 1
            crop_scale = 0.9

            # Convert to TF Tensor and record original data type (should be tf.uint8)
            image = tf.convert_to_tensor(np.array(image))
            orig_dtype = image.dtype

            # Convert to data type tf.float32 and values between [0,1]
            image = tf.image.convert_image_dtype(image, tf.float32)

            # Crop and then resize back to original size
            image = crop_and_resize(image, crop_scale, batch_size)

            # Convert back to original data type
            image = tf.clip_by_value(image, 0, 1)
            image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

            # Convert back to PIL Image
            image = Image.fromarray(image.numpy())
            image = image.convert("RGB")

        # Build VLA prompt
        prompt = f"In: Will taking this sequence of actions lead the robot towards the goal of {lang}? The actions are: {action_tokens}?\nOut:"

        # Process inputs.
        inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

        # Get action.
        action = vla.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        return action
    def resize_image(img, resize_size):
        """
        Takes numpy array corresponding to a single image and returns resized image as numpy array.

        NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                        the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
        """
        assert isinstance(resize_size, tuple)
        # Resize to image size expected by model
        img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
        img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
        img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
        img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
        img = img.numpy()
        return img