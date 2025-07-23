from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
from robomimic.models.obs_core import CropRandomizer
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
import pdb
import numpy as np
from lovely_tensors import lovely
import json

class DiffusionUnetHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            negate_failure_losses=0,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config
            config.observation.encoder.rgb.core_kwargs.backbone_kwargs.pretrained=True
            if 'obs' in shape_meta and 'language_goal' in shape_meta['obs']:
                print('MAKING VISUAL CORE LANGUAGE CONDITIONED')
                config.observation.encoder.rgb.core_class='VisualCoreLanguageConditioned'
                config.observation.encoder.rgb.core_kwargs.backbone_class='ResNet18ConvFiLM'
            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
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
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim #137 -> the encoded 2 images + 9dim eef poses (3robot0_eef_pos, 4robot0_eef_quat, 2gripper_pos)
        self.action_dim = action_dim #10
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        self.negate_failure_losses = negate_failure_losses

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
        print('WHAT IS THE GENERATOR')
        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator, requires_grad=True
        )
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)
        classifier_pred = {'classifier_policy_global_cond': {}, 'global_cond': {}}
        # initial_pred = classifier_policy.model(condition_data[condition_mask], scheduler.timesteps[0], local_cond=local_cond, global_cond=classifier_policy_global_cond)
        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            if classifier_policy:
                # 2.5 compute classifier graident
                timesteps = torch.zeros(trajectory.shape[0], device=trajectory.device).long()+t
                labels=torch.zeros(trajectory.shape[0]).unsqueeze(dim=-1).to(trajectory.device)+guided_towards
                #compute the gradient which points in the direction of steepest ascent
                guidance_gradient, cpoutput= classifier_policy.compute_classifier_gradient( trajectory,  global_cond=classifier_policy_global_cond, timesteps=timesteps, label=labels)
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
                pdb.set_trace()
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
        B, _ = value.shape[:2]
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
            #this_nobs['robot0_eef_pos']==nobs['robot0_eef_pos'][:,-2:].reshape(-1,*nobs['robot0_eef_pos'].shape[2:]) THIS IS NOT TRUE if u sent in 8 OBSERVATIONS IN NOBS!!!!!!
            if nobs['robot0_eef_pos'].shape[1]>2:
                print('OBSREVATION SPACE MORE THAN 2')
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
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        # nsample, inital_pred, classifier_pred = self.conditional_sample(
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
            #TODO: reduce the output dimensionality of the language goal so it doesnt overpower the other lowdim keys
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs,
                lambda x: x[:,-self.n_obs_steps:,...].reshape(-1,*x.shape[2:]))
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

    def compute_loss(self, batch, classifier_policy=None, guidance_scale=None):
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
            #TODO: reduce the output dimensionality of the language goal so it doesnt overpower the other lowdim keys
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs,
                lambda x: x[:,-self.n_obs_steps:,...].reshape(-1,*x.shape[2:]))
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

        if classifier_policy:
            assert False==True
            '''if its a negative sample, only use the classifier loss, if its a positive sample, use both classifier and mse loss'''
            pred_trajectory=torch.zeros(noisy_trajectory.shape)
            for idx in range(len(timesteps)):
                pred_trajectory[idx] = self.noise_scheduler.step(pred[idx:idx+1], timesteps[idx], noisy_trajectory[idx:idx+1]).prev_sample

            # labels=torch.ones(pred_trajectory.shape[0])[:,np.newaxis]
            device = noisy_trajectory.device
            # guidance_gradient = classifier_policy.compute_classifier_gradient( pred_trajectory.to(device),  global_cond=global_cond, timesteps=torch.zeros(labels.shape[0]).to(device), label=labels.to(device))
            # guidance_gradient = torch.abs(guidance_gradient)
            # guidance_gradient = torch.abs(guidance_gradient)
            # classifier_loss = classifier_policy.compute_loss(batch)
            classifier_output = classifier_policy.model(pred_trajectory.to(device), timestep=torch.zeros(pred_trajectory.shape[0]).to(device), local_cond=local_cond, global_cond=classifier_policy_global_cond)
            classifier_probs = torch.sigmoid(classifier_output)
            assert (classifier_probs>1).sum()==0 and (classifier_probs<0).sum()==0
            classifier_log_probs = torch.log(classifier_probs)[:,0]
            total_rewards = batch['total_rewards']
            loss[total_rewards==0]=0

            # guidance_scale = (1/10) * (loss.mean() / guidance_gradient.mean())
            # #round_to_nearest_power_of_10
            # log10_guidance_scale = math.log10(guidance_scale)
            # nearest_power = round(log10_guidance_scale)
            # guidance_scale = 10 ** nearest_power
            # print('loss', loss.mean(), 'guidance_gradient', guidance_gradient.mean(),'guidance_scale',guidance_scale,'guidance_gradient*guidance_scale',guidance_gradient.mean()*guidance_scale)
            # loss += guidance_scale * guidance_gradient
            loss +=-classifier_log_probs * guidance_scale

        if self.negate_failure_losses != 0:
            assert False==True
            '''NEGATE THE LOSS if ITS A FAILED EXECUTION'''
            pdb.set_trace()
            total_rewards = batch['total_rewards']
            total_rewards[total_rewards==0]= -1 * self.negate_failure_losses
            loss = loss.mean(axis=1) * total_rewards

        loss = loss.mean()
        if classifier_policy:
            # return loss, mse_loss.mean(), (guidance_scale * guidance_gradient).mean()
            return loss, mse_loss.mean(), classifier_log_probs.mean() * guidance_scale
        return loss
