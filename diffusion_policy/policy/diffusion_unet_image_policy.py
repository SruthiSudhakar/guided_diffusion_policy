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
# from prismatic.vla.action_tokenizer import ActionTokenizer
from torchvision import transforms

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
            classifier_processor=None,
            image_obs=None, decode_first=True,
            language_goal=None, grad_steps=None,
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

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step( model_output, t, trajectory,  generator=generator, **kwargs  ).prev_sample
            if t==0 and classifier_policy:
                current_guidance_scale=float(guidance_scale)
                if adaptive_guidance=='linear':
                    current_guidance_scale*=trajectory_step/max_steps
                if adaptive_guidance=='keypoints':
                    guidance_scales_list= [0]*100
                    guidance_scales_list[10:27]=[current_guidance_scale] * 17
                    guidance_scales_list[32:]=[current_guidance_scale] * (100-32)
                    current_guidance_scale = guidance_scales_list[trajectory_step]

            if t==0 and classifier_policy and current_guidance_scale>0:
                #TODO: 1. deal with the fact that only 8 actions per input, 2. do per sample grad instead of across batch grad.
                torch.set_grad_enabled(True)
                try:
                    c_input = torch.cat([trajectory[:,:8,:].view(-1,56),classifier_processor.to(trajectory.device)],dim=1).requires_grad_(True)
                except:
                    pdb.set_trace()
                classifier_outputs = classifier_policy(c_input)
                class_grad = torch.autograd.grad(classifier_outputs.sum(), c_input)[0]
                traj_grad=class_grad[:,:56].view(-1,8,7)
                trajectory = trajectory.clone()
                trajectory[:, :8, :] += guidance_scale* traj_grad  # Now safe, not a view of a leaf
                """
                action_tokenizer=ActionTokenizer(classifier_processor.tokenizer)
                
                prompts=[]
                images=[]
                for idx in range(trajectory.shape[0]):
                    traj = trajectory[idx].cpu().numpy()
                    action_tokens=action_tokenizer(traj)
                    # action_tokens = [token.replace('\u202d', "ً") for token in action_tokens]
                    action_tokens=' '.join(action_tokens)
                    lang_idx = idx if idx<len(language_goal) else 0
                    prompts.append(f"In: Will taking this sequence of actions lead the robot towards the goal of {language_goal[lang_idx]}? The actions are: {action_tokens}\nOut: ")
                    images.append(transforms.ToPILImage()(image_obs['robot0_eye_in_hand_image'][idx]))
                try:
                    tokenized_inputs=classifier_processor(prompts, images, padding=True, truncation=True,).to(classifier_policy.device, dtype=torch.bfloat16)
                except:
                    print('hey something wrong')
                    pdb.set_trace()
                input_embeddings=None
                for i in range(grad_steps):
                    torch.set_grad_enabled(True)
                    #get input imbeddings, then run through model and get output logits
                    cpoutput = classifier_policy(input_ids=tokenized_inputs['input_ids'], attention_mask=tokenized_inputs["attention_mask"], pixel_values=tokenized_inputs['pixel_values'], inputs_embeds=input_embeddings, return_dict=True)

                    action_logits = cpoutput.logits[:,classifier_policy.vision_backbone.featurizer.patch_embed.num_patches :]
                    last_valid_indices = tokenized_inputs["attention_mask"].sum(axis=1) - 1  # Get last valid token index
                    batch_indices = torch.arange(tokenized_inputs["attention_mask"].size(0))  # [0, 1, 2, ..., batch_size-1]
                    print(f'{current_guidance_scale} {i}/{grad_steps} action_preds:', action_logits.argmax(dim=-1)[batch_indices,last_valid_indices])
                    # #get gradient of the "class 1" token wrt to the input embeddings (torch.Size([1, 195, 4096]))
                    grad = torch.autograd.grad(action_logits[batch_indices,last_valid_indices,classifier_processor.tokenizer.vocab[str(int(guided_towards))]].sum(), cpoutput.input_embeddings)[0] #Returns a tensor of the same shape as input_embeddings, containing how much each embedding contributes to the selected logits

                    # #add the gradient of the input embeddings to the original input embedding to get a modified embedding
                
                    if decode_first:
                        print('decoding first')
                        input_embeddings = grad 
                    else:
                        print('adding first')
                        input_embeddings = cpoutput.input_embeddings + float(current_guidance_scale) * grad # torch.Size([1, 195, 4096])
                
                modified_actions_embedded=input_embeddings
                #get the token ids of the modified embeddings                
                embedding_matrix = classifier_policy.language_model.model.embed_tokens.weight
                similarities = torch.matmul(modified_actions_embedded, embedding_matrix.T)
                modified_action_tokens = torch.argmax(similarities, dim=-1)  # Get most similar token ID

                #mask the modified actions to only places where actions were in the input
                actions_in_gt=tokenized_inputs['input_ids'].to(classifier_policy.device) #torch.Size([1, 195])
                action_mask = (actions_in_gt > action_tokenizer.action_token_begin_idx) & (actions_in_gt <= (action_tokenizer.action_token_begin_idx + action_tokenizer.vocab_size))
                modified_action_tokens = modified_action_tokens[action_mask]

                #detokenize the actions back to numbers
                modified_actions_detokenized =  action_tokenizer.decode_token_ids_to_actions(modified_action_tokens.cpu().numpy())
                
                shape,device, dtype= trajectory.shape, trajectory.device, trajectory.dtype
                if decode_first:
                    old_mean=trajectory.mean()
                    trajectory += float(current_guidance_scale) * torch.tensor(modified_actions_detokenized).to(device).to(dtype=dtype).reshape(trajectory.shape)
                    print('diff: ', old_mean, trajectory.mean())
                else:
                    og_action_tokens=tokenized_inputs['input_ids'][action_mask]
                    print('diff: ', (modified_action_tokens-og_action_tokens).to(torch.float32).mean())
                    trajectory = torch.tensor(modified_actions_detokenized).to(device).to(dtype=dtype).reshape(trajectory.shape)

                action_norm_stats = classifier_policy.get_action_stats('roboturk')
                norm_mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
                action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
                actions = np.where( norm_mask, 0.5 * (modified_actions_detokenized + 1) * (action_high - action_low) + action_low, modified_actions_detokenized, )

                gtactions = tokenized_inputs['input_ids'][action_mask]
                gtactions = action_tokenizer.decode_token_ids_to_actions(gtactions.cpu().numpy())

                #convert it back to the trajectory shape
                gtactions = torch.tensor(gtactions.reshape(shape)).to(dtype=dtype)

                #unnormalize actions
                action_norm_stats = classifier_policy.get_action_stats('robocasa_chunk_v1_p1')
                mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
                action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
                actions = np.where(mask,0.5 * (gtactions + 1) * (action_high - action_low) + action_low,gtactions,)
                """
                
                # if t==0 and classifier_policy and get_class_scores:
                classifier_pred['classifier_policy_global_cond']['before']= 0 #action_logits.argmax(dim=2)#nn.Sigmoid()(cpoutput)[:,0]
                classifier_pred['classifier_policy_global_cond']['after']= 0 # action_logits.argmax(dim=2) #nn.Sigmoid()(classifier_policy.model(trajectory, timesteps, local_cond=None, global_cond=classifier_policy_global_cond))[:,0]

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask].to(trajectory.dtype)        

        return trajectory, classifier_pred

    # def get_class_score(self, trajectory, obs_dict: Dict[str, torch.Tensor], classifier_processor=None, classifier_policy=None, language_goal=None) -> Dict[str, torch.Tensor]:
    #         with torch.no_grad():
    #             nobs = self.normalizer.normalize(obs_dict)
    #             image_obs=dict_apply(nobs, lambda x: x[:,-1,...])
    #             action_tokenizer=ActionTokenizer(classifier_processor.tokenizer)
                    
    #             prompts=[]
    #             images=[]
    #             for idx in range(trajectory.shape[0]):
    #                 traj = trajectory[idx].cpu().numpy()
    #                 action_tokens=action_tokenizer(traj)
    #                 # action_tokens = [token.replace('\u202d', "ً") for token in action_tokens]
    #                 action_tokens=' '.join(action_tokens)
    #                 lang_idx = idx if idx<len(language_goal) else 0
    #                 prompts.append(f"In: Will taking this sequence of actions lead the robot towards the goal of {language_goal[lang_idx]}? The actions are: {action_tokens}\nOut: ")
    #                 images.append(transforms.ToPILImage()(image_obs['robot0_eye_in_hand_image'][idx]))
    #             try:
    #                 tokenized_inputs=classifier_processor(prompts, images, padding=True, truncation=True,).to(classifier_policy.device, dtype=torch.bfloat16)
    #             except:
    #                 print('hey something wrong')
    #             cpoutput = classifier_policy(input_ids=tokenized_inputs['input_ids'], attention_mask=tokenized_inputs["attention_mask"], pixel_values=tokenized_inputs['pixel_values'], return_dict=True)
    #             action_logits = cpoutput.logits[:,classifier_policy.vision_backbone.featurizer.patch_embed.num_patches :].detach().cpu()
    #             last_valid_indices = (tokenized_inputs["attention_mask"].sum(axis=1) - 1).detach().cpu()  # Get last valid token index
    #             batch_indices = torch.arange(tokenized_inputs["attention_mask"].size(0))  # [0, 1, 2, ..., batch_size-1]
    #             guided_towards=1
    #             print(f'action_preds:', action_logits.argmax(dim=-1)[batch_indices,last_valid_indices])
    #             return action_logits[batch_indices,last_valid_indices,classifier_processor.tokenizer.vocab[str(int(guided_towards))]]



    
    def predict_action(self, obs_dict: Dict[str, torch.Tensor], classifier_processor=None, classifier_policy=None, grad_steps=None, guidance_scale=None, guided_towards=None, trajectory_step=None, adaptive_guidance=None, max_steps=None, get_class_scores=False, decode_first=True, language_goal=None) -> Dict[str, torch.Tensor]:
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
            classifier_processor=classifier_processor,
            image_obs=dict_apply(nobs, lambda x: x[:,-1,...]),
            decode_first=decode_first,
            language_goal=language_goal,
            grad_steps=grad_steps,
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
            'action_pred': action_pred,
            'global_cond': global_cond
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