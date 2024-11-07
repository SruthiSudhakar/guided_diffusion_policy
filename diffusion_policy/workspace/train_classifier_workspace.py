if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
import copy
import random
import wandb
import pickle
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from accelerate import Accelerator
import pdb
from torch import nn
import omegaconf 
import matplotlib.pyplot as plt

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainClassifierWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionClassifierHybridImagePolicy = hydra.utils.instantiate(cfg.policy)

        obs_encorder_lr = cfg.optimizer.lr
        # if cfg.policy.obs_encoder.rgb_model.weights is not None:
        #     obs_encorder_lr *= 0.1
        #     print('==> reduce pretrained obs_encorder\'s lr')
        obs_encorder_params = list()
        for param in self.model.obs_encoder.parameters():
            if param.requires_grad:
                obs_encorder_params.append(param)
        print(f'obs_encorder params: {len(obs_encorder_params)}')

        param_groups = [
            {'params': self.model.model.parameters()},
            {'params': obs_encorder_params, 'lr': obs_encorder_lr},
        ]

        self.criterion = nn.BCELoss()

        # configure training state
        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        optimizer_cfg.pop('_target_')
        self.optimizer = torch.optim.AdamW(
            params=param_groups,
            **optimizer_cfg
        )

        # configure training state
        self.global_step = 0
        self.epoch = 0

        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys = ['optimizer']

    def visualise_dataloader(self, dl, whichone, id_to_label=None, with_outputs=True):
        total_num_images = len(dl.dataset)
        idxs_seen = []
        class_0_batch_counts = []
        class_1_batch_counts = []
        with tqdm.tqdm(dl, desc=f"Training epoch {self.epoch}", 
            leave=False) as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                classes = batch['success']
                class_ids, class_counts = classes.unique(return_counts=True)
                class_ids = set(class_ids.tolist())
                class_counts = class_counts.tolist()

                if len(class_ids) == 2:
                    class_0_batch_counts.append(class_counts[0])
                    class_1_batch_counts.append(class_counts[1])
                elif len(class_ids) == 1 and 0 in class_ids:
                    class_0_batch_counts.append(class_counts[0])
                    class_1_batch_counts.append(0)
                elif len(class_ids) == 1 and 1 in class_ids:
                    class_0_batch_counts.append(0)
                    class_1_batch_counts.append(class_counts[0])
                else:
                    raise ValueError("More than two classes detected")
                idxs_seen.extend(classes)

        if with_outputs:
            fig, ax = plt.subplots(1, figsize=(50,50))

            ind = np.arange(len(class_0_batch_counts))
            width = 0.35

            ax.bar(
                ind,
                class_0_batch_counts,
                width,
                label=(id_to_label[0] if id_to_label is not None else "0"),
            )
            ax.bar(
                ind + width,
                class_1_batch_counts,
                width,
                label=(id_to_label[1] if id_to_label is not None else "1"),
            )
            ax.set_xticks(ind, ind + 1)
            ax.set_xlabel("Batch index", fontsize=12)
            ax.set_ylabel("No. of images in batch", fontsize=12)
            ax.set_aspect("equal")

            plt.legend()
            plt.savefig(f'classdata_{whichone}.png')

            num_images_seen = len(idxs_seen)

            print(
                f'Avg Proportion of {(id_to_label[0] if id_to_label is not None else "Class 0")} per batch: {(np.array(class_0_batch_counts) / 10).mean()}'
            )
            print(
                f'Avg Proportion of {(id_to_label[1] if id_to_label is not None else "Class 1")} per batch: {(np.array(class_1_batch_counts) / 10).mean()}'
            )
            print("=============")
            print(f"Num. unique images seen: {len(set(idxs_seen))}/{total_num_images}")
        return class_0_batch_counts, class_1_batch_counts, idxs_seen


    def run(self):
        cfg = copy.deepcopy(self.cfg)

        accelerator = Accelerator()
        # wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        # wandb_cfg.pop('project')
        # accelerator.init_trackers(
        #     project_name=cfg.logging.project,
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     init_kwargs={"wandb": wandb_cfg}
        # )

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset_combined: BaseImageDataset
        datasets=[]
        for each_dataset in cfg.task.dataset_path:
            cfg.task.dataset.dataset_path = each_dataset
            datasets.append(hydra.utils.instantiate(cfg.task.dataset))
        dataset_combined = ConcatDataset(datasets)
        assert isinstance(datasets[0], BaseImageDataset)
        train_dataloader = DataLoader(dataset_combined, **cfg.dataloader)
        if 'balance_dataset' in cfg.task and cfg.task.balance_dataset:
            prefix = '__'.join([dataset_path.split('/')[-2] for dataset_path in cfg.task.dataset_path])+'_samples_weight_train_dataloader.npy'
            if os.path.exists(prefix):
                print('USING SAVED SAMPLE WEIGHTS', prefix)
                samples_weight = np.load(prefix)
            else:
                samples_weight=[]
                print('calculating sample weights')
                with tqdm.tqdm(train_dataloader) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        samples_weight.extend(batch['success'].tolist())
                successes_count = sum(samples_weight)
                failures_count = len(samples_weight)-successes_count
                weight = 1. / np.array([successes_count,failures_count])
                samples_weight=np.array(samples_weight)
                samples_weight[np.isclose(samples_weight, 1.0)] = weight[0]
                samples_weight[np.isclose(samples_weight, 0.0)] = weight[1]
                np.save(prefix, samples_weight)
                samples_weight = torch.from_numpy(samples_weight)
                samples_weight = samples_weight.double()
            sampler = WeightedRandomSampler(samples_weight, len(dataset_combined), replacement=True)
            train_dataloader = DataLoader(dataset_combined, **cfg.dataloader, sampler=sampler)
        normalizer = datasets[0].get_multidataset_normalizer(datasets)

        # configure validation dataset
        val_dataset = [dataset.get_validation_dataset() for dataset in datasets]
        val_dataset_combined = ConcatDataset(val_dataset)
        val_dataloader = DataLoader(val_dataset_combined, **cfg.val_dataloader)
        
        if 'balance_dataset' in cfg.task and cfg.task.balance_dataset:
            prefix = '__'.join([dataset_path.split('/')[-2] for dataset_path in cfg.task.dataset_path])+'_samples_weight_val_dataloader.npy'
            if os.path.exists(prefix):
                print('USING SAVED SAMPLE WEIGHTS', prefix)
                samples_weight = np.load(prefix)
            else:
                samples_weight=[]
                print('calcualting sample weights')
                with tqdm.tqdm(val_dataloader) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        samples_weight.extend(batch['success'].tolist())
                successes_count = sum(samples_weight)
                failures_count = len(samples_weight)-successes_count
                weight = 1. / np.array([successes_count,failures_count])
                samples_weight=np.array(samples_weight)
                samples_weight[np.isclose(samples_weight, 1.0)] = weight[0]
                samples_weight[np.isclose(samples_weight, 0.0)] = weight[1]
                np.save(prefix, samples_weight)
            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
            sampler = WeightedRandomSampler(samples_weight, len(dataset_combined), replacement=True)        
            val_dataloader = DataLoader(val_dataset_combined, sampler=sampler, **cfg.val_dataloader)

        print('train dataset:', len(dataset_combined), 'train dataloader:', len(train_dataloader))
        print('val dataset:', len(val_dataset_combined), 'val dataloader:', len(val_dataloader))

        # if 'balance_dataset' in cfg.task and cfg.task.balance_dataset:
        #     class_0_batch_counts, class_1_batch_counts, idxs_seen = self.visualise_dataloader(train_dataloader, 'train', {0: "failures", 1: "successes"})
        #     class_0_batch_counts, class_1_batch_counts, idxs_seen = self.visualise_dataloader(val_dataloader, 'validation', {0: "failures", 1: "successes"})
        #     total_0 = 0
        #     total_1 = 0
        #     print('calculating dataset stats')
        #     for batch in tqdm.tqdm(train_dataloader):
        #         total_0 += (batch['success'] == 0).sum()
        #         total_1 += (batch['success'] == 1).sum()
        #     print('TRAIN DATA WEIGHTED BALANCED', total_0, total_1)
        #     total_0 = 0
        #     total_1 = 0
        #     for batch in tqdm.tqdm(val_dataloader):   
        #         total_0 += (batch['success'] == 0).sum()
        #         total_1 += (batch['success'] == 1).sum()
        #     print(total_0, total_1)
        #     print('VALDATON DATA WEIGHTED BALANCED', total_0, total_1)

        self.model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure logging
        # wandb_run = wandb.init(
        #     dir=str(self.output_dir),
        #     config=OmegaConf.to_container(cfg, resolve=True),
        #     **cfg.logging
        # )
        # wandb.config.update(
        #     {
        #         "output_dir": self.output_dir,
        #     }, 
        #     allow_val_change=True
        # )

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # accelerator
        train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader, val_dataloader, self.model, self.optimizer, lr_scheduler
        )
        device = self.model.device

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(cfg.training.num_epochs):
                self.model.train()

                step_log = dict()
                # ========= train for this epoch ==========

                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                        
                        # always use the latest batch
                        train_sampling_batch = batch
                        # compute loss
                        raw_loss = accelerator.unwrap_model(self.model).compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()


                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()
                        
                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }

                        is_last_batch = (batch_idx == (len(train_dataloader)-1))
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            accelerator.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break
                        
                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0 and accelerator.is_main_process:
                    # unwrap the model to save ckpt
                    model_ddp = self.model
                    self.model = accelerator.unwrap_model(self.model)
                    # ========= eval for this epoch ==========
                    self.model.eval()  
                    valid_loss = list()
                    valid_accuracy = list()
                    with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                        for batch_idx, batch in enumerate(tepoch):
                            # device transfer
                            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                            # compute loss
                            loss, pred = self.model.compute_loss(batch, return_raw_outputs=True)
                            valid_loss.append(loss.item())
                            actual_out = (pred[:,0] > 0.5).float() * 1                            
                            equals = (batch['success'].float()  ==  actual_out.t()) + 0.0
                            valid_accuracy.append(torch.mean(equals).cpu().numpy())
                    step_log['valid_loss'] = np.mean(valid_loss)
                    step_log['valid_accuracy'] = np.mean(valid_accuracy)
                    
                    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}  \tAccuracy: {:.6f}  '  .format(
                        self.epoch, train_loss, step_log['valid_loss'], step_log['valid_accuracy']))
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)

                    # recover the DDP model
                    self.model = model_ddp
                # ========= eval end for this epoch ==========
                # end of epoch
                # log of last step is combined with validation and rollout
                accelerator.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

        accelerator.end_training()
    def run_validation(self):
        cfg = copy.deepcopy(self.cfg)
        # configure dataset
        dataset_combined: BaseImageDataset
        datasets=[]
        for each_dataset in cfg.task.dataset_path:
            cfg.task.dataset.dataset_path = each_dataset
            datasets.append(hydra.utils.instantiate(cfg.task.dataset))
        dataset_combined = ConcatDataset(datasets)
        # assert isinstance(datasets[0], BaseImageDataset)
        # train_dataloader = DataLoader(dataset_combined, **cfg.dataloader)
        normalizer = datasets[0].get_multidataset_normalizer(datasets)
        all_metric_dict={}
        # for each_dataset in datasets:
        # pdb.set_trace()
        # print('RUNNING VALIDATION ON:', each_dataset.dataset_path)
        # configure validation dataset
        # val_dataset = each_dataset.get_validation_dataset()
        val_dataset = [dataset.get_validation_dataset() for dataset in datasets]
        val_dataset_combined = ConcatDataset(val_dataset)
        val_dataloader = DataLoader(val_dataset_combined, **cfg.val_dataloader)
            
        if 'balance_dataset' in cfg.task and cfg.task.balance_dataset:
            assert True==False
            prefix = '__'.join([dataset_path.split('/')[-2] for dataset_path in cfg.task.dataset_path])+'_samples_weight_val_dataloader.npy'
            if os.path.exists(prefix):
                print('USING SAVED SAMPLE WEIGHTS', prefix)
                samples_weight = np.load(prefix)
            else:
                samples_weight=[]
                print('Calculating sampler weights')
                with tqdm.tqdm(val_dataloader) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        samples_weight.extend(batch['success'].tolist())
                successes_count = sum(samples_weight)
                failures_count = len(samples_weight)-successes_count
                weight = 1. / np.array([successes_count,failures_count])
                samples_weight=np.array(samples_weight)
                samples_weight[np.isclose(samples_weight, 1.0)] = weight[0]
                samples_weight[np.isclose(samples_weight, 0.0)] = weight[1]
                np.save(prefix, samples_weight)

            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
            sampler = WeightedRandomSampler(samples_weight, len(dataset_combined), replacement=True)
            val_dataloader = DataLoader(val_dataset, sampler=sampler, **cfg.val_dataloader)
        
        print('val dataset:', len(val_dataset), 'val dataloader:', len(val_dataloader))
        # total_0 = 0
        # total_1 = 0
        # pdb.set_trace()
        # print('calculating stats', )
        # with tqdm.tqdm(val_dataloader) as tepoch:
        #     for batch_idx, batch in enumerate(tepoch):
        #         total_0 += (batch['success'] == 0).sum()
        #         total_1 += (batch['success'] == 1).sum()
        # print('VALIDATION DATA WEIGHTED BALANCED', total_0, total_1)

        self.model.set_normalizer(normalizer)

        device = self.model.device
        self.model.eval()  
        valid_loss = list()
        valid_accuracy = list()
        successes = list()
        objects = list()
        preds = list()
        step_log = dict()
        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
            leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
            for batch_idx, batch in enumerate(tepoch):
                # device transfer
                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                # compute loss
                loss, pred = self.model.compute_loss(batch, return_raw_outputs=True)
                valid_loss.append(loss.item())
                actual_out = (pred[:,0] > 0.5).float() * 1  
                equals = (batch['success'].float()  ==  actual_out.t()) + 0.0
                preds.extend(equals.cpu().numpy())
                valid_accuracy.append(torch.mean(equals).cpu().numpy())
                if 'object' in batch:
                    print('OBJECT IN BATCH')
                    pdb.set_trace()
                    objects.extend(batch['object'])
                successes.extend(batch['success'].cpu().numpy())

        step_log['valid_loss'] = np.mean(valid_loss)
        step_log['valid_accuracy'] = np.mean(valid_accuracy)
        step_log['equals'] = preds
        step_log['gt_successes'] = successes
        step_log['gt_objects'] = objects
        step_log['val_dataset_stats'] = self.print_dataset_stats(val_dataset_combined)
        step_log['train_dataset_stats'] = self.print_dataset_stats(dataset_combined)

        print('\tValidation Loss: {:.6f}  \tAccuracy: {:.6f}  '.format(
            step_log['valid_loss'], step_log['valid_accuracy']))

        # sanitize metric names
        metric_dict = dict()
        for key, value in step_log.items():
            new_key = key.replace('/', '_')
            metric_dict[new_key] = value
            # all_metric_dict[each_dataset.dataset_path]=metric_dict
        # return all_metric_dict
        return metric_dict
    def print_dataset_stats(self, all_datasets):
        all_data = {}
        for dataset in all_datasets.datasets:
            total_episodes = len(dataset.train_mask)
            epsiodes_in_dataset = dataset.train_mask.sum()
            dataset_indices = np.where(dataset.train_mask==True)[0]
            successful_episodes = dataset.replay_buffer.data['success'][dataset_indices].sum()
            fail_epsidoes = epsiodes_in_dataset - successful_episodes
            all_data[dataset.dataset_path.split('/')[-2]]={
                'total_episodes': total_episodes,
                'epsiodes_in_dataset': epsiodes_in_dataset,
                'successful_episodes': successful_episodes,
                'fail_epsidoes': fail_epsidoes
            }
        return all_data
    def display_dataset(self, dataset):
        dataset_indices=np.where(dataset.train_mask==True)
        ep_starts = dataset.replay_buffer.episode_ends[dataset_indices]-dataset.replay_buffer.episode_lengths[dataset_indices]
        for i in range(len(ep_starts)): 
            plt.imsave(f'tempval/img_{i}.png', dataset.replay_buffer.data['agentview_image'][ep_starts[i]+10])

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetImageWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()
