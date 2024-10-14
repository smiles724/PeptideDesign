import os
import time
import argparse

import hydra
import torch
import torch.distributed as distrib
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm as tq
from easydict import EasyDict

from multiflow.data.pep_dataloader import PepDataset, PaddingCollate
from multiflow.experiments.utils import inf_iterator, seed_all, get_logger, get_new_log_dir, count_parameters, get_optimizer, get_scheduler, log_losses, recursive_to, \
    sum_weighted_losses
from multiflow.models.flow_model_pep import PepModel

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
config_path = "../configs"
config_name = "pep_codesign.yaml"


@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def main(cfg: DictConfig):
    seed_all(cfg.train.seed)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--local_rank', type=int, help='Local rank. Necessary for using the torch.distributed.launch utility.')
    # args = parser.parse_args()
    # local_rank = args.local_rank
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    distrib.init_process_group(backend="nccl")

    if cfg.train.debug:
        logger = get_logger('train', None)
    else:
        wandb_name = '%s[%s]' % (config_name.split('.')[0], time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()))
        wandb.init(project=cfg.train.project_name, config=EasyDict(OmegaConf.to_container(cfg, resolve=True)), name=wandb_name)
        log_dir = get_new_log_dir(cfg.train.logdir, prefix='%s' % config_name)
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)

    # Data
    logger.info('Loading datasets...')
    train_dataset = PepDataset(structure_dir=cfg.dataset.train.structure_dir, dataset_dir=cfg.dataset.train.dataset_dir, name=cfg.dataset.train.name, reset=cfg.dataset.train.reset)
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, sampler=train_sampler, collate_fn=PaddingCollate(), num_workers=cfg.train.num_workers, pin_memory=True)
    train_iterator = inf_iterator(train_loader)
    logger.info('Train %d | Val %d' % (len(train_dataset), len(train_dataset)))

    # Model
    logger.info('Building model...')
    model = PepModel(cfg.model)
    logger.info('Using the pretrained FM model from checkpoint: %s' % cfg.model.ckpt_path)
    with open(cfg.model.ckpt_path, 'rb') as f:
        ckpt = torch.load(f, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt['state_dict'])
    logger.info('Number of parameters: %d' % count_parameters(model))

    # Wrap model with DistributedDataParallel (DDP)
    model = DDP(model.to(local_rank), device_ids=[local_rank])

    optimizer = get_optimizer(cfg.train.optimizer, model)
    scheduler = get_scheduler(cfg.train.scheduler, optimizer)
    optimizer.zero_grad()
    it_first = 1

    def train(it):
        model.train()
        batch = recursive_to(next(train_iterator), local_rank)
        loss_dict = model(batch)  # get loss and metrics
        loss = sum_weighted_losses(loss_dict, cfg.train.loss_weights)
        loss.backward()

        # rescue for nan grad
        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    param.grad[torch.isnan(param.grad)] = 0

        orig_grad_norm = clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()

        # Logging
        scalar_dict = {}
        scalar_dict.update({'grad': orig_grad_norm, 'lr': optimizer.param_groups[0]['lr']})
        logstr = log_losses(loss, loss_dict, scalar_dict, it=it, tag='train', logger=logger) if not cfg.train.debug else 'Debug'
        return logstr


    # try:
    #     it_tqdm = tq(range(it_first, config.train.max_iters + 1))
    #     for it in it_tqdm:
    #         train_sampler.set_epoch(it)  # Important: update sampler to reshuffle data each epoch
    #         message = train(it)
    #         it_tqdm.set_description(message)
    #         if it % config.train.val_freq == 0 and local_rank == 0:
    #             ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
    #             torch.save({'config': config, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'iteration': it, }, ckpt_path)
    # except KeyboardInterrupt:
    #     logger.info('Terminating...')
    #     distrib.destroy_process_group()
    it_tqdm = tq(range(it_first, cfg.train.max_iters + 1))
    for it in it_tqdm:
        train_sampler.set_epoch(it)  # Important: update sampler to reshuffle data each epoch
        message = train(it)
        it_tqdm.set_description(message)
        if it % cfg.train.val_freq == 0 and local_rank == 0:
            ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
            torch.save({'config': cfg, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'iteration': it, }, ckpt_path)

if __name__ == '__main__':
    main()
