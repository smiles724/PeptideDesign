import argparse
import os
import shutil
import time
import hydra

import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm as tq
from tqdm.auto import tqdm
from easydict import EasyDict

from multiflow.models.flow_model_pep import PepModel
from multiflow.data.pep_dataloader import PepDataset, PaddingCollate
from multiflow.experiments.utils import BlackHole, inf_iterator, seed_all, get_logger, get_new_log_dir, ScalarMetricAccumulator, count_parameters, get_optimizer, \
    get_scheduler, log_losses, recursive_to, sum_weighted_losses

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
config_path = "../configs"
config_name = "pep_codesign.yaml"


@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def main(cfg: DictConfig):
    seed_all(cfg.train.seed)
    if cfg.train.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
    else:
        wandb_name = '%s[%s]' % (config_name.split('.')[0], time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()))
        wandb.init(project=cfg.train.project_name, config=EasyDict(OmegaConf.to_container(cfg, resolve=True)), name=wandb_name)
        log_dir = get_new_log_dir(cfg.train.logdir, prefix='%s' % config_name)
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
    # logger.info(config)

    # Data
    logger.info('Loading datasets...')
    train_dataset = PepDataset(structure_dir=cfg.dataset.train.structure_dir, dataset_dir=cfg.dataset.train.dataset_dir, name=cfg.dataset.train.name,
                               reset=cfg.dataset.train.reset)
    # val_dataset = PepDataset(structure_dir=config.dataset.val.structure_dir, dataset_dir=config.dataset.val.dataset_dir, name=config.dataset.val.name,
    #                          reset=config.dataset.val.reset)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=PaddingCollate(), num_workers=cfg.train.num_workers, pin_memory=True)
    train_iterator = inf_iterator(train_loader)
    # val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, collate_fn=PaddingCollate(), num_workers=args.num_workers)
    logger.info('Train %d | Val %d' % (len(train_dataset), len(train_dataset)))

    # Model
    logger.info('Building model...')
    model = PepModel(cfg.model).to(cfg.train.device)
    logger.info('Using the pretrained FM model from checkpoint: %s' % cfg.model.ckpt_path)
    with open(cfg.model.ckpt_path, 'rb') as f:
        ckpt = torch.load(f, map_location=torch.device(f'cuda:{cfg.train.device}'))  # TODO: mpp
    model.load_state_dict(ckpt['state_dict'])
    logger.info('Number of trainable parameters: %d M' % (count_parameters(model, grad=True) / 1e6))
    logger.info('Number of total parameters: %d M' % (count_parameters(model) / 1e6))

    # Optimizer & Scheduler
    optimizer = get_optimizer(cfg.train.optimizer, model)
    scheduler = get_scheduler(cfg.train.scheduler, optimizer)
    optimizer.zero_grad()
    it_first = 1

    def train(it):
        model.train()
        batch = recursive_to(next(train_iterator), cfg.train.device)
        loss_dict = model(batch)
        loss = sum_weighted_losses(loss_dict, cfg.train.loss_weights)

        if torch.isnan(loss):
            print('NAN Loss!')
            torch.save({'prot': batch, 'loss': loss, 'loss_dict': loss_dict, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                        'iteration': it, }, os.path.join(log_dir, 'nan.pt'))
            loss = torch.tensor(0., requires_grad=True).to(loss.device)
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

    def validate(it):  # TODO: add validation
        scalar_accum = ScalarMetricAccumulator()
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(tqdm(val_loader, desc='Validate', dynamic_ncols=True)):
                batch = recursive_to(batch, cfg.train.device)
                loss_dict = model(batch)
                loss = sum_weighted_losses(loss_dict, cfg.train.loss_weights)
                scalar_accum.add(name='loss', value=loss, batchsize=len(batch['aa']), mode='mean')
                for k, v in loss_dict['scalar'].items():
                    scalar_accum.add(name=k, value=v, batchsize=len(batch['aa']), mode='mean')

        avg_loss = scalar_accum.get_average('loss')
        summary = scalar_accum.log(it, 'val', logger=logger, writer=writer)
        for k, v in summary.items():
            wandb.log({f'val/{k}': v}, step=it)
        # Trigger scheduler
        if cfg.train.scheduler.type == 'plateau':
            scheduler.step(avg_loss)
        else:
            scheduler.step()
        return avg_loss

    try:
        it_tqdm = tq(range(it_first, cfg.train.max_iters + 1))
        for it in it_tqdm:
            message = train(it)
            it_tqdm.set_description(message)
            if it % cfg.train.val_freq == 0:
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({'config': cfg, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'iteration': it,
                            # 'avg_val_loss': avg_val_loss,
                            }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')


if __name__ == '__main__':
    main()
