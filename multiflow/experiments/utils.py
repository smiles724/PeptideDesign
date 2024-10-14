"""Utility functions for experiments."""
import glob
import logging
import os
import random
import shutil
import subprocess
import time

import GPUtil
import numpy as np
import pandas as pd
import torch
import torch.linalg
import wandb
import yaml
from biotite.sequence.io import fasta
from easydict import EasyDict
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from multiflow.analysis import utils as au
from openfold.utils import rigid_utils

Rigid = rigid_utils.Rigid


class BlackHole(object):
    def __setattr__(self, name, value):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return self


def get_logger(name, log_dir=None, local_rank=0):
    logger = logging.getLogger(name)

    # Check if logger already has handlers to avoid adding them again
    if not logger.hasHandlers():
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        if log_dir is not None:
            file_handler = logging.FileHandler(os.path.join(log_dir, 'log_%d.txt' % local_rank))
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger


def get_new_log_dir(root='./logs', prefix='', tag=''):
    fn = time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime())
    if prefix != '':
        fn = prefix + '_' + fn
    if tag != '':
        fn = fn + '_' + tag
    log_dir = os.path.join(root, fn)
    os.makedirs(log_dir)
    return log_dir


def seed_all(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def int_tuple(argstr):
    return tuple(map(int, argstr.split(',')))


def str_tuple(argstr):
    return tuple(argstr.split(','))


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))
    config_name = os.path.basename(config_path)[:os.path.basename(config_path).rfind('.')]
    return config, config_name


class LengthDataset(torch.utils.data.Dataset):
    def __init__(self, samples_cfg):
        self._samples_cfg = samples_cfg
        all_sample_lengths = range(self._samples_cfg.min_length, self._samples_cfg.max_length + 1, self._samples_cfg.length_step)
        if samples_cfg.length_subset is not None:
            all_sample_lengths = [int(x) for x in samples_cfg.length_subset]
        all_sample_ids = []
        num_batch = self._samples_cfg.num_batch
        assert self._samples_cfg.samples_per_length % num_batch == 0
        self.n_samples = self._samples_cfg.samples_per_length // num_batch

        for length in all_sample_lengths:
            for sample_id in range(self.n_samples):
                sample_ids = torch.tensor([num_batch * sample_id + i for i in range(num_batch)])
                all_sample_ids.append((length, sample_ids))
        self._all_sample_ids = all_sample_ids

    def __len__(self):
        return len(self._all_sample_ids)

    def __getitem__(self, idx):
        num_res, sample_id = self._all_sample_ids[idx]
        batch = {'num_res': num_res, 'sample_id': sample_id, }
        return batch


def dataset_creation(dataset_class, cfg, task):
    train_dataset = dataset_class(dataset_cfg=cfg, task=task, is_training=True, )
    eval_dataset = dataset_class(dataset_cfg=cfg, task=task, is_training=False, )
    return train_dataset, eval_dataset


def get_available_device(num_device):
    return GPUtil.getAvailable(order='memory', limit=8)[:num_device]


def run_easy_cluster(designable_dir, output_dir):
    # designable_dir should be a directory with individual PDB files in it that we want to cluster
    # output_dir is where we are going to save the easy cluster output files

    # Returns the number of clusters
    easy_cluster_args = ['foldseek', 'easy-cluster', designable_dir, os.path.join(output_dir, 'res'), output_dir, '--alignment-type', '1', '--cov-mode', '0', '--min-seq-id', '0',
                         '--tmscore-threshold', '0.5', ]
    process = subprocess.Popen(easy_cluster_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, _ = process.communicate()
    del stdout  # We don't actually need the stdout, we will read the number of clusters from the output files
    rep_seq_fasta = fasta.FastaFile.read(os.path.join(output_dir, 'res_rep_seq.fasta'))
    return len(rep_seq_fasta)


def get_all_top_samples(output_dir, csv_fname='*/*/top_sample.csv'):
    all_csv_paths = glob.glob(os.path.join(output_dir, csv_fname), recursive=True)
    top_sample_csv = pd.concat([pd.read_csv(x) for x in all_csv_paths])
    top_sample_csv.to_csv(os.path.join(output_dir, 'all_top_samples.csv'), index=False)
    return top_sample_csv


def calculate_diversity(output_dir, metrics_df, top_sample_csv, designable_csv_path):
    designable_samples = top_sample_csv[top_sample_csv.designable]
    designable_dir = os.path.join(output_dir, 'designable')
    os.makedirs(designable_dir, exist_ok=True)
    designable_txt = os.path.join(designable_dir, 'designable.txt')
    if os.path.exists(designable_txt):
        os.remove(designable_txt)
    with open(designable_txt, 'w') as f:
        for _, row in designable_samples.iterrows():
            sample_path = row.sample_path
            sample_name = f'sample_id_{row.sample_id}_length_{row.length}.pdb'
            write_path = os.path.join(designable_dir, sample_name)
            shutil.copy(sample_path, write_path)
            f.write(write_path + '\n')
    if metrics_df['Total codesignable'].iloc[0] <= 1:
        metrics_df['Clusters'] = metrics_df['Total codesignable'].iloc[0]
    else:
        add_diversity_metrics(designable_dir, metrics_df, designable_csv_path)


def add_diversity_metrics(designable_dir, designable_csv, designable_csv_path):
    clusters = run_easy_cluster(designable_dir, designable_dir)
    designable_csv['Clusters'] = clusters
    designable_csv.to_csv(designable_csv_path, index=False)


def calculate_pmpnn_consistency(output_dir, designable_csv, designable_csv_path):
    # output dir points to directory containing length_60, length_61, ... etc folders
    sample_dirs = glob.glob(os.path.join(output_dir, 'length_*/sample_*'))
    average_accs = []
    max_accs = []
    for sample_dir in sample_dirs:
        pmpnn_fasta_path = os.path.join(sample_dir, 'self_consistency', 'seqs', 'sample_modified.fasta')
        codesign_fasta_path = os.path.join(sample_dir, 'self_consistency', 'codesign_seqs', 'codesign.fa')
        pmpnn_fasta = fasta.FastaFile.read(pmpnn_fasta_path)
        codesign_fasta = fasta.FastaFile.read(codesign_fasta_path)
        codesign_seq = codesign_fasta['codesign_seq_1']
        accs = []
        for seq in pmpnn_fasta:
            num_matches = sum([1 if pmpnn_fasta[seq][i] == codesign_seq[i] else 0 for i in range(len(pmpnn_fasta[seq]))])
            total_length = len(pmpnn_fasta[seq])
            accs.append(num_matches / total_length)
        average_accs.append(np.mean(accs))
        max_accs.append(np.max(accs))
    designable_csv['Average PMPNN Consistency'] = np.mean(average_accs)
    designable_csv['Average Max PMPNN Consistency'] = np.mean(max_accs)
    designable_csv.to_csv(designable_csv_path, index=False)


def calculate_pmpnn_designability(output_dir, designable_csv, designable_csv_path):
    sample_dirs = glob.glob(os.path.join(output_dir, 'length_*/sample_*'))
    try:
        single_pmpnn_results = []
        top_pmpnn_results = []
        for sample_dir in sample_dirs:
            all_pmpnn_folds_df = pd.read_csv(os.path.join(sample_dir, 'pmpnn_results.csv'))
            single_pmpnn_fold_df = all_pmpnn_folds_df.iloc[[0]]
            single_pmpnn_results.append(single_pmpnn_fold_df)
            min_index = all_pmpnn_folds_df['bb_rmsd'].idxmin()
            top_pmpnn_df = all_pmpnn_folds_df.loc[[min_index]]
            top_pmpnn_results.append(top_pmpnn_df)
        single_pmpnn_results_df = pd.concat(single_pmpnn_results, ignore_index=True)
        top_pmpnn_results_df = pd.concat(top_pmpnn_results, ignore_index=True)
        designable_csv['Single seq PMPNN Designability'] = np.mean(single_pmpnn_results_df['bb_rmsd'].to_numpy() < 2.0)
        designable_csv['Top seq PMPNN Designability'] = np.mean(top_pmpnn_results_df['bb_rmsd'].to_numpy() < 2.0)
        designable_csv.to_csv(designable_csv_path, index=False)
    except:
        # TODO i think it breaks when one process gets here first
        print("calculate pmpnn designability didnt work")


def get_pylogger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = ("debug", "info", "warning", "error", "exception", "fatal", "critical")
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))
    return logger


def flatten_dict(raw_dict):
    """Flattens a nested dict."""
    flattened = []
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            flattened.extend([(f'{k}:{i}', j) for i, j in flatten_dict(v)])
        else:
            flattened.append((k, v))
    return flattened


def save_traj(sample: np.ndarray, bb_prot_traj: np.ndarray, x0_traj: np.ndarray, diffuse_mask: np.ndarray, output_dir: str, aa_traj=None, clean_aa_traj=None,
              write_trajectories=True, ):
    """Writes final sample and reverse diffusion trajectory.

    Args:
        bb_prot_traj: [noisy_T, N, 37, 3] atom37 sampled diffusion states.
            T is number of time steps. First time step is t=eps,
            i.e. bb_prot_traj[0] is the final sample after reverse diffusion.
            N is number of residues.
        x0_traj: [clean_T, N, 37, 3] atom37 predictions of clean data at each time step.
        res_mask: [N] residue mask.
        diffuse_mask: [N] which residues are diffused.
        output_dir: where to save samples.
        aa_traj: [noisy_T, N] amino acids (0 - 20 inclusive).
        clean_aa_traj: [clean_T, N] amino acids (0 - 20 inclusive).
        write_trajectories: bool Whether to also write the trajectories as well
                                 as the final sample

    Returns:
        Dictionary with paths to saved samples.
            'sample_path': PDB file of final state of reverse trajectory.
            'traj_path': PDB file os all intermediate diffused states.
            'x0_traj_path': PDB file of C-alpha x_0 predictions at each state.
        b_factors are set to 100 for diffused residues
        residues if there are any.
    """

    # Write sample.
    diffuse_mask = diffuse_mask.astype(bool)
    sample_path = os.path.join(output_dir, 'sample.pdb')
    prot_traj_path = os.path.join(output_dir, 'bb_traj.pdb')
    x0_traj_path = os.path.join(output_dir, 'x0_traj.pdb')

    # Use b-factors to specify which residues are diffused.
    b_factors = np.tile((diffuse_mask * 100)[:, None], (1, 37))

    noisy_traj_length, num_res, _, _ = bb_prot_traj.shape
    clean_traj_length = x0_traj.shape[0]
    assert sample.shape == (num_res, 37, 3)
    assert bb_prot_traj.shape == (noisy_traj_length, num_res, 37, 3)
    assert x0_traj.shape == (clean_traj_length, num_res, 37, 3)

    if aa_traj is not None:
        assert aa_traj.shape == (noisy_traj_length, num_res)
        assert clean_aa_traj is not None
        assert clean_aa_traj.shape == (clean_traj_length, num_res)

    sample_path = au.write_prot_to_pdb(sample, sample_path, b_factors=b_factors, no_indexing=True, aatype=aa_traj[-1] if aa_traj is not None else None, )
    if write_trajectories:
        prot_traj_path = au.write_prot_to_pdb(bb_prot_traj, prot_traj_path, b_factors=b_factors, no_indexing=True, aatype=aa_traj, )
        x0_traj_path = au.write_prot_to_pdb(x0_traj, x0_traj_path, b_factors=b_factors, no_indexing=True, aatype=clean_aa_traj, )
    return {'sample_path': sample_path, 'traj_path': prot_traj_path, 'x0_traj_path': x0_traj_path, }


def get_dataset_cfg(cfg):
    if cfg.data.dataset == 'pdb':
        return cfg.pdb_dataset
    raise ValueError(f'Unrecognized dataset {cfg.data.dataset}')


def get_optimizer(cfg, model):
    if cfg.type == 'adam':
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(cfg.beta1, cfg.beta2,))
    elif cfg.type == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, )
    raise NotImplementedError('Optimizer not supported: %s' % cfg.type)


def get_scheduler(cfg, optimizer):
    if cfg.type is None:
        return BlackHole()
    elif cfg.type == 'plateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=cfg.factor, patience=cfg.patience, min_lr=cfg.min_lr, )
    elif cfg.type == 'multistep':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.milestones, gamma=cfg.gamma, )
    elif cfg.type == 'exp':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma, )
    elif cfg.type is None:
        return BlackHole()
    raise NotImplementedError('Scheduler not supported: %s' % cfg.type)


def log_losses(loss, loss_dict, scalar_dict, it, tag, logger=BlackHole(), ):
    logstr = '[%s] It %05d' % (tag, it)
    logstr += ' | loss %.2f' % loss.item()
    for k, v in loss_dict.items():
        logstr += ' | %s %.3f' % (k, v.item())
    for k, v in scalar_dict.items():
        logstr += ' | %s %.3f' % (k, v.item() if isinstance(v, torch.Tensor) else v)

    for k, v in loss_dict.items():
        wandb.log({f'train/loss_{k}': v}, step=it)
    for k, v in scalar_dict.items():
        wandb.log({f'train/{k}': v}, step=it)

    return logstr


class ScalarMetricAccumulator(object):

    def __init__(self):
        super().__init__()
        self.accum_dict = {}
        self.count_dict = {}

    @torch.no_grad()
    def add(self, name, value, batchsize=None, mode=None):
        assert mode is None or mode in ('mean', 'sum')

        if mode is None:
            delta = value.sum()
            count = value.size(0)
        elif mode == 'mean':
            delta = value * batchsize
            count = batchsize
        elif mode == 'sum':
            delta = value
            count = batchsize
        delta = delta.item() if isinstance(delta, torch.Tensor) else delta

        if name not in self.accum_dict:
            self.accum_dict[name] = 0
            self.count_dict[name] = 0
        self.accum_dict[name] += delta
        self.count_dict[name] += count

    def log(self, it, tag, logger=BlackHole(), writer=BlackHole()):
        summary = {k: self.accum_dict[k] / self.count_dict[k] for k in self.accum_dict}
        logstr = '[%s] Iter %05d' % (tag, it)
        for k, v in summary.items():
            logstr += ' | %s %.4f' % (k, v)
            writer.add_scalar('%s/%s' % (tag, k), v, it)
            wandb.log({f'{tag}/{k}': v}, step=it)
        logger.info(logstr)

    def get_average(self, name):
        return self.accum_dict[name] / self.count_dict[name]


def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor):
        try:
            return obj.cuda(device=device, non_blocking=True)
        except RuntimeError:
            return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to(o, device=device) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, device=device) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device=device) for k, v in obj.items()}
    return obj


def sum_weighted_losses(losses, weights):
    """
    Args:
        losses:     Dict of scalar tensors.
        weights:    Dict of weights.
    """
    loss = 0
    for k in losses.keys():
        if weights is None:
            loss = loss + losses[k]
        else:
            loss = loss + weights[k] * losses[k]
    return loss


def count_parameters(model, grad=False):
    if grad:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
