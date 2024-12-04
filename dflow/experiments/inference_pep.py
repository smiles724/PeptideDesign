import gc
import os
from copy import deepcopy

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm.auto import tqdm
import torch.nn.functional as F

from dflow.data.pep_dataloader import PepDataset, PaddingCollate
from dflow.experiments.utils import seed_all, recursive_to, process_dic
from dflow.models.flow_model_pep import PepModel
from dflow.data.torsion import full_atom_reconstruction, get_heavyatom_mask
from dflow.data.writers import save_pdb

collate_fn = PaddingCollate(eight=False)
config_path = "../configs"
config_name = "pep_codesign.yaml"


@hydra.main(version_base=None, config_path=config_path, config_name=config_name)
def main(cfg: DictConfig):
    seed_all(2024)
    sam_cfg = cfg.sample
    device = sam_cfg.device
    save_path = os.path.join(sam_cfg.output, '{}_{}_{}_{}'.format(sam_cfg.ckpt_path, sam_cfg.num_steps, sam_cfg.num_samples, sam_cfg.llm))
    if sam_cfg.x_mirror:
        save_path += '_x_mirror'
    elif sam_cfg.y_mirror:
        save_path += '_y_mirror'
    elif sam_cfg.z_mirror:
        save_path += '_z_mirror'
    os.makedirs(save_path, exist_ok=True)

    dataset = PepDataset(structure_dir=cfg.dataset.val.structure_dir, dataset_dir=cfg.dataset.val.dataset_dir, name=cfg.dataset.val.name,
                         x_mirror=sam_cfg.x_mirror, reset=cfg.dataset.val.reset)
    model = PepModel(cfg.model).to(device)
    ckpt = torch.load(sam_cfg.ckpt_path, map_location=device)
    print(f'Loading model successfully from {sam_cfg.ckpt_path}')
    model.load_state_dict(process_dic(ckpt['model']), strict=False)
    model.eval()

    dic = {'id': [], 'len': [], 'tran': [], 'best_aar': [], 'avg_aar': [], 'rot': []}
    aar_dic = {}
    tq = tqdm(range(len(dataset)))
    for i in tq:
        try:
            data_list = [deepcopy(dataset[i]) for _ in range(sam_cfg.num_samples)]
            batch = recursive_to(collate_fn(data_list), device)
            name = batch['id'][0]
            traj_1 = model.sample(batch, num_steps=sam_cfg.num_steps, angle_purify=sam_cfg.angle_purify, llm=sam_cfg.llm)

            ca_dist = torch.sqrt(torch.sum((traj_1[-1]['trans'] - traj_1[-1]['trans_1']) ** 2 * batch['generate_mask'][..., None].cpu().long()) / (
                        torch.sum(batch['generate_mask']) + 1e-8).cpu())  # rmsd
            rot_dist = torch.sqrt(torch.sum((traj_1[-1]['rotmats'] - traj_1[-1]['rotmats_1']) ** 2 * batch['generate_mask'][..., None, None].long().cpu()) / (
                        torch.sum(batch['generate_mask']) + 1e-8).cpu())  # rmsd

            pep_len = batch["generate_mask"][0].sum().item()
            aar = torch.sum((traj_1[-1]['seqs'] == traj_1[-1]['seqs_1']) * batch['generate_mask'].long().cpu(), dim=-1) / pep_len

            aar_dic[name] = aar.tolist()
            best_aar, avg_aar = max(aar), aar.mean()
            dic['tran'].append(ca_dist.item())
            dic['rot'].append(rot_dist.item())
            dic['best_aar'].append(best_aar.item())
            dic['avg_aar'].append(aar.mean().item())
            dic['id'].append(name)
            dic['len'].append(pep_len)
            tq.set_description(f"tran:{sum(dic['tran']) / (i + 1):.2f} | rot:{sum(dic['rot']) / (i + 1):.2f} | best_aar: {sum(dic['best_aar']) / (i + 1):.3f} |"
                               f" aar:{sum(dic['avg_aar']) / (i + 1):.3f} | len:{pep_len:.0f}")  # average metrics
        except Exception as e:
            print(f"Error occurred during training: {e}")  # OOD
            torch.cuda.empty_cache()
            continue
        # free
        torch.cuda.empty_cache()
        gc.collect()

        # meta data
        samples = traj_1[-1]
        batch = recursive_to(batch, 'cpu')
        chain_id = [list(item) for item in zip(*batch['chain_id'])][0]  # fix chain id in collate func
        icode = [' ' for _ in range(len(chain_id))]  # batch icode have same problem
        nums = len(batch['id'])
        if samples['angles'].shape[-1] == 7:
            samples['angles'] = samples['angles'][..., 2:]
        pos_ha, _, _ = full_atom_reconstruction(R_bb=samples['rotmats'], t_bb=samples['trans'], angles=samples['angles'], aa=samples['seqs'])  # (32,L,14,3), instead of 15, ignore OXT masked
        pos_ha = F.pad(pos_ha, pad=(0, 0, 0, 15 - 14), value=0.)  # (32,L,A,3) pos14 A=14
        pos_new = torch.where(batch['generate_mask'][:, :, None, None], pos_ha, batch['pos_heavyatom'])
        mask_new = get_heavyatom_mask(samples['seqs'])
        aa_new = samples['seqs']
        if sam_cfg.x_mirror:  # inverse back
            pos_new[..., 0] *= -1
            batch['pos_heavyatom'][..., 0] *= -1

        os.makedirs(f'{save_path}/{name}/', exist_ok=True)
        for j in range(nums):
            data_saved = {'chain_nb': batch['chain_nb'][0], 'chain_id': chain_id, 'resseq': batch['resseq'][0], 'icode': icode, 'aa': aa_new[j], 'mask_heavyatom': mask_new[j],
                          'pos_heavyatom': pos_new[j], }
            save_pdb(data_saved, path=f'{save_path}/{name}/sample_{j}.pdb')
        gt_data = {'chain_nb': batch['chain_nb'][0], 'chain_id': chain_id, 'resseq': batch['resseq'][0], 'icode': icode, 'aa': batch['aa'][0],
                   'mask_heavyatom': batch['mask_heavyatom'][0], 'pos_heavyatom': batch['pos_heavyatom'][0], }
        save_pdb(gt_data, path=f'{save_path}/{name}/gt.pdb')

    dic = pd.DataFrame(dic)
    dic.to_csv(f'{save_path}/outputs.csv', index=None)
    aar_dic = pd.DataFrame(aar_dic)
    aar_dic.to_csv(f'{save_path}/aar.csv', index=None)


if __name__ == '__main__':
    main()
