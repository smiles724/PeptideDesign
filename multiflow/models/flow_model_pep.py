import esm
import torch
import torch.nn.functional as F
from torch import nn

import multiflow.data.torus as torus
from multiflow.data import all_atom
from multiflow.data import so3_utils
from multiflow.data import utils as du
from multiflow.data.pep_constants import BBHeavyAtom
from multiflow.data.pep_dataloader import PaddingCollate
from multiflow.data.residue_constants import restype_order
from multiflow.data.torsion import torsions_mask
from multiflow.models import ipa_pytorch
from multiflow.models.edge_feature_net import EdgeFeatureNet
from multiflow.models.geometry import construct_3d_basis
from multiflow.models.llm.esm2_adapter import ESM2WithStructuralAdatper
from multiflow.models.node_feature_net import NodeFeatureNet
from multiflow.models.utils import AngularEncoding, sample_from, uniform_so3, clampped_one_hot, zero_center_part

collate_fn = PaddingCollate(eight=False)
# transform residue index to the ESM-form
esm_index_transform = {20: 32}  # <mask> is 32 in ESM, different from <unk> (20); but MultiFlow use 20 for both <mask> and <unk>
esm_index_transform_back = []
esm_vocab = esm.Alphabet.from_architecture('ESM-1b').tok_to_idx
for i, j in restype_order.items():
    esm_index_transform[j] = esm_vocab[i]
    esm_index_transform_back.append(esm_vocab[i])


class PepModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._interpolant_cfg = cfg.interpolant
        self._ipa_conf = cfg.ipa
        self._model_conf = cfg
        node_embed_size = self._model_conf.node_embed_size
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE)
        self.node_feature_net = NodeFeatureNet(self._model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(self._model_conf.edge_features)
        self.angles_embedder = AngularEncoding(num_funcs=6)  # 25*7=175, for competitive embedding size
        self.fixed_receptor = cfg.ipa.fixed_receptor
        self.res_feat_mixer = nn.Sequential(nn.Linear(node_embed_size + self.angles_embedder.get_out_dim(in_dim=7), node_embed_size), nn.ReLU(),
                                            nn.Linear(node_embed_size, node_embed_size), )
        self.angle_net = nn.Sequential(nn.Linear(node_embed_size, node_embed_size), nn.ReLU(), nn.Linear(node_embed_size, node_embed_size), nn.ReLU(),
                                       nn.Linear(node_embed_size, 7))  # 7 angles
        self.llm_decoder = ESM2WithStructuralAdatper.from_pretrained(args=self._model_conf.llm, name=self._model_conf.llm_name)
        # Attention trunk
        self.trunk = nn.ModuleDict()
        for b in range(self._ipa_conf.num_blocks):
            self.trunk[f'ipa_{b}'] = ipa_pytorch.InvariantPointAttention(self._ipa_conf)
            self.trunk[f'ipa_ln_{b}'] = nn.LayerNorm(self._ipa_conf.c_s)
            tfmr_in = self._ipa_conf.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(d_model=tfmr_in, nhead=self._ipa_conf.seq_tfmr_num_heads, dim_feedforward=tfmr_in, batch_first=True,
                                                          dropout=self._model_conf.transformer_dropout, norm_first=False)
            self.trunk[f'seq_tfmr_{b}'] = nn.TransformerEncoder(tfmr_layer, self._ipa_conf.seq_tfmr_num_layers, enable_nested_tensor=False)
            self.trunk[f'post_tfmr_{b}'] = ipa_pytorch.Linear(tfmr_in, self._ipa_conf.c_s, init="final")
            self.trunk[f'node_transition_{b}'] = ipa_pytorch.StructureModuleTransition(c=self._ipa_conf.c_s)
            self.trunk[f'bb_update_{b}'] = ipa_pytorch.BackboneUpdate(self._ipa_conf.c_s, use_rot_updates=True)

            if b < self._ipa_conf.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(node_embed_size=self._ipa_conf.c_s, edge_embed_in=edge_in,
                                                                                edge_embed_out=self._model_conf.edge_embed_size, )
        self.aatype_pred_net = nn.Sequential(nn.Linear(node_embed_size, node_embed_size), nn.ReLU(), nn.Linear(node_embed_size, node_embed_size), nn.ReLU(),
                                             nn.Linear(node_embed_size, self._model_conf.aatype_pred_num_tokens), )

        self.K = self._interpolant_cfg.seqs.num_classes
        self.k = self._interpolant_cfg.seqs.simplex_value

    def seq_to_simplex(self, seqs):
        return clampped_one_hot(seqs, self.K).float() * self.k * 2 - self.k  # (B,L,K)

    def corrupt_batch(self, batch):
        num_batch, num_res = batch['aa'].shape
        gen_mask, res_mask, angle_mask = batch['generate_mask'].long(), batch['res_mask'].long(), batch['torsion_angle_mask'].long()  # [..., 2:]
        rotmats_1 = construct_3d_basis(batch['pos_heavyatom'][:, :, BBHeavyAtom.CA], batch['pos_heavyatom'][:, :, BBHeavyAtom.C], batch['pos_heavyatom'][:, :, BBHeavyAtom.N])
        trans_1_c = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]   # already centered when constructing dataset
        seqs_1 = batch['aa']
        angles_1 = batch['torsion_angle']  # [..., 2:]
        seqs_1_simplex = self.seq_to_simplex(seqs_1)

        with torch.no_grad():
            t = torch.rand((num_batch, 1), device=batch['aa'].device)
            t = t * (1 - 2 * self._interpolant_cfg.t_normalization_clip) + self._interpolant_cfg.t_normalization_clip  # avoid 0
            batch['t'] = t
            # corrupt trans
            trans_0 = torch.randn((num_batch, num_res, 3), device=batch['aa'].device) * self._interpolant_cfg.trans.sigma  # scale with sigma?
            trans_0_c, _ = zero_center_part(trans_0, gen_mask, res_mask)
            trans_t = (1 - t[..., None]) * trans_0_c + t[..., None] * trans_1_c
            batch['trans_t_c'] = torch.where(batch['generate_mask'][..., None], trans_t, trans_1_c)
            # corrupt rotmats
            rotmats_0 = uniform_so3(num_batch, num_res, device=batch['aa'].device)
            rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
            batch['rotmats_t'] = torch.where(batch['generate_mask'][..., None, None], rotmats_t, rotmats_1)
            # corrup angles
            angles_0 = torus.tor_random_uniform(angles_1.shape, device=batch['aa'].device, dtype=angles_1.dtype)  # (B,L,5)
            angles_t = torus.tor_geodesic_t(t[..., None], angles_1, angles_0)
            batch['angles_t'] = torch.where(batch['generate_mask'][..., None], angles_t, angles_1)
            # corrupt seqs
            seqs_0_simplex = self.k * torch.randn_like(seqs_1_simplex)  # (B,L,K)
            seqs_t_simplex = ((1 - t[..., None]) * seqs_0_simplex) + (t[..., None] * seqs_1_simplex)  # (B,L,K)
            seqs_t_simplex = torch.where(batch['generate_mask'][..., None], seqs_t_simplex, seqs_1_simplex)
            seqs_t_prob = F.softmax(seqs_t_simplex, dim=-1)  # (B,L,K)
            seqs_t = sample_from(seqs_t_prob)  # (B,L)
            batch['seqs_t'] = torch.where(batch['generate_mask'], seqs_t, seqs_1)
        return batch

    def decode(self, so3_t, trans_t_c, rotmats_t, aatypes_t, angles_t, node_mask, edge_mask, diffuse_mask, chain_index, res_index, ):
        # Initialize node and edge embeddings
        node_embed = self.node_feature_net(t=so3_t, res_mask=node_mask, diffuse_mask=diffuse_mask, chain_index=chain_index, pos=res_index, aatypes=aatypes_t,
                                           angles=angles_t)
        edge_embed = self.edge_feature_net(node_embed, trans_t_c, edge_mask, diffuse_mask, chain_index)
        node_embed = self.res_feat_mixer(torch.cat([node_embed, self.angles_embedder(angles_t)], dim=-1))
        node_embed = node_embed * node_mask[..., None]
        edge_embed = edge_embed * edge_mask[..., None]

        # Main trunk
        curr_rigids = du.create_rigid(rotmats_t, trans_t_c)
        curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        for b in range(self._ipa_conf.num_blocks):
            ipa_embed = self.trunk[f'ipa_{b}'](node_embed, edge_embed, curr_rigids, node_mask)
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f'ipa_ln_{b}'](node_embed + ipa_embed)
            seq_tfmr_out = self.trunk[f'seq_tfmr_{b}'](node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool))
            node_embed = node_embed + self.trunk[f'post_tfmr_{b}'](seq_tfmr_out)
            node_embed = self.trunk[f'node_transition_{b}'](node_embed)
            node_embed = node_embed * node_mask[..., None]
            rigid_update = self.trunk[f'bb_update_{b}'](node_embed * node_mask[..., None])
            if self.fixed_receptor:
                curr_rigids = curr_rigids.compose_q_update_vec(rigid_update, (node_mask * diffuse_mask)[..., None])
            else:
                curr_rigids = curr_rigids.compose_q_update_vec(rigid_update, node_mask[..., None])
            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans_1 = curr_rigids.get_trans()
        pred_rotmats_1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs_1_ipa = self.aatype_pred_net(node_embed)
        pred_angles_1 = self.angle_net(node_embed) % (2 * torch.pi)  # inductive bias to bound between (0,2pi)
        return pred_trans_1, pred_rotmats_1, pred_seqs_1_ipa, pred_angles_1, node_embed

    def forward(self, batch, ):
        batch = self.corrupt_batch(batch)
        num_batch, num_res = batch['aa'].shape
        gen_mask, res_mask, angle_mask = batch['generate_mask'].long(), batch['res_mask'].long(), batch['torsion_angle_mask'].long()
        node_mask = batch['res_mask'].long()
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = batch['generate_mask'].long()

        pos_heavyatom = batch['pos_heavyatom']
        seqs_1 = batch['aa']
        angles_1 = batch['torsion_angle']
        trans_1_c = pos_heavyatom[:, :, BBHeavyAtom.CA]  # already centered when constructing dataset
        rotmats_1 = construct_3d_basis(pos_heavyatom[:, :, BBHeavyAtom.CA], pos_heavyatom[:, :, BBHeavyAtom.C], pos_heavyatom[:, :, BBHeavyAtom.N])

        chain_index = batch['chain_nb']
        res_index = batch['res_nb']

        t = so3_t = cat_t = batch['t']
        seqs_t = batch['seqs_t']
        angles_t = batch['angles_t']
        rotmats_t = batch['rotmats_t']
        trans_t_c = batch['trans_t_c']

        # denoise
        pred_trans_1, pred_rotmats_1, pred_seqs_1_ipa, pred_angles_1, node_embed = self.decode(t, trans_t_c, rotmats_t, seqs_t, angles_t,
                                                                                               node_mask, edge_mask, diffuse_mask, chain_index, res_index)
        pred_trans_1_c, _ = zero_center_part(pred_trans_1, gen_mask, res_mask)
        pred_trans_1_c = pred_trans_1  # implicitly enforce zero center in gen_mask, in this way, we dont need to move receptor when sampling # TODO: ???

        pred_logits_wo_mask = pred_seqs_1_ipa.clone()
        if self._model_conf.aatype_pred_num_tokens == du.NUM_TOKENS + 1:
            pred_logits_wo_mask[..., du.MASK_TOKEN_INDEX] = -1e9  # prob. of unknown/<mask> is ~0
        pred_aatypes = sample_from(F.softmax(pred_logits_wo_mask, dim=-1))
        pred_aatypes = torch.where(batch['generate_mask'], pred_aatypes, torch.clamp(seqs_1, 0, 19))

        # transform to the ESM codebook
        pred_aatypes_esm = pred_aatypes.clone()
        for key, value in esm_index_transform.items():
            pred_aatypes_esm[pred_aatypes == key] = value
        pred_aatypes_esm[~node_mask.bool()] = 1  # 1 for <pad> in ESM. No predicted <mask>, don't worry! # TODO: [gen_mask]?
        esm_logits = self.llm_decoder(tokens=pred_aatypes_esm, encoder_out=node_embed)  # no t is added into llm
        pred_seqs_1_prob = esm_logits[..., esm_index_transform_back]
        pred_seqs_1 = sample_from(F.softmax(pred_seqs_1_prob, dim=-1))
        pred_seqs_1 = torch.where(batch['generate_mask'], pred_seqs_1, torch.clamp(seqs_1, 0, 19))

        norm_scale = 1 / (1 - torch.min(t[..., None], torch.tensor(self._interpolant_cfg.t_normalization_clip)))  # yim etal.trick, 1/1-t

        # trans vf loss
        trans_loss = torch.sum((pred_trans_1_c - trans_1_c) ** 2 * gen_mask[..., None], dim=(-1, -2)) / (torch.sum(gen_mask, dim=-1) + 1e-8)  # (B,)
        trans_loss = torch.mean(trans_loss)

        # rots vf loss
        gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, rotmats_1)
        pred_rot_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
        rot_loss = torch.sum(((gt_rot_vf - pred_rot_vf) * norm_scale) ** 2 * gen_mask[..., None], dim=(-1, -2)) / (torch.sum(gen_mask, dim=-1) + 1e-8)  # (B,)
        rot_loss = torch.mean(rot_loss)

        # bb aux loss
        gt_bb_atoms = all_atom.to_atom37(trans_1_c, rotmats_1)[:, :, :3]
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1_c, pred_rotmats_1)[:, :, :3]
        bb_atom_loss = torch.sum((gt_bb_atoms - pred_bb_atoms) ** 2 * gen_mask[..., None, None], dim=(-1, -2, -3)) / (torch.sum(gen_mask, dim=-1) + 1e-8)  # (B,)
        bb_atom_loss = torch.mean(bb_atom_loss)

        # seqs vf loss
        seqs_loss = F.cross_entropy(pred_seqs_1_prob.view(-1, pred_seqs_1_prob.shape[-1]), torch.clamp(seqs_1, 0, 19).view(-1), reduction='none').view(
            pred_seqs_1_prob.shape[:-1])  # (N,L), not softmax
        seqs_loss = torch.sum(seqs_loss * gen_mask, dim=-1) / (torch.sum(gen_mask, dim=-1) + 1e-8)
        seqs_loss = torch.mean(seqs_loss)

        seqs_loss_ipa = F.cross_entropy(pred_seqs_1_ipa.view(-1, pred_seqs_1_ipa.shape[-1]), torch.clamp(seqs_1, 0, 19).view(-1), reduction='none').view(
            pred_seqs_1_ipa.shape[:-1])  # (N,L), not softmax
        seqs_loss_ipa = torch.sum(seqs_loss_ipa * gen_mask, dim=-1) / (torch.sum(gen_mask, dim=-1) + 1e-8)
        seqs_loss_ipa = torch.mean(seqs_loss_ipa)

        # angle vf loss
        angle_mask_loss = torsions_mask.to(batch['aa'].device)
        angle_mask_loss = angle_mask_loss[pred_seqs_1.reshape(-1)].reshape(num_batch, num_res, -1)  # (B,L,5)
        angle_mask_loss = torch.cat([angle_mask_loss, angle_mask_loss], dim=-1)  # (B,L,10)
        angle_mask_loss = torch.logical_and(batch['generate_mask'][..., None].bool(), angle_mask_loss)
        gt_angle_vf = torus.tor_logmap(angles_t, angles_1)
        gt_angle_vf_vec = torch.cat([torch.sin(gt_angle_vf), torch.cos(gt_angle_vf)], dim=-1)
        pred_angle_vf = torus.tor_logmap(angles_t, pred_angles_1)
        pred_angle_vf_vec = torch.cat([torch.sin(pred_angle_vf), torch.cos(pred_angle_vf)], dim=-1)
        angle_loss = torch.sum(((gt_angle_vf_vec - pred_angle_vf_vec) * norm_scale) ** 2 * angle_mask_loss, dim=(-1, -2)) / (
                torch.sum(angle_mask_loss, dim=(-1, -2)) + 1e-8)  # (B,)
        angle_loss = torch.mean(angle_loss)

        # angle aux loss
        angles_1_vec = torch.cat([torch.sin(angles_1), torch.cos(angles_1)], dim=-1)
        pred_angles_1_vec = torch.cat([torch.sin(pred_angles_1), torch.cos(pred_angles_1)], dim=-1)
        torsion_loss = torch.sum((pred_angles_1_vec - angles_1_vec) ** 2 * angle_mask_loss, dim=(-1, -2)) / (torch.sum(angle_mask_loss, dim=(-1, -2)) + 1e-8)  # (B,)
        torsion_loss = torch.mean(torsion_loss)

        return {"trans_loss": trans_loss, 'seqs_loss_ipa': seqs_loss_ipa, 'rot_loss': rot_loss, 'bb_atom_loss': bb_atom_loss, 'seqs_loss': seqs_loss, 'angle_loss': angle_loss,
                'torsion_loss': torsion_loss, }

    @torch.no_grad()
    def sample(self, batch, num_steps=100, ):
        num_batch, num_res = batch['aa'].shape
        gen_mask, res_mask = batch['generate_mask'], batch['res_mask']
        angle_mask_loss = torsions_mask.to(batch['aa'].device)

        # initial noise
        rotmats_1 = construct_3d_basis(batch['pos_heavyatom'][:, :, BBHeavyAtom.CA], batch['pos_heavyatom'][:, :, BBHeavyAtom.C], batch['pos_heavyatom'][:, :, BBHeavyAtom.N])
        trans_1 = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]
        seqs_1 = batch['aa']
        angles_1 = batch['torsion_angle']
        node_mask = batch['res_mask'].long()
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = batch['generate_mask'].long()
        chain_index = batch['chain_nb']
        res_index = batch['res_nb']
        # prepare for denoise
        trans_1_c = trans_1  # already centered when constructing dataset
        seqs_1_simplex = self.seq_to_simplex(seqs_1)

        # bb
        rotmats_0 = uniform_so3(num_batch, num_res, device=batch['aa'].device)
        rotmats_0 = torch.where(batch['generate_mask'][..., None, None], rotmats_0, rotmats_1)
        trans_0 = torch.randn((num_batch, num_res, 3), device=batch['aa'].device)  # scale with sigma?
        # move center and receptor
        trans_0_c, center = zero_center_part(trans_0, gen_mask, res_mask)
        trans_0_c = torch.where(batch['generate_mask'][..., None], trans_0_c, trans_1_c)

        # angle
        angles_0 = torus.tor_random_uniform(angles_1.shape, device=batch['aa'].device, dtype=angles_1.dtype)  # (B,L,5)
        angles_0 = torch.where(batch['generate_mask'][..., None], angles_0, angles_1)

        # seq
        seqs_0_simplex = self.k * torch.randn((num_batch, num_res, self.K), device=batch['aa'].device)
        seqs_0_prob = F.softmax(seqs_0_simplex, dim=-1)
        seqs_0 = sample_from(seqs_0_prob)
        seqs_0 = torch.where(batch['generate_mask'], seqs_0, seqs_1)
        seqs_0_simplex = torch.where(batch['generate_mask'][..., None], seqs_0_simplex, seqs_1_simplex)

        # Set-up time
        ts = torch.linspace(1.e-2, 1.0, num_steps)
        t_1 = ts[0]
        clean_traj = []
        rotmats_t_1, trans_t_1_c, angles_t_1, aatypes_t_1, seqs_t_1_simplex = rotmats_0, trans_0_c, angles_0, seqs_0, seqs_0_simplex

        # denoise loop
        for t_2 in ts[1:]:
            t = torch.ones((num_batch, 1), device=batch['aa'].device) * t_1

            # denoise
            pred_trans_1, pred_rotmats_1, pred_aatypes_1_prob, pred_angles_1, node_embed = self.decode(t, trans_t_1_c, rotmats_t_1, aatypes_t_1, angles_t_1,
                                                                                                       node_mask, edge_mask, diffuse_mask, chain_index, res_index, )

            # rots
            pred_rotmats_1 = torch.where(batch['generate_mask'][..., None, None], pred_rotmats_1, rotmats_1)
            # trans, move center
            pred_trans_1_c = torch.where(batch['generate_mask'][..., None], pred_trans_1, trans_1_c)  # move receptor also
            # angles
            pred_angles_1 = torch.where(batch['generate_mask'][..., None], pred_angles_1, angles_1)
            # seqs
            pred_seqs_1 = sample_from(F.softmax(pred_aatypes_1_prob, dim=-1))  # deterministic is worse
            pred_seqs_1 = torch.where(batch['generate_mask'], pred_seqs_1, seqs_1)

            # pred_aatypes_esm = pred_seqs_1.clone()
            # for key, value in esm_index_transform.items():
            #     pred_aatypes_esm[pred_seqs_1 == key] = value
            # pred_aatypes_esm[~node_mask.bool()] = 1  # 1 for <pad> in ESM. No predicted <mask>, don't worry!
            # esm_logits = self.llm_decoder(tokens=pred_aatypes_esm, encoder_out=node_embed)  # no t is added into llm
            # pred_seqs_1_prob = esm_logits[..., esm_index_transform_back]
            # pred_seqs_1 = sample_from(F.softmax(pred_seqs_1_prob, dim=-1))

            pred_seqs_1 = torch.where(batch['generate_mask'], pred_seqs_1, seqs_1)

            pred_seqs_1_simplex = self.seq_to_simplex(pred_seqs_1)
            # seq-angle
            torsion_mask = angle_mask_loss[pred_seqs_1.reshape(-1)].reshape(num_batch, num_res, -1)  # (B,L,5)
            pred_angles_1 = torch.where(torsion_mask.bool(), pred_angles_1, torch.zeros_like(pred_angles_1))

            clean_traj.append({'rotmats': pred_rotmats_1.cpu(), 'trans': pred_trans_1_c.cpu(), 'angles': pred_angles_1.cpu(), 'seqs': pred_seqs_1.cpu(),
                               'seqs_simplex': pred_seqs_1_simplex.cpu(), 'rotmats_1': rotmats_1.cpu(), 'trans_1': trans_1_c.cpu(), 'angles_1': angles_1.cpu(),
                               'seqs_1': seqs_1.cpu()})
            # reverse step, also only for gen mask region
            d_t = (t_2 - t_1) * torch.ones((num_batch, 1), device=batch['aa'].device)
            # Euler step
            trans_t_2 = trans_t_1_c + (pred_trans_1_c - trans_0_c) * d_t[..., None]
            trans_t_2_c = torch.where(batch['generate_mask'][..., None], trans_t_2, trans_1_c)  # move receptor also
            rotmats_t_2 = so3_utils.geodesic_t(d_t[..., None] * 10, pred_rotmats_1, rotmats_t_1)
            rotmats_t_2 = torch.where(batch['generate_mask'][..., None, None], rotmats_t_2, rotmats_1)
            # angles
            angles_t_2 = torus.tor_geodesic_t(d_t[..., None], pred_angles_1, angles_t_1)
            angles_t_2 = torch.where(batch['generate_mask'][..., None], angles_t_2, angles_1)
            # seqs
            seqs_t_2_simplex = seqs_t_1_simplex + (pred_seqs_1_simplex - seqs_0_simplex) * d_t[..., None]
            seqs_t_2 = sample_from(F.softmax(seqs_t_2_simplex, dim=-1))
            seqs_t_2 = torch.where(batch['generate_mask'], seqs_t_2, seqs_1)
            # seq-angle
            torsion_mask = angle_mask_loss[seqs_t_2.reshape(-1)].reshape(num_batch, num_res, -1)  # (B,L,5)
            angles_t_2 = torch.where(torsion_mask.bool(), angles_t_2, torch.zeros_like(angles_t_2))

            rotmats_t_1, trans_t_1_c, angles_t_1, aatypes_t_1, seqs_t_1_simplex = rotmats_t_2, trans_t_2_c, angles_t_2, seqs_t_2, seqs_t_2_simplex
            t_1 = t_2

        # final step
        t_1 = ts[-1]
        t = torch.ones((num_batch, 1), device=batch['aa'].device) * t_1

        # denoise
        pred_trans_1, pred_rotmats_1, pred_aatypes_1_prob, pred_angles_1, _ = self.decode(t, trans_t_1_c, rotmats_t_1, aatypes_t_1, angles_t_1, node_mask,
                                                                                          edge_mask, diffuse_mask, chain_index, res_index, )

        # orientations
        pred_rotmats_1 = torch.where(batch['generate_mask'][..., None, None], pred_rotmats_1, rotmats_1)
        # move center
        pred_trans_1_c = torch.where(batch['generate_mask'][..., None], pred_trans_1, trans_1_c)  # move receptor also
        # angles
        pred_angles_1 = torch.where(batch['generate_mask'][..., None], pred_angles_1, angles_1)
        # seqs
        pred_seqs_1 = sample_from(F.softmax(pred_aatypes_1_prob, dim=-1))
        pred_seqs_1 = torch.where(batch['generate_mask'], pred_seqs_1, seqs_1)
        pred_seqs_1_simplex = self.seq_to_simplex(pred_seqs_1)
        # seq-angle
        torsion_mask = angle_mask_loss[pred_seqs_1.reshape(-1)].reshape(num_batch, num_res, -1)  # (B,L,5)
        pred_angles_1 = torch.where(torsion_mask.bool(), pred_angles_1, torch.zeros_like(pred_angles_1))

        clean_traj.append(
            {'rotmats': pred_rotmats_1.cpu(), 'trans': pred_trans_1_c.cpu(), 'angles': pred_angles_1.cpu(), 'seqs': pred_seqs_1.cpu(), 'seqs_simplex': pred_seqs_1_simplex.cpu(),
             'rotmats_1': rotmats_1.cpu(), 'trans_1': trans_1_c.cpu(), 'angles_1': angles_1.cpu(), 'seqs_1': seqs_1.cpu()})

        return clean_traj
