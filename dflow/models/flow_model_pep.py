import torch
import torch.nn.functional as F
from torch import nn

import dflow.data.torus as torus
from dflow.data import all_atom
from dflow.data import so3_utils
from dflow.data import utils as du
from dflow.models import ipa_pytorch
from dflow.models.edge_feature_net import EdgeFeatureNet
from dflow.models.node_feature_net import NodeFeatureNet
from dflow.models.node import NodeEmbedder
from dflow.models.edge import EdgeEmbedder
from dflow.models.geometry import construct_3d_basis
from dflow.models.llm.esm2_adapter import ESM2WithStructuralAdatper
from dflow.data.pep_constants import max_num_heavyatoms, BBHeavyAtom, chi_angles_mask
from dflow.models.utils import AngularEncoding, sample_from, uniform_so3, zero_center_part, esm_index_transform, esm_index_transform_back, clampped_one_hot, get_time_embedding

torsions_mask_vocab_all = torch.zeros([21, 7]).float()  # 0-19, X
for i in range(20):
    torsions_mask_vocab_all[i] = torch.tensor([True, True, True] + chi_angles_mask[i]).float()
torsions_mask_vocab = torsions_mask_vocab_all[:, 2:]


class PepModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._interpolant_cfg = cfg.interpolant
        self._ipa_conf = cfg.ipa
        self._model_conf = cfg
        self.net_type = cfg.net_type
        node_embed_size = cfg.node_embed_size
        self.rigids_scaling = cfg.rigids_scaling
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda y: y * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda y: y * du.NM_TO_ANG_SCALE)
        if cfg.net_type == 'frameflow':  # FrameFlow arch
            self.node_feature_net = NodeFeatureNet(cfg.node_features)
            self.edge_feature_net = EdgeFeatureNet(cfg.edge_features)
        elif cfg.net_type == 'pepflow':
            self.node_feature_net = NodeEmbedder(node_embed_size, max_num_heavyatoms)
            self.edge_feature_net = EdgeEmbedder(self._model_conf.edge_embed_size, max_num_heavyatoms)
            self.angles_embedder = AngularEncoding(num_funcs=5)  # 21*7=147, for competitive embedding size
            self.current_seq_embedder = nn.Embedding(22, self._ipa_conf.c_s)
            self.res_feat_mixer = nn.Sequential(nn.Linear(node_embed_size * 3 + self.angles_embedder.get_out_dim(in_dim=5), node_embed_size), nn.ReLU(),
                                                nn.Linear(node_embed_size, node_embed_size), )
        self.fixed_receptor = cfg.ipa.fixed_receptor
        self.angle_net = nn.Sequential(nn.Linear(node_embed_size, node_embed_size), nn.ReLU(), nn.Linear(node_embed_size, node_embed_size), nn.ReLU(),
                                       nn.Linear(node_embed_size, 5))  # 7 angles
        self.aatype_pred_net = nn.Sequential(nn.Linear(node_embed_size, node_embed_size), nn.ReLU(), nn.Linear(node_embed_size, node_embed_size), nn.ReLU(),
                                             nn.Linear(node_embed_size, self._model_conf.aatype_pred_num_tokens), )

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

            if b < self._ipa_conf.num_blocks - 1:  # No edge update on the last block.
                edge_in = self._model_conf.edge_embed_size
                self.trunk[f'edge_transition_{b}'] = ipa_pytorch.EdgeTransition(node_embed_size=self._ipa_conf.c_s, edge_embed_in=edge_in,
                                                                                edge_embed_out=self._model_conf.edge_embed_size, )
        self.use_llm = self._model_conf.llm.use
        self.llm_decoder = ESM2WithStructuralAdatper.from_pretrained(args=self._model_conf.llm.cfg, name=self._model_conf.llm.name)

        self.K = self._model_conf.aatype_pred_num_tokens
        self.k = self._interpolant_cfg.aatypes.simplex_value

    def seq_to_simplex(self, seqs):
        return clampped_one_hot(seqs, self.K).float() * self.k * 2 - self.k  # (B,L,K)

    def _aatypes_euler_step_purity(self, d_t, t, logits_1, aatypes_t):
        _aatypes_cfg = self._interpolant_cfg.aatypes
        batch_size, num_res, S = logits_1.shape

        logits_1_wo_mask = logits_1[:, :, 0:-1]  # (B, D, S-1)
        pt_x1_probs = F.softmax(logits_1_wo_mask / _aatypes_cfg.temp, dim=-1)  # (B, D, S-1)
        max_logprob = torch.max(torch.log(pt_x1_probs), dim=-1)[0]  # (B, D)
        # bias so that only currently masked positions get chosen to be unmasked
        max_logprob = max_logprob - (aatypes_t != du.MASK_TOKEN_INDEX).float() * 1e9
        sorted_max_logprobs_idcs = torch.argsort(max_logprob, dim=-1, descending=True)  # (B, D)

        unmask_probs = (d_t * ((1 + _aatypes_cfg.noise * t) / (1 - t)).to(logits_1.device)).clamp(max=1)  # scalar

        number_to_unmask = torch.binomial(count=torch.count_nonzero(aatypes_t == du.MASK_TOKEN_INDEX, dim=-1).float(), prob=unmask_probs)
        unmasked_samples = torch.multinomial(pt_x1_probs.view(-1, S - 1), num_samples=1).view(batch_size, num_res)

        D_grid = torch.arange(num_res, device=logits_1.device).view(1, -1).repeat(batch_size, 1)
        mask1 = (D_grid < number_to_unmask.view(-1, 1)).float()
        inital_val_max_logprob_idcs = sorted_max_logprobs_idcs[:, 0].view(-1, 1).repeat(1, num_res)
        masked_sorted_max_logprobs_idcs = (mask1 * sorted_max_logprobs_idcs + (1 - mask1) * inital_val_max_logprob_idcs).long()
        mask2 = torch.zeros((batch_size, num_res), device=logits_1.device)
        mask2.scatter_(dim=1, index=masked_sorted_max_logprobs_idcs, src=torch.ones((batch_size, num_res), device=logits_1.device))
        unmask_zero_row = (number_to_unmask == 0).view(-1, 1).repeat(1, num_res).float()
        mask2 = mask2 * (1 - unmask_zero_row)
        aatypes_t = aatypes_t * (1 - mask2) + unmasked_samples * mask2

        # re-mask
        u = torch.rand(batch_size, num_res, device=logits_1.device)
        re_mask_mask = (u < d_t * _aatypes_cfg.noise).float()
        aatypes_t = aatypes_t * (1 - re_mask_mask) + du.MASK_TOKEN_INDEX * re_mask_mask
        return aatypes_t.long()

    def corrupt_batch(self, batch):
        num_batch, num_res = batch['aa'].shape
        gen_mask, res_mask, angle_mask = batch['generate_mask'].long(), batch['res_mask'].long(), batch['torsion_angle_mask'].long()  # [..., 2:]
        rotmats_1 = construct_3d_basis(batch['pos_heavyatom'][:, :, BBHeavyAtom.CA], batch['pos_heavyatom'][:, :, BBHeavyAtom.C], batch['pos_heavyatom'][:, :, BBHeavyAtom.N])
        trans_1_c = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]  # already centered when constructing dataset
        seqs_1 = batch['aa']
        angles_1 = batch['torsion_angle']
        with torch.no_grad():
            t = torch.rand((num_batch, 1), device=batch['aa'].device)
            t = t * (1 - 2 * self._interpolant_cfg.t_normalization_clip) + self._interpolant_cfg.t_normalization_clip  # avoid 0
            # corrupt trans
            trans_0 = torch.randn((num_batch, num_res, 3), device=batch['aa'].device) * self._interpolant_cfg.trans.sigma  # scale with sigma?
            trans_0_c, _ = zero_center_part(trans_0, gen_mask, res_mask)
            trans_t = (1 - t[..., None]) * trans_0_c + t[..., None] * trans_1_c
            trans_t_c = torch.where(batch['generate_mask'][..., None], trans_t, trans_1_c)
            # corrupt rotmats
            rotmats_0 = uniform_so3(num_batch, num_res, device=batch['aa'].device)
            rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
            rotmats_t = torch.where(batch['generate_mask'][..., None, None], rotmats_t, rotmats_1)
            # corrup angles
            angles_0 = torus.tor_random_uniform(angles_1.shape, device=batch['aa'].device, dtype=angles_1.dtype)  # (B,L,5)
            angles_t = torus.tor_geodesic_t(t[..., None], angles_1, angles_0)
            angles_t = torch.where(batch['generate_mask'][..., None], angles_t, angles_1)
            # corrupt seqs
            if self._interpolant_cfg.aatypes.interpolant_type == 'masking':
                seqs_t = seqs_1.clone()
                corruption_mask = torch.rand(num_batch, num_res, device=batch['aa'].device) < (1 - t)  # (B, L)
                seqs_t[corruption_mask] = du.MASK_TOKEN_INDEX
                seqs_t = seqs_t * res_mask + du.MASK_TOKEN_INDEX * (1 - res_mask)
            elif self._interpolant_cfg.aatypes.interpolant_type == 'uniform':
                seqs_1_simplex = self.seq_to_simplex(seqs_1)
                seqs_0_simplex = self.k * torch.randn_like(seqs_1_simplex)  # (B,L,K)
                seqs_t_simplex = ((1 - t[..., None]) * seqs_0_simplex) + (t[..., None] * seqs_1_simplex)  # (B,L,K)
                seqs_t_simplex = torch.where(batch['generate_mask'][..., None], seqs_t_simplex, seqs_1_simplex)
                seqs_t_prob = F.softmax(seqs_t_simplex, dim=-1)  # (B,L,K)
                seqs_t = sample_from(seqs_t_prob)  # (B,L)
            else:
                raise ValueError(f"Unknown aatypes interpolant type {self._interpolant_cfg.aatypes.interpolant_type}")
            seqs_t = torch.where(batch['generate_mask'], seqs_t, seqs_1)
        return t, seqs_t, trans_t_c, rotmats_t, angles_t

    def structural_plm_forward(self, pred_aatypes, node_embed, diffuse_mask, add_special_tokens=False):
        # transform to the ESM codebook
        pred_aatypes_esm = pred_aatypes.clone()
        for key, value in esm_index_transform.items():
            pred_aatypes_esm[pred_aatypes == key] = value
        pred_aatypes_esm[~diffuse_mask.bool()] = 1  # 1 for <pad> in ESM. No predicted <mask>, don't worry! # ESM is bad at PPI, so input single seq
        if add_special_tokens:
            pred_aatypes_esm[:, 0] = 0  # <cls> token
            pred_aatypes_esm = torch.cat([pred_aatypes_esm, torch.ones(len(node_embed), 1, device=node_embed.device).long() * 2], dim=1)  # <eos> token
            node_embed = torch.cat([node_embed, torch.zeros(len(node_embed), 1, node_embed.shape[-1], device=node_embed.device)], dim=1)
            esm_logits = self.llm_decoder(tokens=pred_aatypes_esm, encoder_out=node_embed)[:, :-1]  # no t is added into llm
        else:
            esm_logits = self.llm_decoder(tokens=pred_aatypes_esm, encoder_out=node_embed)
        if self._model_conf.aatype_pred_num_tokens == du.NUM_TOKENS + 1:
            pred_seqs_1_prob = esm_logits[..., esm_index_transform_back + [3]]  # predict 'X'
        else:  # only predict 20 aa types
            pred_seqs_1_prob = esm_logits[..., esm_index_transform_back]
        return pred_seqs_1_prob

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(timesteps[:, 0], self._model_conf.node_embed_size, max_positions=2056)[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb

    def decode(self, batch, t, seqs_t, trans_t_c, rotmats_t, angles_t, gen_mask, ):
        num_batch, num_res = batch['aa'].shape
        node_mask = batch['res_mask'].long()
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        if self.net_type == 'frameflow':  # FrameFlow arch
            node_embed = self.node_feature_net(t=t, res_mask=node_mask, diffuse_mask=gen_mask, chain_index=batch['chain_nb'], pos=batch['res_nb'], aatypes=seqs_t,
                                               angles=angles_t)
            edge_embed = self.edge_feature_net(node_embed, trans_t_c, edge_mask, gen_mask, batch['chain_nb'])
        else:
            context_mask = torch.logical_and(batch['mask_heavyatom'][:, :, BBHeavyAtom.CA], ~batch['generate_mask'])
            node_embed = self.node_feature_net(batch['aa'], batch['res_nb'], batch['chain_nb'], batch['pos_heavyatom'], batch['mask_heavyatom'], structure_mask=context_mask,
                                               sequence_mask=context_mask)
            edge_embed = self.edge_feature_net(batch['aa'], batch['res_nb'], batch['chain_nb'], batch['pos_heavyatom'], batch['mask_heavyatom'], structure_mask=context_mask,
                                               sequence_mask=context_mask)
            node_embed = self.res_feat_mixer(
                torch.cat([node_embed, self.current_seq_embedder(seqs_t), self.embed_t(t, node_mask), self.angles_embedder(angles_t).reshape(num_batch, num_res, -1)], dim=-1))

        curr_rigids = du.create_rigid(rotmats_t, trans_t_c)
        if self.rigids_scaling:
            curr_rigids = self.rigids_ang_to_nm(curr_rigids)
        node_embed = node_embed * node_mask[..., None]
        edge_embed = edge_embed * edge_mask[..., None]
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
                curr_rigids = curr_rigids.compose_q_update_vec(rigid_update, (node_mask * gen_mask)[..., None])
            else:
                curr_rigids = curr_rigids.compose_q_update_vec(rigid_update, node_mask[..., None])
            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        if self.rigids_scaling:
            curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans_1 = curr_rigids.get_trans()
        pred_rotmats_1 = curr_rigids.get_rots().get_rot_mats()
        pred_seqs_1_ipa = self.aatype_pred_net(node_embed)
        pred_angles_1 = self.angle_net(node_embed) % (2 * torch.pi)  # inductive bias to bound between (0,2pi)
        return pred_seqs_1_ipa, pred_trans_1, pred_rotmats_1, pred_angles_1, node_embed

    def forward(self, batch, ):
        if batch['torsion_angle'].shape[-1] == 7:
            batch['torsion_angle'] = batch['torsion_angle'][..., 2:]
        t, seqs_t, trans_t_c, rotmats_t, angles_t = self.corrupt_batch(batch)
        num_batch, num_res = batch['aa'].shape
        gen_mask = batch['generate_mask'].long()
        pos_heavyatom = batch['pos_heavyatom']
        seqs_1 = batch['aa']
        angles_1 = batch['torsion_angle']
        trans_1_c = pos_heavyatom[:, :, BBHeavyAtom.CA]  # already centered when constructing dataset
        rotmats_1 = construct_3d_basis(pos_heavyatom[:, :, BBHeavyAtom.CA], pos_heavyatom[:, :, BBHeavyAtom.C], pos_heavyatom[:, :, BBHeavyAtom.N])

        # denoise
        pred_seqs_1_ipa, pred_trans_1_c, pred_rotmats_1, pred_angles_1, node_embed = self.decode(batch, t, seqs_t, trans_t_c, rotmats_t, angles_t, gen_mask)
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
        seqs_loss = torch.tensor(0.0, device=pred_seqs_1_ipa.device)
        if self.use_llm:
            if self._interpolant_cfg.aatypes.interpolant_type == 'masking':
                pred_aatypes = torch.argmax(pred_seqs_1_ipa, dim=-1)
            elif self._interpolant_cfg.aatypes.interpolant_type == 'uniform':
                pred_aatypes = sample_from(F.softmax(pred_seqs_1_ipa, dim=-1))
            pred_seqs_1_prob = self.structural_plm_forward(pred_aatypes, node_embed, gen_mask)
            seqs_loss = F.cross_entropy(pred_seqs_1_prob.view(-1, pred_seqs_1_prob.shape[-1]), torch.clamp(seqs_1, 0, 19).view(-1), reduction='none').view(
                pred_seqs_1_prob.shape[:-1])  # (N,L), not softmax
            seqs_loss = torch.sum(seqs_loss * gen_mask, dim=-1) / (torch.sum(gen_mask, dim=-1) + 1e-8)
            seqs_loss = torch.mean(seqs_loss)

        seqs_loss_ipa = F.cross_entropy(pred_seqs_1_ipa.view(-1, pred_seqs_1_ipa.shape[-1]), torch.clamp(seqs_1, 0, 19).view(-1), reduction='none').view(
            pred_seqs_1_ipa.shape[:-1])  # (N,L), not softmax
        seqs_loss_ipa = torch.sum(seqs_loss_ipa * gen_mask, dim=-1) / (torch.sum(gen_mask, dim=-1) + 1e-8)
        seqs_loss_ipa = torch.mean(seqs_loss_ipa)

        # angle vf loss, teacher forcing
        angle_mask_loss = torsions_mask_vocab.to(batch['aa'].device)
        angle_mask_loss = angle_mask_loss[seqs_1.reshape(-1)].reshape(num_batch, num_res, -1)  # (B,L,5)
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
    def sample(self, batch, num_steps=100, angle_purify=False, llm=False):
        if batch['torsion_angle'].shape[-1] == 7:
            batch['torsion_angle'] = batch['torsion_angle'][..., 2:]
        num_batch, num_res = batch['aa'].shape
        gen_mask, res_mask = batch['generate_mask'].long(), batch['res_mask']
        angle_mask_loss = torsions_mask_vocab.to(batch['aa'].device)
        rotmats_1 = construct_3d_basis(batch['pos_heavyatom'][:, :, BBHeavyAtom.CA], batch['pos_heavyatom'][:, :, BBHeavyAtom.C], batch['pos_heavyatom'][:, :, BBHeavyAtom.N])
        trans_1 = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]
        seqs_1 = batch['aa']
        angles_1 = batch['torsion_angle']
        trans_1_c = trans_1  # already centered when constructing dataset

        # bb
        rotmats_0 = uniform_so3(num_batch, num_res, device=seqs_1.device)
        rotmats_0 = torch.where(batch['generate_mask'][..., None, None], rotmats_0, rotmats_1)
        trans_0 = torch.randn((num_batch, num_res, 3), device=seqs_1.device)  # scale with sigma?
        # move center and receptor
        trans_0_c, center = zero_center_part(trans_0, gen_mask, res_mask)
        trans_0_c = torch.where(batch['generate_mask'][..., None], trans_0_c, trans_1_c)
        # angle
        angles_0 = torus.tor_random_uniform(angles_1.shape, device=angles_1.device, dtype=angles_1.dtype)  # (B,L,7)
        angles_0 = torch.where(batch['generate_mask'][..., None], angles_0, angles_1)
        # seq
        if self._interpolant_cfg.aatypes.interpolant_type == 'masking':
            seqs_0 = torch.ones(num_batch, num_res, device=seqs_1.device).long() * du.MASK_TOKEN_INDEX
        elif self._interpolant_cfg.aatypes.interpolant_type == 'uniform':
            seqs_1_simplex = self.seq_to_simplex(seqs_1)
            seqs_0_simplex = self.k * torch.randn((num_batch, num_res, self.K), device=batch['aa'].device)
            seqs_0_prob = F.softmax(seqs_0_simplex, dim=-1)
            seqs_0 = sample_from(seqs_0_prob)
            seqs_0 = torch.where(batch['generate_mask'], seqs_0, seqs_1)
            seqs_0_simplex = torch.where(batch['generate_mask'][..., None], seqs_0_simplex, seqs_1_simplex)
            seqs_t_1_simplex = seqs_0_simplex

        # Set-up time
        ts = torch.linspace(1.e-2, 1.0, num_steps)
        t_1 = ts[0]
        clean_traj = []
        rotmats_t_1, trans_t_1_c, angles_t_1, aatypes_t_1 = rotmats_0, trans_0_c, angles_0, seqs_0

        # denoise loop
        for t_2 in ts[1:]:
            with torch.no_grad():  # denoise
                t = torch.ones((num_batch, 1), device=seqs_1.device) * t_1
                pred_aatypes_1_prob, pred_trans_1, pred_rotmats_1, pred_angles_1, node_embed = self.decode(batch, t, aatypes_t_1, trans_t_1_c, rotmats_t_1, angles_t_1, gen_mask)

            # rots, trans -- move center (PepFlow not move), angles, seqs
            pred_rotmats_1 = torch.where(batch['generate_mask'][..., None, None], pred_rotmats_1, rotmats_1)
            pred_trans_1_c = torch.where(batch['generate_mask'][..., None], pred_trans_1, trans_1_c)
            pred_angles_1 = torch.where(batch['generate_mask'][..., None], pred_angles_1, angles_1)
            if self._interpolant_cfg.aatypes.interpolant_type == 'masking':
                pred_aatypes = torch.argmax(pred_aatypes_1_prob, dim=-1)
            elif self._interpolant_cfg.aatypes.interpolant_type == 'uniform':
                pred_aatypes = sample_from(F.softmax(pred_aatypes_1_prob, dim=-1))
            pred_seqs_1 = torch.where(batch['generate_mask'], pred_aatypes, seqs_1)
            if llm:
                pred_seqs_1_prob = self.structural_plm_forward(pred_seqs_1, node_embed, gen_mask)
                pred_seqs_1 = torch.argmax(pred_seqs_1_prob, dim=-1)
                pred_seqs_1 = torch.where(batch['generate_mask'], pred_seqs_1, seqs_1)

            # seq-angle
            if angle_purify:
                torsion_mask = angle_mask_loss[pred_seqs_1.reshape(-1)].reshape(num_batch, num_res, -1)  # (B,L,5)
                pred_angles_1 = torch.where(torsion_mask.bool(), pred_angles_1, torch.zeros_like(pred_angles_1))

            clean_traj.append(
                {'rotmats': pred_rotmats_1.cpu(), 'trans': pred_trans_1_c.cpu(), 'angles': pred_angles_1.cpu(), 'seqs': pred_seqs_1.cpu(), 'rotmats_1': rotmats_1.cpu(),
                 'trans_1': trans_1_c.cpu(), 'angles_1': angles_1.cpu(), 'seqs_1': seqs_1.cpu()})
            # reverse step, also only for gen mask region
            d_t = t_2 - t_1
            # Euler step
            if self._interpolant_cfg.aatypes.interpolant_type == 'masking':
                seqs_t_2 = self._aatypes_euler_step_purity(d_t, t_1, pred_aatypes_1_prob, aatypes_t_1)
            elif self._interpolant_cfg.aatypes.interpolant_type == 'uniform':
                pred_seqs_1_simplex = self.seq_to_simplex(pred_seqs_1)
                seqs_t_2_simplex = seqs_t_1_simplex + (pred_seqs_1_simplex - seqs_0_simplex) * d_t
                seqs_t_2 = sample_from(F.softmax(seqs_t_2_simplex, dim=-1))
                seqs_t_1_simplex = seqs_t_2_simplex

            seqs_t_2 = torch.where(batch['generate_mask'], seqs_t_2, seqs_1)
            trans_t_2 = trans_t_1_c + (pred_trans_1_c - trans_0_c) * d_t
            trans_t_2_c = torch.where(batch['generate_mask'][..., None], trans_t_2, trans_1_c)  # move receptor also
            if self._interpolant_cfg.rots.sample_schedule == 'linear':
                scaling = 1 / (1 - t_1)
            elif self._interpolant_cfg.rots.sample_schedule == 'exp':
                scaling = self._interpolant_cfg.rots.exp_rate
            else:
                raise ValueError(f'Unknown sample schedule {self._interpolant_cfg.rots.sample_schedule}')
            rotmats_t_2 = so3_utils.geodesic_t(d_t * scaling, pred_rotmats_1, rotmats_t_1)
            rotmats_t_2 = torch.where(batch['generate_mask'][..., None, None], rotmats_t_2, rotmats_1)
            angles_t_2 = torus.tor_geodesic_t(d_t, pred_angles_1, angles_t_1)
            angles_t_2 = torch.where(batch['generate_mask'][..., None], angles_t_2, angles_1)
            if angle_purify:  # seq-angle
                torsion_mask = angle_mask_loss[seqs_t_2.long().reshape(-1)].reshape(num_batch, num_res, -1)  # (B,L,5)
                angles_t_2 = torch.where(torsion_mask.bool(), angles_t_2, torch.zeros_like(angles_t_2))

            rotmats_t_1, trans_t_1_c, angles_t_1, aatypes_t_1 = rotmats_t_2, trans_t_2_c, angles_t_2, seqs_t_2
            t_1 = t_2

        # final step
        t = torch.ones((num_batch, 1), device=seqs_1.device) * ts[-1]
        with torch.no_grad():
            pred_aatypes_1_prob, pred_trans_1, pred_rotmats_1, pred_angles_1, node_embed = self.decode(batch, t, aatypes_t_1, trans_t_1_c, rotmats_t_1, angles_t_1, gen_mask)
        pred_rotmats_1 = torch.where(batch['generate_mask'][..., None, None], pred_rotmats_1, rotmats_1)
        pred_trans_1_c = torch.where(batch['generate_mask'][..., None], pred_trans_1, trans_1_c)
        pred_angles_1 = torch.where(batch['generate_mask'][..., None], pred_angles_1, angles_1)
        if self._interpolant_cfg.aatypes.interpolant_type == 'masking':
            pred_aatypes = torch.argmax(pred_aatypes_1_prob, dim=-1)
        elif self._interpolant_cfg.aatypes.interpolant_type == 'uniform':
            pred_aatypes = sample_from(F.softmax(pred_aatypes_1_prob, dim=-1))
        pred_seqs_1 = torch.where(batch['generate_mask'], pred_aatypes, seqs_1)
        if llm:
            pred_seqs_1_prob = self.structural_plm_forward(pred_seqs_1, node_embed, gen_mask)
            pred_seqs_1 = torch.argmax(pred_seqs_1_prob, dim=-1)
            pred_seqs_1 = torch.where(batch['generate_mask'], pred_seqs_1, seqs_1)

        # seq-angle
        if angle_purify:
            torsion_mask = angle_mask_loss[pred_seqs_1.reshape(-1)].reshape(num_batch, num_res, -1)  # (B,L,5)
            pred_angles_1 = torch.where(torsion_mask.bool(), pred_angles_1, torch.zeros_like(pred_angles_1))
        clean_traj.append({'rotmats': pred_rotmats_1.cpu(), 'trans': pred_trans_1_c.cpu(), 'angles': pred_angles_1.cpu(), 'seqs': pred_seqs_1.cpu(), 'rotmats_1': rotmats_1.cpu(),
                           'trans_1': trans_1_c.cpu(), 'angles_1': angles_1.cpu(), 'seqs_1': seqs_1.cpu()})
        return clean_traj
