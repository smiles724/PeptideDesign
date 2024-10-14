import esm
import torch
import torch.nn.functional as F
from torch import nn

from multiflow.data import utils as du
from multiflow.data.residue_constants import restype_order
from multiflow.models import ipa_pytorch
from multiflow.models.edge_feature_net import EdgeFeatureNet
from multiflow.models.llm.esm2_adapter import ESM2WithStructuralAdatper
from multiflow.models.node_feature_net import NodeFeatureNet
from multiflow.models.utils import AngularEncoding, sample_from

# transform residue index to the ESM-form
esm_index_transform = {20: 32}  # <mask> is 32 in ESM, different from <unk> (20); but MultiFlow use 20 for both <mask> and <unk>
esm_index_transform_back = []  # esm_index_transform_back += [3]  if predict 'X'
esm_vocab = esm.Alphabet.from_architecture('ESM-1b').tok_to_idx  # TODO: separate <mask> and <unk>
for i, j in restype_order.items():
    esm_index_transform[j] = esm_vocab[i]
    esm_index_transform_back.append(esm_vocab[i])


class FlowModel(nn.Module):

    def __init__(self, model_conf):
        super(FlowModel, self).__init__()
        self._model_conf = model_conf
        self._ipa_conf = model_conf.ipa
        node_embed_size = self._model_conf.node_embed_size
        self.rigids_ang_to_nm = lambda x: x.apply_trans_fn(lambda x: x * du.ANG_TO_NM_SCALE)
        self.rigids_nm_to_ang = lambda x: x.apply_trans_fn(lambda x: x * du.NM_TO_ANG_SCALE)
        self.node_feature_net = NodeFeatureNet(model_conf.node_features)
        self.edge_feature_net = EdgeFeatureNet(model_conf.edge_features)
        self.angles_embedder = AngularEncoding(num_funcs=6)  # 25*7=175, for competitive embedding size
        self.res_feat_mixer = nn.Sequential(nn.Linear(node_embed_size + self.angles_embedder.get_out_dim(in_dim=7), node_embed_size), nn.ReLU(),
                                            nn.Linear(node_embed_size, node_embed_size), )
        self.angle_net = nn.Sequential(nn.Linear(node_embed_size, node_embed_size), nn.ReLU(), nn.Linear(node_embed_size, node_embed_size), nn.ReLU(),
                                       nn.Linear(node_embed_size, 7))  # 7 angles

        self.sample_deterministic = model_conf.llm.sample_deterministic
        self.llm_decoder = ESM2WithStructuralAdatper.from_pretrained(args=model_conf.llm, name=model_conf.llm_name)

        self.aatype_pred_net = nn.Sequential(nn.Linear(node_embed_size, node_embed_size), nn.ReLU(), nn.Linear(node_embed_size, node_embed_size), nn.ReLU(),
                                             nn.Linear(node_embed_size, self._model_conf.aatype_pred_num_tokens), )

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

    def forward(self, input_feats, ):
        node_mask = input_feats['res_mask'].long()
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = input_feats['diffuse_mask']
        chain_index = input_feats['chain_idx']
        res_index = input_feats['res_idx']
        so3_t = input_feats['so3_t']
        trans_t = input_feats['trans_t']
        rotmats_t = input_feats['rotmats_t']
        aatypes_t = input_feats['aatypes_t'].long()
        angles_t = input_feats['angles_t'].to(trans_t.dtype)

        # Initialize node and edge embeddings
        node_embed = self.node_feature_net(t=so3_t, res_mask=node_mask, diffuse_mask=diffuse_mask, chain_index=chain_index, pos=res_index, aatypes=aatypes_t,
                                           angles=angles_t)
        edge_embed = self.edge_feature_net(node_embed, trans_t, edge_mask, diffuse_mask, chain_index)
        node_embed = self.res_feat_mixer(torch.cat([node_embed, self.angles_embedder(angles_t)], dim=-1))

        # Initial rigids
        curr_rigids = du.create_rigid(rotmats_t, trans_t)

        # Main trunk
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
            curr_rigids = curr_rigids.compose_q_update_vec(rigid_update, (node_mask * diffuse_mask)[..., None])
            if b < self._ipa_conf.num_blocks - 1:
                edge_embed = self.trunk[f'edge_transition_{b}'](node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        curr_rigids = self.rigids_nm_to_ang(curr_rigids)
        pred_trans = curr_rigids.get_trans()
        pred_rotmats = curr_rigids.get_rots().get_rot_mats()

        pred_logits_ipa = self.aatype_pred_net(node_embed)
        if self._model_conf.aatype_pred_num_tokens == du.NUM_TOKENS + 1:  # 20 + 1 mask/unknown token = 21
            pred_logits_wo_mask = pred_logits_ipa.clone()
            pred_logits_wo_mask[..., du.MASK_TOKEN_INDEX] = -1e9  # prob. of unknown/<mask> is ~0
            if self.sample_deterministic:
                pred_aatypes = pred_logits_wo_mask.argmax(-1)
            else:
                pred_aatypes = sample_from(F.softmax(pred_logits_wo_mask, dim=-1))
        else:
            if self.sample_deterministic:
                pred_aatypes = pred_logits_ipa.argmax(-1)
            else:
                pred_aatypes = sample_from(F.softmax(pred_logits_ipa, dim=-1))  # (B, L), sample predicted sequences

        # transform to the ESM codebook
        pred_aatypes_esm = pred_aatypes.clone()
        for key, value in esm_index_transform.items():
            pred_aatypes_esm[pred_aatypes == key] = value
        pred_aatypes_esm[~node_mask.bool()] = 1  # 1 for <pad> in ESM. No predicted <mask>, don't worry!
        esm_logits = self.llm_decoder(tokens=pred_aatypes_esm, encoder_out=node_embed)  # no t is added into llm
        esm_logits = esm_logits[..., esm_index_transform_back]
        if self._model_conf.aatype_pred_num_tokens == du.NUM_TOKENS + 1:   # masking training strategy
            esm_logits_wo_mask = esm_logits.clone()
            esm_logits_wo_mask[..., du.MASK_TOKEN_INDEX] = -1e9  # prob. of unknown/<mask> is ~0
            pred_aatypes = esm_logits_wo_mask.argmax(-1)  # esm logits for final seq prediction
        else:
            pred_aatypes = sample_from(F.softmax(esm_logits, dim=-1))  # (B, L), sample predicted sequences

        # angle prediction
        pred_angles = self.angle_net(node_embed) % (2 * torch.pi)  # inductive bias to bound between (0,2pi)
        return {'pred_trans': pred_trans, 'pred_rotmats': pred_rotmats, 'pred_logits': esm_logits, 'pred_aatypes': pred_aatypes, 'pred_angles': pred_angles,
                'pred_logits_ipa': pred_logits_ipa, }
