import torch
from torch import nn
from dflow.models.utils import get_index_embedding, get_time_embedding, AngularEncoding


class NodeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(NodeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        self.aatype_embedding = nn.Embedding(21, self.c_s)  # Always 21 because of 20 amino acids + 1 for unk/<mask>

        embed_size = self._cfg.c_pos_emb + self.c_s + self._cfg.c_timestep_emb + 1
        if self._cfg.embed_angle:
            self.angles_embedder = AngularEncoding(num_funcs=5)  # 21*5=105
            embed_size += self.angles_embedder.get_out_dim(in_dim=5)
        if self._cfg.embed_chain:
            embed_size += self._cfg.c_pos_emb
        self.linear = nn.Sequential(nn.Linear(embed_size, self.c_s), nn.ReLU(), nn.Linear(self.c_s, self.c_s), nn.ReLU(), nn.Linear(self.c_s, self.c_s), nn.LayerNorm(self.c_s), )

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(timesteps[:, 0], self.c_timestep_emb, max_positions=2056)[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(self, *, t, res_mask, diffuse_mask, chain_index, pos, aatypes, angles):
        # s: [b]
        # [b, n_res, c_pos_emb]
        num_batch, num_res = aatypes.shape
        pos_emb = get_index_embedding(pos, self.c_pos_emb, max_len=2056)  # for residue index
        pos_emb = pos_emb * res_mask.unsqueeze(-1)

        # [b, n_res, c_timestep_emb], so3_t == r3_t, diffusion_mask as indicator
        input_feats = [self.aatype_embedding(aatypes), pos_emb, diffuse_mask[..., None], self.embed_t(t, res_mask)]
        if self._cfg.embed_angle:
            input_feats.append(self.angles_embedder(angles).reshape(num_batch, num_res, -1))  # no mask to avoid light data leakage
        if self._cfg.embed_chain:
            input_feats.append(get_index_embedding(chain_index, self.c_pos_emb, max_len=100))

        input_feats = torch.cat(input_feats, dim=-1)
        return self.linear(input_feats)
