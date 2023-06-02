import torch
import torch.nn as nn
from .spatial_cross_attention import sca
from .temporal_self_attention import tsa

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.embed_dim = config["embed_dim"]
        self.config = config
        self.num_query = self.config["num_query"]
        self.num_heads = self.config["num_heads"]
        self.tsa = tsa(self.config)
        self.sca = sca(self.config)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        self.norm = nn.LayerNorm(self.embed_dim)
        self.prev_embed = None
        

    def forward(self, query, all_img_feats):
        bs, _, _, h, w = all_img_feats.shape
        ref_2d = self.get_reference_points(h, w, bs)
        query = query.unsqueeze(0).repeat(bs, 1, 1)
        all_feats = all_img_feats.flatten(3).permute(0,1,3,2)  # [8,10,120,256]
        _,num_imgs,resolu,_ = all_feats.shape
        output = None

        for i in range(num_imgs):
            single_feat = all_feats[:, i, :, :].view(bs, resolu, self.num_heads, -1)
            output = self.tsa(query, self.prev_embed)
            output = self.sca(output, single_feat, ref_2d, h, w) 
            output = output.mean(1).unsqueeze(1) 
            output = self.mlp(output) + output
            output = self.norm(output)
            # output = output.relu()
            self.prev_embed = output
            # self.prev_embed = None # 消融实验
            
        self.prev_embed = None

        return output 
    
    def get_reference_points(self, H=4, W=11, bs=8, device='cuda'):
        ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H - 0.5, H, dtype=torch.float, device=device), torch.linspace(0.5, W - 0.5, W, dtype=torch.float, device=device))
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_2d = torch.stack((ref_x, ref_y), -1)
        ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
        return ref_2d