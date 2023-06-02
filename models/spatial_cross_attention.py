import torch
import torch.nn as nn
import mmcv
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch


class sca(nn.Module): # spatial cross attention
    def __init__(self, config):
        super(sca,self).__init__()
        self.config = config
        # self.bs = bs # hard code
        self.num_heads = self.config["num_heads"] 
        self.num_pts = self.config["num_sam_pts"]
        self.num_levels = self.config["num_levels"] 
        self.embed_dim = self.config["embed_dim"]
        self.sampling_offset = nn.Linear(self.embed_dim, self.num_heads*self.num_pts*self.num_levels*2)
        self.attention_weights = nn.Linear(self.embed_dim, self.num_heads*self.num_pts*self.num_levels)
        self.norm = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query,  single_feat, ref_2d, h, w): 
        """
        query:[bs,1,embed_dim]
        single_feat:[bs, h*w, num_heads, embed_dim/num_heads] 
        """
        bs, num_query, _, _ = single_feat.shape
        query = query.repeat(1, h*w, 1) 
        sampling_offsets = self.sampling_offset(query) 
        sampling_offsets = sampling_offsets.relu()
        sampling_offsets = sampling_offsets.view(bs, h*w, self.num_heads,  self.num_levels, self.num_pts, 2) 
        spatial_shapes = torch.tensor([[h, w]], device=query.device)
        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = ref_2d[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.relu()
        attention_weights = attention_weights.view(bs, h*w, self.num_heads,  self.num_levels, self.num_pts) 
        attention_weights = attention_weights.softmax(-1)
        value = single_feat.view(bs, num_query, self.num_heads, -1) 
        output = multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)
        output  = self.norm(self.dropout(output) + query)
        return output