import torch.nn as nn

class tsa(nn.Module): # temporal self attention
    def __init__(self, config):
        super(tsa,self).__init__()
        self.config = config
        self.num_heads = self.config["num_heads"] # hard code
        self.embed_dim = self.config["embed_dim"]
        self.temporal_attn = nn.MultiheadAttention(self.embed_dim, self.num_heads, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, query, prev_embed=None): 
        """
        query:[bs,1,256]
        prev_embed:[bs,1,256]
        """
        bypass = query
        if prev_embed is None:
            prev_embed = query
        output, _ = self.temporal_attn(query, prev_embed, prev_embed)
        output = self.norm(output + bypass)
        return output