import torch
import torch.nn as nn
import torch.nn.functional as F
from .multi_arcface import MultiArcFace


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder,self).__init__()
        self.config = config
        self.mul_arcface = MultiArcFace(self.config["mlp_out_channel"], self.config["n"] * self.config["out_class_num"], self.config["n"], s=20, m=0.5, easy_margin=True) 


    def forward(self, agent_all_feature, lable=None):
        B, K, _,_ = agent_all_feature.shape 
        prob = self.mul_arcface(agent_all_feature,lable)
        prob = prob.view(B, self.config['n'], self.config['out_class_num'], 1) 
        res, idx = torch.max(prob, dim=1) 
        # res = torch.mean(prob, dim=1)
        res = F.softmax(res, dim=1) 
        return res
