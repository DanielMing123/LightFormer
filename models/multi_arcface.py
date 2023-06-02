import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import math

class MultiArcFace(nn.Module):
    """
    Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample 1024
            out_features: size of each output sample 80
            s: norm of input feature
            m: margin
            cos(theta + m)
    """
    def __init__(self, in_features, out_features, n, s=20.0, m=0.5, easy_margin=False):
        super(MultiArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.cluster_centres = n
        self.out_class_num = out_features // n
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        if label is None:
            input = input.squeeze()
            input = input.view(1,1024)
            return F.linear(F.normalize(input), F.normalize(self.weight))
            
        input = input.squeeze(1)
        input = input.squeeze(2)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        B,_ = cosine.shape
        cosine = cosine.view(B, self.cluster_centres, self.out_class_num)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m 

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            
        one_hot = label
        one_hot = one_hot.unsqueeze(1)
        one_hot = one_hot.repeat(1,self.cluster_centres,1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  
        output *= self.s

        return output