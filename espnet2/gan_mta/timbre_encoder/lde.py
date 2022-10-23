import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

__all__ = ['LDE']

###############################################################################
### LDE
###############################################################################

# TODO
# tidy the code

class LDE(nn.Module):
    def __init__(self, D, input_dim, with_bias=False, distance_type='norm', network_type='att', pooling='mean'):
        super(LDE, self).__init__()
        self.dic = nn.Parameter(torch.randn(D, input_dim)) # input_dim by D (dictionary components)
        nn.init.uniform_(self.dic.data, -1, 1)
        self.wei = nn.Parameter(torch.ones(D)) # non-negative assigning weight in Eq(4) in LDE paper
        if with_bias: # Eq(4) in LDE paper
            self.bias = nn.Parameter(torch.zeros(D))
        else:
            self.bias = 0
        assert distance_type == 'norm' or distance_type == 'sqr'
        if distance_type == 'norm':
            self.dis = lambda x: torch.norm(x, p=2, dim=-1)
        else:
            self.dis = lambda x: torch.sum(x**2, dim=-1)
        assert network_type == 'att' or network_type == 'lde'
        if network_type == 'att':
            self.norm = lambda x: F.softmax(-self.dis(x) * self.wei + self.bias, dim = -2)
        else:
            self.norm = lambda x: F.softmax(-self.dis(x) * (self.wei ** 2) + self.bias, dim = -1)
        assert pooling == 'mean' or pooling == 'mean+std'
        self.pool = pooling
        # regularization maybe

    def forward(self, x):
        # print(x.size()) # (B, T, F)
        # print(self.dic.size()) # (D, F)
        r = x.view(x.size(0), x.size(1), 1, x.size(2)) - self.dic # residaul vector
        # print(r.size()) # (B, T, D, F)
        w = self.norm(r).view(r.size(0), r.size(1), r.size(2), 1) # numerator without r in Eq(5) in LDE paper
        # print(self.norm(r).size()) # (B, T, D)
        # print(w.size()) # (B, T, D, 1)
        w = w / (torch.sum(w, dim=1, keepdim=True) + 1e-9) #batch_size, timesteps, component # denominator of Eq(5) in LDE paper
        if self.pool == 'mean':
            x = torch.sum(w * r, dim=1) # Eq(5) in LDE paper
        else:
            x1 = torch.sum(w * r, dim=1) # Eq(5) in LDE paper
            x2 = torch.sqrt(torch.sum(w * r ** 2, dim=1)+1e-8) # std vector
            x = torch.cat([x1, x2], dim=-1)
        return x.view(x.size(0), -1)
