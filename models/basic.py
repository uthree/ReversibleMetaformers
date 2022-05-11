import random
import numpy as np

import torch
import torch.nn as nn

def apply_stochastic_depth(seq : nn.Sequential, max_p=1.0, min_p=0.5):
    return nn.Sequential(*[Stochastic(mod, p) for p,mod in zip(np.linspace(max_p, min_p, len(seq)), seq)])

class Stochastic(nn.Module):
    def __init__(self, module, p=0.5):
        super(Stochastic, self).__init__()
        print(p)
        self.probability = p
        self.module = module

    def forward(self, *args, **kwargs):
        if random.random() <= self.probability or self.training:
            return self.module(*args, **kwargs)
        else:
            return args

def string2activation(s):
    if s == 'gelu':
        return nn.GELU()
    if s == 'relu':
        return nn.ReLU()

# Dummy layer for rv.ReversibleSequence
class IrreversibleBlock(nn.Module):
    def __init__(self, f_block, g_block, split_along_dim):
        self.f_block, self.g_block = f_block, g_block
        self.spit_along_dim = split_along_dim
    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=self.split_along_dim)
        y1, y2 = None, None
        y1 = x1 + self.f_block(x2)
        y2 = x2 + self.g_block(y1)

        return torch.cat([y1, y2], dim=self.split_along_dim)
