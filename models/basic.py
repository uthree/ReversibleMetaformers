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



