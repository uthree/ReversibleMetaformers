import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def apply_stochastic_depth(seq: nn.Sequential, max_p=1.0, min_p=0.5):
    return nn.Sequential(*[Stochastic(mod, p) for p, mod in zip(np.linspace(max_p, min_p, len(seq)), seq)])


class Stochastic(nn.Module):
    def __init__(self, module, p=0.5):
        super(Stochastic, self).__init__()
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
    def __init__(self, f_block, g_block, split_along_dim=1):
        super(IrreversibleBlock, self).__init__()
        self.f_block, self.g_block = f_block, g_block
        self.split_along_dim = split_along_dim

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=self.split_along_dim)
        y1, y2 = None, None
        y1 = x1 + self.f_block(x2)
        y2 = x2 + self.g_block(y1)

        return torch.cat([y1, y2], dim=self.split_along_dim)

# two layer perceptron


class TwoLP(nn.Module):
    def __init__(self, d_model, hidden_dim, activation=nn.GELU()):
        super(TwoLP, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.activation = activation
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))

# some module with gate
# module must be take [*, *, d_model] shape and returns [*, *, d_model].
# hint : parameter of gate works as summary of this expert module.


class Expert(nn.Module):
    def __init__(self, d_model, module, name=None):
        super(Expert, self).__init__()
        self.gate = nn.Linear(d_model, 1, bias=False)
        self.module = module
        if not name:
            self.name = f'Unnnamed expert module {hex(random.randint(0, 2**32))}'
        else:
            self.name=name

    def forward(self, x):
        return self.module(x)

# expert must have to have attribute: gate, module, name.


class MixtureOfExperts(nn.Module):
    def __init__(self, d_model, experts=[], num_available_experts=4, logger=nn.Identity()):
        super(MixtureOfExperts, self).__init__()
        self.experts = experts
        self.d_model = d_model
        self.logger = logger  # logger function. required callable
        self.num_available_experts = num_available_experts

    # x: [*, *, d_model]
    def forward(self, x):
        # caluclate key of gate
        k = torch.sum(x, dim=[0, 1]) # [d_model]
        # weight of each gate
        # gate(k) : shape=[1]
        gw = torch.stack([e.gate(k) for e in self.experts]) # [number of experts]
        gw = gw.squeeze(1)
        gw, indexes = torch.topk(gw, min(self.num_available_experts, len(self.experts)))

        available_experts = [self.experts[i] for i in indexes]
        self.logger(f"selected experts: {', '.join([e.name for e in available_experts])}")

        gw = F.softmax(torch.stack([ expert.gate(x) for expert in available_experts], dim=2).squeeze(3), dim=2)
        self.logger(gw.shape)

        # call available experts
      
        print(x.shape)
        x = sum([expert(x) * weight.swapaxes(0,1).unsqueeze(-1) for expert, weight in zip(available_experts, gw.swapaxes(0,2))])
        return x
        
    def append(self, expert):
        self.experts.append(expert)


