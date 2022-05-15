import numpy as np

import torch
import torch.nn as nn

from basic import Stochastic, TwoLP, IrreversibleBlock, MixtureOfExperts, Expert
import revtorch as rv

# Inputs [batch_size, seq_len, d_model]
# Outputs [batch_size, seq_len, d_model]

# if you need use it as self-attention like, for example,
# hint: set submodule == nn.Conv1d(), swap axes (1, 2), before and after of passing this module.
class WithLSHSort(nn.Module):
    def __init__(self,
            d_model=512,
            n_heads=8,
            submodule=nn.Identity(),
            eps=1e-4
            ):
        super(WithLSHSort, self).__init__()
        assert d_model % n_heads == 0, f"d_model must be able to devided by n_heads"
        self.hash = nn.ModuleList([nn.Linear(d_model // n_heads, 2) for _ in range(d_model // n_heads)])
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.mod = submodule
        self.eps = 1e-4

    def forward(self, x):
        # caluclate indexes

        projected = torch.cat([self.hash[n](head) for n, head in zip(range(self.n_heads), torch.split(x, self.d_head, dim=2))], dim=2)
        h_x, h_y = torch.split(projected, self.n_heads, dim=2) # [batch_size, seq_len, nheads] where h_x, h_y
        angles = torch.arctan(h_x / (h_y + self.eps)) # [batch_size, seq_len, n_heads] # calculate angle of vector
        indexes = torch.argsort(angles, 1) # [batch_size, seq_len, n_heads]
        indexes = torch.unsqueeze(indexes, dim=3).expand(-1, -1, -1, self.d_head) # [batch_size, seq_len, n_heads, d_head]
        
        # split heads
        x = x.reshape(x.shape[0], x.shape[1], self.n_heads, self.d_head)

        # sort heads
        x = torch.gather(x, 1, indexes)

        # concatenate heads
        x = x.reshape(x.shape[0], x.shape[1], self.d_model)
                
    
        # call module
        x = self.mod(x)
        
        # split heads
        x = x.reshape(x.shape[0], x.shape[1], self.n_heads, self.d_head)

        
        # scatter
        x = torch.scatter(torch.zeros_like(x) ,1, indexes, x)

        # concatenate heads
        x = x.reshape(x.shape[0], x.shape[1], self.d_model)
        return x

# convolution with swap axes.
class Conv1dForLSHSort(nn.Module):
    def __init__(self, d_model, kernel_size, stride, padding, padding_mode='circular', **kwargs):
        super(Conv1dForLSHSort, self).__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, stride, padding, padding_mode=padding_mode, **kwargs)
    def forward(self, x):
        x = x.swapaxes(1,2)
        x = self.conv(x)
        x = x.swapaxes(1,2)
        return x


# convolution with swap axes.
class Conv1dForLSHSort(nn.Module):
    def __init__(self, d_model, kernel_size, stride, padding, padding_mode='circular', **kwargs):
        super(Conv1dForLSHSort, self).__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size, stride, padding, padding_mode=padding_mode, **kwargs)
    def forward(self, x):
        x = x.swapaxes(1,2)
        x = self.conv(x)
        x = x.swapaxes(1,2)
        return x


# convolution sequece with LSH sort
# For example usage, replace Transformer's MultiHeadAttention to it
class LSHConv(nn.Module):
    def __init__(self, d_model, n_heads, kernel_size=3, stride=1, padding=1, padding_mode='circular', groups=None, bias=True):

        super(LSHConv, self).__init__()

        if not groups:
            groups = n_heads
        submodule = Conv1dForLSHSort(d_model, kernel_size, stride, padding, padding_mode, groups=groups, bias=bias)
        self.lsh_module = WithLSHSort(d_model, n_heads, submodule)

    def forward(self,x):
        return self.lsh_module(x)

    def forward(self,x):
        return self.lsh_module(x)

# LEAD: Lesser-required-computability Efficient Approximated Data transformer
class LEAD(nn.Module):
    def __init__(
            self,
            d_model=256,
            n_heads=8,
            n_layers=12,
            d_expert_ffn=256,
            n_experts=4,
            layer_drop_probability=0.5,
            spatial_mixer_class=LSHConv,
            reversible=True,
            spatial_mixer_kwargs={
                    'kernel_size': 3,
                    'stride' : 1,
                    'padding' : 1,
                }
            ):
        block_class = rv.ReversibleBlock if reversible else IrreversibleBlock
        seq_init    = (lambda blocks: rv.ReversibleSequence(nn.ModuleList(blocks))) if reversible else (lambda blocks: nn.Sequential(*blocks))
        super(LEAD, self).__init__()
        
        seq = []
        self.moes = []
        self.d_model = d_model
        for i, d_prob in zip(range(n_layers), np.linspace(1.0, layer_drop_probability)):
            moe = MixtureOfExperts(
                    d_model,
                    [Expert(
                        d_model,
                        TwoLP(
                            d_model,
                            d_expert_ffn),
                        name=f"FFN of Layer:{i} No.:{j}"
                        ) for j in range(n_experts)],
                    num_available_experts=n_experts, logger=print)
            self.moes.append(moe)
            seq.append(
                block_class(
                    Stochastic(
                        nn.Sequential(# F block: spatial mixer
                            nn.LayerNorm(d_model),
                            spatial_mixer_class(d_model, n_heads, **spatial_mixer_kwargs),
                            ),
                        p=d_prob
                        ),
                    Stochastic(
                        nn.Sequential( # G block: FeedForward
                            nn.LayerNorm(d_model),
                            moe,
                            ), p=d_prob
                        ),
                    split_along_dim=2
                    )
                )
            self.seq = seq_init(seq)
        
        self.num_available_experts_ = n_experts

    def forward(self, x):
        x = torch.repeat_interleave(x, repeats=2, dim=2)
        x = self.seq(x)
        x1, x2 = torch.chunk(x, 2, dim=2)
        x = (x1 + x2) / 2
        return x

    @property
    def num_avairable_experts(self):
        return self.num_available_experts_

    @num_avairable_experts.setter
    def num_available_experts(self, num):
        self.num_available_experts_ = num
        for moe in self.moes:
            moe.num_available_experts = num
    
    def add_expert_mlp(self, name='Unnamed Expert', dim=None):
        d_model = self.d_model
        if not dim:
            dim = self.d_model
        for i, moe in enumerate(self.moes):
            moe.append(Expert(d_model, TwoLP(d_model, dim), name=f"{name} of Layer {i}"))

