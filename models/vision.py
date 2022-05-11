import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic import *

import numpy as np
import revtorch as rv

# for S2-MLP type models.
# input: [batch_size, channels, height, width]
class SpatialShift2d(nn.Module):
    def __init__(
            self,
            shift_directions=[[1,0], [-1,0], [0,1], [0,-1]], # List of shift directions. [X, Y], if you need to identity mapping, set this value [0, 0]
            padding_mode='replicate',
            ):
        super(SpatialShift2d, self).__init__()
        # caluclate padding range. I like one-line code lol.
        (l, r), (t, b) = [[f(d) for f in [lambda a:abs(min(a+[0])), lambda b:abs(max(*b+[0]))]] for d in [list(c) for c in zip(*shift_directions)]]
        self.pad_size = [l, r, t, b]
        self.num_directions = len(shift_directions)
        self.shift_directions = shift_directions
        self.padding_mode = padding_mode

    def forward(
            self,
            x,
            ):
        x = F.pad(x, self.pad_size, mode=self.padding_mode) # pad
        # caluclate channels of each section
        c = x.shape[1]
        sections = [c//self.num_directions]*self.num_directions
        # concatenate remainder of channels to last section
        sections[-1] += c % self.num_directions
        # save height and width
        h, w = x.shape[2], x.shape[3]

        # split
        x = torch.split(x, sections, dim=1)

        l,r,t,b = self.pad_size
        print(self.pad_size)
        # clip each sections.
        x = torch.cat([s[:, :, t:h-b, l:w-r] for (s, d) in zip(x, self.shift_directions)], dim=1)
        return x

# for MLP-Mixer models

# input: [batch_size, channels, height, width]
# output: [batch_size, seq_len, patch_dim]
class Image2Seq(nn.Module):
    def __init__(self, channels, image_size, patch_size):
        super(Image2Seq, self).__init__()
        if type(patch_size) == int:
            patch_size = [patch_size, patch_size] # [height, width]
        self.patch_size = patch_size
        if type(image_size) == int:
            image_size = [image_size, image_size] # [height, width]
        self.image_size = image_size
        self.channels = channels
        self.num_patch = [image_size[0] // patch_size[0], image_size[1] // patch_size[1]]

    def forward(self, x):
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        x = x.swapaxes(1, 2)
        return x

# input: [batch_size, seq_len, patch_dim]
# output: [batch_size, channels, height, width]
class Seq2Image(nn.Module):
    def __init__(self, channels, image_size, patch_size):
        super(Seq2Image, self).__init__()
        if type(patch_size) == int:
            patch_size = [patch_size, patch_size] # [height, width]
        self.patch_size = patch_size
        if type(image_size) == int:
            image_size = [image_size, image_size] # [height, width]
        self.image_size = image_size
        self.channels = channels

    def forward(self, x):
        x = x.swapaxes(1, 2)
        x = F.fold(x, output_size=self.image_size, kernel_size=self.patch_size, stride=self.patch_size)
        return x

class MLP(nn.Module):
    def __init__(self, dim, activation=nn.ReLU()):
        super(MLP, self).__init__()
        self.fc1, fc2 = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.act = activation

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class PatchWiseMLP(nn.Module):
    def __init__(self, dim,  activation=nn.ReLU()):
        super(PatchWiseMLP, self).__init__()
        self.fc1, fc2 = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.act = activation

    def forward(self, x):
        x = x.swapaxes(1, 2)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = x.swapaxes(1, 2)
        return x


# MLP Mixer
# If patch_size and num_layers given list, this model swichs pyramid structure mode automatically.
class MLPMixer(nn.Module):
    def __init__(
            self,
            input_channels=None,
            output_channels=None,
            num_classes=None,
            dim_input_vector=None,
            image_size=256,
            patch_size=16,
            d_model=64,
            num_layers=1,
            scale_factor=2,
            scale='down',
            stochastic_depth_probability=1.0,
            reversible=True,
            activation='gelu'
            ):
        super(MLPMixer, self).__init__()
        
        num_layers = [num_layers] if type(num_layers) == int else num_layers
        # create entry flow
        self.entry_flow = []
        if input_channels == None:
            input_channels = d_model
        if dim_input_vector:
            self.entry_flow.append(
                    nn.Sequential(
                        nn.Linear(dim_input_vector, image_size**2 *input_channels),
                        nn.Unflatten(d_model,patch_size**2)))
        else:
            self.entry_flow.append(Image2Seq(input_channels, image_size, patch_size))
        self.entry_flow = nn.Sequential(*self.entry_flow)
        
        # hidden layers
        hidden_blocks = []
        block_class = rv.ReversibleBlock if reversible else IrreversibleBlock
        seq_init    = lambda blocks: rv.ReversibleSequence(nn.ModuleList(blocks)) if reversible else nn.Sequential
        for nlayers in num_layers:
            for _ in range(nlayers):
                hidden_blocks.append(
                        block_class(
                            nn.Sequential( # f block
                            
                            )
                        ))
            if scale == 'down':
                patch_size = patch_size * scale_factor
            elif scale == 'up':
                patch_size = patch_size // scale_factor
        hidden_blocks = apply_stochastic_depth(hidden_blocks, 1.0, stochastic_depth_probability)
        self.mid_flow = seq_init(*hidden_blocks)
