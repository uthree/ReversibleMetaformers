import torch
import torch.nn as nn
import torch.nn.functional as F

from basic import *

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
    def __init__(self, dim, activation=nn.ReLU(), swap_axes=None):
        super(MLP, self).__init__()
        self.fc1, self.fc2 = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.act = activation
        self.swap_axes=swap_axes

    def forward(self, x):
        if self.swap_axes:
            x = x.swapaxes(*self.swap_axes)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        if self.swap_axes:
            x = x.swapaxes(*self.swap_axes)
        return x

# MLP Mixer
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
            stochastic_depth_probability=1.0,
            reversible=True,
            activation='gelu',
            ):
        super(MLPMixer, self).__init__()

        # create entry flow
        self.entry_flow = []
        if input_channels == None:
            input_channels = d_model
        if dim_input_vector:
            self.entry_flow.append(
                    nn.Sequential(
                        nn.Linear(dim_input_vector, image_size**2 *input_channels),
                        nn.Unflatten(d_model, (image_size//patch_size)**2)))
        else:
            self.entry_flow.append(
                    nn.Sequential(
                        Image2Seq(input_channels, image_size, patch_size),
                        nn.Linear(patch_size**2*input_channels, d_model)))
        self.entry_flow = nn.Sequential(*self.entry_flow)
        
        # hidden layers
        hidden_blocks = []
        block_class = rv.ReversibleBlock if reversible else IrreversibleBlock
        seq_init    = (lambda blocks: rv.ReversibleSequence(nn.ModuleList(blocks))) if reversible else (lambda blocks: nn.Sequential(*blocks))
        for prob in np.linspace(1.0, stochastic_depth_probability, num_layers):
            hidden_blocks.append(
                block_class(
                    Stochastic(
                        nn.Sequential( # f block spatial mixer mlp
                            nn.LayerNorm(d_model),
                            MLP((image_size//patch_size)**2, string2activation(activation), swap_axes=(1,2))),
                        p=prob),
                    Stochastic(
                        nn.Sequential( # g block channelwise MLP
                            nn.LayerNorm(d_model),
                            MLP(d_model, string2activation(activation))
                        ),
                        p=prob),

                    split_along_dim=2))
        self.mid_flow = seq_init(hidden_blocks)
        exit_flow = []

        # create exit flow
        if output_channels == None:
            output_channels = d_model
        exit_flow.append(
                nn.Linear(d_model, output_channels))
        if num_classes:
            exit_flow.append(Seq2Image(output_channels, image_size, patch_size))
        else:
            exit_flow.append(nn.AvgPool1d(image_size))
        self.exit_flow = nn.Sequential(*exit_flow)

    def forward(self, x):
        x = self.entry_flow(x)
        x = torch.repeat_interleave(x, repeats=2, dim=2)
        x = self.mid_flow(x)
        x1, x2 = torch.chunk(x, 2, dim=2)
        x = (x1 + x2) / 2
        x = self.exit_flow(x)
        return x

# test code
image = torch.rand(10, 3, 128, 128)
model = MLPMixer(input_channels=3, image_size=128, patch_size=4, d_model=64, num_classes=10, num_layers=10, reversible=True)
model(image)
