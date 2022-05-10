import torch
import torch.nn as nn
import torch.nn.functional as F

import revtorch as rv

class SpatialShift(nn.Module):
    def __init__(
            self,
            shift_directions=[[1,0], [-1,0], [0,1], [0,-1]], # List of shift directions. [X, Y], if you need to identity mapping, set this value [0, 0]
            padding_mode='replicate',
            ):
        super(SpatialShift, self).__init__()
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

