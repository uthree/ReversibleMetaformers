#example usage
from models.sequential import WithLSHSort
from tqdm import tqdm
import torch
import torch.nn as nn
 
class Conv1dForLSHSort(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Conv1dForLSHSort, self).__init__()
        self.conv = nn.Conv1d(*args, **kwargs)
    def forward(self, x):
        x = x.swapaxes(1,2)
        x = self.conv(x)
        x = x.swapaxes(1,2)
        return x

seq = torch.randn(10, 1000, 512)
model_our = WithLSHSort(512, submodule=Conv1dForLSHSort(512, 512, 3, 1, 1, padding_mode='circular'))
model_att = nn.MultiheadAttention(512, num_heads=8)

from tqdm import tqdm
print("LSH Conv")
for _ in tqdm(range(1000)):
    model_our(seq)
print("Multihead Attn.")
for _ in tqdm(range(1000)):
    model_att(seq, seq, seq)

