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

model_our = WithLSHSort(512, submodule=Conv1dForLSHSort(512, 512, 3, 1, 1, padding_mode='circular', groups=8))
model_att = nn.MultiheadAttention(512, num_heads=8, batch_first=True)

from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
out_model = model_our.to(device)
model_att = model_att.to(device)
print("LSH Conv")
for _ in tqdm(range(1000)):
    seq = torch.randn(1, 5000, 512).to(device)
    model_our(seq)
print("Multihead Attn.")
for _ in tqdm(range(1000)):
    seq = torch.randn(1, 5000, 512).to(device)
    model_att(seq, seq, seq)

