import torch
import torch.nn as nn

# Inputs [batch_size, seq_len, d_model]
# Outputs [batch_size, seq_len, d_model]

# if you need use it as self-attention like, for example,
# hint: set submodule == nn.Conv1d(), swap axes (1, 2), before and after of passing this module.
class WithLSHSort(nn.Module):
    def __init__(self,
            d_model=512,
            n_heads=8,
            segnemt_size=4,
            submodule=nn.Identity()
            ):
        super(WithLSHSort, self).__init__()
        assert d_model % n_heads == 0, f"d_model must be able to devided by n_heads"
        self.hash = nn.Linear(d_model, 2*n_heads, bias=False) # TODO: Splity by heads
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def forward(self, x):
        # caluclate indexes

         projected = self.hash(x) #[batch_size d_model, n_heads*2] # Project to 2-dimentional space
        h_x, h_y = torch.split(projected, self.n_heads, dim=2) # [batch_size, seq_len, nheads] where h_x, h_y
        angles = torch.arctan(h_x / h_y) # [batch_size, seq_len, n_heads] # calculate angle of vector
        indexes = torch.argsort(angles, 1) # [batch_size, seq_len, n_heads]
        indexes = torch.unsqueeze(indexes, dim=3).expand(-1, -1, -1, self.d_head) # [batch_size, seq_len, n_heads, d_head]
        
        # split by heads
        x = torch.stack(torch.split(x, self.n_heads, dim=2), dim=3) # [batch_size, seq_len, nheads, dim_head]
        print(x.shape)

        # sort heads
        torch.gather(x, 2, indexes)
        

# test

seq = torch.randn(4, 3, 512)
model = WithLSHSort()
model(seq)
        
