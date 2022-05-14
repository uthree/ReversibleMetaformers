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
        self.hash = nn.Linear(d_model, 2*n_heads, bias=False) # TODO: Split by heads
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.mod = submodule

    def forward(self, x):
        # caluclate indexes

        projected = self.hash(x) #[batch_size d_model, n_heads*2] # Project to 2-dimentional space
        h_x, h_y = torch.split(projected, self.n_heads, dim=2) # [batch_size, seq_len, nheads] where h_x, h_y
        angles = torch.arctan(h_x / h_y) # [batch_size, seq_len, n_heads] # calculate angle of vector
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
        #print(indexes.shape, x.shape)
        x = torch.scatter(torch.zeros_like(x) ,1, indexes, x)

        # concatenate heads
        x = x.reshape(x.shape[0], x.shape[1], self.d_model)
        return x



