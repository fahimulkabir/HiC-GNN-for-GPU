import torch
import torch.nn as nn
import torch.nn.functional as F

class UniversalSAGEConv(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=False, bias=True):
        super().__init__()
        self.normalize = normalize
        
        # Standard Linear layers (Universally compatible)
        self.lin_l = nn.Linear(in_channels, out_channels, bias=bias)
        self.lin_r = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x, adj_matrix):
        """
        x: [N, Features]
        adj_matrix: [N, N] (Normalized Adjacency)
        """
        # 1. Message Passing = Matrix Multiplication
        # This is the "Magic" that replaces torch-geometric kernels
        # A * X aggregates neighbors
        out = torch.matmul(adj_matrix, x)
        
        # 2. Update and Skip Connection
        out = self.lin_l(out) + self.lin_r(x)
        
        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)
            
        return out