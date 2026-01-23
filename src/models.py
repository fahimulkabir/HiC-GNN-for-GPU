import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import UniversalSAGEConv

class UniversalHiCGNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 512 input -> 128 hidden -> 128 hidden -> 3 (XYZ)
        self.conv1 = UniversalSAGEConv(512, 128)
        self.conv2 = UniversalSAGEConv(128, 128)
        self.dec = nn.Linear(128, 3) 
    
    def forward(self, x, adj):
        x = F.relu(self.conv1(x, adj))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, adj)
        x = self.dec(x)
        
        # Output Distance Matrix
        return torch.cdist(x, x, p=2)

    def get_structure(self, x, adj):
        x = F.relu(self.conv1(x, adj))
        x = self.conv2(x, adj)
        return self.dec(x)