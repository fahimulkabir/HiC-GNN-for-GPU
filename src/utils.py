import torch

def prepare_tensors(norm_matrix, embeddings, device):
    """Converts numpy arrays to GPU tensors."""
    # Normalized Adjacency for GraphSAGE (D^-1 * A)
    adj = torch.tensor(norm_matrix, dtype=torch.float32)
    adj = adj + torch.eye(adj.shape[0]) # Self loops
    
    degree = adj.sum(dim=1)
    degree_inv = degree.pow(-1)
    degree_inv[torch.isinf(degree_inv)] = 0
    d_inv = torch.diag(degree_inv)
    
    norm_adj_tensor = torch.matmul(d_inv, adj).to(device)
    feat_tensor = torch.tensor(embeddings, dtype=torch.float32).to(device)
    
    return feat_tensor, norm_adj_tensor

def contact_to_distance(contacts, alpha):
    """Converts contact freq to distance: d = (1/c)^alpha"""
    c = torch.tensor(contacts, dtype=torch.float32) + 1e-8
    d = (1.0 / c) ** alpha
    d.fill_diagonal_(0)
    # Normalize max to 1.0
    d = d / torch.max(torch.nan_to_num(d, posinf=0))
    return d