import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import argparse
from scipy.stats import spearmanr

from .normalization import preprocess_hic_file, kr_normalization
from .embeddings import generate_embeddings
from .utils import prepare_tensors, contact_to_distance
from .models import UniversalHiCGNN

def train_hic_gnn(filepath, device_name="cuda"):
    # 1. Setup Device
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    
    # 2. Load & Normalize Data (Python implementation)
    print(" -> Loading Data...")
    raw_matrix, _ = preprocess_hic_file(filepath)
    np.fill_diagonal(raw_matrix, 0)
    
    print(" -> Normalizing (KR)...")
    norm_matrix = kr_normalization(raw_matrix)
    
    # 3. Generate Embeddings
    embeddings = generate_embeddings(raw_matrix)
    
    # 4. Prepare Tensors
    features, adj_tensor = prepare_tensors(norm_matrix, embeddings, device)
    
    # 5. Training Loop (Sweep over conversions)
    conversions = np.arange(0.1, 2.0, 0.1)
    best_score = -1
    best_struct = None
    
    print(" -> Starting Training...")
    
    for alpha in conversions:
        model = UniversalHiCGNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Ground Truth
        truth = contact_to_distance(raw_matrix, alpha).to(device)
        
        # Early Stopping vars
        prev_loss = float('inf')
        
        model.train()
        for epoch in range(1000): # Max epochs
            optimizer.zero_grad()
            output_dist = model(features, adj_tensor)
            
            loss = criterion(output_dist, truth)
            loss.backward()
            optimizer.step()
            
            # Convergence check
            if abs(prev_loss - loss.item()) < 1e-8:
                break
            prev_loss = loss.item()
            
        # Evaluation
        model.eval()
        with torch.no_grad():
            struct = model.get_structure(features, adj_tensor)
            dist_out = torch.cdist(struct, struct).cpu().numpy()
            dist_truth = truth.cpu().numpy()
            
            # Spearman correlation on upper triangle
            idx = np.triu_indices(dist_truth.shape[0], k=1)
            score = spearmanr(dist_truth[idx], dist_out[idx])[0]
            
            print(f"   Alpha: {alpha:.1f} | Loss: {loss.item():.6f} | dSCC: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_struct = struct.cpu().numpy()

    print(f"\nFinal Result -> Best dSCC: {best_score:.4f}")
    return best_score, best_struct

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to input file")
    args = parser.parse_args()
    
    train_hic_gnn(args.file)