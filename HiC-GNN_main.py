import numpy as np
from ge import LINE
import sys
import utils
import networkx as nx 
import os
from models import Net
import torch
from torch.nn import MSELoss
from torch.optim import Adam
from scipy.stats import spearmanr
import ast 
import argparse

if __name__ == "__main__":

    # =========================
    # DEVICE SETUP (NEW)
    # =========================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not(os.path.exists('Outputs')):
        os.makedirs('Outputs')

    if not(os.path.exists('Data')):
        os.makedirs('Data')

    parser = argparse.ArgumentParser(description='Generate embeddings and train a HiC-GNN model.')
    parser.add_argument('filepath', type=str)
    parser.add_argument('-c', '--conversions', type=str, default='[.1,.1, 2]')
    parser.add_argument('-bs', '--batchsize', type=int, default=128)
    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-lr', '--learningrate', type=float, default=.001)
    parser.add_argument('-th', '--threshold', type=float, default=1e-8)
    args = parser.parse_args()

    filepath = args.filepath
    batch_size = args.batchsize
    epochs = args.epochs
    lr = args.learningrate
    thresh = args.threshold

    conversions = ast.literal_eval(args.conversions)
    if len(conversions) == 3:
        conversions = list(np.arange(conversions[0], conversions[2], conversions[1]))
    elif len(conversions) == 1:
        conversions = [conversions[0]]
    else:
        raise Exception('Invalid conversion input.')

    name = os.path.splitext(os.path.basename(filepath))[0]

    adj = np.loadtxt(filepath)
    if adj.shape[1] == 3:
        print('Converting coordinate list format to matrix.')
        adj = utils.convert_to_matrix(adj)

    np.fill_diagonal(adj, 0)
    np.savetxt(f'Data/{name}_matrix.txt', adj, delimiter='\t')
    os.system(f'Rscript normalize.R {name}_matrix')
    normed = np.loadtxt(f'Data/{name}_matrix_KR_normed.txt')

    # =========================
    # Embeddings (still CPU only, by design)
    # =========================
    G = nx.from_numpy_matrix(adj)
    embed = LINE(G, embedding_size=512, order='second')
    embed.train(batch_size=batch_size, epochs=epochs, verbose=1)

    embeddings = np.asarray(list(embed.get_embeddings().values()))
    np.savetxt(f'Data/{name}_embeddings.txt', embeddings)

    print(f'Created embeddings: Data/{name}_embeddings.txt')

    # =========================
    # GPU-aware data creation
    # =========================
    data = utils.load_input(normed, embeddings, device)

    tempmodels, tempspear, tempmse, model_list = [], [], [], []

    # =========================
    # Training on GPU
    # =========================
    for conversion in conversions:
        print(f'Training model using conversion value {conversion}.')

        model = Net().to(device)
        criterion = MSELoss()
        optimizer = Adam(model.parameters(), lr=lr)

        truth = utils.cont2dist(data.y, conversion).to(device)

        oldloss = 1
        lossdiff = 1

        while lossdiff > thresh:
            model.train()
            optimizer.zero_grad()

            out = model(data.x.float(), data.edge_index)
            loss = criterion(out.float(), truth.float())

            loss.backward()
            optimizer.step()

            lossdiff = abs(oldloss - loss)
            oldloss = loss

            print(f'Loss: {loss.item():.6f}', end='\r')

        # =========================
        # Evaluation (move to CPU for scipy)
        # =========================
        coords = model.get_model(data.x.float(), data.edge_index)
        out = torch.cdist(coords, coords)

        idx = torch.triu_indices(data.y.shape[0], data.y.shape[1], offset=1, device=device)

        dist_truth = truth[idx[0], idx[1]].detach().cpu().numpy()
        dist_out = out[idx[0], idx[1]].detach().cpu().numpy()

        SpRho = spearmanr(dist_truth, dist_out)[0]

        tempspear.append(SpRho)
        tempmodels.append(coords)
        tempmse.append(loss)
        model_list.append(model)

    # =========================
    # Save best model
    # =========================
    idx_best = tempspear.index(max(tempspear))
    repmod = tempmodels[idx_best]
    repspear = tempspear[idx_best]
    repmse = tempmse[idx_best]
    repconv = conversions[idx_best]
    repnet = model_list[idx_best]

    print(f'Optimal conversion factor: {repconv}')
    print(f'Optimal dSCC: {repspear}')

    with open(f'Outputs/{name}_log.txt', 'w') as f:
        f.write(f'Optimal conversion factor: {repconv}\n')
        f.write(f'Optimal dSCC: {repspear}\n')
        f.write(f'Final MSE loss: {repmse}\n')

    torch.save(repnet.state_dict(), f'Outputs/{name}_weights.pt')
    utils.WritePDB(repmod.detach().cpu().numpy() * 100,
                   f'Outputs/{name}_structure.pdb')

    print("Training complete.")
