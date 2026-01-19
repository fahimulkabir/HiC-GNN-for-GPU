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

    parser = argparse.ArgumentParser(description='Generalize a trained model to new data.')
    parser.add_argument('list_trained', type=str)
    parser.add_argument('list_untrained', type=str)
    parser.add_argument('-c', '--conversions', type=str, default='[.1,.1, 2]')
    parser.add_argument('-bs', '--batchsize', type=int, default=128)
    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-lr', '--learningrate', type=float, default=.001)
    parser.add_argument('-th', '--threshold', type=float, default=1e-8)

    args = parser.parse_args()

    filepath_trained = args.list_trained
    filepath_untrained = args.list_untrained
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

    name_trained = os.path.splitext(os.path.basename(filepath_trained))[0]
    name_untrained = os.path.splitext(os.path.basename(filepath_untrained))[0]

    list_trained = np.loadtxt(filepath_trained)
    list_untrained = np.loadtxt(filepath_untrained)

    # ---------- preprocessing stays unchanged ----------
    if not os.path.isfile(f'Data/{name_trained}_matrix.txt'):
        adj_trained = utils.convert_to_matrix(list_trained)
        np.fill_diagonal(adj_trained, 0)
        np.savetxt(f'Data/{name_trained}_matrix.txt', adj_trained)

    if not os.path.isfile(f'Data/{name_untrained}_matrix.txt'):
        adj_untrained = utils.convert_to_matrix(list_untrained)
        np.fill_diagonal(adj_untrained, 0)
        np.savetxt(f'Data/{name_untrained}_matrix.txt', adj_untrained)

    matrix_trained = np.loadtxt(f'Data/{name_trained}_matrix.txt')
    matrix_untrained = np.loadtxt(f'Data/{name_untrained}_matrix.txt')

    if not os.path.isfile(f'Data/{name_trained}_matrix_KR_normed.txt'):
        os.system(f'Rscript normalize.R {name_trained}_matrix')

    if not os.path.isfile(f'Data/{name_untrained}_matrix_KR_normed.txt'):
        os.system(f'Rscript normalize.R {name_untrained}_matrix')

    normed_trained = np.loadtxt(f'Data/{name_trained}_matrix_KR_normed.txt')
    normed_untrained = np.loadtxt(f'Data/{name_untrained}_matrix_KR_normed.txt')

    # ---------- embeddings (still CPU only by design) ----------
    if not os.path.isfile(f'Data/{name_trained}_embeddings.txt'):
        G = nx.from_numpy_matrix(matrix_trained)
        embed = LINE(G, embedding_size=512, order='second')
        embed.train(batch_size=batch_size, epochs=epochs, verbose=1)
        embeddings_trained = np.asarray(list(embed.get_embeddings().values()))
        np.savetxt(f'Data/{name_trained}_embeddings.txt', embeddings_trained)

    if not os.path.isfile(f'Data/{name_untrained}_embeddings.txt'):
        G = nx.from_numpy_matrix(matrix_untrained)
        embed = LINE(G, embedding_size=512, order='second')
        embed.train(batch_size=batch_size, epochs=epochs, verbose=1)
        embeddings_untrained = np.asarray(list(embed.get_embeddings().values()))
        np.savetxt(f'Data/{name_untrained}_embeddings.txt', embeddings_untrained)

    embeddings_trained = np.loadtxt(f'Data/{name_trained}_embeddings.txt')
    embeddings_untrained = np.loadtxt(f'Data/{name_untrained}_embeddings.txt')

    # =========================
    # GPU-AWARE DATA CREATION (NEW)
    # =========================
    data_trained = utils.load_input(normed_trained, embeddings_trained, device)
    data_untrained = utils.load_input(normed_untrained, embeddings_untrained, device)

    # =========================
    # TRAINING ON GPU (NEW)
    # =========================
    if not os.path.isfile(f'Outputs/{name_trained}_weights.pt'):
        tempmodels, tempspear, tempmse, model_list = [], [], [], []

        for conversion in conversions:
            model = Net().to(device)
            criterion = MSELoss()
            optimizer = Adam(model.parameters(), lr=lr)

            truth = utils.cont2dist(data_trained.y, conversion).to(device)

            oldloss = 1
            lossdiff = 1

            while lossdiff > thresh:
                model.train()
                optimizer.zero_grad()
                out = model(data_trained.x.float(), data_trained.edge_index)
                loss = criterion(out.float(), truth.float())
                loss.backward()
                optimizer.step()
                lossdiff = abs(oldloss - loss)
                oldloss = loss

            coords = model.get_model(data_trained.x.float(), data_trained.edge_index)
            out = torch.cdist(coords, coords)

            idx = torch.triu_indices(data_trained.y.shape[0], data_trained.y.shape[1], offset=1, device=device)
            dist_truth = truth[idx[0], idx[1]].detach().cpu().numpy()
            dist_out = out[idx[0], idx[1]].detach().cpu().numpy()

            SpRho = spearmanr(dist_truth, dist_out)[0]

            tempspear.append(SpRho)
            tempmodels.append(coords)
            tempmse.append(loss)
            model_list.append(model)

        idx_best = tempspear.index(max(tempspear))
        repnet = model_list[idx_best]
        repmod = tempmodels[idx_best]

        torch.save(repnet.state_dict(), f'Outputs/{name_trained}_weights.pt')
        utils.WritePDB(repmod.detach().cpu().numpy()*100, f'Outputs/{name_trained}_structure.pdb')

    # =========================
    # LOAD MODEL ON GPU (NEW)
    # =========================
    model = Net().to(device)
    model.load_state_dict(torch.load(f'Outputs/{name_trained}_weights.pt', map_location=device))
    model.eval()

    fitembed = utils.domain_alignment(list_trained, list_untrained, embeddings_trained, embeddings_untrained)
    data_untrained_fit = utils.load_input(normed_untrained, fitembed, device)

    temp_spear, temp_models = [], []

    for conversion in conversions:
        truth = utils.cont2dist(data_untrained_fit.y, conversion).to(device)

        coords = model.get_model(data_untrained_fit.x.float(), data_untrained_fit.edge_index)
        out = torch.cdist(coords, coords)

        idx = torch.triu_indices(truth.shape[0], truth.shape[1], offset=1, device=device)
        dist_truth = truth[idx[0], idx[1]].detach().cpu().numpy()
        dist_out = out[idx[0], idx[1]].detach().cpu().numpy()

        SpRho = spearmanr(dist_truth, dist_out)[0]
        temp_spear.append(SpRho)
        temp_models.append(coords)

    best = temp_spear.index(max(temp_spear))
    repmod = temp_models[best]

    utils.WritePDB(repmod.detach().cpu().numpy()*100,
                   f'Outputs/{name_untrained}_generalized_structure.pdb')

    print("Done.")
