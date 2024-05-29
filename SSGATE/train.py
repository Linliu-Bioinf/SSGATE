#!/bin/env python3
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import phenograph


from .model import ssmiDGATE
from .utils import Transfer_pytorch_Data
from .triplet_loss import batch_hard_triplet_loss



def train(adata1, adata2, hidden_dims1 = 128, hidden_dims2 = 128, out_dims = 30, n_epochs=200, lr=0.001, epochs_init = 100, cluster_update_epoch = 100, key_added='ssmi_embed', gradient_clipping=5.,  weight_decay=0.0001, verbose=True, random_seed=0, save_loss=False, save_reconstrction=False, sigma = 0.1,margin = 1.0, device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), feat1 = "PCA", feat2 = 'fullproteins'):

#adata1: spatial transcriptomics
#adata2: spatial proteomics
    # seed_everything()
    seed=random_seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    if 'nbrs_net' not in adata1.uns.keys() and adata2.uns.keys():
        raise ValueError("nbrs_net is not existed! Run Cal_Nbrs_Net first!")

    data1 = Transfer_pytorch_Data(adata1, feat = feat1)
    data2 = Transfer_pytorch_Data(adata2, feat = feat2)


    model = ssmiDGATE(data1.x.shape[1], hidden_dims1, data2.x.shape[1], hidden_dims2, out_dims).to(device)
    data1 = data1.to(device)
    data2 = data2.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    #Initiating the training!!!
    loss_list = []
    x_losses = []
    y_losses = []
    triplet_losses = []
#    if n_epochs >= 500:
#        epochs_init = 200
#    else:
#        epochs_init = 100

    for epoch in tqdm(range(1, epochs_init+1)):
        model.train()
        optimizer.zero_grad()
        h_x, h_y, z, z_x, z_y, x_hat, y_hat = model(data1.x, data2.x, data1.edge_index, data2.edge_index)
        loss_x = F.mse_loss(data1.x, x_hat)
        loss_y = F.mse_loss(data2.x, y_hat)
        loss = loss_x + loss_y
        loss.backward()
        loss_list.append(loss)
        x_losses.append(loss_x)
        y_losses.append(loss_y)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()
    model.eval()
    h_x, h_y, z, z_x, z_y, x_hat, y_hat = model(data1.x, data2.x, data1.edge_index, data2.edge_index)
    #Do the model training!!!
    for epoch in tqdm(range(epochs_init, n_epochs)):
        if epoch % cluster_update_epoch == 0 or epoch == epochs_init:
            labels_x, _, _ = phenograph.cluster(h_x.to('cpu').detach().numpy())
            labels_x = torch.tensor(labels_x)
            labels_y, _, _ = phenograph.cluster(h_y.to('cpu').detach().numpy())
            labels_y = torch.tensor(labels_y)

        model.train()
        optimizer.zero_grad()
        h_x, h_y, z, z_x, z_y, x_hat, y_hat = model(data1.x, data2.x, data1.edge_index, data2.edge_index)

        triplet_loss_x = batch_hard_triplet_loss(labels_x, z.to("cpu"), margin = margin)
        triplet_loss_x = triplet_loss_x.to(device)
        triplet_loss_y = batch_hard_triplet_loss(labels_y, z.to("cpu"), margin = margin)
        triplet_loss_y = triplet_loss_y.to(device)
        triplet_loss = triplet_loss_x + triplet_loss_y

        mse_loss_x = F.mse_loss(data1.x, x_hat)
        mse_loss_y = F.mse_loss(data2.x, y_hat)
        mse_loss = mse_loss_x + mse_loss_y

        loss = mse_loss + sigma * triplet_loss
        loss_list.append(loss)
        x_losses.append(mse_loss_x)
        y_losses.append(mse_loss_y)
        triplet_losses.append(triplet_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()

    model.eval()
    h_x, h_y, z, z_x, z_y, x_hat, y_hat = model(data1.x, data2.x, data1.edge_index, data2.edge_index)
    embed_z = z.to('cpu').detach().numpy()
    z_x = z_x.to('cpu').detach().numpy()
    z_y = z_y.to('cpu').detach().numpy()
    adata1.obsm[key_added] = embed_z
    adata2.obsm[key_added] = embed_z
    if save_loss:
        adata1.uns['train_loss'] = loss
    if save_reconstrction:
        ReX_x = x_hat.to('cpu').detach().numpy()
        ReX_y = y_hat.to('cpu').detach().numpy()
        ReX_y[ReX_y<0] = 0
        ReX_x[ReX_x<0] = 0
        adata1.obsm["ssmi_rex"] = ReX_x
        adata1.obsm["z_x"] = z_x
        adata2.obsm["ssmi_rex"] = ReX_y
        adata2.obsm["z_y"] = z_y
#Plot loss history
    print("Ploting losses!")
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["figure.figsize"] = [12,2]
    losses = np.array(torch.tensor(loss_list, device = 'cpu'))
    x_losses = np.array(torch.tensor(x_losses, device = 'cpu'))
    y_losses = np.array(torch.tensor(y_losses, device = 'cpu'))
    triplet_losses = np.array(torch.tensor(triplet_losses, device = 'cpu'))
    fig = plt.figure()
    ax1 = fig.add_subplot(1,4,1)
    ax1.plot(range(len(losses)), losses, c = np.array([255, 71, 90]) / 255.)
    plt.title("Total loss")
    plt.xlabel("Epoch")
    ax2 = fig.add_subplot(1,4,2)
    ax2.plot(range(len(x_losses)), x_losses, c = np.array([255, 71, 90]) / 255.)
    plt.title("MSE loss of x")
    plt.xlabel("Epoch")    
    ax3 = fig.add_subplot(1,4,3)
    ax3.plot(range(len(y_losses)), y_losses, c = np.array([255, 71, 90]) / 255.)
    plt.title("MSE loss of y")
    plt.xlabel("Epoch")    
    ax3 = fig.add_subplot(1,4,4)
    ax3.plot(range(len(triplet_losses)), triplet_losses, c = np.array([255, 71, 90]) / 255.)
    plt.title("Triplet loss of z")
    plt.xlabel("Epoch")        
    plt.show()
    return adata1, adata2



