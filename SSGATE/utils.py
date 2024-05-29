#!/bin/env python3

import numpy as np 
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
import sklearn.neighbors



def Transfer_pytorch_Data(adata, feat = "PCA"):
    assert('nbrs_net' in adata.uns.keys())
    G_df = adata.uns['nbrs_net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)

    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])

    edgeList = np.nonzero(G)
    if feat == "PCA" and "X_pca" in adata.obsm:
        X = sp.csr_matrix(adata.obsm["X_pca"])
    elif feat == "hvg" and "highly_variable" not in adata.var.keys():
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=500)
        X = adata[:, adata.var['highly_variable']].X
    elif feat == "hvg" and "highly_variable" in adata.var.keys():
        X = adata[:, adata.var['highly_variable']].X
    else:
        X = adata.X
        
    if type(X) == np.ndarray:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(X))  # .todense()
    else:
        data = Data(edge_index=torch.LongTensor(np.array(
            [edgeList[0], edgeList[1]])), x=torch.FloatTensor(X.todense()))  # .todense()
    return data


def Cal_Nbrs_Net(adata, feat = 'X_pca', rad_cutoff=None, k_cutoff=None, model='Radius', verbose=True):
# feat = {'X_pca', "spatial"}
    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(adata.obsm[feat])
    coor.index = adata.obs.index
#    coor.columns = ['imagerow', 'imagecol']

    if model == 'Radius':
        nbrs = sklearn.neighbors.NearestNeighbors(radius=rad_cutoff).fit(coor)
        distances, indices = nbrs.radius_neighbors(coor, return_distance=True)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices[it].shape[0], indices[it], distances[it])))

    if model == 'KNN':
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=k_cutoff+1).fit(coor)
        distances, indices = nbrs.kneighbors(coor)
        KNN_list = []
        for it in range(indices.shape[0]):
            KNN_list.append(pd.DataFrame(zip([it]*indices.shape[1],indices[it,:], distances[it,:])))

    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']

    nbrs_net = KNN_df.copy()
    nbrs_net = nbrs_net.loc[nbrs_net['Distance']>0,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    nbrs_net['Cell1'] = nbrs_net['Cell1'].map(id_cell_trans)
    nbrs_net['Cell2'] = nbrs_net['Cell2'].map(id_cell_trans)
    if verbose:
        print('The graph contains %d edges, %d cells.' %(nbrs_net.shape[0], adata.n_obs))
        print('%.4f neighbors per cell on average.' %(nbrs_net.shape[0]/adata.n_obs))

    adata.uns['nbrs_net'] = nbrs_net
    return adata


def Stats_Nbrs_Net(adata):
    import matplotlib.pyplot as plt
    Num_edge = adata.uns['nbrs_net']['Cell1'].shape[0]
    Mean_edge = Num_edge/adata.shape[0]
    plot_df = pd.value_counts(pd.value_counts(adata.uns['nbrs_net']['Cell1']))
    plot_df = plot_df/adata.shape[0]
    fig, ax = plt.subplots(figsize=[3,2])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)'%Mean_edge)
    ax.bar(plot_df.index, plot_df)


def prune_net(adata):
    assert('nbrs_net' in adata.uns.keys())
    if 'pre_label' not in adata.obs.keys():
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs = 25)
        sc.tl.leiden(adata, resolution = 0.2, key_added = "pre_label")    
    assert('pre_label' in adata.obs.keys())
    Graph_df = adata.uns["nbrs_net"].copy()
    print("++++++Pruning the spatial graph!++++++")
    print("%d edges before pruning." % Graph_df.shape[0])
    label = adata.obs["pre_label"]
    pro_labels_dict = dict(zip(list(label.index), label))
    Graph_df['Cell1_label'] = Graph_df['Cell1'].map(pro_labels_dict)
    Graph_df['Cell2_label'] = Graph_df['Cell2'].map(pro_labels_dict)
    Graph_df = Graph_df.loc[Graph_df['Cell1_label']==Graph_df['Cell2_label'],]
    print('%d edges after pruning.' %Graph_df.shape[0])
    adata.uns["nbrs_Net"] = Graph_df
    return adata

def clr_normalization(adata):
    #Centered Log-Ratio normalization (CLR): clr(i) = ln(x_i/g(x)); g(x) = n^(x_2 ...x_n)
    if sp.issparse(adata.X):
        GM = np.exp(np.sum(np.log(adata.X.toarray() + 1)/adata.X.shape[1], axis = 1))
        adata.X = np.log(adata.X.toarray()/GM[:, None] + 1)
    else:
        GM = np.exp(np.sum(np.log(adata.X + 1)/adata.X.shape[1], axis = 1))
        adata.X = np.log(adata.X/GM[:, None] + 1)
    return adata



