#!/bin/env python3

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from .utils import clr_normalization 


def preprocess_cluster(adata_st, adata_sp, res_st = 0.2, res_sp = 0.2, show_fig = True, figsize = (8,8), s = 10, spot_size = 10):
    print('Original ST Data Info: %d cells * %d genes.' % (adata_st.shape[0], adata_st.shape[1]))
    adata_st.X = adata_st.X.astype(np.float)
    sc.pp.normalize_total(adata_st, target_sum = 1e4)
    sc.pp.log1p(adata_st)
    sc.pp.highly_variable_genes(adata_st, flavor = 'seurat_v3', n_top_genes = 4000)
    adata_st = adata_st[:, adata_st.var['highly_variable']]
#    sc.pp.scale(adata_st, max_value = 10)
    sc.tl.pca(adata_st, svd_solver = 'arpack')
    sc.pp.neighbors(adata_st)
    sc.tl.umap(adata_st)
    sc.tl.leiden(adata_st, resolution = res_st, key_added = 'pre_label')


    print('Original SP Data Info: %d cells * %d genes.' % (adata_sp.shape[0], adata_sp.shape[1]))
    adata_sp.X = adata_sp.X.astype(np.float)
    adata_sp = clr_normalization(adata_sp)
    sc.pp.neighbors(adata_sp)
    sc.tl.umap(adata_sp)
    sc.tl.leiden(adata_sp, resolution = res_sp, key_added = 'pre_label')

    if show_fig:
        if "spatial" in adata_st.obsm.keys() and adata_sp.obsm.keys():
            fig, axs = plt.subplots(2, 2, figsize = figsize)
            sc.pl.umap(adata_st, color = "pre_label", s = s, ax = axs[0, 0], show = False, title = "mRNA")
            sc.pl.umap(adata_sp, color = "pre_label", s = s, ax = axs[0, 1], show = False, title = "Protein")
            sc.pl.spatial(adata_st, color = "pre_label", spot_size = spot_size, ax = axs[1, 0], show = False, title = "mRNA")
            sc.pl.spatial(adata_sp, color = "pre_label", spot_size = spot_size, ax = axs[1, 1], show = False, title = "Protein")
        else:
            fig, axs = plt.subplots(1, 2, figsize = figsize)
            sc.pl.umap(adata_st, color = "pre_label", s = s, ax = axs[0], show = False, title = "mRNA")
            sc.pl.umap(adata_sp, color = "pre_label", s = s, ax = axs[1], show = False, title = "Protein")
        fig.tight_layout(pad = 1.0)
        plt.show()
    
    return adata_st, adata_sp



