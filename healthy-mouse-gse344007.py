#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 15:47:38 2020

@author: Joan Smith

Pulmonary Fibrosis - Mouse Dataset from 
Single-Cell Transcriptomic Analysis of Human Lung Reveals Complex Multicellular Changes During Pulmonary Fibrosis
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE121611

AKA GSE121611
"""

#%% Imports

import pandas as pd
import scanpy as sc
from matplotlib import pyplot as plt 
import matplotlib as mpl
import scipy.sparse

#%% Set scanpy and matplotlib settings
sc.settings.figdir = 'data/cluster-plots/healthy-mouse/'
sc.settings.verbosity = 4

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({
    'font.sans-serif': 'Arial',
    'font.family': 'sans-serif',
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    })

#%% Get genes

#note genes for the second mouse are identical, so no need to duplicate
genes = pd.read_csv('data/raw-data/GSM344007/GSM3440071_SC01_genes.tsv', sep='\t', names=['key', 'name'])
genes['name'] = genes['name'].str.upper()

#%% Read and log normalize single cell data
adata01 = sc.read_mtx('data/raw-data/GSM344007/GSM3440071_SC01_matrix.mtx')
adata02 = sc.read_mtx('data/raw-data/GSM344007/GSM3440072_SC02_matrix.mtx')
data01 = adata01.X.transpose()
data02 = adata02.X.transpose()

adata = sc.AnnData(scipy.sparse.vstack((data01, data02), format='csr'))
adata.var = genes.set_index('name')
adata.var_names_make_unique()
print('Healthy Mouse: ', adata.X.max())
sc.pp.log1p(adata, base=2)

#%% Exploratory plots

# Unused in the main analysis, but used to parameter search for filtering cutoff.
def exploratory_plots(adata):
    plt.figure()
    plt.hist(adata.obs.n_genes, bins=500)
    plt.title('Healthy Mouse per cell')
    print(adata.obs['n_genes'].min())
    
    minimum = adata.obs['n_genes'].min()
    plt.xlabel('# Genes. Min:' + str(minimum))
    plt.ylabel('# Cells')
    plt.savefig('data/cluster-plots/healthy-mouse/healthy_mouse_genes_cell_exploratory.pdf')
    
#%% Perform Clustering

sc.pp.filter_cells(adata, min_genes=500)
sc.pp.highly_variable_genes(adata, min_mean=0.0, max_mean=13, min_disp=0.5)

sc.tl.pca(adata, zero_center=False)
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)
sc.tl.tsne(adata, perplexity=50)
sc.tl.leiden(adata)

#%% Plot clusters and clusters with highlighted genes

markers = pd.read_csv('data/highlighted_genes.csv', header=None, names=['gene', 'cluster'])
markers['gene'] = markers['gene'].str.upper()
markers = markers[markers['gene'].isin(genes['name'])]

markers['title'] = markers['gene'] + '+: ' + markers['cluster']
markers = markers.set_index('gene')
markers.loc['PTPRC']['title'] = 'PTPRC (CD45)+: Immune Cells'

ax = sc.pl.tsne(adata, color=['leiden'],
           title=['Mouse Lung'], 
           color_map='plasma',
           size=25,
           save='_labeled_leiden_healthy_mouse_gse3440071_markers_500gene.pdf',
           show=False) 

ax = sc.pl.tsne(adata, color=['leiden'],
           title=['Mouse Lung'], 
           color_map='plasma',
           size=25,
           show=False) 
ax.legend().remove()
plt.savefig('data/cluster-plots/healthy-mouse/leiden_healthy_mouse_gse3440071_markers_500gene.pdf')

for i, g in markers.iterrows():
    sc.pl.tsne(adata, color=i,
           title=g['title'],
           color_map='plasma',
           size=25,
           save='_' + i + '_healthy_mouse_gse3440071_markers_500gene.pdf',
           show=False) 
#%% Collect top-ranked genes for each group

sc.tl.rank_genes_groups(adata, groupby='leiden', n_genes=10)
pd.DataFrame(adata.uns['rank_genes_groups']['names']).to_csv('data/cluster-plots/healthy-mouse/healthy_mouse_groups.csv')

#%% Calculate correlations with ACE2

df = adata.to_df()
corr = df.corrwith(df['ACE2']).sort_values(ascending=False)
corr.to_csv('data/cluster-plots/healthy-mouse/ACE2_correlates_healthy_mouse_GSM3440071.csv')

#%% Stacked Violin Plot of labeled clusters and markers

violin_markers = pd.read_csv('data/violin_markers_mouse.csv', header=None)
violin_markers[0] = violin_markers[0].str.upper()
violin_markers = violin_markers[violin_markers[0].isin(genes['name'].values)]

cluster_key = pd.read_csv('data/healthy_mouse_cluster_key.csv', header=None, names=['cluster', 'cell_type'])
cluster_key['cluster'] = cluster_key['cluster'].astype(str)
cluster_key_map = cluster_key.set_index('cluster')['cell_type'].to_dict()

adata.obs['Cell Types'] = adata.obs['leiden'].map(cluster_key_map, na_action='ignore').astype('category')


axes_list = sc.pl.stacked_violin(adata, var_names=violin_markers[0].values, groupby='Cell Types', show=False, swap_axes=True)
[i.yaxis.set_ticks([]) for i in axes_list]
ax = plt.gca()
ax.get_xaxis().set_label_text('')
ax.figure.set_size_inches(5, 6)
plt.savefig('data/cluster-plots/healthy-mouse/cell_type_stacked_violin2.pdf', bbox_inches='tight')
