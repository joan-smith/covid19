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
import os

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
#%%
sc.pp.normalize_total(adata, target_sum=1e6)
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
LEARNING_RATE = 1000
EARLY_EXAGGERATION = 12
RESOLUTION = 1
PERPLEXITY=30
N_PCS = 50

sc.pp.filter_cells(adata, min_genes=500)
sc.pp.highly_variable_genes(adata)

sc.tl.pca(adata)
sc.pp.neighbors(adata)
sc.tl.tsne(adata, learning_rate=LEARNING_RATE, n_jobs=8, early_exaggeration=EARLY_EXAGGERATION, perplexity=PERPLEXITY, n_pcs=N_PCS)
sc.tl.leiden(adata, resolution=RESOLUTION)
sc.pl.tsne(adata, color='leiden')

params = {'learning_rate': LEARNING_RATE,
          'early_exaggeration':EARLY_EXAGGERATION,
          'resolution': RESOLUTION,
          'perplexity': PERPLEXITY,
          'n_pcs': N_PCS,
          'genes': 'all'
         }
sc.pl.tsne(adata, color=['leiden'], size=25, save='_leiden.pdf', color_map='plasma')
pd.Series(params).to_csv(os.path.join(sc.settings.figdir, 'params.txt'))
adata.uns['joan_cluster_params'] = params
adata.write(os.path.join(sc.settings.figdir, 'adata.h5ad'))
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
plt.savefig(os.path.join(sc.settings.figdir, 'leiden_healthy_mouse_gse3440071.pdf'))

for i, g in markers.iterrows():
    sc.pl.tsne(adata, color=i,
           title=g['title'],
           color_map='plasma',
         
            size=25,
           save='_' + i + '_healthy_mouse_gse3440071_markers_500gene.pdf',
           show=False) 
    
#%% 
adata = sc.read_h5ad(os.path.join(sc.settings.figdir, 'adata.h5ad'))
#%% Collect top-ranked genes for each group

sc.tl.rank_genes_groups(adata, groupby='leiden', n_genes=20)
pd.DataFrame(adata.uns['rank_genes_groups']['names']).to_csv(os.path.join(sc.settings.figdir, 'healthy_mouse_groups.csv'))

adata.write(os.path.join(sc.settings.figdir, 'adata.h5ad'))
#%% Calculate correlations with ACE2

df = adata.to_df()
corr = df.corrwith(df['ACE2']).sort_values(ascending=False)
corr.to_csv(os.path.join(sc.settings.figdir, 'ACE2_correlates_healthy_mouse_GSM3440071.csv'))


#%% Cluster 14 and ACE2 pos

adata.obs.loc[adata.obs['leiden'] == '14','ace2_goblet'] = 'goblet'
adata.obs.loc[adata.to_df()['ACE2'] > 0, 'ace2_goblet'] = 'ace2_pos'
sc.tl.rank_genes_groups(adata, groupby='ace2_goblet', key_added='ACE2_goblet', groups=['ace2_pos', 'goblet'])
pd.DataFrame(adata.uns['ACE2_goblet']['names']).to_csv(os.path.join(sc.settings.figdir, 'ACE2_pos_v_goblet_rankings.csv'))

#%% Trackplot Plot of labeled clusters and markers
cluster_key = pd.read_csv(os.path.join(sc.settings.figdir, 'key.csv'), header=None, names=['cluster', 'cell_type'])
cluster_key['cluster'] = cluster_key['cluster'].astype(str)
cluster_key_map = cluster_key.set_index('cluster')['cell_type'].to_dict()

adata.obs['Cell Types'] = adata.obs['leiden'].map(cluster_key_map, na_action='ignore').astype('category')
sc.pl.tsne(adata, color=['Cell Types'], size=25, save='_leiden.pdf', color_map='plasma', title='Mouse Lung')

trackplot_markers = pd.read_csv(os.path.join(sc.settings.figdir, 'trackplot_mouse.csv'), header=None)
axes_list = sc.pl.tracksplot(adata, var_names=trackplot_markers[0].values, groupby='Cell Types', figsize=(18,3))
[i.yaxis.set_ticks([]) for i in axes_list]
ax = plt.gca()
ax.set_xlabel('')
plt.savefig(os.path.join(sc.settings.figdir, 'tracksplot_cell_types.pdf'), bbox_inches='tight')
