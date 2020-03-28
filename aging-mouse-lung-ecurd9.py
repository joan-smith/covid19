#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 09:40:32 2020

@author: Joan Smith

https://www.ebi.ac.uk/gxa/sc/experiments/E-CURD-9/results/tsne
"""


#%%

import pandas as pd
import scanpy as sc
from matplotlib import pyplot as plt
import matplotlib as mpl

#%% Set scanpy and matplotlib settings
sc.settings.figdir = 'data/cluster-plots/aging-mouse/'
sc.settings.verbosity = 4

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({
    'font.sans-serif': 'Arial',
    'font.family': 'sans-serif',
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    })
#%% Collect gene names and annotation file

gene_names = pd.read_csv('data/raw-data/Mus_musculus.GRCm38.99.chr.gtf', sep='\t', comment='#', header=None)

gene_names.columns = ['chr', 'annotation_src', 'feature_type', 'start', 'end', 'score', 'genomic_strand', 'genomic_phase', 'extra']
gene_names['gene_name'] = gene_names['extra'].str.extract(pat='gene_name "(.*?)";')
gene_names['gene_id'] = gene_names['extra'].str.extract(pat='gene_id "(.*?)";')
gene_names['transcript_type'] = gene_names['extra'].str.extract(pat='transcript_type "(.*?)";')
gene_names = gene_names[gene_names['feature_type'] == 'gene']

annotation = gene_names[['gene_id', 'gene_name', 'chr']].groupby('gene_name').head(1)
annotation['gene_name'] = annotation['gene_name'].str.upper()
annotation = annotation.set_index('gene_id')

#%% Read and log normalize single cell data

adata = sc.read_mtx('data/raw-data/E-CURD-9/E-CURD-9.aggregated_filtered_normalised_counts.mtx')
data = adata.X.toarray().T
adata = sc.AnnData(data)

cols = pd.read_csv('data/raw-data/E-CURD-9/E-CURD-9.aggregated_filtered_normalised_counts.mtx_cols', header=None)
rows = pd.read_csv('data/raw-data/E-CURD-9/E-CURD-9.aggregated_filtered_normalised_counts.mtx_rows', sep='\t', header=None)
adata.var = rows.set_index(0).join(annotation)[['gene_name']].reset_index().set_index('gene_name')
adata.var.index = adata.var.index.astype('str')
adata.var_names_make_unique()
adata.obs = cols
sc.pp.log1p(adata, base=2)

#%% Exploratory Plots

# Unused in the main analysis, but used to parameter search for filtering cutoff.
def exploratory_plots(adata):
    plt.figure()
    plt.hist(adata.obs.n_genes, bins=500)
    plt.title('Aging Mouse per cell')
    print(adata.obs['n_genes'].min())
    
    minimum = adata.obs['n_genes'].min()
    plt.xlabel('# Genes. Min:' + str(minimum))
    plt.ylabel('# Cells')
    plt.show()
    plt.savefig('data/cluster-plots/aging-mouse/aging_mouse_genes_cell.pdf')

#%% Perform Clustering

sc.pp.filter_cells(adata, min_genes=500)
sc.pp.highly_variable_genes(adata, min_mean=0.0, max_mean=13, min_disp=0.5)

sc.tl.pca(adata)
sc.pp.neighbors(adata, n_neighbors=30, n_pcs=30)
sc.tl.tsne(adata, perplexity=25)
sc.tl.leiden(adata)

#%% Plot Clusters with highlighted genes

markers = pd.read_csv('data/highlighted_genes.csv', header=None, names=['gene', 'cluster'])
markers['gene'] = markers['gene'].str.upper()
markers = markers[markers['gene'].isin(list(annotation['gene_name'].values))]

markers['title'] = markers['gene'] + '+: ' + markers['cluster']
markers = markers.set_index('gene')
markers.loc['PTPRC']['title'] = 'PTPRC (CD45)+: Immune Cells'


ax = sc.pl.tsne(adata, color=['leiden'],
           title=['Mouse Lung'], 
           color_map='plasma',
           size=25,
           save='_labeled_leiden_aging_mouse_ecurd9_markers_500gene.pdf',
           show=False) 

ax = sc.pl.tsne(adata, color=['leiden'],
           title=['Mouse Lung'], 
           color_map='plasma',
           size=25,
           show=False) 
ax.legend().remove()
plt.savefig('data/cluster-plots/aging-mouse/leiden_aging_mouse_ecurd9_potential_markers_500gene.pdf')

for i, g in markers.iterrows():
    sc.pl.tsne(adata, color=i,
           title=g['title'],
           color_map='plasma',
           size=25,
           save='_' + i + '_aging_mouse_ecurd9_potential_markers_500gene.pdf',
           show=False) 
    

#%% Collect top ranked genes for each group

sc.tl.rank_genes_groups(adata, groupby='leiden', n_genes=10)
pd.DataFrame(adata.uns['rank_genes_groups']['names']).to_csv('data/cluster-plots/aging-mouse/aging_mouse_groups.csv')

#%% Calculate Correlations with ACE2

df = adata.to_df()
corr = df.corrwith(df['ACE2']).sort_values(ascending=False)
print('normal ace2')
print(corr)
corr.to_csv('data/cluster-plots/aging-mouse/ACE2_correlates_aging_mouse_ecurd9.csv')

#%%

violin_markers = pd.read_csv('data/violin_markers_mouse.csv', header=None)
violin_markers[0] = violin_markers[0].str.upper()
violin_markers = violin_markers[violin_markers[0].isin(adata.var.index.values)]

cluster_key = pd.read_csv('data/aging_mouse_cluster_key.csv', header=None, names=['cluster', 'cell_type'])
cluster_key['cluster'] = cluster_key['cluster'].astype(str)
cluster_key_map = cluster_key.set_index('cluster')['cell_type'].to_dict()

adata.obs['Cell Types'] = adata.obs['leiden'].map(cluster_key_map, na_action='ignore').astype('category')


axes_list = sc.pl.stacked_violin(adata, var_names=violin_markers[0].values, groupby='Cell Types', swap_axes=True, show=False, )
[i.yaxis.set_ticks([]) for i in axes_list]
ax = plt.gca()
ax.get_xaxis().set_label_text('')
ax.figure.set_size_inches(5, 6)
plt.savefig('data/cluster-plots/aging-mouse/cell_type_stacked_violin2.pdf', bbox_inches='tight')