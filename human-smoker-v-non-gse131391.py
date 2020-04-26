#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 10:31:08 2020

@author: Joan Smith
Human Smoker v. Non. Single Cell
"""

#%%
import pandas as pd
from matplotlib import pyplot as plt
import scanpy as sc
import matplotlib as mpl

import os

#%% Set scanpy and matplotlib settings
sc.settings.figdir = 'data/cluster-plots/human-smoker-non/'
sc.settings.verbosity = 4

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({
    'font.sans-serif': 'Arial',
    'font.family': 'sans-serif',
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    })

#%% Collect Gene names and annotation file

cell_counts = pd.read_csv('data/raw-data/GSE131391/GSE131391_count_matrix.txt', sep='\t', comment='!', index_col=0)
gene_names = pd.read_csv('data/raw-data/Homo_sapiens.GRCh38.99.gtf', sep='\t', comment='#', header=None)
                                  
gene_names.columns = ['chr', 'annotation_src', 'feature_type', 'start', 'end', 'score', 'genomic_strand', 'genomic_phase', 'extra']
gene_names['gene_name'] = gene_names['extra'].str.extract(pat='gene_name "(.*?)";')
gene_names['gene_id'] = gene_names['extra'].str.extract(pat='gene_id "(.*?)";')
gene_names['transcript_type'] = gene_names['extra'].str.extract(pat='transcript_type "(.*?)";')
gene_names = gene_names[gene_names['feature_type'] == 'gene']

annotation = gene_names[['gene_id', 'gene_name', 'chr']].groupby('gene_name').head(1)
annotation = annotation.set_index('gene_id')

#%% Read and normalize single cell data

cell_counts = annotation.join(cell_counts, how='inner').set_index('gene_name')
cell_counts = cell_counts.drop('chr', axis=1)

#%%
adata = sc.AnnData(cell_counts.T)
sc.pp.filter_cells(adata, min_genes=500)
sc.pp.normalize_total(adata, target_sum=1e6)
sc.pp.log1p(adata, base=2)

#%% Exploratory plots

def exploratory_plots(adata):
    plt.figure()
    plt.hist(adata.obs.n_genes, bins=500)
    plt.title('Human Smoker-v-Non Genes per cell')
    print(adata.obs['n_genes'].min())
    
    minimum = adata.obs['n_genes'].min()
    plt.xlabel('# Genes. Min:' + str(minimum))
    plt.ylabel('# Cells')
    plt.savefig(os.path.join(sc.settings.figdir, 'human_smoker_v_non_genes_cell.pdf')

#%% Clustering all data together

sc.pp.filter_cells(adata, min_genes=500)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=6, min_disp=0.5)

sc.tl.pca(adata)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
sc.tl.tsne(adata)
sc.tl.leiden(adata)

#%% Collect markers

markers = pd.read_csv('data/highlighted_genes.csv', header=None, names=['gene', 'cluster'])
markers['gene'] = markers['gene'].str.upper()
markers = markers[markers['gene'].isin(list(annotation['gene_name'].values))]

markers['title'] = markers['gene'] + '+: ' + markers['cluster']
markers = markers.set_index('gene')
markers.loc['PTPRC']['title'] = 'PTPRC (CD45)+: Immune Cells'

#%% Plot combined smokers+non-smokers with highlighted genes

ax = sc.pl.tsne(adata, color=['leiden'],
           title=['Human Bronchial Epithelium'], 
           color_map='plasma',
           save='_labeled_leiden_both_human_smoker_v_non_gse131391_potential_markers_500gene.pdf',
           show=False) 

ax = sc.pl.tsne(adata, color=['leiden'],
           title=['Human Bronchial Epithelium'], 
           color_map='plasma',
           show=False) 
ax.legend().remove()

plt.savefig(os.path.join(sc.settings.figdir, 'leiden_tsne_both_human_smoker_v_non_gse131391_potential_markers_500gene.pdf'))

for i, g in markers.iterrows():
    axes_list = sc.pl.tsne(adata, color=i,
           title=g['title'],
           color_map='plasma',
           save='_' + i + '_both_human_smoker_v_non_gse131391_potential_markers_500gene.pdf',
           show=False) 
#%% Collect top ranked genes for each combined group

sc.tl.rank_genes_groups(adata, groupby='leiden', n_genes=10)
pd.DataFrame(adata.uns['rank_genes_groups']['names']).to_csv(os.path.join(sc.settings.figdir, 'human_smoker_v_non_both_ranked_gene_groups.csv'))


