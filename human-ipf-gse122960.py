#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 2020

@author: Joan Smith

https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE122960
"""
#%%
import scipy.sparse as sp_sparse
import tables
import pandas as pd
import os
import glob
import scanpy as sc
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

#%%

sc.settings.figdir = 'data/cluster-plots/gse1229560_ipf/'
sc.settings.verbosity = 4

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({
    'font.sans-serif': 'Arial',
    'font.family': 'sans-serif',
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    })


#%%
def get_series_from_h5(filename):
    with tables.open_file(filename, 'r') as f:
        mat_group = f.get_node(f.root, 'GRCh38')
        barcodes = f.get_node(mat_group, 'barcodes').read().astype(str)
        gene_names = f.get_node(mat_group, 'gene_names').read().astype(str)
        data = getattr(mat_group, 'data').read()
        indices = getattr(mat_group, 'indices').read()
        indptr = getattr(mat_group, 'indptr').read()
        shape = getattr(mat_group, 'shape').read()
        matrix = sp_sparse.csc_matrix((data, indices, indptr), shape=shape)
        return  matrix, barcodes, gene_names.tolist()
    
#%%
p = 'data/raw-data/GSE122960_RAW/*filtered_*.h5'
dfs = []
genes = []
cells = []
donors = []
globs = [i for i in glob.glob(p) if 'IPF' in i or 'Cryobiopsy' in i]
for f in globs:
    print(f)
    matrix, barcodes, gene_names = get_series_from_h5(f)
    donor_id = '_'.join(f.split('/')[-1].split('_')[0:3])
    dfs.append(matrix)
    cells.append(barcodes)
    donors.append([donor_id]*barcodes.shape[0])
    genes.append(gene_names)
#%%
#Verifies all gene names are present, in the same order, for each matrix

def verify_genes(gene_names):
    gene_names_df = pd.DataFrame(genes)
    assert (np.array([gene_names_df[i].value_counts().iloc[0] for i in gene_names_df]) < len(gene_names)).sum() == 0
verify_genes(genes)    


#%% Smoker v. Non

#Never = 0, Former = 1, Active = 2
smoker_v_non_donor_table = {
    'Donor_01': 0,
    'Donor_02': 1,
    'Donor_03': 0,
    'Donor_04': 0,
    'Donor_05': 2,
    'Donor_06': 0,
    'Donor_07': 2,
    'Donor_08': 0 }
smoker_v_non_disease_table = {
    'IPF_01': 1,
    'IPF_02': 0,
    'IPF_03': 1,
    'IPF_04': 0,
    'HP_01':  0,
    'SSc-ILD_01': 0,
    'SSc-ILD_02': 0,
    'Myositis-ILD_01': 0,
    'Cryobiopsy_01': 1
    }

#%% Collect into AnnData object

adata = sc.AnnData(sp_sparse.hstack(dfs).T)
adata.var = pd.DataFrame(genes[0], index=genes[0], columns=['name'])
adata.var_names_make_unique()
obs =  pd.DataFrame({'barcodes': np.hstack(cells), 'donor_id': np.hstack(donors)})
obs['donor-barcode'] = obs['barcodes'] + '_' + obs['donor_id']
obs = obs.set_index('donor-barcode')

adata.obs = obs
adata.obs['donor'] = adata.obs.donor_id.str.split('_', n=1, expand=True)[1]
#%%
adata.obs['smoker'] = adata.obs.donor.map(smoker_v_non_donor_table)
active_smokers = adata[adata.obs['smoker'] == 2].copy()
never_smokers = adata[adata.obs['smoker'] == 0].copy()

#%% Normalize data
sc.pp.normalize_total(adata, target_sum=1e6)
sc.pp.log1p(adata, base=2)

#%%
def exploratory_plots(adata):
    
    # Check normalization
    num_non_int = (adata.to_df().applymap(float.is_integer) == False).sum().sum()
    print('Num non-int: ', num_non_int)
    
    plt.figure()
    sc.pp.filter_cells(adata, min_genes=0)
    sc.pp.filter_genes(adata, min_cells=0)

    plt.hist(adata.obs.n_genes, bins=500)
    plt.title('IPF per cell')
    print('Min:', adata.obs['n_genes'].min())
    
    minimum = adata.obs['n_genes'].min()
    maximum = adata.obs['n_genes'].max()
    print('Max:', maximum)
    plt.xlabel('# Genes. Min:' + str(minimum))
    plt.ylabel('# Cells')
    
    plt.figure()
    plt.hist(adata.var.n_cells, bins=500)
    plt.title('IPF per gene')
    print('Min:', adata.var['n_cells'].min())
    
    sc.pl.pca_variance_ratio(adata, log=True)

exploratory_plots(adata)

#%%
sc.pp.filter_cells(adata, min_genes=500)
sc.pp.highly_variable_genes(adata)

sc.tl.pca(adata)
sc.pp.neighbors(adata)
#%%
LEARNING_RATE = 1000
EARLY_EXAGGERATION = 12
RESOLUTION = 1.25
PERPLEXITY=130

sc.tl.tsne(adata, learning_rate=LEARNING_RATE, n_jobs=8, early_exaggeration=EARLY_EXAGGERATION, perplexity=PERPLEXITY)
sc.tl.leiden(adata, resolution=RESOLUTION)

params = {'learning_rate': LEARNING_RATE,
          'early_exaggeration':EARLY_EXAGGERATION,
          'resolution': RESOLUTION,
          'perplexity': PERPLEXITY,
          'genes': 'all',
          'files': globs}
pd.Series(params).to_csv(os.path.join(sc.settings.figdir, 'params.txt'))
adata.write(os.path.join(sc.settings.figdir, 'adata.h5ad'))

#%%
adata = sc.read_h5ad(os.path.join(sc.settings.figdir, 'adata.h5ad'))
params = pd.read_csv(os.path.join(sc.settings.figdir, 'params.txt'), index_col=0).to_dict('dict')['0']

#%%
markers = pd.read_csv('data/highlighted_genes.csv', header=None, names=['gene', 'cluster'])
markers['gene'] = markers['gene'].str.upper()
markers = markers[markers['gene'].isin(gene_names)]

markers['title'] = markers['gene'] + '+: ' + markers['cluster']
markers = markers.set_index('gene')
markers.loc['PTPRC']['title'] = 'PTPRC (CD45)+: Immune Cells'
markers.loc['leiden'] = ['Leiden', 'Clusters']

addl_genes = pd.read_csv('data/additional_ipf_genes.csv', header=None) 
addl_genes['title'] = addl_genes[0]
addl_genes = addl_genes.set_index(0)
markers = markers.append(addl_genes)

#%%
for i, g in markers.iterrows():
    sc.pl.tsne(adata, color=i,
           title=g['title'],
           color_map='plasma',
           size=25,
           save='_' + i + '_all.pdf',
           show=False)

#%%
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test', n_genes=20)
pd.DataFrame(adata.uns['rank_genes_groups']['names']).to_csv(os.path.join(sc.settings.figdir, 'rank_groups.csv'))
adata.write(os.path.join(sc.settings.figdir, 'adata.h5ad'))

#%% Correlations

df = adata.to_df()
corr = df.corrwith(df['ACE2']).sort_values(ascending=False)
print('correlations with ACE2')
print(corr.head(20)) 
corr.to_csv(os.path.join(sc.settings.figdir, 'ACE2_correlates.csv'))   
    
#%%
cluster_labels = pd.read_csv(os.path.join(sc.settings.figdir, 'key.csv'), header=None, names=['cluster', 'label'], dtype='str')
cluster_labels.loc[cluster_labels['label'].isna(),'label'] = cluster_labels.loc[cluster_labels['label'].isna(), 'cluster']
cluster_labels_dict = cluster_labels.set_index('cluster')['label'].to_dict()
adata.obs['Cell Types'] = adata.obs.leiden.map(cluster_labels_dict).astype('category')

ax = sc.pl.tsne(adata, color=['Cell Types'],
           size=25,
           title='Human Lung',
           save='_' + 'labeled_clusters.pdf')
adata.write(os.path.join(sc.settings.figdir, 'adata.h5ad'))
#%%
sc.tl.rank_genes_groups(adata, 'Cell Types', method='t-test', n_genes=20)
pd.DataFrame(adata.uns['rank_genes_groups']['names']).to_csv(os.path.join(sc.settings.figdir, 'labeled_clusters_rank_groups.csv'))
adata.write(os.path.join(sc.settings.figdir, 'adata.h5ad'))

#%%
trackplot_markers = pd.read_csv(os.path.join(sc.settings.figdir, 'trackplot_human.csv'), header=None)
axes_list = sc.pl.tracksplot(adata, var_names=trackplot_markers[0].values, groupby='Cell Types', figsize=(18,3))
[i.yaxis.set_ticks([]) for i in axes_list]
ax = plt.gca()
ax.set_xlabel('')
plt.savefig(os.path.join(sc.settings.figdir, 'tracksplot_cell_types.pdf'), bbox_inches='tight')
