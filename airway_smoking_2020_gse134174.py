#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 13:38:41 2020

@author: Joan Smith
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE134174
"""
#%%

import pandas as pd
import scanpy as sc
from matplotlib import pyplot as plt 
import matplotlib as mpl
import os
import numpy as np
#%%
sc.settings.figdir = 'data/cluster-plots/airway-smoking-2020-gse134174/'
sc.settings.verbosity = 4

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({
    'font.sans-serif': 'Arial',
    'font.family': 'sans-serif',
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    })

#%% Load prenormed data and metadata
input_file = 'data/raw-data/airway-smoking-GSE134174/GSE134174_Processed_invivo_norm.txt'
metadata = pd.read_csv('data/raw-data/airway-smoking-GSE134174/GSE134174_Processed_invivo_metadata.txt', sep='\t', index_col=0)
adata = sc.read_text(input_file).T
adata.obs = metadata
all_adata = adata.copy()
VARIANT = 'all'

#%%
def exploratory_plots(adata):
    
    num_non_int = (adata.to_df().applymap(float.is_integer) == False).sum().sum()
    print('Num non-int: ', num_non_int)
    plt.figure()
    sc.pp.filter_cells(adata, min_genes=0)
    plt.hist(adata.obs.n_genes, bins=500)
    plt.title('Genes per cell')    
    minimum = adata.obs['n_genes'].min()
    plt.xlabel('# Genes. Min:' + str(minimum))
    plt.ylabel('# Cells')

    plt.figure()
    sc.pl.highest_expr_genes(adata, n_top=20, )
    sc.pl.pca_variance_ratio(adata, log=True)
    plt.show()    
exploratory_plots(adata)

#%% HEAVY AND NEVER
adata = all_adata[all_adata.obs.Smoke_status.isin(['heavy', 'never'])].copy()
VARIANT='heavy_and_never'
sc.settings.figdir = 'data/cluster-plots/airway-smoking-2020-gse134174-heavy-and-never'

#%%
sc.pp.highly_variable_genes(adata)
sc.tl.pca(adata)
sc.pp.neighbors(adata)

#%%
LEARNING_RATE = 1000
EARLY_EXAGGERATION = 12
RESOLUTION = 2.5 #2 
PERPLEXITY=30
N_PCS = 15

sc.tl.tsne(adata, learning_rate=LEARNING_RATE, n_jobs=8, early_exaggeration=EARLY_EXAGGERATION, perplexity=PERPLEXITY, n_pcs=N_PCS)
sc.tl.leiden(adata, resolution=RESOLUTION)

params = {'learning_rate': LEARNING_RATE,
          'early_exaggeration':EARLY_EXAGGERATION,
          'resolution': RESOLUTION,
          'perplexity': PERPLEXITY,
          'n_pcs': N_PCS,
          'genes': 'all',
          'variant': VARIANT,
          'files': input_file}
sc.pl.tsne(adata, color=['leiden'], size=25, save='_leiden.pdf', color_map='plasma')
pd.Series(params).to_csv(os.path.join(sc.settings.figdir, 'params.txt'))
adata.uns['joan_cluster_params'] = params
adata.write(os.path.join(sc.settings.figdir, 'adata.h5ad'))
#%%
#%%
#%%
#%%
markers = pd.read_csv(os.path.join(sc.settings.figdir, 'markers.csv'), header=None, names=['gene', 'cluster'])
markers['gene'] = markers['gene'].str.upper()
markers = markers[markers['gene'].isin(adata.var.index.values)]

markers['title'] = markers['gene'] + '+: ' + markers['cluster']
markers = markers.set_index('gene')
markers.loc['PTPRC']['title'] = 'PTPRC (CD45)+: Immune Cells'
markers.loc['leiden'] = ['Leiden', 'Clusters']
#%%
adata = sc.read_h5ad(os.path.join(sc.settings.figdir, 'adata.h5ad'))
params = pd.read_csv(os.path.join(sc.settings.figdir, 'params.txt'), index_col=0).to_dict('dict')['0']
VARIANT = params['variant']
#%%
for i, g in markers.iterrows():
    sc.pl.tsne(adata, color=i,
           title=g['title'],
           color_map='plasma',
           size=25,
           save='_' + i + '_airway_smoking_2020.pdf',
           show=False)
    
#%%
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test', n_genes=20)
pd.DataFrame(adata.uns['rank_genes_groups']['names']).to_csv(os.path.join(sc.settings.figdir, 'rank_groups.csv'))
adata.write(os.path.join(sc.settings.figdir, 'adata.h5ad'))


#%%
genes = ['ACE2', 'KRT5', 'TP63', 'BCAM', 'NGFR']
df = adata.to_df()
for g in genes:
    print(g)
    g_df = df.join(adata.obs['Smoke_status'])[[g, 'Smoke_status']]
    g_df = g_df[g_df[g] > 0].pivot(columns='Smoke_status')
    g_df.to_csv(os.path.join(sc.settings.figdir, g +'_smoking_status.csv'))

#%%

#%% Correlations

df = adata.to_df()
corr = df.corrwith(df['ACE2']).sort_values(ascending=False)
print('correlations with ACE2')
print(corr.head(20)) 
corr.to_csv(os.path.join(sc.settings.figdir, 'ACE2_correlates.csv'))   


#%%
markers_for_pos_counts = markers.drop('leiden').copy()
markers_for_pos_counts.loc['MKI67'] = ['MKI67', 'MKI67']
markers_for_pos_counts.loc['PCNA'] = ['PCNA', 'PCNA']
markers_for_pos_counts.loc['TOP2A'] = ['TOP2A', 'TOP2A']

## pos for every marker, # cells
df = adata.to_df()
pos_counts = (df[markers_for_pos_counts.index] > 0).sum()
pos_counts['cell_counts'] = len(df.index)
pos_counts.to_csv(os.path.join(sc.settings.figdir, 'single_positive_counts.csv'), header=['single pos counts'])
                  
# # double pos w ace2, # 
double_pos =  (df[markers_for_pos_counts.index] > 0).apply(lambda x: x & (df['ACE2'] > 0)).sum()
double_pos['cell_counts'] = len(df.index)
double_pos.to_csv(os.path.join(sc.settings.figdir, 'double_positive_counts.csv'), header=['double pos counts'])

#%%
df = adata.to_df()
ace2_muc5ac_double_pos = (df['MUC5AC'] > 0) & (df['ACE2'] > 0)
ace2_muc5ac = df.loc[ace2_muc5ac_double_pos][['ACE2']].join(adata.obs.Smoke_status)
ace2_muc5ac.pivot(columns='Smoke_status').to_csv(os.path.join(sc.settings.figdir, 'ACE2_MUC5AC_double_pos.csv'))

#%%
cluster_labels = pd.read_csv(os.path.join(sc.settings.figdir, 'key.csv'), header=None, names=['cluster', 'label'], dtype='str')
cluster_labels.loc[cluster_labels['label'].isna(),'label'] = cluster_labels.loc[cluster_labels['label'].isna(), 'cluster']
cluster_labels_dict = cluster_labels.set_index('cluster')['label'].to_dict()
adata.obs['Cell Types'] = adata.obs.leiden.map(cluster_labels_dict).astype('category')

sc.pl.tsne(adata, color=['Cell Types'], size=25, save='_manually_labeled.pdf',  color_map='plasma', title='Human Trachea')
adata.write(os.path.join(sc.settings.figdir, 'adata.h5ad'))

sc.tl.rank_genes_groups(adata, 'Cell Types', method='t-test', n_genes=100, key_added='cell_types_gene_groups')
pd.DataFrame(adata.uns['cell_types_gene_groups']['names']).to_csv(os.path.join(sc.settings.figdir, 'manually_labeled_rank_groups_ttest.csv'))

#%%
adata.obs.groupby('Cell Types')['Smoke_status'].value_counts().to_csv(os.path.join(sc.settings.figdir, 'cell_type_smoker_counts.csv'))
ace2_pos_cell_types = adata.to_df()[(adata.to_df()['ACE2'] > 0)][['ACE2']].join(adata.obs[['Cell Types', 'Smoke_status']])
ace2_pos_cell_types.groupby(['Cell Types', 'Smoke_status']).count().to_csv(os.path.join(sc.settings.figdir, 'cell_type_ace2_pos_counts.csv'))

#%%
adata.obs.loc[(adata.to_df()['ACE2'] > 0), 'ace2_pos'] = 'ace2_pos'
adata.obs.loc[(adata.to_df()['ACE2'] <= 0), 'ace2_pos'] = 'ace2_neg'
sc.tl.rank_genes_groups(adata, groupby='ace2_pos', key_added='ACE2_pos')
#%%

adata.obs['ACE2_goblet'] = np.nan
adata.obs.loc[(adata.obs['Cell Types'] == 'Goblet/Club cells'), 'ACE2_goblet'] = 'Goblet/Club'
adata.obs.loc[(adata.obs['Cell Types'] == 'Ciliated cells'), 'ACE2_goblet'] = 'Ciliated'
adata.obs.loc[(adata.obs['Cell Types'] == 'Basal cells'), 'ACE2_goblet'] = 'Basal'
adata.obs.loc[(adata.to_df()['ACE2'] > 0), 'ACE2_goblet'] = 'ACE2+'


adata_ace2_goblet = adata[adata.obs.dropna(subset=['ACE2_goblet']).index].copy()
adata_ace2_goblet.obs['ACE2_goblet'] = adata_ace2_goblet.obs['ACE2_goblet'].astype(pd.CategoricalDtype(ordered=True))
adata_ace2_goblet.obs['ACE2_goblet'] = adata_ace2_goblet.obs['ACE2_goblet'].cat.reorder_categories(['ACE2+', 'Goblet/Club', 'Ciliated', 'Basal'], ordered=True)


heatmap_genes = {'ACE2+ markers': adata.uns['ACE2_pos']['names']['ace2_pos'][1:11],
                 'Goblet/Club markers': adata.uns['cell_types_gene_groups']['names']['Goblet/Club cells'][0:10],
                 'Ciliated markers': adata.uns['cell_types_gene_groups']['names']['Ciliated cells'][0:10],
                 'Basal markers': adata.uns['cell_types_gene_groups']['names']['Basal cells'][0:10],
                 }

ax = sc.pl.dotplot(adata_ace2_goblet, groupby='ACE2_goblet', var_names=heatmap_genes, #save='_ACE2_marker_goblet_ciliated.png', 
              mean_only_expressed=True, var_group_rotation=0.0, show=False, figsize=(12, 2.5))
all_axes = plt.gcf().get_axes()
all_axes[0].set_ylabel('')

plt.savefig(os.path.join(sc.settings.figdir, 'dotplot_ace2_marker_goblet_ciliated_basal.pdf'), bbox_inches='tight')

