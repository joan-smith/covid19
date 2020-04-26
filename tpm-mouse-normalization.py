#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:24:32 2020

@author: Joan Smith
"""

import scanpy as sc
import pandas as pd

#%%

gene_names = pd.read_csv('data/raw-data/Mus_musculus.GRCm38.99.chr.gtf', sep='\t', comment='#', header=None)
gene_names.columns = ['chr', 'annotation_src', 'feature_type', 'start', 'end', 'score', 'genomic_strand', 'genomic_phase', 'extra']
gene_names['gene_name'] = gene_names['extra'].str.extract(pat='gene_name "(.*?)";')
gene_names['gene_id'] = gene_names['extra'].str.extract(pat='gene_id "(.*?)";')
gene_names['transcript_id'] = gene_names['extra'].str.extract(pat='transcript_id "(.*?)";')

gene_names = gene_names[gene_names['feature_type'] == 'gene']
annotation = gene_names[['gene_id', 'gene_name', 'chr']].groupby('gene_name').head(1)
annotation = annotation.join(gene_transcript_lens, on='gene_id')
annotation = annotation.set_index('gene_id')

#%%% GSE 75715
counts = pd.read_excel('data/raw-data/GSE75715_DE_results_RNA-seq.xlsx', header=[1, 2, 3], index_col=0)
counts = counts.set_index(counts.columns[0], append=True)
counts.index = counts.index.rename(['gene_id', 'gene_name'])
counts.columns = counts.columns.rename(['Day', 'Genetic Background', 'Replicate'])

adata = sc.AnnData(counts.T)
sc.pp.normalize_total(adata, target_sum=1e6)
adata.to_df().T.to_csv('data/raw-data/GSE74715_cpm.csv')

#%% GSE 1302040

annotation = annotation.reset_index().set_index('gene_name')
gse1302040_counts = pd.read_csv('data/raw-data/GSE132040/GSE132040_190214_A00111_0269_AHH3J3DSXX_190214_A00111_0270_BHHMFWDSXX (4).csv', index_col=0)

adata = sc.AnnData(gse1302040_counts.T)
sc.pp.normalize_total(adata, target_sum=1e6)

adata.to_df().T.to_csv('data/raw-data/GSE132040/GSE132040_cpm.csv')
