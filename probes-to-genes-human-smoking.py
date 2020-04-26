#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 18:57:27 2020

@author: joan
"""

import collections
import scipy.sparse as sp_sparse
import tables
import pandas as pd
import os
import glob
import scanpy as sc
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np
import sys

#%%

sc.settings.figdir = 'data/Data/Human smoking/Volcano plots'
sc.settings.verbosity = 4

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams.update({
    'font.sans-serif': 'Arial',
    'font.family': 'sans-serif',
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    })

#%%
gpl570_files = ['GSE13933_trachea.csv', 'gse64614_SAE.csv', 'gse22047_large.csv']

gpl570 = pd.read_csv(os.path.join(sc.settings.figdir, 'GPL570.txt'),index_col=0, sep='\t', comment='#', dtype='str')
gpl570 = gpl570[['Gene Symbol']]
gpl570 = gpl570.dropna(how='any')
gpl570['Gene Symbol'] = '\'' + gpl570['Gene Symbol']

for i in gpl570_files:
    f = pd.read_csv(os.path.join(sc.settings.figdir, i), index_col=0, comment='!')
    mean_centered = f.apply(lambda x: x-x.mean(), axis=1)
    probe_and_symbol = mean_centered.join(gpl570)
    mapped = probe_and_symbol.groupby('Gene Symbol').agg(np.mean)
    mapped.to_csv(os.path.join(sc.settings.figdir, 'genes', i))
#%%

gene_names = pd.read_csv('data/raw-data/Homo_sapiens.GRCh38.99.gtf', sep='\t', comment='#', header=None, dtype='str')
                                  
gene_names.columns = ['chr', 'annotation_src', 'feature_type', 'start', 'end', 'score', 'genomic_strand', 'genomic_phase', 'extra']
gene_names['gene_name'] = gene_names['extra'].str.extract(pat='gene_name "(.*?)";')
gene_names['gene_id'] = gene_names['extra'].str.extract(pat='gene_id "(.*?)";')
gene_names['transcript_type'] = gene_names['extra'].str.extract(pat='transcript_type "(.*?)";')
gene_names['gene_biotype'] =  gene_names['extra'].str.extract(pat='gene_biotype "(.*?)";')
gene_names['transcript_biotype'] =  gene_names['extra'].str.extract(pat='transcript_biotype "(.*?)";')
gene_names['gene_name'] = '\'' + gene_names['gene_name']
ensembl_annotation = gene_names[['gene_id', 'gene_name']].set_index('gene_id')

f = pd.read_csv(os.path.join(sc.settings.figdir, 'gse79209.csv'), index_col=0, comment='!')
mean_centered = f.apply(lambda x: x-x.mean(), axis=1)
probe_and_symbol = mean_centered.join(ensembl_annotation)
mapped = probe_and_symbol.groupby('gene_name').agg(np.mean)
mapped.to_csv(os.path.join(sc.settings.figdir, 'genes', 'GSE79209.csv'))

#%%
f = pd.read_csv(os.path.join(sc.settings.figdir, 'gse135188.csv'), index_col=0, comment='!')
mean_centered = f.apply(lambda x: x-x.mean(), axis=1)
probe_and_symbol = mean_centered.join(ensembl_annotation)
mapped = probe_and_symbol.groupby('gene_name').agg(np.mean)
mapped.to_csv(os.path.join(sc.settings.figdir, 'genes', 'GSE135188.csv'))
