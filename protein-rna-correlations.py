#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 20:27:33 2020

@author: Joan Smith
"""

#%%

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt

# Column A: Gene
# Column B: Pearson Corr Coeff
# Column C: # of cell lines

DIRECTORY = 'data/Data/RNA-protein correlation'
#%%
protein = pd.read_csv(os.path.join(DIRECTORY, 'protein_quant_current_normalized.csv'))
rna = pd.read_csv(os.path.join(DIRECTORY, 'CCLE_expression.csv'), index_col=0)
cell_line_key = pd.read_csv(os.path.join(DIRECTORY, 'sample_info (3).csv'), index_col=0)
#%%
rna = rna.join(cell_line_key[['CCLE_Name']], how='inner').set_index('CCLE_Name')
rna.columns =  [i.split(' ')[0] for i in rna.columns]
#%%
protein = protein.set_index('Gene_Symbol')
protein = protein.drop(['SW948_LARGE_INTESTINE_TenPx11', 'CAL120_BREAST_TenPx02', 'HCT15_LARGE_INTESTINE_TenPx30'], axis=1)
protein.columns = pd.Series(protein.columns).str.rsplit('_', n=1, expand=True)[0]
#%%
overlapping_cell_lines = set(protein.columns) & set(rna.index.values)

rna = rna.loc[overlapping_cell_lines]
protein = protein[overlapping_cell_lines].T
#%%
def corr(protein_series):
    df = pd.DataFrame([protein_series, rna[protein_series.name]]).T
    df = df.dropna(how='any')
    c = df.corr().iloc[0,1]
    count = df.shape[0]
    return pd.Series({'corr': c, 'count': count})

gene_overlap = set(rna.columns) & set(protein.columns)
protein = protein[gene_overlap]
rna = rna[gene_overlap]

corrs = protein.apply(corr)
#%%
corrs.T.to_csv(os.path.join(DIRECTORY, 'correlation_output.csv'))

