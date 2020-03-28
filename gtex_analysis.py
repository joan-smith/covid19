#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 19:26:43 2020

@author: Joan Smith
"""

#%%
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

gtex = pd.read_csv("data/raw-data/GTEX/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_tpm.gct", sep="\t", header=2, index_col=1)
gtex = gtex.drop('Name', axis=1).astype(float)
attributes = pd.read_csv("data/raw-data/GTEX/GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt", sep='\t', index_col=0, dtype=None)
phenotypes = pd.read_csv("data/raw-data/GTEX/GTEx_Analysis_v8_Annotations_SubjectPhenotypesDS.txt", sep='\t', index_col=0)
phenotypes[['SEX']] = phenotypes[['SEX']].replace(2, 'F')
phenotypes[['SEX']] = phenotypes[['SEX']].replace(1, 'M')

#%%
def set_labels(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)

#%%
def plot_by_pheno(ace2_tissue_w_pheno, col, tissue):
    fig, ax = plt.subplots()
    plt.title(tissue + '\nACE2 by ' + col.lower())
    pheno = [(i[0], i[1].ACE2.dropna()) for i in ace2_tissue_w_pheno.groupby(col)]
    labels, split = zip(*pheno)
    ax.violinplot(split, showmeans=True)
    set_labels(ax, labels)
    
    ace2_tissue_w_pheno.reset_index().pivot_table(columns='SEX', values='ACE2', index='index').to_csv('gtex_' + tissue.lower() + '_sex.csv')
    ace2_tissue_w_pheno.reset_index().pivot_table(columns='AGE', values='ACE2', index='index').to_csv('gtex_' + tissue.lower() + '_age.csv')

    
#%% LUNG
ace2 = gtex.loc['ACE2']
samples_w_lung = attributes[attributes['SMTS'] == 'Lung'].index
ace2_lung = gtex.loc['ACE2'][samples_w_lung].astype(float)
ace2_lung.index = ace2_lung.index.str[0:10]

ace2_lung_w_pheno = phenotypes.join(ace2_lung, how='inner')
plot_by_pheno(ace2_lung_w_pheno, 'AGE', 'Lung')
plot_by_pheno(ace2_lung_w_pheno, 'SEX', 'Lung')

#%% Esophogeal Mucosa

samples_w_esoph_muc = attributes[attributes['SMTSD'] == 'Esophagus - Mucosa'].index
ace2_esoph = gtex.loc['ACE2'][samples_w_esoph_muc].astype(float)
ace2_esoph.index = ace2_esoph.index.str[0:10]

ace2_esoph_w_pheno = phenotypes.join(ace2_esoph, how='inner')
plot_by_pheno(ace2_esoph_w_pheno, 'AGE', 'Esophagus - Mucosa')
plot_by_pheno(ace2_esoph_w_pheno, 'SEX', 'Esophagus - Mucosa')

#%% Salivary

samples_w_salivary = attributes[attributes['SMTS'] == 'Salivary Gland'].index
ace2_sal = gtex.loc['ACE2'][samples_w_salivary].astype(float)
ace2_sal.index = ace2_sal.index.str[0:10]

ace2_sal_w_pheno = phenotypes.join(ace2_sal, how='inner')
plot_by_pheno(ace2_sal_w_pheno, 'AGE', 'Salivary Gland')
plot_by_pheno(ace2_sal_w_pheno, 'SEX', 'Salivary Gland')

#%% All Tissue

#%% Plot All Tissue

ace2_tissue = gtex.loc[['ACE2']].T.join(attributes)
ace2_tissue['ACE2'] = np.log2(ace2_tissue['ACE2'] + 1)

fig, ax = plt.subplots()
plt.title('ACE2 by tissue')

order = ace2_tissue.groupby('SMTS')['ACE2'].apply(np.mean).sort_values()
print(order)
g = {i[0]: i[1].ACE2.dropna() for i in ace2_tissue.groupby('SMTS')}
ordered_g= [(k, g[k]) for k in order.index]
labels, split = zip(*ordered_g)
ax.violinplot(split, showmeans=True)
set_labels(ax, labels)
plt.xticks(rotation=45)
plt.show()

#%% Export Data for All Tissue

ace2_tissue.reset_index().pivot_table(columns='SMTS', values='ACE2', index='index').to_csv('ace2_by_tissue.csv')

