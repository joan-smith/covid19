#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 15:43:34 2020

@author: Joan Smith

Human pack-years regressions
"""

#%%

import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf

gse76925_ace2 = pd. read_csv("data/Data/Human smoking - packyears/pack-years.csv")

#%%

gse76925_ace2.loc[gse76925_ace2['Race'] == 'C', 'Race'] = 0
gse76925_ace2.loc[gse76925_ace2['Race'] != 0, 'Race'] = 1

gse76925_ace2['Sex'] = gse76925_ace2['Sex'].replace({'M': 1, 'F': 0})
#%%

def regress(data, cols, plot=False):
    data = data.dropna(how='any', subset=cols)
    
    formula = 'ACE2 ~ ' + '+'.join(cols)
    model = smf.ols(formula, data=data)
    results = model.fit()
    
    if len(cols) == 1 and plot:
        plt.figure()
        plt.title(formula)
        plt.plot(data[cols[0]], data['ACE2'], 'o')
        plt.plot(data[cols[0]],  model.predict(results.params), color='red')
        plt.xlabel(cols[0])
        plt.ylabel('ACE2 expression')
        

    df = pd.DataFrame({'p': results.pvalues, 'se': results.bse, 'beta': results.params})
    df.index = formula + ': ' + df.index 
    print(df)
    return df
    
#%%
sex_regress = regress(gse76925_ace2, ['Sex'])
race_regress = regress(gse76925_ace2, ['Race'])
age_regress = regress(gse76925_ace2, ['Age'])
bmi_regress = regress(gse76925_ace2, ['BMI'])
copd_regress = regress(gse76925_ace2, ['COPD'])
packyears_regress = regress(gse76925_ace2, ['Packyears'])

pack_age_sex_regress = regress(gse76925_ace2, ['Packyears', 'Age', 'Sex'])
pack_age_sex_bmi_regress = regress(gse76925_ace2, ['Packyears', 'Age', 'Sex', 'BMI'])
pack_age_sex_bmi_copd_regress = regress(gse76925_ace2, ['Packyears', 'Age', 'Sex', 'BMI', 'COPD'])

pack_age_sex_bmi_race_regress = regress(gse76925_ace2, ['Packyears', 'Age', 'Sex', 'BMI', 'Race'])
pack_age_bmi_race_regress = regress(gse76925_ace2, ['Packyears', 'Age', 'BMI', 'Race'])


output = pd.concat((sex_regress, race_regress, age_regress, bmi_regress, copd_regress, packyears_regress,
                    pack_age_sex_regress, pack_age_sex_bmi_regress, pack_age_sex_bmi_copd_regress,
                    pack_age_sex_bmi_race_regress, pack_age_bmi_race_regress))
output.to_csv('data/gse131391_regressions.csv')

