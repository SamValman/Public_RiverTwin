# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 13:56:35 2022

@author: lgxsv2
"""

import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 

#%% Sort out the models 
# What models:
    # 10, 20, 30, big,  ANN, OTSU

m10 = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\RiverTwin\2022_11_29_PaperFigures\Results\M10_Results.csv"
m20 = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\RiverTwin\2022_11_29_PaperFigures\Results\M20_Results.csv"
m32 = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\RiverTwin\2022_11_29_PaperFigures\Results\M32_Results.csv"

otsu = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\RiverTwin\2022_11_29_PaperFigures\Results\otsu_Results.csv" # move otsu results to where needed
ANN = r""
VGG16 = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\RiverTwin\2022_11_29_PaperFigures\Results\VGG16_results.csv"
VGG16 = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\RiverTwin\ZZ_results\NewVGG16results.csv"
def sortDf(df):
    
    df = pd.read_csv(df)
    df = df[df['id'].str[:2]!='ZZ']
    df = df['macro']
    print(df.mean())
    return df

# temp changes
m10 = sortDf(m10)
m20 = sortDf(m20)
m32 = sortDf(m32)

otsu = sortDf(otsu)
# ANN = sortDf(ANN)
VGG16 = sortDf(VGG16)

# ANN = m10






#%%


#%%
def violinModels(listDfs):

    
    fig, ax = plt.subplots(1,1)
    
    
    #colours
    vps = ax.violinplot(listDfs)
    for pc in vps['bodies']:
        pc.set_color('grey')
        pc.set_edgecolor('grey')
    for partname in ('cbars','cmins','cmaxes'):
        vp = vps[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)
   
    
   #labels 
    ax.set_xlabel('Classification Models')
    ax.set_ylabel('F1 scores')
    
    places = [1,2,3,4,5] #,6]
    bname = ['Tile size \n 10','Tile size \n 20','Tile size \n 32', 'Very Deep \n Neural Net',  'OTSU'] #'Multilayer \n Peceptron',
    ax.set_xticks(places)#, labels=bname)
    ax.set_xticklabels(bname, rotation=90)#, rotation=90)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    plt.tight_layout()

    
    
    plt.show()
    
    

#%%
a = violinModels([m10, m20, m32, VGG16, otsu])

