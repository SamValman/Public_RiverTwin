# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 09:58:41 2022

@author: lgxsv2
"""
import matplotlib.pyplot as plt 
import pandas as pd
import numpy as np 
from matplotlib.patches import Patch

df = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\RiverTwin\2022_11_29_PaperFigures\Results\M20_Results.csv"

df = pd.read_csv(df)

plt.close('all')
def violinBiome(df):
    df = df[df['id'].str[:2]!='ZZ']

    df = df[['macro','biome']]
    
    task = []
    bname = []
    for i in df['biome'].unique():
        t = df[df['biome']==i]
        t = t['macro']
        task.append( t)
        bname.append(i)
    
    fig, ax = plt.subplots(1,1)
    
    counter = 0
    
    #colours
    vps = ax.violinplot(task)
    for pc in vps['bodies']:
        counter+=1
        if counter in [1,6,7,8]:
            pc.set_color('navy')
            pc.set_edgecolor('navy')
           
        elif counter in [2,3,4,12]:
            pc.set_color('purple')
            pc.set_edgecolor('purple')
        else:
            pc.set_color('darkred')
            pc.set_edgecolor('darkred')
    for partname in ('cbars','cmins','cmaxes'):
        vp = vps[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1)
   
    
   #labels 
    ax.set_xlabel('Ecoregions')
    ax.set_ylabel('F1 scores')
    
    places = [1,2,3,4,5,6,7,8,9,10,11,12]
    bname = ['Montane \n grasslands',
     'Tundra',
     'Mediterranean',
     'Desert',
     'Tropical \n moist broadleaf',
     'Tropical \n dry broadleaf',
     'Temperate \n broadleaf',
     'Temperate \n coniferous',
     'Taiga',
     'Tropical \n grasslands',
     'Temperate \n grasslands',
     'Flooded \n grasslands \n and savanas']
    ax.set_xticks(places)#, labels=bname)
    ax.set_xticklabels(bname, rotation=90)#, rotation=90)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    
    lg = [
          Patch(facecolor='darkred', edgecolor='darkred'),
          Patch(facecolor='purple', edgecolor='purple'),
          Patch(facecolor='navy', edgecolor='navy')
         ]
    plt.legend(lg, ['good', 'varied','poor'], loc='lower right')

    plt.tight_layout()

    
    
    plt.show()
    
    
    return vps

a = violinBiome(df)
