# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 16:10:39 2023

@author: lgxsv2
"""
# *** check river width in log m or whatever 
# *** legend
# *** change instrument order

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from scipy.stats import boxcox
from scipy import stats
from scipy import optimize as opt
from matplotlib.patches import Patch
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms


#%% Dataset sorting
########################################################################
# Width and NIR
df = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\RiverTwin\2022_11_29_PaperFigures\Results\M20_Results.csv"
nir = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\RiverTwin\2022_11_29_PaperFigures\Results\NIR.csv"

def sortDf(df, nir):
    
    df = pd.read_csv(df)
    nir = pd.read_csv(nir)
    nir['f1'] = df['f1']
    nir['macro'] = df['macro']

    df = nir[nir['id'].str[:2]!='ZZ']
    df = df.dropna()
    df = df.drop(12)

    df['width_rank'] = df['width'].rank(ascending = 1)
    df['nir_rank'] = df['IQRb'].rank(ascending = 1)

    return df

nir_width_df = sortDf(df, nir)


########################################################################
# Meta and anthro
fn = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\RiverTwin\2022_11_29_PaperFigures\Results\m20_withmeta.csv"
#lg==larger df
lg = pd.read_csv(fn)
lg = lg.drop(12)
lg['class']=nir_width_df['class']

########################################################################
# months
fn = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\RiverTwin\2022_11_29_PaperFigures\Results\M20_Results.csv"
def seasonSort(fn):
    df = pd.read_csv(fn)
    df = df[df['id'].str[:2]!='ZZ']
    months = []
    for index, row in df.iterrows():
        d = row.date
        m = int(d[4:6])
        
        if row.season == 'N':
            m = int(m)
        elif m >=6:
            m = m-5
        else:
            m = m+7
        # months since mid winter
        months.append(m)
    df['months']=months
    df = df.drop(12)
    

    return df
sea = seasonSort(fn) 
sea['class']=nir_width_df['class']
 

#%% the figure 

#close early checks
plt.close('all')


fig, axes = plt.subplots(3, 3, sharey=True, figsize=(10, 10)) # , figsize=(10, 10))
# set titles 
xlab = ["River Width (log m)","River Type",  "Anthropogenic Features",
         "Distance from Equator (degrees)", "Azimuth (degrees)", "Months from Solstice",
         "Satellite Model","Near Infrared Range (5-95%)", "Confidence of Clear Image (%)"]

# Loop  and set titles
for i, ax in enumerate(axes.flat):
    # ax.scatter(x, y, alpha=0.5)
    # ax.set_ylabel('F1 score')
    ax.set_xlabel(xlab[i])
fig.text(0.001, 0.5, 'F1 score', va='center', rotation='vertical')


###################################################################
#River Width (log m) and NIR
# df will be another temporary var that will be overwritten with the other dataset
df = nir_width_df
plt.ylim(bottom = 0.4, top=1.04)


temp = df[df['class'] == 1 ]
f1 = temp.macro
w = np.log(temp.width)
NIR = (temp.IQR)
axes[0][0].scatter(w, f1, facecolors="navy", edgecolors='black')
axes[2][1].scatter(NIR, f1, facecolors="navy", edgecolors='black') 

# trendline
x = np.log(df.width)
y = df.macro
# def logg(x,a,b):
#     return a+b*np.log(x)
# pars, cov = opt.curve_fit(f=logg, xdata=x, ydata=y, p0=[0, 0],
#                           bounds=(-np.inf, np.inf))
# axes[0][0].plot(sorted(x), logg(sorted(x), *pars), color='black')
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
axes[0][0].plot(x, p(x), color='black') # , width=2)

# make log
axes[0][0].set_xticks([4,6,8])
axes[0][0].set_xticklabels([50,400,3000])


temp = df[df['class'] == 2 ]
f1 = temp.macro
w = np.log(temp.width)
NIR = (temp.IQR)
axes[0][0].scatter(w, f1, facecolors="purple", edgecolors='black')
axes[2][1].scatter(NIR, f1, facecolors="purple", edgecolors='black')   

temp = df[df['class'] == 3 ]
f1 = temp.macro
w = np.log(temp.width)
NIR = (temp.IQR)
axes[0][0].scatter(w, f1, facecolors="darkred", edgecolors='black')
axes[2][1].scatter(NIR, f1, facecolors="darkred", edgecolors='black') 

#%% the scatter parts of lg 
df = lg
order_lg = ["PS2", "PS2.SD", "PSB.SD"]
custom_labels_lg = ["Dove Classic", "Dove-R", "SuperDove"]

# Reorder the labels for axes[2][0]

invisible_x = ["PS2", "PS2.SD", "PSB.SD"]
invisible_y = [0.8,0.8,0.8]
axes[2][0].scatter(invisible_x, invisible_y, marker='o', color='none')
axes[2][0].set_xticks(np.arange(0,3 ))
axes[2][0].set_xticklabels(custom_labels_lg)

#divide into classes
temp = df[df['class'] == 1 ]
f1 = temp.f1
dfe = abs(temp.lat)
a = temp.azimuth
CI = temp.clear_confidence_percent
Instrument = temp.instrument


# good
axes[1][0].scatter(dfe, f1, facecolors='navy', edgecolors='black')
axes[1][1].scatter(a, f1, facecolors='navy', edgecolors='black')
axes[2][2].scatter(CI, f1, facecolors='navy', edgecolors='black')
axes[2][0].scatter(Instrument, f1, facecolors='navy', edgecolors='black')


temp = df[df['class'] == 2 ]
f1 = temp.f1
dfe = abs(temp.lat)
a = temp.azimuth
CI = temp.clear_confidence_percent
Instrument = temp.instrument
#medium 
axes[1][0].scatter(dfe, f1, facecolors="purple", edgecolors='black')
axes[1][1].scatter(a, f1, facecolors="purple", edgecolors='black')
axes[2][2].scatter(CI, f1, facecolors="purple", edgecolors='black')
axes[2][0].scatter(Instrument, f1, facecolors="purple", edgecolors='black')



temp = df[df['class'] == 3 ]
f1 = temp.f1
dfe = abs(temp.lat)
a = temp.azimuth
CI = temp.clear_confidence_percent
Instrument = temp.instrument
#poor
axes[1][0].scatter(dfe, f1, facecolors="darkred", edgecolors='black')
axes[1][1].scatter(a, f1, facecolors="darkred", edgecolors='black')
axes[2][2].scatter(CI, f1, facecolors="darkred", edgecolors='black')
axes[2][0].scatter(Instrument, f1, facecolors="darkred", edgecolors='black')

#%% Months/seasons
df = sea
#divide into classes
temp = df[df['class'] == 1 ]
f1 = temp.macro
m = temp.months
axes[1][2].scatter(m, f1, facecolors='navy', edgecolors='black')
# 
temp = df[df['class'] == 2 ]
f1 = temp.macro
m = temp.months
axes[1][2].scatter(m, f1, facecolors="purple", edgecolors='black')
#
temp = df[df['class'] == 3 ]
f1 = temp.macro
m = temp.months
axes[1][2].scatter(m, f1, facecolors="darkred", edgecolors='black')


#%%
# Anthropogenic # create dfs for each category
no_anthro = lg[lg['tot']==0].iloc[:,1]
urban = lg[lg['urban']==1].iloc[:,1]
fields = lg[lg['fields']==1].iloc[:,1]
in_channel = lg[lg['in']==1].iloc[:,1]
alls = lg.iloc[:,1]

listDfs = [ no_anthro, urban, fields,in_channel]
labels = ['None \n (n=19)', 'Urbanisation \n (n=14)', 'Agriculture \n (n=11)', 'In-Channel Features \n (n=9)']
vps = axes[0][2].violinplot(listDfs)
for pc in vps['bodies']:
    pc.set_color('grey')
    pc.set_edgecolor('grey')
for partname in ('cbars','cmins','cmaxes'):
    vp = vps[partname]
    vp.set_edgecolor('black')
    vp.set_linewidth(1)
    
# rename x axis ticks 
axes[0][2].set_xticks(np.arange(1, len(labels) + 1))
axes[0][2].set_xticklabels(labels, rotation=45)
#%% now where the fuck is the river type data 
meandering = sea[sea['morpho_option1']=='Meandering']['macro']
braided = sea[sea['morpho_option1']=='Braided']['macro']
anastomosed = sea[sea['morpho_option1']=='Anastomosed']['macro']
channelised = sea[sea['morpho_option1']=='Urban']['macro']

listDfs = [meandering, braided, anastomosed, channelised]
labels = ['Meandering \n (n=24)', 'Braided \n (n=4)', 'Anastomosed \n (n=3)', 'Channelised \n (n=5)']

vps = axes[0][1].violinplot(listDfs)
for pc in vps['bodies']:
    pc.set_color('grey')
    pc.set_edgecolor('grey')
for partname in ('cbars','cmins','cmaxes'):
    vp = vps[partname]
    vp.set_edgecolor('black')
    vp.set_linewidth(1)
axes[0][1].set_xticks(np.arange(1, len(labels) + 1))
axes[0][1].set_xticklabels(labels, rotation=45)


#%%
# legend 



lg = [
      Line2D([0],[0], marker='o',color='w', markerfacecolor='darkred', markersize=10),
      Line2D([0],[0], marker='o',color='w', markerfacecolor='purple', markersize=10),
      Line2D([0],[0], marker='o',color='w', markerfacecolor='navy', markersize=10)]

axes[0][0].legend(lg, ['good', 'varied','poor'], loc='lower right', title='Biome group')

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

# Iterate through the subplots and add labels to the left of the figure
for i, ax in enumerate(axes.ravel()):
    ax.annotate(labels[i], xy=(-0.07, 0.85), xycoords='axes fraction', fontsize=12, weight='bold')

    # if labels[i] in ['d','e','f']:
    #     ax.annotate(labels[i], xy=(0.01, 0.8), xycoords='axes fraction', fontsize=12, weight='bold')
    # elif labels[i] in ['g','h','i']:
    #     ax.annotate(labels[i], xy=(-0.1, 0.9), xycoords='axes fraction', fontsize=12, weight='bold')

    # else:
    #     ax.annotate(labels[i], xy=(0.01, 0.9), xycoords='axes fraction', fontsize=12, weight='bold')



plt.subplots_adjust(wspace=0.2, hspace=0.2)





plt.tight_layout()
fp = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\RiverTwin\2022_11_29_PaperFigures\2023_01\Figure7_CausePanel\2023_10_20_0F7.jpeg"
plt.savefig(fp, dpi=800)


raise SystemExit()
