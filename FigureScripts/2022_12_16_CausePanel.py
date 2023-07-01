# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 15:24:37 2022

@author: lgxsv2
"""

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from scipy.stats import boxcox
from scipy import stats
from scipy import optimize as opt

df = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\RiverTwin\2022_11_29_PaperFigures\Results\M20_Results.csv"
nir = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\RiverTwin\2022_11_29_PaperFigures\Results\NIR.csv"

def sortDf(df, nir):
    
    df = pd.read_csv(df)
    nir = pd.read_csv(nir)
    nir['f1'] = df['f1']
    nir['macro'] = df['macro']

    df = nir[nir['id'].str[:2]!='ZZ']
    df = df.dropna()
    df['width_rank'] = df['width'].rank(ascending = 1)
    df['nir_rank'] = df['range'].rank(ascending = 1)

    return df

df = sortDf(df, nir)


#%%
plt.close('all')
fig, ax = plt.subplots(1,2, sharey=True)








temp = df[df['class'] == 1 ]
f1 = temp.macro
w = np.log(temp.width)
NIR = (temp.IQR)
ax[0].scatter(w, f1, facecolors="navy", edgecolors='black')
ax[1].scatter(NIR, f1, facecolors="navy", edgecolors='black') 

temp = df[df['class'] == 2 ]
f1 = temp.macro
w = np.log(temp.width)
NIR = (temp.IQR)
ax[0].scatter(w, f1, facecolors="purple", edgecolors='black')
ax[1].scatter(NIR, f1, facecolors="purple", edgecolors='black')   

temp = df[df['class'] == 3 ]
f1 = temp.macro
w = np.log(temp.width)
NIR = (temp.IQR)
ax[0].scatter(w, f1, facecolors="darkred", edgecolors='black')
ax[1].scatter(NIR, f1, facecolors="darkred", edgecolors='black') 


ax[0].set_ylabel('F1 score')
ax[0].set_xlabel('Width (m)')
ax[1].set_xlabel('Near Infrared Inter quartile range')

# axis limits
plt.ylim(top=1.01)
ax[0].set_xticks([4,6,8])
ax[0].set_xticklabels([50,400,3000])
# [2.302585092994046,4.605170185988092,6.907755278982137,8.006367567650246], [10,100,1000,3000])

f1 = df.macro
w = np.log(df.width)
NIR = (df.IQR)
#statistics for width
x = w
y = f1
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
rsq1, pval1 = stats.pearsonr(x,y)

#add trendline to plot
ax[0].plot(x, p(x), color='black') # , width=2)
        
# r^2
        
# put in box 

# textstr = '\n'.join((
#     r'$Pearson-r=%.2f$' % (rsq, ),
#     r'$p-value=%.2f$' % (pval, )))
        
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# upper = y.max()-0.4
# plt.text(-2000, upper, textstr, fontsize=14,
# verticalalignment='top', bbox=props)

x = NIR
y = f1
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
rsq, pval = stats.pearsonr(x,y)

#add trendline to plot
ax[1].plot(x, p(x), color='black')
from matplotlib.patches import Patch
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

lg = [
      Line2D([0],[0], marker='o',color='w', markerfacecolor='darkred', markersize=10),
      Line2D([0],[0], marker='o',color='w', markerfacecolor='purple', markersize=10),
      Line2D([0],[0], marker='o',color='w', markerfacecolor='navy', markersize=10)]

ax[0].legend(lg, ['good', 'varied','poor'], loc='lower right', title='Biome group')

        






# ax[0].axvline(4.09)

plt.show()
#%%
plt.close('all')
fig, ax = plt.subplots(1,2, sharey=True)








temp = df[df['class'] == 1 ]
f1 = temp.macro
w = np.log(temp.width)
NIR = (temp.IQR)
ax[0].scatter(w, f1, facecolors="navy", edgecolors='black')
ax[1].scatter(NIR, f1, facecolors="navy", edgecolors='black') 

temp = df[df['class'] == 2 ]
f1 = temp.macro
w = np.log(temp.width)
NIR = (temp.IQR)
ax[0].scatter(w, f1, facecolors="purple", edgecolors='black')
ax[1].scatter(NIR, f1, facecolors="purple", edgecolors='black')   

temp = df[df['class'] == 3 ]
f1 = temp.macro
w = np.log(temp.width)
NIR = (temp.IQR)
ax[0].scatter(w, f1, facecolors="darkred", edgecolors='black')
ax[1].scatter(NIR, f1, facecolors="darkred", edgecolors='black') 


ax[0].set_ylabel('F1 score')
ax[0].set_xlabel('Width (m)')
ax[1].set_xlabel('Near Infrared Inter quartile range')

# axis limits
plt.ylim(top=1.01)
ax[0].set_xticks([4,6,8])
ax[0].set_xticklabels([50,400,3000])
# [2.302585092994046,4.605170185988092,6.907755278982137,8.006367567650246], [10,100,1000,3000])


################################################################################
f1 = df.macro
w = np.log(df.width)
NIR = (df.IQR)
#statistics for width
x = w
y = f1
#%%

def exponential(x, a, b):
    return a*np.exp(b*x)
def powerlaw(x,a,b):
    return a*np.power(x,b)
def logg(x,a,b):
    return a+b*np.log(x)


pars, cov = opt.curve_fit(f=logg, xdata=x, ydata=y, p0=[0, 0],
                          bounds=(-np.inf, np.inf))

# ax[0].set_xlim(x[0]-1)

# x_dummy = x.append(pd.Series([20])) # list(range(20,49))
# sigma_ab = np.sqrt(np.diagonal(cov))
# bound_upper = logg(sorted(x), *(pars + sigma_ab))
# bound_lower = logg(sorted(x), *(pars - sigma_ab))
# ax[0].fill_between(sorted(x), bound_lower, bound_upper,
#                  color = 'black', alpha = 0.15)


ax[0].plot(sorted(x), logg(sorted(x), *pars), color='black')
from sklearn.metrics import r2_score
r_squared_2 = r2_score(y, logg(x, *pars), multioutput='variance_weighted')

r1, p1 = stats.spearmanr(x, y)




###############################################################################
# r^2
        
# put in box 

# textstr = '\n'.join((
#     r'$Pearson-r=%.2f$' % (rsq, ),
#     r'$p-value=%.2f$' % (pval, )))
        
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# upper = y.max()-0.4
# plt.text(-2000, upper, textstr, fontsize=14,
# verticalalignment='top', bbox=props)

x = NIR
y = f1
z = np.polyfit(x, y, 1)
p = np.poly1d(z)
rsq, pval = stats.pearsonr(x,y)

#add trendline to plot
ax[1].plot(x, p(x), color='black')
from matplotlib.patches import Patch
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

lg = [
      Line2D([0],[0], marker='o',color='w', markerfacecolor='darkred', markersize=10),
      Line2D([0],[0], marker='o',color='w', markerfacecolor='purple', markersize=10),
      Line2D([0],[0], marker='o',color='w', markerfacecolor='navy', markersize=10)]

ax[0].legend(lg, ['good', 'varied','poor'], loc='lower right', title='Biome group')

        






# ax[0].axvline(4.09)

plt.show()


