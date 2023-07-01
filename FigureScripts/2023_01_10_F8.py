# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 10:14:24 2023

@author: lgxsv2
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy import stats
#%%

post = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\RiverTwin\2022_11_29_PaperFigures\Results\Comb_FTOnly.csv"
pre = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\RiverTwin\2022_11_29_PaperFigures\Results\OttawaFTPre_results.csv"
#pre
a1 = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\RiverTwin\2022_11_29_PaperFigures\Results\M20_Results.csv"
#post
a2 = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\RiverTwin\2022_11_29_PaperFigures\Results\Comb_FTALL.csv"


#put F1 together
post = pd.read_csv(post)
pre = pd.read_csv(pre)

df = pd.DataFrame()

df['id']=pre['id']
df['pre']=pre['macro']
df['post']=post['macro']



#put all together 
a1 = pd.read_csv(a1)
a2 = pd.read_csv(a2)
al = pd.DataFrame()
al['id']=a1['id']
al['pre']=a1['macro']
al['post']=a2['macro']
al = al[:37]

x = df.pre
y = df.post

plt.figure()
plt.scatter(x, y, c='black')
plt.scatter(al['pre'], al.post)
plt.ylabel('F1 value after fine tuning')
plt.xlabel('F1 original model')


# z = np.polyfit(x, y, 1)
# p = np.poly1d(z)
# #add trendline to plot
# plt.plot(x, p(x), color='black') # , width=2)

# # r^2
# rsq, pval = stats.pearsonr(x,y)

# # put in box 

# textstr = '\n'.join((
# r'$Pearson-r=%.2f$' % (rsq, ),
# r'$p-value=%.2f$' % (pval, )))

# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# upper = y.max()-0.001
# plt.text(0.75, upper, textstr, fontsize=14, verticalalignment='top', bbox=props)

#%%

fn = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\RiverTwin\2022_11_29_PaperFigures\Results\M20_results.csv"

df = pd.read_csv(fn)
df = df[['id', 'macro']][:-4]
print(len(df))

for i in [0.99, 0.9, 0.85, 0.8, 0.7, 0.6]:
    a = df[df['macro']>=i]
    print(i)
    print(len(a))