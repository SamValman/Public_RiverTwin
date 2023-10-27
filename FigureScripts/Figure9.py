# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:12:12 2023

@author: lgxsv2
"""

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd

post = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\RiverTwin\ZZ_results\FTOttawa_additional5\results.csv"
pre = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\RiverTwin\ZZ_results\PRE_FTOttawa_additional5\results.csv"
pre = pd.read_csv(pre)
post = pd.read_csv(post)
prea = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\RiverTwin\2022_11_29_PaperFigures\Results\M20_Results.csv"
posta = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\RiverTwin\2022_11_29_PaperFigures\Results\Comb_FTALL.csv"
prea = pd.read_csv(prea)
posta = pd.read_csv(posta)
#%%
labels = list(pre.id)
ndf = pd.DataFrame()
ndf['pre'] = pre['macro']
ndf['post'] = post['macro']
ndf['dif'] = ndf['post']-ndf['pre']
ndf = ndf.sort_values('pre', ascending=False)


pr = ndf.pre
po = ndf.post

#%%

plt.close('all')

x = np.arange(len(labels))  # the label locations
width = 0.3  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, pr, 0.3, label='M20', edgecolor='black',facecolor='none', hatch='/////' )
rects2 = ax.bar(x + width/2, po, 0.3, label='Fine Tuned', edgecolor='black', facecolor='darkgrey' )

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('F1 Score')
ax.set_xlabel('Fine tune test images \n sorted by M20 F1 score')# difference in F1 score

ax.set_ylim([0.4,1])
ax.set_xticks([])
# ax.set_xticklabels(range(1, len(labels)+1))
ax.legend(bbox_to_anchor=(1, 1))

# ax.bar_label(rects1, padding=3)
# ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()

#%% try histogram

dif = po - pr
dif = posta.macro - prea.macro

plt.figure()
plt.hist(dif)
plt.show()























