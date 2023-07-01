# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 14:39:54 2022

@author: lgxsv2
"""

import os
import glob
import pandas as pd


os.chdir(r'D:\Code\RiverTwin\2022_12_08_unPacked')
# os.chdir(r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\RiverTwin\2022_12_08_unPacked')

from testSuccess import testSuccess
from otsuPackage import otsuPackage

#%%

imPath = r"D:\Training_data\test\*.tif" 
# imPath = r'D:\FineTuning\test'
output = r'D:\Code\RiverTwin\ZZ_results\otsu'
names = []
f1s = []
mac=[]  
#%%


for i in glob.glob(imPath):
    
    im = otsuPackage(i, output)
    
    im_name = i.split('\\')[-1]
 
    fp = os.path.join(output, 'o1')
    fp = os.path.join(fp,im_name)
    
    r = testSuccess(image_fp=fp, output=output)
    
    
    names.append(im_name[:-4])
    f1s.append(r['f1-score']['weighted avg'])
    mac.append(r['f1-score']['macro avg'])
    
#%%
      
df = pd.DataFrame({'id':names,'f1':f1s, 'macro':mac })

fn = os.path.join(output, 'results.csv')
df.to_csv(fn)