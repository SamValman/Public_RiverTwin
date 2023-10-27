# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:44:06 2023

@author: lgxsv2
"""

###############################################################################
# TEST
# ###############################################################################
import os
import skimage.io as io 
import gc
import glob
import pandas as pd
#%%
os.chdir(r'D:\Code\RiverTwin\2022_12_08_unPacked')
gc.collect()

from testSuccess import testSuccess

def F1Score(folder, outfn):
    
    names = []
    f1s = []
    for i in glob.glob(folder):
        
        # P3 = io.imread(i)
        try:
            r = testSuccess(image_fp=i, time=False, output=False, display_image=False,
                            save_image=False, fcn=True)
        except Exception as error:
            print("HERE:", error)
            print(i.split('\\')[-1][:-4])
            print("HERE:", error)
            continue
        
        names.append(i.split('\\')[-1][:-4])
        f1s.append(r['f1-score']['macro avg'])
    
    df = pd.DataFrame({'id':names, 'f1':f1s})

    df.to_csv(outfn)
    return df

#%%
output_name = r"D:\Code\RiverTwin\2022_11_29_PaperFigures\Results\tira.csv"
folder = r'D:\Code\RiverTwin\ZZ_results\tira_2\*.tif'

F1Score(folder, output_name)