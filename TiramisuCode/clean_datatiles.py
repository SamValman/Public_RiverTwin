# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 09:43:08 2023

@author: lgxsv2
"""

import skimage.io as IO
import matplotlib.pyplot as plt
import glob 
import os 
import numpy as np


fp_validate = r'C:\Users\lgxsv2\TrainingData\ZZ_Tiramasu\VA\*.jpg'
fp_train = r'C:\Users\lgxsv2\TrainingData\ZZ_Tiramasu\TR\*.jpg'

def remove_corner_pixels(fp):
    '''
    leave some room for ambiguity with >20 required

    Parameters
    ----------
    fp : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    tick = 0
    for i in glob.glob(fp):
        t = IO.imread(i)
        azp = np.all(t == 0, axis=-1)
        azp = np.count_nonzero(azp)
        if azp>20:
            os.remove(i)
            tick+=1
            root, ext = os.path.splitext(i)
            li = root + ".png"
            os.remove(li)
    print(tick, ' removed')

    #17214
    # total 44,700
#%%
remove_corner_pixels(fp_train)
remove_corner_pixels(fp_validate)
# fp_validate = r'C:\Users\lgxsv2\TrainingData\ZZ_Tiramasu\validate_nc3\*.jpg'
# for i in glob.glob(fp_validate):
#     tile = IO.imread(i)
#     break
