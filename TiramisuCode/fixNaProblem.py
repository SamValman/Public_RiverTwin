# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:07:20 2023

@author: lgxsv2
"""

import skimage.io as IO
import numpy as np 
import os 
import glob 

path = r"D:\Training_data\desert\f\*.tif"
for i in glob.glob(path):
    print(i)
    try:
        im = IO.imread(i)
        im[im == -999] = 0
        im[im == 2] = 0
        name = 'Mask_'+(i.split('\\')[-1])
        ofn = os.path.join(r"D:\Training_data\desert\doodless", name)
        IO.imsave(ofn, im, check_contrast=False)
    except IndexError:
        print('ERROR:  ',i)

