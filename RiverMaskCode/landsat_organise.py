# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 15:15:49 2022

@author: lgxsv2
"""

import skimage.io as IO
import numpy as np

def LandsatOrgangise(fn=r"D:\FineTuning\LS_test\LC09_L2SP_138043_20220130_20220201_02_T1_SR_", out_fn='complete.tif'):

    B = fn+'B2.tif'
    G = fn+'B3.tif'
    R = fn+'B4.tif'
    NIR = fn+'B5.tif'

    
    B = IO.imread(B)
    G = IO.imread(G)
    R = IO.imread(R)
    NIR = IO.imread(NIR)
    
    t = np.stack([B,G, R, NIR], axis=-1)
    
    IO.imsave(out_fn, t, check_contrast=False)


