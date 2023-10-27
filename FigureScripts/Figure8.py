# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 12:04:03 2023

@author: lgxsv2
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 14:27:20 2023

@author: lgxsv2
"""

import pandas as pd
import numpy as np
import skimage.io as IO
import matplotlib.pyplot as plt
import rasterio as rio


#%%

def tif_projection(fn_wm, fn_raw='x', out='x', verbose=False):
    '''
    For some reason returns water as 1 and land as 0
    Adds geolocation data to P3 imagery 
    
    Parameters
    ----------
    fn_wm : Str fn
        water mask p3 from River Twin Water Mask.
    fn_raw : Str fn
        raw tiff image used for the water mask.
    out : Str fn
        fn to save new located image. 

    Returns
    -------
    saves located image.

    '''
    
    
    with rio.open(fn_wm) as m:
        im = m.read(1)
        profile_old = m.profile
    if verbose:
        print('old profile:')
        print(profile_old)
    
    with rio.open(fn_raw) as i:
        profile = i.profile
        label = i.read(1)
        profile['count']=1
    if verbose:
        print('new profile:')
        print(profile)
    
    # return profile
    label[label==2]=0
    im = np.not_equal(im, label).astype(int)

    # save file with new/old profile
    with rio.open(out, 'w', **profile) as l:
        l.write(im,1)
    if verbose:
        print('complete')

#%% testing and running this function

# fn = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\11_Canada\temp_pythonFiles\1_A3.tif"
# fnr = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\11_Canada\temp_pythonFiles\1_A3_psscene_analytic_sr_udm2\files\20211105_111447_1054_3B_AnalyticMS_SR_harmonized_clip.tif"
# im = IO.imread(fn)

# out_fn  = r"C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\11_Canada\temp_pythonFiles\1_A3_wm.tif"
# tif_projection(fn, fnr, out_fn)
import glob 
import os 

wmPaths = r'E:\forFigure8\noCSVModelOutput\*.tif'
rawPath = r'E:\forFigure8'
outRoot = r"E:\forFigure8\csvmodelOut"


for i in glob.glob(wmPaths):
    
    print(i.split('\\')[-1])
    split = i.split('\\')[-1]
    
    fn = i 
    fnr = os.path.join(rawPath, split)
    out_fn = os.path.join(outRoot, split)
    tif_projection(fn, fnr, out_fn)

# im = IO.imread(out_fn)

