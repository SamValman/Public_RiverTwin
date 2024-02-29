# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 16:29:17 2024

@author: lgxsv2
"""
import rasterio as rio

def tifProjection(fn_in, fn_raw='x', out='x'):
    '''
    Adds geolocation data to imagery 
    **REQUIRES: temporary saving of image to be located.
    
    Parameters
    ----------
    fn_in : Str fn
        water mask p3 from River Twin Water Mask.Or other un located image
    fn_raw : Str fn
        raw tiff image used for the water mask.
    out : Str fn
        fn to save new located image. 

    Returns
    -------
    saves located image.

    '''
    
    
    with rio.open(fn_in) as m:
        im = m.read()
        profile_old = m.profile
    
    print('old profile:')
    print(profile_old)
    
    with rio.open(fn_raw) as i:
        profile = i.profile
        profile['count']=1
    print('new profile:')
    print(profile)
    
    # return profile

    # save file with new/old profile
    with rio.open(out, 'w', **profile) as l:
        l.write(im)
    print('complete')