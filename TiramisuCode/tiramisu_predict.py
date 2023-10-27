# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 12:24:43 2023

@author: lgxsv2
"""

# this needs to read in the files and predict them 
# at which point I will check them here and then if this is working
# we will put a function that scrolls through them.

import tensorflow as tf
from skimage import io, transform
import numpy as np
from focal_loss import SparseCategoricalFocalLoss
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from skimage.util import view_as_windows
import gc
import glob
import os 
# from osgeo import gdal_array


from tensorflow.keras import utils

#%%


# can use skimage io
def getModel(fcn_fp=r'D:\Code\RiverTwin\ZZ_Models\tira_MO\model'):
    return tf.keras.models.load_model(fcn_fp) 


def Normaliseto8bit(image):
    if image.shape[2]>3:
        image=image[:,:,1:]
    image = image.astype(np.float32)

    image = ((image/6000)*255)
    image = np.clip(image, 0, 255)
    image = image.astype(np.uint8)
    return image


def normalize(input_tiles):
    input_tiles = input_tiles.astype(np.float32)
    input_tiles = input_tiles / 127.5
    input_tiles = input_tiles - 1.0
    return input_tiles

def tileImage(im, tile_size=224):
    '''
    '''
    # lab = io.imread(r"D:\Training_data\test_labels\label_SAC\5_C3.tif")
    
    # remove edge cells from image so it can be tiled precisely
    height, length = im.shape[:2]
    height, length = int(height//tile_size), int(length//tile_size)
    # get the new max axis lengths
    y_axis, x_axis = (height*tile_size), (length*tile_size)
    
    im = im[:y_axis, :x_axis,:] # make them divisable by tile size used later
    # lab = lab [:y_axis, :x_axis]
    # empty list to be appended to
    # im_ls = []

    # for each tile on each axis 
    for m in range(height):
        for n in range(length):
            #multiply by size
            
            temp_m = (m+1)*tile_size 
            temp_n = (n+1)*tile_size
            # cut out the tile
            band_tile = im[(temp_m-tile_size): temp_m, (temp_n-tile_size):temp_n, :]
            # lab_tile = lab[(temp_m-tile_size): temp_m, (temp_n-tile_size):temp_n]
            # lab_tile = np.uint8(lab_tile)
            # lab_tile[lab_tile==2]=0
            # if lab_tile.max()>1:
                # print('ERRRROROORROR')
            bandPath = r'C:\Users\lgxsv2\TrainingData\ZZ_Tiramasu\p'
            bandPath = os.path.join(bandPath, ('tile_'+str(m)+'_'+str(n)+'_.jpg'))
            io.imsave(bandPath, band_tile, check_contrast=False, quality=95)
            # labPath = os.path.join(r'C:\Users\lgxsv2\TrainingData\ZZ_Tiramasu\p', ('tile_'+str(m)+'_'+str(n)+'_.png'))
            # io.imsave(labPath, lab_tile, check_contrast=False)



    gc.collect()
    return (y_axis, x_axis)  

def saveClear(im, fp):
    split = os.path.split(fp)[-1]
    fp = os.path.join(r'D:\Code\RiverTwin\ZZ_results\tira_2', split)
    io.imsave(fp, im, check_contrast=False)
    ls = [r'C:\Users\lgxsv2\TrainingData\ZZ_Tiramasu\p', r'C:\Users\lgxsv2\TrainingData\ZZ_Tiramasu\temp_tile_predictions']
    for direct in ls:
        [os.remove(entry.path) for entry in os.scandir(direct) if entry.is_file()]
    print('finished: ', str(split))


#%%
def predict(pathToImage, model):
    im = io.imread(pathToImage)
    
    # im = gdal_array.LoadFile(pathToImage)
    # im=np.flipud(np.rollaxis(im, 0, 3))
    
    im = Normaliseto8bit(im)
    axes = tileImage(im)    

    tilePath = r'C:\Users\lgxsv2\TrainingData\ZZ_Tiramasu\p\*.jpg'

    finalOutput = np.empty(axes, dtype=np.uint8)

    for i in glob.glob(tilePath): 
        one = io.imread(i)
        
        one = np.expand_dims(one, axis=0)
        p3 = model.predict(one)
        p3 = np.argmax(p3, axis=-1)
        p3 = np.squeeze(p3, axis=0)
        
        #For saving 
        p3 = np.uint8(p3)
        if p3.max()>1:
            print('errororor')
        # save this here 
        # lab = io.imread((i[:-3]+'png'))
        outPath= r'C:\Users\lgxsv2\TrainingData\ZZ_Tiramasu\temp_tile_predictions'
        split = os.path.split(i)[-1]
        outPath = os.path.join(outPath, split)
        io.imsave(outPath, p3, check_contrast=False)
        
        #extracting
        mn = os.path.split(i)[-1].split('_')
        m, n = int(mn[1]), int(mn[2])
        # reasigning
        m = (m+1)*224
        n = (n+1)*224
        
        finalOutput[(m-224): m, (n-224):n] = p3
    

        
        
   


        
        
    return finalOutput




#%%
model = getModel()

for i in glob.glob(r'D:\Training_data\test\*.tif'):
    op = predict(i, model)
    saveClear(op, i)

#%%
###############################################################################
# TEST
# ###############################################################################
os.chdir(r'D:\Code\RiverTwin\2022_12_08_unPacked')
gc.collect()

from testSuccess import testSuccess
names = []
f1s = []
for i in glob.glob(r'D:\Code\RiverTwin\ZZ_results\tira_1\*.tif'):
    P3 = io.imread(i)
    r = testSuccess(image_fp=P3, time=False, output=False, display_image=False,
                    save_image=False)
    names.append(i.split('\\')[-1][:-4])
    f1s.append(r['f1-score']['macro avg'])
    
import pandas as pd
df = pd.DataFrame({'id':names, 'f1':f1s})

output_name = r"D:\Code\RiverTwin\2022_11_29_PaperFigures\Results\tira.csv"
df.to_csv(output_name)
