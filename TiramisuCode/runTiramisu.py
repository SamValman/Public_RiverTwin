# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:40:32 2023

Author: Sam Valman

"""
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
import matplotlib.pyplot as plt
#%% functions!!!!
gc.collect()
def tile_for_CNNPrediction(im, tile_size):
    '''
    '''
    # remove edge cells from image so it can be tiled precisely
    height, length = im.shape[:2]
    height, length = int(height//tile_size), int(length//tile_size)
    # get the new max axis lengths
    y_axis, x_axis = (height*tile_size), (length*tile_size)
    
    im = im[:y_axis, :x_axis,:] # make them divisable by tile size used later
    
    # empty list to be appended to
    im_ls = []
    
    # for each tile on each axis 
    for m in range(height):
        for n in range(length):
            #multiply by size
            temp_m = (m+1)*tile_size 
            temp_n = (n+1)*tile_size
            # cut out the tile
            band_tile = im[(temp_m-tile_size): temp_m, (temp_n-tile_size):temp_n, :]
            im_ls.append(band_tile)
            # turn list into an array
    im_ls = np.array(im_ls)
    gc.collect()
    return im_ls  

def normalize(input_tiles):
# (input_image: tf.Tensor) -> tuple:
    """Rescale the pixel values of the images between -1.0 and 1.0
    compared to [0,255] originally.

    Parameters
    ----------
    input_image : tf.Tensor
        Tensorflow tensor containing an image of size [SIZE,SIZE,3].
    input_mask : tf.Tensor
        Tensorflow tensor containing an annotation of size [SIZE,SIZE,1].

    Returns
    -------
    tuple
        Normalized image and its annotation.
    """
    # input_image = tf.cast(input_image, tf.float32)
    # input_image = tf.math.divide(input_image, 127.5)
    # input_image = tf.math.add(input_image, -1)
    input_tiles = input_tiles.astype(np.float32)
    input_tiles = input_tiles / 127.5
    input_tiles = input_tiles - 1.0
    return input_tiles




def makePrediction(tiles, model, im, tile_size=(224,224)):
    # num_tiles = len(tiles)
    # predicted_image = np.empty((num_tiles, tile_size[0], tile_size[1]), dtype=np.uint8)




##    im stuff
    height, length = im.shape[:2]
    height, length = int(height//tile_size[0]), int(length//tile_size[0])
    # get the new max axis lengths
    y_axis, x_axis = (height*tile_size[0]), (length*tile_size[0])
    
    im = im[:y_axis, :x_axis,:] # make them divisable by tile size used later
    tiles = normalize(tiles)
    # Loop through each tile and make predictions
    predictions_tile = model.predict(tiles, batch_size=3)
    ls =[]
    for i in predictions_tile:
        predicted_classes_tile = np.argmax(i, axis=-1)
        ls.append(predicted_classes_tile)
    gc.collect()
    ls = np.array(ls)
    pr = ls.reshape(y_axis,x_axis)



    
    return pr



























#########################################################################
##**START**##
#########################################################################

#%%

fcn_fp = r"C:\Users\lgxsv2\TrainingData\ZZ_Tiramasu_weights.05.hdf5"
fcn_fp = r"C:\Users\lgxsv2\TrainingData\ZZ_Tiramasu_o3.02.hdf5"
fcn_fp= r"C:\Users\lgxsv2\TrainingData\ZZ_Tiramasu_o3.01.hdf5"
fcn_fp = r"D:\Code\RiverTwin\ZZ_Models\tiramisu10OverfitX\model"
fcn_fp= r"C:\Users\lgxsv2\TrainingData\ZZ_Tiramasu_o3.01.hdf5"
fcn_fp= r"C:\Users\lgxsv2\TrainingData\ZZ_Tiramasu_fixedGood.01.hdf5"
fcn_fp = r"C:\Users\lgxsv2\TrainingData\ZZ_Tiraloss2.01.hdf5"

model = tf.keras.models.load_model(fcn_fp)

tile_size = (224, 224)  # Size of each tile in pixels
fn = r"C:\Users\lgxsv2\Downloads\tile_1.npy"
one = np.load(fn)
# fn = r"C:\Users\lgxsv2\Downloads\tile_2.npy"
# two = np.load(fn)
# fn = r"C:\Users\lgxsv2\Downloads\tile_3.npy"
# three = np.load(fn)

a = model.predict(one, batch_size=3)
aa= np.argmax(a[0], axis=-1)

#%%
def predictAndSave(fn, model,tile_size):
    
    im = io.imread(fn)
    im = im.astype(np.float32)

    im1 = im[:,:,[0,1,3]] #im[:,:,:-1]
    im = ((im1*255)/65535).astype(np.uint8)
    input_tiles = tile_for_CNNPrediction(im, 224)
    p3 = makePrediction(input_tiles, model, im)

    o_fn = os.path.join(r'D:\Code\RiverTwin\ZZ_results\tira_2', fn.split('\\')[-1])
    io.imsave(o_fn, p3)





fp = r'D:\Training_data\test\*.tif'
fp = r'D:\Training_data\train\*.tif'
a = 1
for fn in glob.glob(fp):
    if fn.split('\\')[-1] == '1_A1.tif':
    #     continue
    # if a ==10:

        predictAndSave(fn, model,tile_size)
    # a+=1
    if a ==15:
        break
#%%








###############################################################################
# TEST
###############################################################################
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



















# Using tiramisu

# either per tile metrics or some way of putting tile in and returning per tile 


# #%%
# import tensorflow as tf
# from skimage import io
# import numpy as np

# fcn_fp = r"C:\Users\lgxsv2\TrainingData\ZZ_Tiramasu_10.hdf5"
# # Replace 'your_model.hdf5' with the path to your .hdf5 model file
# model = tf.keras.models.load_model(fcn_fp)

# # im file path
# im_path = r"D:\Training_data\test\1_A3.tif"
# im = io.imread(im_path)

# im = ((im*255)/65535).astype(np.uint8)

# #%%

# predictions = model.predict(im)
