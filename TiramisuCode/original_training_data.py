# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 14:02:57 2023

@author: lgxsv2
"""

# this file is needed because the augmentation only increases data and does not confirm the inclusion of the rest of the dataset

import skimage.io as io
from osgeo import gdal_array
import numpy as np
import glob
import os
import gc


#%%
maskFiles=glob.glob('D:\Training_data\desert\doodless\Mask*.tif', recursive=True)
outrootT='C:/Users/lgxsv2/TrainingData/ZZ_Tiramasu/train_OGsmall/image_' #edit the folder, but leave the: image_
outrootV='C:/Users/lgxsv2/TrainingData/ZZ_Tiramasu/validate_OGsmall/image_'
#%%




BaseSize=112
flipmask=True #flip the mask due to gdal odd reading
ValidationSplit=0.2

#%% functions 
def tile_for_CNNPrediction(im,label, tile_size):
    '''
    '''
    # remove edge cells from image so it can be tiled precisely
    height, length = im.shape[:2]
    height, length = int(height//tile_size), int(length//tile_size)
    # get the new max axis lengths
    y_axis, x_axis = (height*tile_size), (length*tile_size)
    
    im = im[:y_axis, :x_axis,:] # make them divisable by tile size used later
    label = label[:y_axis, :x_axis]

    # empty list to be appended to
    im_ls = []
    label_ls = []

    # for each tile on each axis 
    for m in range(height):
        for n in range(length):
            #multiply by size
            temp_m = (m+1)*tile_size 
            temp_n = (n+1)*tile_size
            # cut out the tile
            band_tile = im[(temp_m-tile_size): temp_m, (temp_n-tile_size):temp_n, :]
            label_tile = label[(temp_m-tile_size): temp_m, (temp_n-tile_size):temp_n]

            im_ls.append(band_tile)
            label_ls.append(label_tile)

            # turn list into an array
    im_ls = np.array(im_ls)
    label_ls = np.array(label_ls)

    gc.collect()
    return im_ls , label_ls 

#%%
pool=(np.asarray(range(9999999))) #int removed
np.random.shuffle(pool)
count=1

for i in range(len(maskFiles)):
    
    if flipmask:
        mask=np.flipud(gdal_array.LoadFile(maskFiles[i]))
    else:
        mask=gdal_array.LoadFile(maskFiles[i])
    #edit this line carefully so that it reconstructs the filepath of the image associated to a mask
    imFile=os.path.join(os.path.join(os.path.dirname(os.path.dirname(maskFiles[i])), 'train'),os.path.basename(maskFiles[i])[5:])
    image = gdal_array.LoadFile(imFile)
    image=np.flipud(np.rollaxis(image, 0, 3))

    if image.shape[2]>3:
        image=image[:,:,[0,1,3]]
    image = image.astype(np.float32)

    image = ((image*255)/65535).astype(np.uint8)
    
    ims, labs = tile_for_CNNPrediction(image, mask, BaseSize)
    
    for image_light, mask_light in zip(ims, labs):
            if np.random.random()>ValidationSplit:
                io.imsave(outrootT+str(pool[count]).zfill(7)+'.jpg', image_light, check_contrast=False, quality=95)
                io.imsave(outrootT+str(pool[count]).zfill(7)+'.png', mask_light, check_contrast=False)
                count+=1
            else:
                io.imsave(outrootV+str(pool[count]).zfill(7)+'.jpg', image_light, check_contrast=False, quality=95)
                io.imsave(outrootV+str(pool[count]).zfill(7)+'.png', mask_light, check_contrast=False)
                count+=1
    print('image done')
print('final_count: ', count)

