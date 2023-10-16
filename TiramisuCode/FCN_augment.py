#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 13:41:07 2022

@author: patrice
edited by Sam Valman
"""


import albumentations as A
import skimage.io as io
from osgeo import gdal_array
from osgeo import gdal
import numpy as np
import glob
import os

#%%
'''
To start, you need a conda env with alnumentations (from pip) and osgeo

Then, get a set of images of at least 1000 x 1000.  Store both as geotifs with the mask and the image having a different, but predictable name, 
eg S2image_001.tif for the image and Mask_001.tif for the associated mask.  The mask needs to cover the whole sample image and 
give a class for each pixel

Then create destination folders for training and validation.

Script will produce Target samples for each image.  these will be saved as image_XXXXXXX.jpg for the image and
image_XXXXXXX.png for the mask.  Ready for the FCN training code

'''

maskFiles=glob.glob('D:\Training_data\desert\doodle_good\Mask*.tif', recursive=True)
outrootT='C:/Users/lgxsv2/TrainingData/ZZ_Tiramasu/train_nc3/image_' #edit the folder, but leave the: image_
outrootV='C:/Users/lgxsv2/TrainingData/ZZ_Tiramasu/validate_nc3/image_'
#%%
BaseSize=224
MaxScale=5
test=False #a few samples to test that data is ok.  Check that mask and images match,  If not, change the flipmask variable.
Target=1000 #if test is false, generate this many samples PER TILE - this is not per tile this is per image
flipmask=True #flip the mask due to gdal odd reading
ValidationSplit=0.2
ls = []


#define the augmentation. Here we randomly select images with sizes from 224 to 224* 5 and resize them to 224.  Then apply random flips
aug = A.Compose([
    A.RandomSizedCrop(min_max_height=[BaseSize, BaseSize*MaxScale], height=BaseSize, width=BaseSize, w2h_ratio=1, always_apply=True, p=1.0),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.Transpose(p=0.5),              
    A.RandomRotate90(p=0.5)]
        )


#create a random sequence of numbers to minimise correlation in the image sequence
# new pool so doesnt overlap with original images
new_pool_start = 9999999

# Create the new pool with the same length as the original pool
pool = np.asarray(range(new_pool_start, new_pool_start + new_pool_start))
np.random.shuffle(pool)

#loop through the tiles
count=1
if test:
    # for i in range(len(maskFiles)):

    #     if flipmask:
    #         mask=np.flipud(gdal_array.LoadFile(maskFiles[i]))
            
    #     else:
    #         mask=gdal_array.LoadFile(maskFiles[i])
    #     #edit this line carefully so that it reconstructs the filepath of the image associated to a mask
    #     imFile=os.path.join(os.path.join(os.path.dirname(os.path.dirname(maskFiles[i])), 'train'),os.path.basename(maskFiles[i])[5:])
    #     image=np.flipud(np.rollaxis(gdal_array.LoadFile(imFile), 0, 3))
    #     if image.shape[2]>3:
    #         image=image[:,:,1:]
    
    
    #     for n in range(2):
    #         augmented = aug(image=image, mask=mask)
    #         image_light = augmented['image'] 
    #         image_light = image_light.astype(np.float32)

    #         image_light = ((image_light*255)/65535).astype(np.uint8)
    #         #.astype(np.uint8)
    #         # image_light = (image_light - np.min(image_light)) / (np.max(image_light) - np.min(image_light)) * 255
    #         # image_light = image_light.astype(np.uint8)
            
            
            
    #         mask_light = augmented['mask'] #.astype(np.uint8)
    #         # mask_light = mask_light.astype(np.float32)

    #         # make sure same method is applied to the holdout image.
    #         # mask_light = ((mask_light - np.min(mask_light)) / (np.max(mask_light) - np.min(mask_light)) * 255).astype(np.uint8)
    #         # mask_light = ((mask_light*255)/65535).astype(np.uint8)

    #         if np.random.random()>ValidationSplit:
    #             io.imsave(outrootT+str(pool[count]).zfill(7)+'.jpg', image_light, check_contrast=False, quality=95)
    #             io.imsave(outrootT+str(pool[count]).zfill(7)+'.png', mask_light, check_contrast=False)
    #             count+=1
    #         else:
    #             io.imsave(outrootV+str(pool[count]).zfill(7)+'.jpg', image_light, check_contrast=False, quality=95)
    #             io.imsave(outrootV+str(pool[count]).zfill(7)+'.png', mask_light, check_contrast=False)
    #             count+=1
        print('Finished test augmentation. Check images. Delete if they are incorect')
else:    

    for i in range(len(maskFiles)):
        
        try:        
            if flipmask:
                mask=np.flipud(gdal_array.LoadFile(maskFiles[i]))
            else:
                mask=gdal_array.LoadFile(maskFiles[i])
            #edit this line carefully so that it reconstructs the filepath of the image associated to a mask
            imFile=os.path.join(os.path.join(os.path.dirname(os.path.dirname(maskFiles[i])), 'train'),os.path.basename(maskFiles[i])[5:])
            image = gdal_array.LoadFile(imFile)
            image=np.flipud(np.rollaxis(image, 0, 3))

            if image.shape[2]>3:
                image=image[:,:,1:]
            image = image.astype(np.float32)
            # common_mean = 3000
            # image = image+(common_mean - image.mean())
            # image = image/3000
            # image = ((image*255)/3000).astype(np.uint8)
            # image=np.flipud(np.rollaxis(image, 0, 3))

            n=0
            while n < Target:
                augmented = aug(image=image, mask=mask)
                
                image_light = augmented['image']#.astype(np.uint8)
                azp = np.all(image_light == 0, axis=-1)
                azp = np.count_nonzero(azp)
                if azp>20:
                    continue
                else:
                    n+=1 
                    # image_light = image_light.astype(np.float32)
                    # image_light = image_light/3000
                    image_light = ((image_light*255)/3000).astype(np.uint8)
                    # # image_light = ((image_light - np.min(image_light)) / (np.max(image_light) - np.min(image_light)) * 255).astype(np.uint8)
                    # image_light = ((image_light*255)/65535).astype(np.uint8)
                    
                    mask_light = augmented['mask'].astype(np.uint8)
     
    
                    
                    if np.random.random()>ValidationSplit:
                        io.imsave(outrootT+str(pool[count]).zfill(7)+'.jpg', image_light, check_contrast=False, quality=95)
                        io.imsave(outrootT+str(pool[count]).zfill(7)+'.png', mask_light, check_contrast=False)
                        count+=1
                    else:
                        io.imsave(outrootV+str(pool[count]).zfill(7)+'.jpg', image_light, check_contrast=False, quality=95)
                        io.imsave(outrootV+str(pool[count]).zfill(7)+'.png', mask_light, check_contrast=False)
                        count+=1
            print('Finished augmentation')
        except ValueError:
            print('too SMall')
            ls.append(maskFiles[i])