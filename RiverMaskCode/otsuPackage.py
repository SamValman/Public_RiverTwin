"""
###############################################################################
Project: RT 1
###############################################################################
Water Mask comparison otsu

Created on 2022_11_07

@author: lgxsv2
"""
import os
import glob
import skimage.io as IO
import numpy as np 
from skimage.filters import threshold_otsu
import cv2
#%%


def otsuPackage(image=r"D:\Training_data\test\*.tif", op_folder=r'D:\\Code\\RiverTwin\\ZZ_Results\\otsux'):
        '''
    only requires folder not image

    Parameters
    ----------
    image : TYPE, optional
        DESCRIPTION. The default is r"D:\Training_data\test".
    op_folder : TYPE, optional
        DESCRIPTION. The default is r'D:\\Code\\RiverTwin\\ZZ_Results\\otsux'.

    Returns
    -------
    None.

        '''
    
        # saves folder string length for later 
        # path_len = len(folder)+1
        
        # # lists all files to be tested. #prints len of list
        # file_ls =  glob.glob(os.path.join(folder, '*.tif'))
        # print('number of files: ', len(file_ls))
        
        
        
        im_name = image.split('\\')[-1]
        path = os.path.join(op_folder, 'o1')
        fp = os.path.join(path, im_name)
 
        #create path for actual images
        path = op_folder + '\\o1'

        if os.path.exists(op_folder):
            print('saving to existing dir')
        else:
            #make main dir
            os.mkdir(op_folder)
            os.mkdir(path)
            
        
        

            #carrys out the automate function on the file
        p = automate(image, save=True, op_fn=fp)
        return p

def automate(fn,save=True, op_fn='op'):
    
    #read image
    im = IO.imread(fn)
    
    #seperates the bands needed for NDWI (MNDWI not available)
    green = im[:,:,1].astype(np.float64)
    NIR = im[:,:,3].astype(np.float64)
    
    # gets NDWI value
    NDWI = (green-NIR)/(green+NIR)
    
    #empties unneeded containers
    green=None
    NIR=None
    
    #remove nans without changing file shape
    test = NDWI[~np.isnan(NDWI)]
    #find Otsu threshold
    thresh = threshold_otsu(test)
    test=None
    
    
    binary = NDWI > thresh
    print(thresh)
    
    # use remove specal function over the whole image
    final = remove_specal(binary)
    #put in same format as test data
    final = 1-final 
    #save the image.
    IO.imsave(op_fn, final, check_contrast=False)
    return final 
        
        
def remove_specal(im, kernel_size=3):
    
    #reverse image because dilation works on one class primarily 
    # our method seems to bleed land into water therefore this is the way we want it 
    im_reversed = 1-im 
    
    # needs to be this format for cv2
    input_im = im_reversed.astype('uint8')
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilution = cv2.dilate(input_im, kernel)
    
    #revert to normal 
    output = dilution
    return output    