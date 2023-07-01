# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 17:46:51 2022

@author: lgxsv2
"""

import skimage.io as IO
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from matplotlib import colors
import matplotlib.gridspec as gridspec

cmap = colors.ListedColormap(['white', 'red'])
bounds=[0,5,10]

fp = r''
im = '2_N3'

def plotF4_fours(fp, imName, keepTitles=True):
    
    p1 = os.path.join(fp, 'p1')
    p2 = os.path.join(fp, 'p2')
    p3 = os.path.join(fp, 'p3')
    
    im = r'D:\Training_data\test'
    lab = r'D:\Training_data\test_labels\label_SAC'
    # im = os.path.join(fp, 'test')
    # lab = os.path.join(fp, 'label')
    
    imName += '.tif'
    
    p1 = IO.imread(os.path.join(p1, imName))
    p2 = IO.imread(os.path.join(p2, imName))
    p3 = IO.imread(os.path.join(p3, imName))
    
    im = IO.imread(os.path.join(im, imName))
    lab = IO.imread(os.path.join(lab, imName))
    print(im.shape)
    print(lab.shape)
    print(p1.shape)
    print(p2.shape)
    print(p3.shape)
    # normalize im
    im = keras.utils.normalize(im)
    
    fig, ax = plt.subplots(1, 5)
    
    #remove gaps between images.!!
    gs1 = gridspec.GridSpec(1, 5)
    gs1.update(wspace=0, hspace=0.05)
           
    # remove box outlines
    for i in range(0,5):
        ax[i].axis('off')
        # if i !=2:
        #     ax[i].set_xlim(2000,6500)
        # elif i ==2:
        #     ax[i].set_xlim(90,335)

    # remove background color!!
    cmap = colors.ListedColormap(['white', 'navy'])
    cmap2 = colors.ListedColormap(['white','navy', 'white'])

    bounds=[0,5,10]

    ax[0].imshow(im)
    ax[2].imshow(p1, cmap=cmap)
    ax[3].imshow(p2, cmap=cmap)
    
    ax[4].imshow(p3, cmap=cmap)
    ax[1].imshow(lab, cmap=cmap2)
    
    # remove xy ticks
    plt.setp(ax, xticks=[], yticks=[])
    
    #rest of these !!
    if keepTitles:
        ax[0].set_title('Raw Image')
        ax[1].set_title('Labelled Image')
        ax[2].set_title('CNN Prediction')
        ax[3].set_title('ANN Prediction')
        ax[4].set_title('Final Prediction')


plt.close('all')
fp = r'C:\Users\lgxsv2\OneDrive - The University of Nottingham\PhD\yr_2\2022_PhD_PaperOneWriting\.cCNN_images\M20'
fp = r'D:\Code\RiverTwin\ZZ_results\M20'
imName = '2_N3'
plotF4_fours(fp, imName, keepTitles=False)
    
    
    
    
    
    
    
    
    
    
    