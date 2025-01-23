# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 15:54:45 2022

@author: lgxsv2
"""
from sklearn import metrics 
import pandas as pd
import skimage.io as IO
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def testSuccess(image_fp, output, time=False,
                display_image=True, save_image=True,
                label_fp=r'D:\Training_data\test_labels\label_SAC'):
    '''
    provides success statistics for an image and can print the image
    **Want to add an agreement disagreement graph when time permits**

    Parameters
    ----------
    image_fp : str
        P3 result or equivelant filepath.
    output : str
        path of output files.
    display_image : Boolean, optional
        Do you want an image comparison output. The default is True.
    save_image : Boolean, optional
        do you want the image saved to file. The default is True.

    Returns
    -------
    r : df
        returns the classification report.

    '''
    # read the image in
    im = IO.imread(image_fp)
    
    # find label
    # Section 1: finds the corresponding image and the name of these:
        # will error out if none exists
    label, lname = findLabel(image_fp, label_fp)

 # put label and image into the same format
    label = np.where(label==2, 0, label)
    #capture eronous probelms
    if label.min() == -999:
        label = np.where(label==-999, 0, label)
        
    #Print report 
    # Section 2: sklearn classification report: saves at output
    r = report(im, label, output, lname, time)
    
    

    #print image
    # Section 3: if asked will display a side by side comparison and save
    if display_image:
        plotimage(im, label, lname, save=save_image, output=output)
    
    

    return r


#%% Section 1: findLabel

def findLabel(image_fp, label_fp): 
    '''
    Finds the image name and the corresponding label
    will error out if this is the case

    Parameters
    ----------
    image_fp : str
        im path.

    Returns
    -------
    label : array
        test labelled image.
    lname : str
        name of images.

    '''
    #split path for image name
    lname = os.path.basename(image_fp)
    print(lname)

    #hardcoded position of test labels
    label_path = os.path.join(label_fp, lname)
    
    # load this label image
    label = IO.imread(label_path)
    
    return label, lname




#%% Section 2 print report

def report(label, im, output, lname, time):
    '''
    provides classification report

    Parameters
    ----------
    label : array
        "true" image.
    im : array
        test image.
    output : str
        path for output.
    lname : str
        image name.

    Returns
    -------
    None.

    '''
    #reshape to format required by classification report
    label = label.reshape(-1,1)
    im = im.reshape(-1,1)
    

    r = metrics.classification_report(label, im ,target_names=(['Water','Land']), output_dict=True)
    r = pd.DataFrame(r).transpose()
    try:
        if time != False:
            r = combineReportandTime(r, time)
    except ValueError:
        r = combineReportandTime(r, time)

    
    lname = lname[:-3]+'csv'
    if output!=False:
        fp = os.path.join(output, lname)
        r.to_csv(fp)
    

    return r

#%%
def combineReportandTime(r,time):
    
    wa = r.loc['weighted avg']
    
    wa_df = pd.DataFrame(wa).transpose().reset_index(drop=True)
    wa_df = wa_df.T.reset_index().T
    wa_df = wa_df.rename(columns={0:5, 1:6, 2:7, 3:8}).reset_index()
    df = wa_df.join(time)
    
    return df 
    
    
    
#%% Section 3 image plot 

def plotimage(im, label, lname, save, output):
    '''
    plots side by side image comparison

    Parameters
    ----------
    im : array
        test image.
    label : array
        true image.
    lname : str
        image name.
    save : boolean
        save or not.
    output : str
        save path.

    Returns
    -------
    None.

    '''
    fig, axes = plt.subplots(1,2)
    axes[0].imshow(im)
    axes[0].set_title('Modelled image')
    axes[1].imshow(label)
    axes[1].set_title('"True" image')

    plt.suptitle(lname)
    
    if save:
        output = os.path.join(output, "ims")
        if os.path.exists(output):
            print('saving to existing dir')
        else:
            #make main dir
            os.mkdir(output)
        
        fp = os.path.join(output, lname)
        plt.savefig(fp, dpi=600)
    
    
    

    
    
    
    
    
    
    
