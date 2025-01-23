# -*- coding: utf-8 -*-
"""
###############################################################################
Project: River Twin: WaterMask
###############################################################################
run water mask on image

Created on Wed Oct 26 16:42:02 2022

@author: lgxsv2
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from codecarbon import OfflineEmissionsTracker
import pandas as pd

import skimage.io as IO
import cv2

from tensorflow.keras.models import load_model
from tensorflow.keras.layers import  Dense, Normalization
from tensorflow.keras import Sequential, optimizers
import tensorflow_addons as tfa

import gc
#%% overall wrapper
def RiverTwinWaterMask(image_fp, tileSize=32, model=r"D:\Code\RiverTwin\ZZ_ModelOutputs\2022_11_01_Base\model", output=r'D:\\Code\\RiverTwin\\ZZ_Results\\x'):
    '''
    Wrapper to predict an image

    Parameters
    ----------
    image_fp : str
        file path for image being predicted.
    tileSize : int, optional
        tile size of CNN saved . The default is 32.
    model : str, optional
        directory of CNN model. The default is r"D:\Code\RiverTwin\ZZ_ModelOutputs\2022_11_01_Base\model".
    output: str
        output directory for data. The default is r'D:\\Code\\RiverTwin\\ZZ_Results\\x'
        just change x KEEP THE R
    Returns
    -------
    P1 : array
        CNN prediction output.
    P2 : array
        ANN prediction output.
    P3 : array
        final prediction output.
    time_taken: DT object
        monitors how long code took to run per image
    imName: str
        Image name to tie to time

    '''
    # start time taken and emission tracke
    #log level removes verbosity
    tracker = OfflineEmissionsTracker(country_iso_code="GBR", log_level='critical')
    tracker.start()
    
    # Section 1, take image and predict with pretrained CNN
    # Returns argmaxxed CNN, im and ActualPrediction to make a mask from
    P1, im, ActualPrediction = CNNPrediction(image_fp=image_fp, model=model, tileSize=tileSize)
    
    # Section 2, ANN
    # semantic segmentation output
    P2 = ANNPrediction(P1=P1, im=im, tile_size=tileSize, AP=ActualPrediction)
    
    # Section 3, morphological operators
    P3 = remove_specal(P2)
    
    #end timer
    tracker.stop()
    
    ### Save images
    time = save_imgs(P1,P2,P3, output, image_fp, tracker)

    gc.collect()
    
    #end timer
    
    
    # retu
    return P1,P2,P3,time




 

#%% Section 1: CNN
def CNNPrediction(image_fp, model, tileSize):
    '''
    Carrys out rough prediction using pre-trained CNN

    Parameters
    ----------
    image_fp : str
        image path.
    model : str
        model path str (dir level).
    tileSize : int
        model tilesize trained at .

    Returns
    -------
    P1 : array
        Prediction 1 output.
    im : array
        the orginal read in image to be predicted.

    '''
    ### Section 1: A
    # tile the image to produce training data.
    # No need for y label as just prediction
    X_test, im = imageFormat(image_fp, tileSize)
    
    # change to int64
    X_test = np.int64(X_test)
    
    # load the pretrained CNN
    CNN = load_model(model)
    # Carryout prediction 
    CNNPrediction = CNN.predict(X_test)
    
    ## Section 1: B Argmax function
    # set to tiled so same func can be used for the ANN
    
    P1 = argmax(prediction=CNNPrediction, 
                            im = im,
                            tile_size=tileSize, 
                            model_type='Tiled'
                            )
    
    return P1, im, CNNPrediction



#%% Section One:A image formatting

def imageFormat(image_fp, tileSize):
    '''
    loads and formats image

    Parameters
    ----------
    image_fp : str
        image path.
    tileSize : int
        CNN tileSize.

    Returns
    -------
    im_ls : array
        list of testing tiles.
    im : array
        original read in image for other sections.

    '''
    # read image to array
    im = IO.imread(image_fp)
    
    # S1:A:1
    # tiles to same dimensions as CNN
    im_ls = tile_for_CNNPrediction(im, tileSize)
    return im_ls, im
    
    


### Section 1:A:1 tile for CNN
def tile_for_CNNPrediction(im, tile_size):
    '''

    Parameters
    ----------
    im : array
        image.
    tile_size : int
        CNN tile size.

    Returns
    -------
    im_ls : array
        tiles.

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
    return im_ls   

#%% Section 2:B Argmax

def argmax(prediction, im, tile_size, model_type='Not'):
    '''
    Parameters
    ----------
    prediction : array
        Direct result from CNN/ANN prediction.
    im : array
        Original image array for shape
    model_type: str 
        Tiled or Not 
        Not == ANN
    tile_size : int
    
    Returns
    -------
    argmax_image : array
        Argmax result of ANN or CNN prediction
    '''
    # height and length of orignal image
    # y/x of tiled image
    height, length = im.shape[:2]
    if model_type == 'Tiled':
        y_axis, x_axis = int(height//tile_size), int(length//tile_size)
    else:
        prediction = prediction.reshape((height*length), 2)
        

    ls = []
    # uses numpys argmax for finding the most likely tile land use
    for i in prediction:
        temp = np.argmax(i)
        ls.append(temp)
        #in case of an issue
        if type(temp)!=np.int64:
            print(temp)
            break
    
    #converts list to array
    argmax_result = np.array(ls)
    
    # reshapes array to original image shape
    if model_type=='Tiled':  
        argmax_image = argmax_result.reshape((y_axis, x_axis))
    else:
        argmax_image = argmax_result.reshape((height, length))
        
    return argmax_image

#%% Section 2: ANN Prediction

def ANNPrediction(P1, im, tile_size, AP):
    '''

    Parameters
    ----------
    P1 : array
        argmaxed CNN prediction.
    im : array
        original image.
    tile_size : int
        tilesize of cnn.

    Returns
    -------
    P2 : array
        Prediction 2 semantic prediction.

    '''
    # Section 2: A
    # get in the correct format (pixel) for ANN
    X_train, y_train = ANNImageFormat(P1,im, tile_size, AP)
    
    # Section 2: B
    # fits the ANN based on this training data 
    ANN = fit_ANN(X_train, y_train)
    
    # 
    
    # predicts the whole image based on this ANN
    P2 = ANN.predict(im)
    
    # gets the final argmax
    P2 = argmax(prediction=P2, im=im, tile_size=tile_size, model_type='semantic')

    
    return P2
    
#%% Section 2: A ANNImageFormat

def ANNImageFormat(P1, im, tile_size, AP):
    '''
    Puts CNN prediction into ANN format

    Parameters
    ----------
    P1 : array
        CNN prediction.
    im : array
        image read in.
    tile_size : int
        CNN tile size.
    AP: 
        actual prediction

    Returns
    -------
    X_train : array
        pixel values to train ANN (n, 4).
    y_train : array
        label values to train ANN (n,).

    '''
    
    # remove edge cells from image so it can be tiled precisely
    height, length = im.shape[:2]
    height, length = int(height//tile_size), int(length//tile_size)
    y_axis, x_axis = (height*tile_size), (length*tile_size)
    

    
    X_train = im[:y_axis,:x_axis, :]
    
    
    
    
    
    # empty list to be appended to 
    row_ls=[]
    
    

    #for every row in the df
    for i in P1:
        #create temp list
        ls = []

        
        # for every tile in the row
        for j in i:
            values = [j]*(tile_size**2)
            v = np.array(values).reshape((tile_size,tile_size))
            ls.append(v)

                
                
                
                
        row = np.concatenate(ls, axis=1)
        row_ls.append(row)
        
    label_image = np.concatenate(row_ls)
    label_image.reshape( y_axis, x_axis )
    
    
    # X_train, y_train = maskUnsurePixels(X_train, label_image, AP, y_axis,
    #                                     x_axis, tile_size, height, length)
    #reshape for train/test
    ########IF MASKING UNSURE PIXELS THEN COMMENT THIS OUT
    pixel_number = y_axis*x_axis
    y_train = label_image.reshape(pixel_number, 1)
    X_train = X_train.reshape(pixel_number, 4)

    return X_train, y_train 

def maskUnsurePixels(X_train, y_train, mask, y_axis, x_axis, tile_size, height, length):
    
    
    # selects best of the values (doesnt matter which)
    foo = lambda x : x[np.argmax(x)]
    # does the indexing 
    HV = np.array([foo(xi) for xi in mask])
    mask =  np.where((np.isnan(HV)) | (HV < 0.60), False, True) 
    mask = mask.reshape(height, length)
    
    row_ls=[]
        #for every row in the df
    for i in mask:
          #create temp list
        ls = []
    # for every tile in the row
        for j in i:
            values = [j]*(tile_size**2)
            v = np.array(values).reshape((tile_size,tile_size))
            ls.append(v)
        row = np.concatenate(ls, axis=1)
        row_ls.append(row)
    mask = np.concatenate(row_ls)
    mask = mask.reshape( y_axis, x_axis )
    
    
    X_train = X_train[~np.array(mask)]    
    y_train = y_train[~np.array(mask)]   
    
    pixel_number = y_train.shape[0]
    y_train = y_train.reshape(pixel_number, 1)
    
    print('df')
    return X_train, y_train


#%% Section 2: B


def fit_ANN(X_train, y_train):
    '''
    Trains the ANN

    Parameters
    ----------
    X_train : array
        (n, 4).
    y_train : array
        (n,1).

    Returns
    -------
    ANN : model
        sequential model.

    '''
    ### This is the model from ice but will be changed to fit the working order set by carboneau
    ANN = Sequential()
    
    # normalise input
    # ANN.add(Normalization())
    
    #try 3 dense layers
    ANN.add(Dense(64,  activation='relu'))
    ANN.add(Dense(64, activation='relu'))
    ANN.add(Dense(64, activation='relu'))
    
    # softmax final layer
    ANN.add(Dense(2, activation='softmax'))
    
    optim = optimizers.Adam()
    # loss = tfa.losses.SigmoidFocalCrossEntropy()
    # ANN.compile(loss=loss,
    #             optimizer=optim, metrics = ['accuracy']) 
    ANN.compile(loss='sparse_categorical_crossentropy',
                optimizer=optim, metrics = ['accuracy']) 
        

    
    # large batch size for speed, 
    # validation should be removed if the epochs are removed
    ANN.fit(X_train, y_train, batch_size=108,
            validation_split=0.0, epochs=1,  verbose=1)
    
    ### Section 2:B:1 
    # graph output - mostly to check success of the model
    # try: 
    #     graphs(ANN)
    # except:
    #     print('issue with graph')
    # try:
    #     graphs(ANN)
    # except:
    #     print('no graph')
    return ANN


### Section 2:B:2

def graphs(model):
    '''
    Prints graphs of ANN model training

    Parameters
    ----------
    model : TF ANN
        to extract history.

    Returns
    -------
    None.

    '''
    # both plots together, requires validation data
    fig, axes = plt.subplots(1,2)
    data = model.history.history
    axes[0].set_title('Loss')
    axes[0].plot(data['loss'], label='Loss')
    axes[0].plot(data['val_loss'], label='Validation Loss')
    axes[0].legend()
    
    axes[1].set_title('Accuracy')
    axes[1].plot(data['accuracy'], label='Accuracy')
    axes[1].plot(data['val_accuracy'], label='Validation Accuracy')
    axes[1].legend()
    plt.show()

    

#%% Section 3: Remove specal 

def remove_specal(im, kernel_size=3):
    '''
    uses morphological operators from image processing
    Parameters
    ----------
    im : array
        P2 out put.
    kernel_size : int, optional
        kernal for operator. The default is 3.

    Returns
    -------
    output : TYPE
        DESCRIPTION.

    '''
    #reverse image because dilation works on one class primarily 
    # our method seems to bleed land into water therefore this is the way we want it 
    im_reversed = 1-im 
    
    # needs to be this format for cv2
    input_im = im_reversed.astype('uint8')
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilution = cv2.dilate(input_im, kernel)
    
    #revert to normal 
    output = 1-dilution
    return output


    
#%%

def save_imgs(P1,P2, P3, output, image_fp, tracker):
    
    
    #paths for each phase
    path1 = os.path.join(output, 'p1')
    path2 = os.path.join(output, 'p2')
    path3 = os.path.join(output, 'p3')

    im_name = os.path.basename(image_fp)
    # im_name = im_name+'f'
    
    #check if already exists
    if os.path.exists(output):
        print('saving to existing dir')
    else:
        #make main dir
        os.makedirs(output, exist_ok=True)
    # add dir inside this for each section

    if os.path.exists(path1):
        print('set up complete')
    else:
        os.mkdir(path1)
        os.mkdir(path2)
        os.mkdir(path3)

    path1 = os.path.join(path1,  im_name)
    path2 = os.path.join(path2,  im_name)
    path3 = os.path.join(path3,  im_name)

    print("path1", path1)
    print("image_fp", image_fp)

    IO.imsave(path1, np.uint8(P1), check_contrast=False)
    IO.imsave(path2, np.uint8(P2), check_contrast=False)
    IO.imsave(path3, P3, check_contrast=False)
    # Done 
    
    name_list = ['time', 'cpu', 'ram', 'gpu', 'emissions']
    em_list = [tracker.final_emissions_data.duration, 
           tracker.final_emissions_data.cpu_energy,
           tracker.final_emissions_data.ram_energy, 
           tracker.final_emissions_data.gpu_energy, 
           tracker.final_emissions_data.emissions]
    df = pd.DataFrame([name_list, em_list])  
    im_name = im_name[-3]+'csv'
    em_path = os.path.join(output,  im_name)
    
    # df.to_csv(em_path)
    return df 
    
    
    
    
    
    
    
    
    
    
    
    