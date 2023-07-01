

# -*- coding: utf-8 -*-
"""
###############################################################################
Project: River Twin: WaterMask PROFILER
###############################################################################
Final River twin water mask model
Created on Wed Oct 26 16:39:34 2022

@author: lgxsv2
"""
#%% packages

import numpy as np
import math
import pandas as pd

#plotting
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'qt')


# should probably only use one keras layers style
from tensorflow import keras
import tensorflow as tf

from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPool2D
from tensorflow.keras import  optimizers
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import callbacks
from tensorflow.keras.applications.vgg16 import VGG16
import os
import imageio
from tqdm import tqdm
import glob

import pathlib
import skimage.io as IO
import datetime
import shutil


import tensorflow_addons as tfa
import gc
#%% Wrapper


def fineTune(newTrainingData=True, trainingFolder='',
                             trainingData=[], balanceTrainingData=1, tileSize=32,
                             epochs=100, bs=6,
                             lr=0.001, lr_type='plain', 
                             loss_type='notfocal', alpha=1, gamma=2,
                             inc_2ndbatch=False, inc_neck=False,
                             outfile='todaysmodel', 
                             saveModel=True
                             ):
    '''
    Trains a CNN to provide the basis for the cCNN water mask.
    Code format is Section number, letter, number. in a nested format. 
    
    Patches needed:
        no extra file path for multiple different balanced runs in a day.
        
        
    Parameters
    ----------
    trainingData : TYPE
        DESCRIPTION.
    balanceTrainingData: int
        Default: None
        if a value it will balance training data.
        1 == balance, >1 == more land than water, <1 == more water than land.
    hyperParameters : dict
        {epochs:int, 'batchsize':int, }.
    tileSize : Int
        tile size used to create pure training tiles
    
    ## Needs to know if to balance or not ## 

    Returns
    -------
    None.

    '''

    

    ##Section One:  Training data 
    if newTrainingData: 
        print('Tiling new training data')
        trainingFolder = CollectAndOrganiseCNNTrainingData(trainingData, 
                                      tileSize, balanceTrainingData)
    else:
        print('Using pre-collected training data')

    ##Section Two: TF records and Datasets
    # relabelled to section two - change and change below
    Xy_train, Xy_test, trainingDataSize = reloadTrainingData(trainingFolder, epochs, bs)
    
    
    #Section Three: Train CNN
    # change as needed 
    CNN = fit_CNN(Xy_train, Xy_test, epochs, tileSize, lr, lr_type, loss_type, alpha, gamma, inc_2ndbatch, inc_neck, trainingDataSize) #epochs, bs,


    #Section Four: Save model
    saveModelAndOutputs(CNN, saveModel, outfile)
    
    
    
###############################################################################   
#%% Section one: Wrapper. 
###############################################################################

def CollectAndOrganiseCNNTrainingData(trainingData, tileSize,
                                      balanceTrainingData):
    '''
    Wrapper to do all the work to collect and organise training data
    
    Parameters
    ----------
    trainingData : list
        list of file names of rivers in the training folder specified .
    tile_size : int
        specified in Train_RiverTwinWaterMask wrapper.
    balanceTrainingData: int
        Degree of training data inbalancing acceptable. 

    Returns
    -------
    X_train : NumpyArray
        data ready for training.
    y_train : NumpyArray
        labels ready for training.

    '''
    ## Section one:A - removes training files wihtout labels
    im_list, label_list = ListUsableFileNames(trainingData)
    print('Total number of images: ', len(im_list))
    
    ## Section one:B - creates pure tiles to train model
    # 0==water, 1==land.
    X_train, y_train = stackTiles(im_list, label_list, tileSize)
    
    # balance training data as requested (saves regardless)
    ## Section one: c function to save and balance training data
    path = pruneAndSave(X_train, y_train, balanceTrainingData, tileSize)
        
        
        
        
    
    return path


#%% Section one: A: 
    
def ListUsableFileNames(trainingData):
    '''
    Checks if images have labels and returns the file path of those that do
    
    Parameters
    ----------
    trainingData : list
        Training images we would like to use.
    

    Returns
    -------
    im_list : list
        list of filepaths for available X_train images.
    label_list : list
        list of filepaths for available y_train label images.
.

    '''
    im_list, label_list = [], []
    
    # list all label files available and cut to just their names
    file_ls =  glob.glob(os.path.join('D:/FineTuning/combined/train_label/', '*.tif'))
    file_ls = [x[35:] for x in file_ls]
    
    for i in trainingData:
        # checks if that image has a label
        if i in file_ls:
            # A :1 function
            im, label = getPaths(i)

            # join to list for return
            im_list.append(im)
            label_list.append(label)

        
    return im_list, label_list   

### A: 1
def getPaths(riverID):
    '''
    gets paths for river images and labels 
    Parameters
    ----------
    riverName : str
        river Image ID name.

    Returns
    -------
    im_ls : list
        list of image filepaths .
    label_ls : list
        list of label image filepaths.

    '''
    # riverID = riverID +'.tif'
    imPath = os.path.join(r'D:/FineTuning/combined/train/', riverID) 
    labPath = os.path.join(r'D:/FineTuning/combined/train_label/', riverID) 
    

    return imPath, labPath


###############################################################################    
#%% Section one: B
###############################################################################

def stackTiles(im_list, label_list, tile_size):
    '''
   Creates (N, tileSize, tileSize, bandNumber) arrays
   from training and label data (0=water, 1 = land)
   
    Parameters
    ----------
    im_list : List
        Training image file paths.
    label_list : List
        Label image file paths.
    tile_size : Int
        Tile size to be maintained throughout.

    Returns
    -------
    X_train : numpy array
    
    y_train : numpy array

    '''
    # empty lists to be filled 
    X_train, y_train = [],[]
    
    for i,m in zip(im_list, label_list):
        #open label image
        label_im = np.int16(IO.imread(m))
        # check that the image is not too large
        # skips if this is the case
        # Gdal options available but not built in 
        try:
            #open related training image
            band_im = np.int16(IO.imread(i))
        except MemoryError:
            print(i, 'too large')
            continue
       
        
        #Section 1:B:1 function to turn individual images to tiles
        X_temp, y_temp = tileForCNN(band_im, label_im, tile_size)
        
        
        # print the image name to show it worked
        print(i)
      
        # remove empyt images - str output from tileForCNN
        if type(X_temp) != str:
            X_train.append(X_temp)
            y_train.append(y_temp)
        else:
            print(X_temp)
            continue
    
    # combine lissts into format (N, tileSize, tileSize, bandNumber)
    # band number is binary for y_train
    X_train = np.concatenate((X_train), axis=0)
    y_train = np.concatenate((y_train), axis=0)
    
    return X_train, y_train

### B:1
def tileForCNN(im, label, tileSize):
    '''
    cuts individual images into tiles 
    Parameters
    ----------
    im : array
        Band image (4 band to be used).
    label : array
        label image (0, 1, 2 for nothing, water, land respectively).
    tileSize: int
        from uppermost function
   
    Returns
    -------
    tiled image as input to CNN.
    '''
    # normalise satellite bands
    # normalised removed for storage etc - will put in with data reading
    # im = keras.utils.normalize(im)

    # remove edge cells from image so it can be tiled precisely
    height, length = label.shape
    height, length = int(height//tileSize), int(length//tileSize)
    y_axis, x_axis = (height*tileSize), (length*tileSize)
    
    # cut to just divisable area 
    # makes them divisable by tile size used later
    im = im[:y_axis, :x_axis,:]
    label = label[:y_axis, :x_axis]
    
    #int of empty tile size
    pure_tile = tileSize**2
   
    #list for output tiles
    im_ls = []
    label_ls = []
    
    # scrolls through image height and length (when multiplied by tile_size)
    # height len already//tilesize
    for m in range(height):
        for n in range(length):
            temp_m = (m+1)*tileSize
            temp_n = (n+1)*tileSize
            # selects this tile out of label image
            label_tile = label[(temp_m-tileSize): temp_m, (temp_n-tileSize):temp_n]
            
            # only pure tiles  (water first)
            if np.count_nonzero(label_tile == 1) == pure_tile:
                # create tile with these values
                band_tile = im[(temp_m-tileSize): temp_m, (temp_n-tileSize):temp_n, :]
                # captures tile errors if they occur 
                if band_tile.shape != (tileSize,tileSize,4):
                    continue
                # label is just a list so only needs int appended 
                im_ls.append(band_tile)
                label_ls.append(1)
    
            # land now - if pure all same as above. 
            elif np.count_nonzero(label_tile == 2) == pure_tile:
                band_tile = im[(temp_m-tileSize): temp_m, (temp_n-tileSize):temp_n, :]
                if band_tile.shape != (tileSize,tileSize,4):
                    continue
                im_ls.append(band_tile)
                label_ls.append(2)
    
    # some images may have no tiles if they were very small or very unpure.
    if len(im_ls)!= 0:        
        im_ls = np.array(im_ls)
        label_ls = np.array(label_ls).reshape((-1,1))
    
        # get format correct 
        # now water 0, land 1 
        label_ls = label_ls - 1
    else: 
        im_ls, label_ls = 'No tiles', 'no Tiles'
                
    return im_ls, label_ls


#%% Section one:C
def pruneAndSave(X_train, y_train, balanceTrainingData, tileSize):
    '''
    organises the balancing and saving of training data for the model

    Parameters
    ----------
    X_train : array
        all X_train.
    y_train : array
        all y_train.
    balanceTrainingData : int
        balance value.
    extra_folder_name : 'str', optional
        DESCRIPTION. The default is ''.
        for having more than one different training set in a day
    epoch: int
    bs : int
        
        

    Returns
    -------
    Xy_train : tf data dataset
        new X_train
    Xy_test: tf data dataset
        new y_train

    '''
    #create_training_directory_to_save_into
    parent_dir='D:/Training_data/temporary_tiles'
    parent_dir = r'C:\Users\lgxsv2\TrainingData'

    
    #extra folder name for two in a day
    directory =  datetime.datetime.today().strftime('%Y_%m_%d')
    # +extra_folder_name If wanted it needs to be added in all levels
    path = os.path.join(parent_dir, directory)
    
   #remove dir if already exisits
    if os.path.exists(path):
        print('overwriting old directory from today')
        shutil.rmtree(path)
    os.mkdir(path)
    
    # add water and land label categories 
    water = os.path.join(path, 'water')
    land = os.path.join(path, 'land')
    
    os.mkdir(water)
    os.mkdir(land)
    name = 0
    
    # go through each tile and place in correct folder
    for tile, label in zip(X_train, y_train): 
        name +=1
        # 1:C:1 function for saving all tiles
        saveTile(tile, label, name, water, land)
    
    # No longer need so remove
    X_train, y_train = None, None
    #function 1:C:2
    # remove tiles from the larger class based on balanceTrainingData input
    prune(water, land, balanceTrainingData)
    
    
    return path
    
    
    
### section 1:C:1
def saveTile(tile, label, name, water_path, land_path):
    '''
    Saves the tiles (all)

    Parameters
    ----------
    tile : array
        individual tile.
    label : array
        tile label.
    name : int
        descriptor for that tile number.
    water_path : str
        folder path.
    land_path : str
        folder path.

    Raises
    ------
    SystemExit
        if there is an error it will shut the model down.

    Returns
    -------
    None.

    '''
    name = str(name) + '.tif'
    # print('norm' , type(tile[1][1][1]))

    if label[0] == 0:
        temp_im_path = os.path.join(water_path, name)
                
        IO.imsave(temp_im_path, (tile), check_contrast=False)
        
       
        
    elif label[0] == 1:
        temp_im_path = os.path.join(land_path, name)
        IO.imsave(temp_im_path, (tile), check_contrast=False)
        
      

    else:
        print('error probably should start looking here')
        print('name')
        raise SystemExit()
       
        
       
### Section 1:C:2
def prune(water_folder_name, land_folder_name, balanceTrainingData):
    '''
    Deletes according to balanceTrainingData to balance training set

    Parameters
    ----------
    water_folder_name : str
        folder path.
    land_folder_name : str
        folder path.
    balanceTrainingData : int
        acceptable balance see wrapper function.

    Returns
    -------
    None.

    '''
    # how many water tiles are there
    water_tiles = glob.glob(water_folder_name + '/*.tif')
    water_tiles = len(water_tiles)
    
    # get all land tiles
    land_tiles = glob.glob(land_folder_name + '/*.tif')
    
    # using balanceTrainingData as a weight to increase or decrease tile overlap. 
    # balanceTrainingData of 1.1 would allow 10% more land than water
    water_tiles = int(water_tiles*balanceTrainingData)
    if water_tiles < len(land_tiles):
        # selects random land images up to number of water tiles. (pre-influenced)
        removable_images = np.random.choice(np.arange(len(land_tiles)), len(land_tiles)-water_tiles, replace=False)
        print('removing ',len(removable_images), ' land tiles to balance dataset')
       
        # actually do the removing
        for i in removable_images:
            os.remove(land_tiles[i])
            
            
            
#%%
###############################################################################
### Section Two: TF records and Datasets
###############################################################################

def reloadTrainingData(folder, epoch, bs):
    
    # get the directory for the tiles
    raw_tiff_dir = pathlib.Path(folder)
    
    # measure quanity of tiles for test train split later
    data_dir = folder + '/*/*.tif'
    trainingDataSize = len(glob.glob(data_dir))
    
    # output file name for tf records file
    tfrecord_file = folder +".tfrecord"
    
    # Section 2: A
    # Convert the TIFF images to TFRecords
    if not os.path.exists(tfrecord_file):
        tiff_to_tfrecords(raw_tiff_dir, tfrecord_file)
    
    # Load the TFRecord dataset
    dataset = tf.data.TFRecordDataset(tfrecord_file)
    
    # Section 2: B
    # Apply the parse function to the dataset - get actual results
    dataset = dataset.map(parse_tfrecord)
    
    # Shuffle the data
    buffer_size = trainingDataSize  # Use a buffer size of 1000 elements
    dataset = dataset.shuffle(buffer_size, reshuffle_each_iteration=True )
    # Xy_train = dataset.batch(bs)
    # Xy_test = 1
    

    
    # split dataset into test train 
    Xy_train = dataset.take(int(trainingDataSize*0.9)).batch(bs)
    Xy_test = dataset.skip(int(trainingDataSize*0.9)).batch(bs)
    # select automated prefectch to speed up process
    AUTOTUNE = tf.data.AUTOTUNE
    Xy_train = Xy_train.prefetch(buffer_size=AUTOTUNE)
    Xy_test = Xy_test.prefetch(buffer_size=AUTOTUNE)


    return Xy_train, Xy_test, trainingDataSize

### Section 2:A: tiff to TFRecords
def tiff_to_tfrecords(tiff_dir, tfrecord_file):

  print("Writing images to tfrecord")

  # Create a TFRecordWriter
  writer = tf.io.TFRecordWriter(tfrecord_file)

  # Iterate through all subdirectories in the tiff directory
  for label_counter, subdir in enumerate(os.listdir(tiff_dir)):
    print("\n", str(subdir), f"mapped to {label_counter}")

    # Iterate through all files in the subdirectory
    for file in tqdm(os.listdir(os.path.join(tiff_dir, subdir))):
      # Load the TIFF image
      tiff_image = imageio.v2.imread(os.path.join(tiff_dir, subdir, file))

      # Convert the TIFF image to a NumPy array
      np_image = np.array(tiff_image)

      # Convert the NumPy array to a byte string
      image_bytes = np_image.tobytes()

      # Create a dictionary with the image data and metadata
      data = {
          'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
          'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[np_image.shape[0]])),
          'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[np_image.shape[1]])),
          'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[np_image.shape[2]])),
          'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label_counter])),
      }

      # Create a tf.train.Example protobuf object
      example = tf.train.Example(features=tf.train.Features(feature=data))

      # Serialize the example and write it to the TFRecord file
      writer.write(example.SerializeToString())

  # Close the TFRecordWriter
  writer.close()

### Section 2: B: get images and labels from tf recordds
def parse_tfrecord(example):
    # Define a function to parse the data from a TFRecord

  # Parse the data from the example
  data = {
      'image_raw': tf.io.FixedLenFeature([], tf.string),
      'height': tf.io.FixedLenFeature([], tf.int64),
      'width': tf.io.FixedLenFeature([], tf.int64),
      'depth': tf.io.FixedLenFeature([], tf.int64),
      'label': tf.io.FixedLenFeature([], tf.int64),
  }
  parsed_data = tf.io.parse_single_example(example, data)

  # Decode the image bytes and reshape the image data
  image = tf.io.decode_raw(parsed_data['image_raw'], tf.uint16)
  image = tf.reshape(image, [parsed_data['height'], parsed_data['width'], parsed_data['depth']])

  # Extract the label data
  label = parsed_data['label']

  # Return the parsed data
  return image, label


    
###############################################################################
#%% Section Three: Train CNN    
###############################################################################

# BUILD all as complicated options then remove as needed
def fit_CNN(Xy_train, Xy_test, epochs, tileSize, lr,
               lr_type, loss_type, alpha, gamma, 
               inc_2ndbatch, inc_neck, 
               trainingDataSize):
    '''
    fits a basic CNN
    Parameters
    ----------
    X_train : array
        shape [ntiles, ].
    y_train : array
        shape [ntiles, ].

    Returns
    -------
    CNN : Model
        trained model.

    '''
    # Transfer model Unused:
        
    # TransferedModel = VGG16(include_top=False, weights=None, input_shape=(tileSize, tileSize, 4)) #input_shape=(40,40,4)
    # flat = keras.layers.Flatten()(TransferedModel.layers[-1].output)
    # class1 = keras.layers.Dense(32, activation='relu')(flat) 
    # output = keras.layers.Dense(2, activation='softmax')(class1)
    # CNN = keras.Model(inputs=TransferedModel.inputs, outputs=output)
        
    # CNN.compile(loss="sparse_categorical_crossentropy", 
    #               optimizer="adam",
    #               metrics=["accuracy"])





    # # # Sequential model
    CNN = keras.Sequential()
    
    # # #Section 3: A - get data aug layer
    # # data_augmentation = daug(tileSize)
    
    # # CNN.add(data_augmentation)

    CNN.add(Conv2D(32, (3, 3), activation='relu', input_shape=(tileSize, tileSize, 4)))
    CNN.add(tf.keras.layers.BatchNormalization())

    CNN.add(MaxPool2D(pool_size=(2,2)))
    
    CNN.add(Conv2D(32, (3,3), activation=("relu")))
    CNN.add(MaxPool2D(pool_size=(2,2)))
    
    CNN.add(Flatten())
    CNN.add(Dense(32, activation='relu'))
    CNN.add(tf.keras.layers.Dropout(0.5))

    CNN.add(Dense(2, activation='softmax'))

    

    
    # # Section 3: B
    loss = loss_options(loss_type, alpha, gamma)
    
    # #compile based on loss function from 2a
    CNN.compile(loss=loss, optimizer=optimizers.Adam(learning_rate=lr),
                metrics=["accuracy"])
   
    # # section 3: B
    # #call back for learning rate decay
    callback = LR_options(lr,lr_type, epochs)

    # # section 3: D
    es = MarochovCallback(threshold=0.95)


    
    # if callback == None:
    #     callback = [es]
    # else:
    #     callback = callback+[es]
    
    # v_steps = int(math.floor(int(trainingDataSize*0.9)/32))
    CNN.fit(Xy_train, epochs=epochs ,
            validation_data=Xy_test) #, validation_steps=v_steps, steps_per_epoch=v_steps, callbacks=callback,

    gc.collect()

        
    return CNN


### Section Three: A
def daug(tileSize):
    # Define the augmentation
    data_augmentation = tf.keras.Sequential(
      [
        tf.keras.layers.RandomFlip(input_shape=(tileSize, tileSize, 4)),
        tf.keras.layers.GaussianNoise(
    0.1, seed=None)
      ]
    )
    return data_augmentation


### Section Two: B
def loss_options(loss_type, alpha, gamma):
    if loss_type == 'focal':
        loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=alpha, gamma=gamma)
        
    else:
        loss = "sparse_categorical_crossentropy"
    return loss


### Section Three: C
def LR_options(lr, lr_type, epochs):
    
    if lr_type == 'plain':
        callback = None   
    else:
        #decay type
        initial_learning_rate = lr
        if lr_type == 'time':
            
            decay = lr / epochs
            def lr_time_based_decay(epoch, lr):
                return initial_learning_rate * 1 / (1 + decay * epoch)
            decay_func = lr_time_based_decay
            
        elif lr_type == 'step':
            def lr_step_decay(epoch, lr):
                drop_rate = 0.5
                epochs_drop = 4.0
                return initial_learning_rate * math.pow(drop_rate, math.floor(epoch/epochs_drop))
            decay_func = lr_step_decay
        else:
            def lr_exp_decay(epoch, lr):
                k = 0.1
                return initial_learning_rate * math.exp(-k*epoch)
            decay_func = lr_exp_decay
        print(decay_func)
        callback = [LearningRateScheduler(decay_func, verbose=1)]
    return callback

### Section Three: D
class MarochovCallback(callbacks.Callback):
    def __init__(self, threshold):
        super(MarochovCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        accuracy = logs["val_accuracy"]
        if accuracy >= self.threshold:
            print('')
            print('Validation accuracy target reached, stopping training')
            print('')
            self.model.stop_training = True
            
###############################################################################
#%% Section Four
###############################################################################

def saveModelAndOutputs(model, saveModel, outfile):
    
    
    # put in the model output section
    parentDir = 'D:/Code/RiverTwin/ZZ_Models'
    path = os.path.join(parentDir, outfile)
    
    try:
        #make an outputs folder
        os.mkdir(path)
    except FileExistsError:
        # if the file exists just make the folder name different by adding an X
        new_output = outfile + 'X'
        # and repeat
        path = os.path.join(parentDir, new_output)
        os.mkdir(path)

        
    
    # Section 4: A
    graphs(model, saveModel, outfile, path)
    
    #Section 4: B
    saveWholeModel(model, path)
    
    #Section 4:C
    saveTrainingEpochs(model, path)
#%%
### Section 4:A
def graphs(model, saveModel, outfile, path):
    
    ### loss
    plt.figure()
    data = model.history.history
    plt.title(outfile+' Loss')
    plt.plot(data['loss'], label='Loss')
    plt.plot(data['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    if saveModel: 
        name = 'loss.png'
        name = os.path.join(path, name)
        plt.savefig(name, dpi=600)
        
    ### Accuracy
    plt.figure()
    data = model.history.history
    plt.title(outfile+' Accuracy')
    plt.plot(data['accuracy'], label='Accuracy')
    plt.plot(data['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    if saveModel: 
        name = 'accuracy.png'
        name = os.path.join(path, name)
        plt.savefig(name, dpi=600)
    
    
### Section 4:B        
def saveWholeModel(model, path):
    name = 'model'
    name = os.path.join(path, name)
    
    #check on first run for file extension
    model.save(name)

    
    
### Section 4:C
def saveTrainingEpochs(model, path):
    hd = model.history.history
    df = pd.DataFrame.from_dict(hd)
    name = 'trainingEpochs.csv'
    name = os.path.join(path, name)
    
    df.to_csv(name)

        
        

    
##############################################################################
def GPU_SETUP():
    '''setup for RTX use of mixed precision
    directly from Carbonneu '''
    #Needs Tensorflow 2.4 and an RTX GPU
    if ('RTX' in os.popen('nvidia-smi -L').read()) and ('2.10' in tf.__version__):
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
        from tensorflow.compat.v1 import ConfigProto
        from tensorflow.compat.v1 import InteractiveSession
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    