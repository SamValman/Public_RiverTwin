# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 12:19:07 2022

@author: lgxsv2
"""
import skimage.io as IO
import glob
import tensorflow as tf
import numpy as np 
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import  Dense, Normalization
from tensorflow.keras import Sequential, optimizers
import matplotlib.pyplot as plt
import pandas as pd
import cv2 

from tensorflow.keras.utils import Sequence

#%%tests
imFP = r'D:\Training_data\train\*'
lbFP = r'D:\Training_data\label_train\*'

# TrainANNPackage(imFP, lbFP)

#%%

def ANNPackage(impath='', mpath=r'D:\Code\RiverTwin\ZZ_ModelOutputs\ANN\ANN', output=r'D:\\Code\\RiverTwin\\ZZ_Results\\ANN'):
    from codecarbon import OfflineEmissionsTracker
    tracker = OfflineEmissionsTracker(country_iso_code="GBR", log_level='critical')
    tracker.start()
    # load model
    ANN = load_model(mpath)
    
    #load test 
    testim = IO.imread(impath)
    
    #predict 
    P1 = ANN.predict(testim)
    P1 = argmax(prediction=P1, im=testim)
    
    # morpholigical
    P2 = remove_specal(P1)
    tracker.stop()
    save_imgs(P1,P2, output, impath, tracker)
    
    return P1,P2


#%%
def TrainANNPackage(imFP, lbFP, saveModel=True, outfile='ANN'):
    '''

    Parameters
    ----------
    imFP : TYPE
        DESCRIPTION.
    lbFP : TYPE
        DESCRIPTION.
    saveModel : TYPE, optional
        DESCRIPTION. The default is True.
    outfile : TYPE, optional
        DESCRIPTION. The default is 'ANN'.

    Returns
    -------
    None.

    '''
   # get files 
    train = glob.glob(imFP)
    label = glob.glob(lbFP)
    
    Xy_train = load_data(train, label)
    
    #create model
    ANN = fit_ANN(Xy_train)
    
    saveModelAndOutputs(ANN, saveModel, 'ANN')
    
   
    
#%% load data 

def load_data(train, label):
    
    
    def genny(filepaths, batchsize):
        for i in filepaths :
            X_trains
        
    
    
    train_set = load_dataset('images/train', (patch_height, patch_width), noise_sigma, batch_size)

    
    class DataGenerator(Sequence):
        def __init__(self, x_set, y_set, batch_size):
            self.x, self.y = x_set, y_set
            self.batch_size = batch_size

        def __len__(self):
            return int(np.ceil(len(self.x) / float(self.batch_size)))

        def __getitem__(self, idx):
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
            return batch_x, batch_y
        
    first = True

    
    for i,l in zip(train,label):
        im = IO.imread(i)
        label = IO.imread(l)
        
        if first:
            dataset = DataGenerator(im, y_train, 32)
            first = False
            print('ran')
        
            
        else:
            ds = tf.data.Dataset.from_tensor_slices((im, label))

            dataset = dataset.concatenate(ds)
    
    dataset = dataset.batch(64)
            
    return dataset
            
def load_data(train, label):


    class DataGenerator(Sequence):
        def __init__(self, x_set, y_set, batch_size):
            self.x, self.y = x_set, y_set
            self.batch_size = batch_size

        def __len__(self):
            return int(np.ceil(len(self.x) / float(self.batch_size)))

        def __getitem__(self, idx):
            batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
            return batch_x, batch_y

    train_gen = DataGenerator(X_train, y_train, 32)
    
    
    
#%%
from keras.preprocessing.image import load_img, img_to_array, list_pictures

def random_crop(image, crop_size):
    height, width = image.shape[1:]
    dy, dx = crop_size
    if width < dx or height < dy:
        return None
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return image[:, y:(y+dy), x:(x+dx)]

def image_generator(list_of_files, tileSize, to_grayscale=True):
    while True:
        filename = np.random.choice(list_of_files)
        try:
            img = img_to_array(load_img(filename, color_mode='rgba'))
        except:
            return
        cropped_img = random_crop(img, tileSize)
        if cropped_img is None:
            continue
        yield cropped_img
        
        
        
        
        
def corrupted_training_pair(images, sigma):
    for img in images:
        target = img
        if sigma > 0:
            source = img + np.random.normal(0, sigma, img.shape)/255.0
        else:
            source = img
        yield (source, target)
def group_by_batch(dataset, batch_size):
    while True:
        try:
            sources, targets = zip(*[next(dataset) for i in xrange(batch_size)])
            batch = (np.stack(sources), np.stack(targets))
            yield batch
        except:
            return
        
def load_dataset(directory, crop_size, sigma, batch_size):
    files = list_pictures(directory)
    generator = image_generator(files, crop_size, scale=1/255.0, shift=0.5)
    generator = corrupted_training_pair(generator, sigma)
    generator = group_by_batch(generator, batch_size)
    return generator
            
#%% fit ANN

def fit_ANN(Xy_train):
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
    ANN.add(Normalization())
    
    #try 3 dense layers
    ANN.add(Dense(64,  activation='relu'))
    ANN.add(Dense(64, activation='relu'))
    # ANN.add(Dense(64, activation='relu'))
    
    # softmax final layer
    ANN.add(Dense(2, activation='softmax'))
    
    optim = optimizers.Adam()
    ANN.compile(loss='sparse_categorical_crossentropy',
                optimizer=optim, metrics = ['accuracy'])
        

    
    # large batch size for speed, 
    # validation should be removed if the epochs are removed
    ANN.fit(Xy_train, epochs=40, batch_size=64,
            validation_split=0.3, verbose=1)
    return ANN

#%% SAVE

def saveModelAndOutputs(model, saveModel, outfile):
    
    
    # put in the model output section
    parentDir = 'D:/Code/RiverTwin/ZZ_ModelOutputs'
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

        
    
    # Section 3: A
    graphs(model, saveModel, outfile, path)
    
    #Section 3: B
    saveWholeModel(model, path)
    
    #Section 3:C
    saveTrainingEpochs(model, path)




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
    
    
### Section 3:B        
def saveWholeModel(model, path):
    name = 'model'
    name = os.path.join(path, name)
    
    #check on first run for file extension
    model.save(name)

    
    
### Section 3:C
def saveTrainingEpochs(model, path):
    hd = model.history.history
    df = pd.DataFrame.from_dict(hd)
    name = 'trainingEpochs.csv'
    name = os.path.join(path, name)
    
    df.to_csv(name)


#%% argmax
def argmax(prediction, im):
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
    

    argmax_image = argmax_result.reshape((height, length))
        
    return argmax_image

#%%
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
import os
def save_imgs(P1,P2, output, image_fp, tracker):
    
    
    #paths for each phase
    path1 = output + '\\p1'
    path2 = output + '\\p2'

    im_name = image_fp.split('\\')[-1]
    
    #check if already exists
    if os.path.exists(output):
        print('saving to existing dir')
    else:
        #make main dir
        os.mkdir(output)
        # add dir inside this for each section
        os.mkdir(path1)
        os.mkdir(path2)
    
    path1 = os.path.join(path1,  im_name)
    path2 = os.path.join(path2,  im_name)

    IO.imsave(path1, P1)
    IO.imsave(path2, P2)
    
    name_list = ['time', 'cpu', 'ram', 'gpu', 'emissions']
    em_list = [tracker.final_emissions_data.duration, 
           tracker.final_emissions_data.cpu_energy,
           tracker.final_emissions_data.ram_energy, 
           tracker.final_emissions_data.gpu_energy, 
           tracker.final_emissions_data.emissions]
    df = pd.DataFrame([name_list, em_list])   
    em_path = os.path.join(output,  im_name)
    df.to_csv(em_path)
    
