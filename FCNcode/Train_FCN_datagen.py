 # -*- coding: utf-8 -*-

__author__ = 'Patrice Carbonneau'
__contact__ = 'patrice.carbonneau@durham.ac.uk'
__copyright__ = '(c) Patrice Carbonneau'
__date__ = 'SEP 2021'
__version__ = '0.1'
__status__ = "initial release"


'''
Name:           Train_FCN_datagen.py
Compatibility:  Python 3.9
Description:    Trains an FCN model with a data generator

Requires:       Tensorflow 2.9, high-end GPU recommended


Model training with a data generator

'''
# #%% Sam added these 
# import tensorflow as tf
# with tf.device('/gpu:0'):
#     a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
#     b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
#     c = tf.matmul(a, b)

# with tf.compat.v1.Session() as sess:
#     print (sess.run(c))
# #%%
#  # Launch the graph in a session.
# with tf.compat.v1.Session() as ses:

#      # Build a graph.
#      a = tf.constant(5.0)
#      b = tf.constant(6.0)
#      c = a * b

#      # Evaluate the tensor `c`.
#      print(ses.run(c))
#%%



'''Control tensorflow verbosity'''
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.chdir(r"D:\Code\RiverTwin\2022_12_08_unPacked\TiramTry\FCNcode")
""" Libraries"""
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import optimizers
from tensorflow.keras.losses import binary_crossentropy
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten
from tiramisu_tf2 import tiramisu
import glob
import skimage.io as io
import tensorflow_io as tfio
from focal_loss import SparseCategoricalFocalLoss
from tensorflow.keras.callbacks import EarlyStopping
import gc
gc.collect()

AUTOTUNE = tf.data.experimental.AUTOTUNE
print(f"Tensorflow ver. {tf.__version__}")

##############################################################################################################
'''User Inputs'''
#images in jpeg and masks in png, both in same folder for either train or validate
TrainFolder='C:/Users/lgxsv2/TrainingData/ZZ_Tiramasu/train_good/' 
ValFolder='C:/Users/lgxsv2/TrainingData/ZZ_Tiramasu/validate_good/' #'location of validation data
GenerateValidation=True #use a data generator for validation
n_valid=5000#if above is False, take this many random samples in a tensor loaded in RAM

checkpoint_filepath='C:/Users/lgxsv2/TrainingData/ZZ_Tiramasu_o3.{epoch:02d}.hdf5'#add a full path and name before the _weights
n_class=2
img_size=(224, 224)
bands=3
eps=3
ModelName='tiramisuModelADO'#ADO==AllDoodleOkay
BATCH_SIZE = 3
BUFFER_SIZE = 200
Lrate=0.00000001
exponent=-0.5
constantLR=True
F10=False

#fine tuning info
FineTune=False
# ModelName='/home/patrice/Documents/FastData/BetaCNNmodels/Beta_jul2021_morebarsfast_multiscale_weights.14.hdf5'


def scheduler(epoch, learning_rate):
    if epoch <= 0:
        return learning_rate
    elif epoch>0:
        if F10:
            return learning_rate*0.5
        elif constantLR:
            return learning_rate
        else:
            return learning_rate*tf.math.exp(exponent)




    
def parse_image(img_path: str) -> dict:
    """Load an image and its annotation (mask) and returning
    a dictionary.

    Parameters
    ----------
    img_path : str
        Image (not the mask) location.

    Returns
    -------
    dict
        Dictionary mapping an image and its annotation.
    """
    #print(img_path)
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    # For one Image path:
    # .../trainset/images/training/ADE_train_00000001.jpg
    # Its corresponding annotation path is:
    # .../trainset/annotations/training/ADE_train_00000001.png
    mask_path = tf.strings.regex_replace(img_path, "images", "annotations")
    mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")
    mask = tf.io.read_file(mask_path)
    # The masks contain a 1-hot class index for each pixels in a multiband tif
    #mask = tfio.experimental.image.decode_tiff(mask)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.uint8)
    mask=tf.squeeze(tf.one_hot(mask, depth=n_class))
    image.set_shape([img_size[0],img_size[0],3])
    mask.set_shape([img_size[0],img_size[0],n_class])
    #mask is one hot in more than 3 classes, use gdal
    #mask=tf.py_function(onehotpreprocess, [mask_path], tf.uint8)


    return {'image': image, 'segmentation_mask': mask}


@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
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
    input_image = tf.cast(input_image, tf.float32)
    input_image = tf.math.divide(input_image, 127.5)
    input_image = tf.math.add(input_image, -1)
    #input_mask=tf.one_hot(input_mask, 6)
    
    return input_image, input_mask

@tf.function
def load_image_train(datapoint: dict) -> tuple:
    """Apply some transformations to an input dictionary
    containing a train image and its annotation.

    Notes
    -----
    An annotation is a regular  channel image.
    If a transformation such as rotation is applied to the image,
    the same transformation has to be applied on the annotation also.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """


    input_image = datapoint['image']
    input_mask = datapoint['segmentation_mask']
    
    
    
    input_image, input_mask = normalize(input_image, input_mask)
     
    return input_image, input_mask


'''Dice Loss functions, not recommended for use'''
def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = Flatten()(y_true)
    y_pred_f = Flatten()(y_pred)
    intersection = tf.math.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.math.reduce_sum(y_true_f) + tf.math.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def MakeValidationTensor(Valfolder, n,n_class, size):
    Vimages=np.zeros((n,size,size,3), dtype='float32')
    Vmasks=np.zeros((n,size,size), dtype='float32')
    Ifiles=glob.glob(Valfolder+'*.jpg')
    samples=np.random.choice(Ifiles, n, replace=False)
    for i in range(n):
        Vimages[i, :,:,:]=io.imread(samples[i]).reshape(1,size,size,3)
        maskname=samples[i][:-4]+'.png'
        Vmasks[i,:,:]=io.imread(maskname).reshape(1,size,size)
    Vimages=(Vimages/127.5)-1
    return Vimages, Vmasks
early_stopping = EarlyStopping(
    monitor='val_loss',  # Metric to monitor (usually validation loss)
    patience=5,           # Number of epochs with no improvement before stopping
    restore_best_weights=True  # Restore model weights from the epoch with the best monitored metric
)  


'''Instantiate FCN'''
model=tiramisu(tile_size=img_size[0],bands=bands,Nclasses=n_class)
Optim = optimizers.RMSprop(learning_rate=Lrate)


if n_class>1:
    model.compile(optimizer=Optim, loss=tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO, gamma=3, alpha=0.05), metrics=['accuracy'])
    if FineTune:
        model.load_weights(ModelName)
    

else:
    model.compile(loss='binary_crossentropy',optimizer=Optim,metrics=['categorical_accuracy'])
    #model.compile(optimizer=Optim, loss=bce_dice_loss, metrics=[dice_loss])
    if FineTune:
        model.load_weights(ModelName)


'''Generate Data'''
num_samples=len(glob.glob(TrainFolder+'*.jpg'))
train_dataset = tf.data.Dataset.list_files(TrainFolder + "*.jpg", seed=42)
train_dataset = train_dataset.map(parse_image)


dataset = {"train": train_dataset}

# -- Train Dataset --#  
dataset['train'] = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=42)
#dataset['train'] = dataset['train'].repeat()
dataset['train'] = dataset['train'].batch(BATCH_SIZE)
dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)
print(str(num_samples)+' training samples')
#load the val tensor
if GenerateValidation:
    num_samples=len(glob.glob(ValFolder+'*.jpg'))
    val_dataset = tf.data.Dataset.list_files(ValFolder + "*.jpg", seed=224)
    val_dataset = val_dataset.map(parse_image)


    vdataset = {"val": val_dataset}

    # -- Train Dataset --#
    vdataset['val'] = vdataset['val'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    vdataset['val'] = vdataset['val'].batch(BATCH_SIZE)
    vdataset['val'] = vdataset['val'].prefetch(buffer_size=AUTOTUNE)
else: 
    print('compiling validation tensors')
    Vimages, Vmasks=MakeValidationTensor(ValFolder, n_valid, n_class, img_size[0])
    
print(str(num_samples)+' validation samples')

    
    

#uncomment this to check that images and labels output by the datagen match
# for e in range(5):
#     print('Epoch', e)
#     batches = 5
#     for x_batch, y_batch in train_ds:
#         for i in range(5):
#             plt.figure()
#             plt.subplot(1,2,1)
#             plt.imshow(np.squeeze(x_batch[i])[:,:,0])
#             I=np.squeeze(x_batch[i])[:,:,:]
#             plt.title(type(I[0,0,0]))
#             plt.ylabel(str(np.amax(I)))
#             plt.xlabel(str(np.amin(I)))
#             plt.subplot(1,2,2)
#             plt.imshow(np.uint8(255*np.squeeze(y_batch[i][:,:,1:4])))
#             string=type(y_batch[i][0])
#             plt.xlabel(string)
#         break





'''Callbacks'''
scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='loss',
    mode='min',
    save_best_only=True,
    save_freq='epoch')


'''Tune and Fit FCN'''


if GenerateValidation:
    history = model.fit(dataset['train'], validation_data = vdataset['val'], epochs = eps,  batch_size = BATCH_SIZE, callbacks=[scheduler,model_checkpoint_callback, early_stopping])
else:
    history = model.fit(dataset['train'], validation_data = (Vimages, Vmasks), epochs = eps,  batch_size = BATCH_SIZE, callbacks=[scheduler,model_checkpoint_callback, early_stopping])



#####################
'''plot the history'''

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)
plt.figure(figsize = (12, 9.5))
plt.subplot(1,2,1)
plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
plt.plot(epochs,val_loss_values, 'b', label = 'Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.subplot(1,2,2)
plt.plot(epochs, acc_values, 'go', label = 'Training acc')
plt.plot(epochs, val_acc_values, 'g', label = 'Validation acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
   

####










#%%





gc.collect()

##############################################################################################################
'''User Inputs'''
#images in jpeg and masks in png, both in same folder for either train or validate
TrainFolder='C:/Users/lgxsv2/TrainingData/ZZ_Tiramasu/train_f/' 
ValFolder='C:/Users/lgxsv2/TrainingData/ZZ_Tiramasu/validate_f/' #'location of validation data
GenerateValidation=True #use a data generator for validation
n_valid=5000#if above is False, take this many random samples in a tensor loaded in RAM

checkpoint_filepath='C:/Users/lgxsv2/TrainingData/ZZ_Tiramasu_o10.{epoch:02d}.hdf5'#add a full path and name before the _weights
n_class=2
img_size=(224, 224)
bands=3
eps=10
ModelName='tiramisuModelADO'#ADO==AllDoodleOkay
BATCH_SIZE = 3
BUFFER_SIZE = 200
Lrate=0.0000001
exponent=-0.5
constantLR=False
F10=False

#fine tuning info
FineTune=False
# ModelName='/home/patrice/Documents/FastData/BetaCNNmodels/Beta_jul2021_morebarsfast_multiscale_weights.14.hdf5'


def scheduler(epoch, learning_rate):
    if epoch <= 0:
        return learning_rate
    elif epoch>0:
        if F10:
            return learning_rate*0.5
        elif constantLR:
            return learning_rate
        else:
            return learning_rate*tf.math.exp(exponent)




    
def parse_image(img_path: str) -> dict:
    """Load an image and its annotation (mask) and returning
    a dictionary.

    Parameters
    ----------
    img_path : str
        Image (not the mask) location.

    Returns
    -------
    dict
        Dictionary mapping an image and its annotation.
    """
    #print(img_path)
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.uint8)

    # For one Image path:
    # .../trainset/images/training/ADE_train_00000001.jpg
    # Its corresponding annotation path is:
    # .../trainset/annotations/training/ADE_train_00000001.png
    mask_path = tf.strings.regex_replace(img_path, "images", "annotations")
    mask_path = tf.strings.regex_replace(mask_path, "jpg", "png")
    mask = tf.io.read_file(mask_path)
    # The masks contain a 1-hot class index for each pixels in a multiband tif
    #mask = tfio.experimental.image.decode_tiff(mask)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.uint8)
    mask=tf.squeeze(tf.one_hot(mask, depth=n_class))
    image.set_shape([img_size[0],img_size[0],3])
    mask.set_shape([img_size[0],img_size[0],n_class])
    #mask is one hot in more than 3 classes, use gdal
    #mask=tf.py_function(onehotpreprocess, [mask_path], tf.uint8)


    return {'image': image, 'segmentation_mask': mask}


@tf.function
def normalize(input_image: tf.Tensor, input_mask: tf.Tensor) -> tuple:
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
    input_image = tf.cast(input_image, tf.float32)
    input_image = tf.math.divide(input_image, 127.5)
    input_image = tf.math.add(input_image, -1)
    #input_mask=tf.one_hot(input_mask, 6)
    
    return input_image, input_mask

@tf.function
def load_image_train(datapoint: dict) -> tuple:
    """Apply some transformations to an input dictionary
    containing a train image and its annotation.

    Notes
    -----
    An annotation is a regular  channel image.
    If a transformation such as rotation is applied to the image,
    the same transformation has to be applied on the annotation also.

    Parameters
    ----------
    datapoint : dict
        A dict containing an image and its annotation.

    Returns
    -------
    tuple
        A modified image and its annotation.
    """


    input_image = datapoint['image']
    input_mask = datapoint['segmentation_mask']
    
    
    
    input_image, input_mask = normalize(input_image, input_mask)
     
    return input_image, input_mask


'''Dice Loss functions, not recommended for use'''
def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = Flatten()(y_true)
    y_pred_f = Flatten()(y_pred)
    intersection = tf.math.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.math.reduce_sum(y_true_f) + tf.math.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def MakeValidationTensor(Valfolder, n,n_class, size):
    Vimages=np.zeros((n,size,size,3), dtype='float32')
    Vmasks=np.zeros((n,size,size), dtype='float32')
    Ifiles=glob.glob(Valfolder+'*.jpg')
    samples=np.random.choice(Ifiles, n, replace=False)
    for i in range(n):
        Vimages[i, :,:,:]=io.imread(samples[i]).reshape(1,size,size,3)
        maskname=samples[i][:-4]+'.png'
        Vmasks[i,:,:]=io.imread(maskname).reshape(1,size,size)
    Vimages=(Vimages/127.5)-1
    return Vimages, Vmasks
early_stopping = EarlyStopping(
    monitor='val_loss',  # Metric to monitor (usually validation loss)
    patience=5,           # Number of epochs with no improvement before stopping
    restore_best_weights=True  # Restore model weights from the epoch with the best monitored metric
)  


'''Instantiate FCN'''
model=tiramisu(tile_size=img_size[0],bands=bands,Nclasses=n_class)
Optim = optimizers.RMSprop(learning_rate=Lrate)


if n_class>1:
    model.compile(optimizer=Optim, loss=tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO, gamma=3, alpha=0.05), metrics=['accuracy'])
    if FineTune:
        model.load_weights(ModelName)
    

else:
    model.compile(loss='binary_crossentropy',optimizer=Optim,metrics=['categorical_accuracy'])
    #model.compile(optimizer=Optim, loss=bce_dice_loss, metrics=[dice_loss])
    if FineTune:
        model.load_weights(ModelName)


'''Generate Data'''
num_samples=len(glob.glob(TrainFolder+'*.jpg'))
train_dataset = tf.data.Dataset.list_files(TrainFolder + "*.jpg", seed=42)
train_dataset = train_dataset.map(parse_image)


dataset = {"train": train_dataset}

# -- Train Dataset --#  
dataset['train'] = dataset['train'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset['train'] = dataset['train'].shuffle(buffer_size=BUFFER_SIZE, seed=42)
#dataset['train'] = dataset['train'].repeat()
dataset['train'] = dataset['train'].batch(BATCH_SIZE)
dataset['train'] = dataset['train'].prefetch(buffer_size=AUTOTUNE)
print(str(num_samples)+' training samples')
#load the val tensor
if GenerateValidation:
    num_samples=len(glob.glob(ValFolder+'*.jpg'))
    val_dataset = tf.data.Dataset.list_files(ValFolder + "*.jpg", seed=224)
    val_dataset = val_dataset.map(parse_image)


    vdataset = {"val": val_dataset}

    # -- Train Dataset --#
    vdataset['val'] = vdataset['val'].map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    vdataset['val'] = vdataset['val'].batch(BATCH_SIZE)
    vdataset['val'] = vdataset['val'].prefetch(buffer_size=AUTOTUNE)
else: 
    print('compiling validation tensors')
    Vimages, Vmasks=MakeValidationTensor(ValFolder, n_valid, n_class, img_size[0])
    
print(str(num_samples)+' validation samples')

    
    

#uncomment this to check that images and labels output by the datagen match
# for e in range(5):
#     print('Epoch', e)
#     batches = 5
#     for x_batch, y_batch in train_ds:
#         for i in range(5):
#             plt.figure()
#             plt.subplot(1,2,1)
#             plt.imshow(np.squeeze(x_batch[i])[:,:,0])
#             I=np.squeeze(x_batch[i])[:,:,:]
#             plt.title(type(I[0,0,0]))
#             plt.ylabel(str(np.amax(I)))
#             plt.xlabel(str(np.amin(I)))
#             plt.subplot(1,2,2)
#             plt.imshow(np.uint8(255*np.squeeze(y_batch[i][:,:,1:4])))
#             string=type(y_batch[i][0])
#             plt.xlabel(string)
#         break





'''Callbacks'''
scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='loss',
    mode='min',
    save_best_only=True,
    save_freq='epoch')


'''Tune and Fit FCN'''


if GenerateValidation:
    history = model.fit(dataset['train'], validation_data = vdataset['val'], epochs = eps,  batch_size = BATCH_SIZE, callbacks=[scheduler,model_checkpoint_callback, early_stopping])
else:
    history = model.fit(dataset['train'], validation_data = (Vimages, Vmasks), epochs = eps,  batch_size = BATCH_SIZE, callbacks=[scheduler,model_checkpoint_callback, early_stopping])


#%%
#####################
'''plot the history'''

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(loss_values) + 1)
plt.figure(figsize = (12, 9.5))
plt.subplot(1,2,1)
plt.plot(epochs, loss_values, 'bo', label = 'Training loss')
plt.plot(epochs,val_loss_values, 'b', label = 'Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.subplot(1,2,2)
plt.plot(epochs, acc_values, 'go', label = 'Training acc')
plt.plot(epochs, val_acc_values, 'g', label = 'Validation acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
   

####
#%%
import pandas as pd
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
#%%
saveModelAndOutputs(model, True, 'tiramisu10Overfit')
        