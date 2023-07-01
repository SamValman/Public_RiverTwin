# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 11:21:14 2022

@author: lgxsv2


"""



#%% Fine tuning with total data

import os
os.chdir(r'D:\Code\RiverTwin\2022_12_08_unPacked')
import gc
gc.collect()
#The only difference betweeen these imports was trying to get profiler to work so I could understand the problem
from FineTune2 import fineTune





FolderContents = ['10_A1.tif', '10_A2.tif', '10_T1.tif', '10_T2.tif', '10_V1.tif', '10_V2.tif', '11_B1.tif', '11_B2.tif', '11_Po1.tif', '11_Po2.tif', '11_Py1.tif', '11_Py2.tif', '12_C1.tif', '12_C2.tif', '12_E1.tif', '12_E2.tif', '12_T1.tif', '12_T2.tif', '13_A1.tif', '13_A2.tif', '13_R1.tif', '13_R2.tif', '13_S1.tif', '13_S2.tif', '1_A1.tif', '1_A2.tif', '1_B1.tif', '1_B2.tif', '1_X1.tif', '1_X2.tif', '2_B1.tif', '2_B2.tif', '2_N1.tif', '2_N2.tif', '2_R1.tif', '2_R2.tif', '4_R1.tif', '4_R2.tif', '4_Th1.tif', '4_Th2.tif', '4_Tr1.tif', '4_Tr2.tif', '5_C1.tif', '5_C2.tif', '5_H1.tif', '5_H2.tif', '5_S1.tif', '5_S2.tif', '6_M1.tif', '6_M2.tif', '6_S1.tif', '6_S2.tif', '6_T1.tif', '6_T2.tif', '7_C1.tif', '7_C2.tif', '7_N1.tif', '7_N2.tif', '7_V1.tif', '7_V2.tif', '8_M1.tif', '8_M2.tif', '8_R1.tif', '8_R2.tif', '8_V1.tif', '8_V2.tif', '9_B1.tif', '9_B2.tif', '9_N1.tif', '9_N2.tif', '9_T1.tif', '9_T2.tif']

trainingfolder = r"C:\Users\lgxsv2\TrainingData\VGG_newLR" # rename todays date to match
# ['FTO02.tif','FTO04.tif','FTO08.tif','FTO11.tif','FTO14.tif',]

#%%
fineTune(newTrainingData=False, trainingData=FolderContents,
                             balanceTrainingData=1, trainingFolder=trainingfolder,
                             outfile='VGG_newLR100epochs',
                              epochs=100, bs=32, lr_type='plain',
                              tileSize=32, lr=0.0001)


# rewrite the load data file to include fine tuned data. 












#%%

#%%TRAINING THE CNN
import os
os.chdir(r'D:\Code\RiverTwin\2022_12_08_unPacked')
import gc
gc.collect()
#The only difference betweeen these imports was trying to get profiler to work so I could understand the problem
from TrainRiverTwinWaterMask import Train_RiverTwinWaterMask, GPU_SETUP



# GPU_SETUP()

#%%

FolderContents = ['10_A1.tif', '10_A2.tif', '10_T1.tif', '10_T2.tif', '10_V1.tif', '10_V2.tif', '11_B1.tif', '11_B2.tif', '11_Po1.tif', '11_Po2.tif', '11_Py1.tif', '11_Py2.tif', '12_C1.tif', '12_C2.tif', '12_E1.tif', '12_E2.tif', '12_T1.tif', '12_T2.tif', '13_A1.tif', '13_A2.tif', '13_R1.tif', '13_R2.tif', '13_S1.tif', '13_S2.tif', '1_A1.tif', '1_A2.tif', '1_B1.tif', '1_B2.tif', '1_X1.tif', '1_X2.tif', '2_B1.tif', '2_B2.tif', '2_N1.tif', '2_N2.tif', '2_R1.tif', '2_R2.tif', '4_R1.tif', '4_R2.tif', '4_Th1.tif', '4_Th2.tif', '4_Tr1.tif', '4_Tr2.tif', '5_C1.tif', '5_C2.tif', '5_H1.tif', '5_H2.tif', '5_S1.tif', '5_S2.tif', '6_M1.tif', '6_M2.tif', '6_S1.tif', '6_S2.tif', '6_T1.tif', '6_T2.tif', '7_C1.tif', '7_C2.tif', '7_N1.tif', '7_N2.tif', '7_V1.tif', '7_V2.tif', '8_M1.tif', '8_M2.tif', '8_R1.tif', '8_R2.tif', '8_V1.tif', '8_V2.tif', '9_B1.tif', '9_B2.tif', '9_N1.tif', '9_N2.tif', '9_T1.tif', '9_T2.tif']

trainingfolder = r"C:\Users\lgxsv2\TrainingData\Balanced32"


#%%
Train_RiverTwinWaterMask(newTrainingData=False, trainingData=FolderContents,
                             balanceTrainingData=1, trainingFolder=trainingfolder,
                             outfile='VGG16_2',
                              epochs=1, bs=32, lr_type='plain',
                              tileSize=32)
#%%


# rename the folder after******************************************************
trainingfolder = r"C:\Users\lgxsv2\TrainingData\Balanced20"
Train_RiverTwinWaterMask(newTrainingData=False, trainingData=FolderContents,
                             balanceTrainingData=1, trainingFolder=trainingfolder,
                             outfile='M20_empty',
                              epochs=10, bs=32, lr_type='plain',
                              tileSize=20)

# try without penultimate layer

gc.collect()




raise SystemExit()


#%%
import gc
gc.collect()

import skimage.io as IO
import numpy as np 

fn = r"C:\Users\lgxsv2\TrainingData\Balanced32\Water\100013.tif"


im = IO.imread(fn)

op = r"C:\Users\lgxsv2\TrainingData\int64.tif"

# # #int16
# IO.imsave(op, np.int64(im))

# im2 = IO.imread(op)

import imageio

tiff_image = imageio.v2.imread(fn)



















