# -*- coding: utf-8 -*-
"""
Run ANN
Created on Tue Nov 29 13:56:42 2022

@author: lgxsv2
"""



import os
import glob
import pandas as pd


os.chdir(r'D:\Code\RiverTwin\2022_12_08_unPacked')
from RiverTwinWaterMask import RiverTwinWaterMask
from testSuccess import testSuccess

    
    

#%%
imPath = r'D:\Training_data\test\*.tif'
fn_model = r"D:\Code\RiverTwin\ZZ_Models\VGG_newLR\model"
output = r'D:\Code\RiverTwin\ZZ_results\VGG16_2023_06_02'

label_fp = r'D:\Training_data\test_labels\label_SAC'



# FT_1 is just load and fit
# ft_2 is freeze and add dense layer




#%%

# iterate through images
names = []
f1s = []
mac=[]

for i in glob.glob(imPath)[:]:

    
    im_name = i.split('\\')[-1]
    print(im_name)
    
    p1,p2,p3, time = RiverTwinWaterMask(image_fp=i,
                                            model=fn_model, tileSize=32,
                                            output=output)
    
    P3 = os.path.join(output, 'p3', i.split('\\')[-1])
    #once to save with time etc
    r = testSuccess(image_fp=P3, output=output, time=time)
    
    # once to save macro avg
    r = testSuccess(image_fp=P3, time=False, output=False, display_image=False,
                    save_image=False)
    
    names.append(im_name[:-4])
    f1s.append(r['f1-score']['weighted avg'])
    mac.append(r['f1-score']['macro avg'])
    
df = pd.DataFrame({'id':names,'f1':f1s, 'macro':mac })

fn = os.path.join(output, 'results.csv')
df.to_csv(fn)





#%%
imPath = r'E:\Mitacs\decharge_fluviale\Rivers\Ste-Marguerite\raw\*.tif'
fn_model = r"D:\Code\RiverTwin\ZZ_Models\M20\model"
output = r"E:\Mitacs\decharge_fluviale\Rivers\Ste-Marguerite\watermask"


for i in glob.glob(imPath)[:]:

    
    im_name = i.split('\\')[-1]
    print(im_name)
    
    p1,p2,p3, time = RiverTwinWaterMask(image_fp=i,
                                            model=fn_model, tileSize=20,
                                            output=output)
#%%
imPath = r'E:\Mitacs\Rivers\Restigouche\raw\*.tif'
output = r"E:\Mitacs\Rivers\Restigouche\watermask"

for i in glob.glob(imPath)[:]:
    im_name = i.split('\\')[-1]
    print(im_name)
    p1,p2,p3, time = RiverTwinWaterMask(image_fp=i,
                                            model=fn_model, tileSize=20,
                                            output=output)
imPath = r'E:\Mitacs\Rivers\Richelieu\raw\*.tif'
output = r"E:\Mitacs\Rivers\Richelieu\watermask"

for i in glob.glob(imPath)[:]:
    im_name = i.split('\\')[-1]
    print(im_name)
    p1,p2,p3, time = RiverTwinWaterMask(image_fp=i,
                                            model=fn_model, tileSize=20,
                                            output=output)
    
    
    
    
    

#%% for the original images 
imPath = r"D:\Training_data\test\*.tif" 
# iterate through images
names = []
f1s = []
mac=[]

for i in glob.glob(imPath)[:]:

    
    im_name = i.split('\\')[-1]
    print(im_name)
    
    p1,p2,p3, time = RiverTwinWaterMask(image_fp=i,
                                            model=fn_model, tileSize=20,
                                            output=output)
    
    P3 = os.path.join(output, 'p3', i.split('\\')[-1])
    #once to save with time etc
    r = testSuccess(image_fp=P3, output=output, time=time)
    
    # once to save macro avg
    r = testSuccess(image_fp=P3, time=False, output=False, display_image=False,
                    save_image=False)
    names.append(im_name[:-4])
    f1s.append(r['f1-score']['weighted avg'])
    mac.append(r['f1-score']['macro avg'])
    
df = pd.DataFrame({'id':names,'f1':f1s, 'macro':mac })

fn = os.path.join(output, 'results2.csv')
df.to_csv(fn)


raise SystemExit()

#%% Just the Test sucess

fp = r'D:\Code\RiverTwin\ZZ_results\VGG16\p3\*.tif'
output = r'D:\Code\RiverTwin\ZZ_results\VGG16'
# names = []
# f1s = []
# mac=[]

for i in glob.glob(fp)[2:]:
    im_name = i.split('\\')[-1]

    r = testSuccess(image_fp=i, output=output)
    names.append(im_name[:-4])
    f1s.append(r['f1-score']['weighted avg'])
    mac.append(r['f1-score']['macro avg'])
    
df = pd.DataFrame({'id':names,'f1':f1s, 'macro':mac })

fn = os.path.join(output, 'results.csv')
df.to_csv(fn)
raise SystemExit()

#%%
fp = r'D:\Code\RiverTwin\ZZ_results\M10\p3\*.tif'
output = r'D:\Code\RiverTwin\ZZ_results\M10'
names = []
f1s = []
mac =[]

for i in glob.glob(fp):
    im_name = i.split('\\')[-1]

    r = testSuccess(image_fp=i, output=output, display_image=False, save_image=False)
    names.append(im_name[:-4])
    f1s.append(r['f1-score']['weighted avg'])
    mac.append(r['f1-score']['macro avg'])


df = pd.DataFrame({'id':names,'f1':f1s, 'macro':mac })

fn = os.path.join(output, 'results.csv')
df.to_csv(fn)



#%% Landsat
i = r"D:\FineTuning\LS_test\LS_B1.tif"



fn_model = r"D:\Code\RiverTwin\ZZ_Models\M20\model"
output = r'D:\Code\RiverTwin\ZZ_results\Landsat'

p1,p2,p3, time = RiverTwinWaterMask(image_fp=i,
                                        model=fn_model, tileSize=20,
                                        output=output)



