# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:55:12 2023

@author: lgxsv2
"""
import numpy as np
import glob
import skimage.io as IO
import matplotlib.pyplot as plt 
import os
plt.close('all')
def h(im):
    im = im[:,:,:]
    i = im.flatten()
    plt.figure()
    plt.title(str(i.max()))
    plt.hist(i)
#%%
fp = r'C:\Users\lgxsv2\TrainingData\ZZ_Tiramasu\validate_nc3\*.jpg'
# fp = r'C:\Users\lgxsv2\TrainingData\ZZ_Tiramasu\train_nc3\*.jpg'

# fp = r'C:\Users\lgxsv2\TrainingData\ZZ_Tiramasu\validate_6\*.jpg'
# D:\Training_data\train
# ls = []
for i in glob.glob(fp):
    im = IO.imread(i)
    root, ext = os.path.splitext(i)
    li = root + ".png"
    try: 
        im = IO.imread(li)
    except:
        print(i)
        os.remove(i)

    # # print(im.max())
    # if im.max() < 1:
    #     print(im.max())
    # h
    # m = im.max()
    # ls.append(m)
    # break
#%%
fp = r'C:\Users\lgxsv2\TrainingData\ZZ_Tiramasu\validate_nc3\*.png'

for i in glob.glob(fp):
    im = IO.imread(i)
    root, ext = os.path.splitext(i)
    li = root + ".jpg"
    try: 
        im = IO.imread(li)
    except:
        print(i)
        os.path.splitext(li)
# im = im[:,:,1:]
# plot = im.flatten()
# plt.figure()
# plt.title('image')

# plt.hist(plot)
# plt.show
# t =im.astype(np.float32)

# t = ((t*255)/65535).astype(np.uint8)
# t = t.flatten()
# plt.figure()
# plt.title('tile')
# plt.hist(t)
# plt.show


















#%%
import json
def open_json_file(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
        
        return data


path= r"D:\20220716_131542_78_2464_metadata.json"
data = open_json_file(path)


















