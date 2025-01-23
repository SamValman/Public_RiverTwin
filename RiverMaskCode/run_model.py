import os
import glob
import pandas as pd
from pathlib import Path
from datetime import datetime

from RiverTwinWaterMask import RiverTwinWaterMask
from testSuccess import testSuccess
from filepaths import FP_DIRECTORY_DATA_TRAIN_GLOB, FP_DIRECTORY_DATA_LABELS, FILE_PATH_MODEL_M20, FILE_PATH_MODEL_M32, \
    FILE_PATH_MODEL_M32_TOMS, FP_OUTPUT_RESULTS

# Configuration
modelPath = FILE_PATH_MODEL_M20 # Change file path to model here
tileSize = 20 # Change tile size here

outputFolder = os.path.join(FP_OUTPUT_RESULTS, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

# iterate through images
names = []
f1s = []
mac = []

for i in glob.glob(FP_DIRECTORY_DATA_TRAIN_GLOB)[:]:
    im_name = os.path.basename(i)
    print(im_name)

    p1, p2, p3, time = RiverTwinWaterMask(image_fp=i,
                                          model=modelPath, tileSize=tileSize,
                                          output=outputFolder)

    P3 = os.path.join(outputFolder, 'p3', im_name)
    # once to save with time etc
    r = testSuccess(image_fp=P3, output=outputFolder, time=time, label_fp=FP_DIRECTORY_DATA_LABELS)

    # once to save macro avg
    r = testSuccess(image_fp=P3, time=False, output=False, display_image=False,
                    save_image=False, label_fp=FP_DIRECTORY_DATA_LABELS)

    names.append(Path(im_name).stem)
    f1s.append(r['f1-score']['weighted avg'])
    mac.append(r['f1-score']['macro avg'])

df = pd.DataFrame({'id': names, 'f1': f1s, 'macro': mac})

fn = os.path.join(outputFolder, 'results.csv')
df.to_csv(fn)
