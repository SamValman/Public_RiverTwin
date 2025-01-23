from file_utils import get_training_file_names
from TrainRiverTwinWaterMask import Train_RiverTwinWaterMask
from filepaths import FP_DIRECTORY_DATA
from datetime import datetime

# Configuration
tileSize = 20 # Change tile size here

trainingFileNames = get_training_file_names(FP_DIRECTORY_DATA)
modelFolderName = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

Train_RiverTwinWaterMask(newTrainingData=True, trainingData=trainingFileNames,
                         balanceTrainingData=1, trainingFolder=FP_DIRECTORY_DATA,
                         outfile=modelFolderName,
                         epochs=1, bs=32, lr_type='plain',
                         tileSize=tileSize)

raise SystemExit()