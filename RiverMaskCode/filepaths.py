import os

FOLDER_DATA_LABEL = "Label"
FOLDER_DATA_TRAIN = "Train"

# Training Data Paths
FP_DIRECTORY_DATA = "./Data"
FP_DIRECTORY_DATA_LABELS = os.path.join(FP_DIRECTORY_DATA, FOLDER_DATA_LABEL)
FP_DIRECTORY_DATA_TRAIN = os.path.join(FP_DIRECTORY_DATA, FOLDER_DATA_TRAIN)
FP_DIRECTORY_DATA_TRAIN_GLOB = os.path.join(FP_DIRECTORY_DATA_TRAIN, "*")

# Output Paths
FP_OUTPUT_DIRECTORY = "./Output"
FP_OUTPUT_RESULTS = os.path.join(FP_OUTPUT_DIRECTORY, "Results")
FP_DIRECTORY_OUTPUT_TEMP = os.path.join(FP_OUTPUT_DIRECTORY, "Temp")
FP_DIRECTORY_OUTPUT_MODEL = os.path.join(FP_OUTPUT_DIRECTORY, "Model")

# Models
FILE_PATH_MODEL_M20 = "./ZZ_Models/M20/model"
FILE_PATH_MODEL_M32 = "./ZZ_Models/M32/model"
FILE_PATH_MODEL_M32_TOMS = "./Output/Model/toms_out/model"