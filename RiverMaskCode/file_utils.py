import os
import warnings

from filepaths import FOLDER_DATA_TRAIN, FOLDER_DATA_LABEL


def get_training_file_names(directory):
    train_path = os.path.join(directory, FOLDER_DATA_TRAIN)
    label_path = os.path.join(directory, FOLDER_DATA_LABEL)

    train_files = [file for file in os.listdir(train_path) if os.path.isfile(os.path.join(train_path, file))]
    label_files = [file for file in os.listdir(label_path) if os.path.isfile(os.path.join(label_path, file))]

    files = []
    for train_file in train_files:
        if train_file not in label_files:
            warnings.warn(f"Training file '{train_file}' does not have a corresponding label. Skipping.")
        else:
            files.append(train_file)

    for label_file in label_files:
        if label_file not in train_files:
            warnings.warn(f"Label file '{label_file}' does not have a corresponding training file. Skipping.")

    return files