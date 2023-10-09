## Tiramisu - comparison to State-of-the-art model-

Code provided by Dr. Patrice Carbonneau and edited for this task by Samuel Valman. 

Original Tiramisu model written by: @author:  Marcin Luksza , MIT license
https://github.com/lukszamarcin/100-tiramisu-keras

Tif masks of land and water should be created in either Semi-Automatic classifier on QGIS, DashDoodler (requires change in file type), or alternative pixel level classification with human input to ensure it matches the spatial location of the image and is not altered by PlanetScopes radiometric issues.

### fixNaProblem
This was required by us for the images that we classified with the QGIS Semi-Automatic Classifier. It should be used to sort out band values before any other step.

### FCN_augment
Written by Patrice Carbonnea - edited.

This file is then used next. You will need to change the locations and make sure all your label images are prefixed with "Mask_". 
This will then using augmentation to increase the training script. 
It also has a test function to check your images work. 
It will save Pngs and Jpegs which is what the Tiramisu model is later set up for. 


### Train_FCN_datagen
Written by Patrice Carbonneau - edited. 
This uses structures built in the **tiramisu_tf2.py** file. The tiramisu_tf2.py file does not need to be edited.

The file names will need to be changed here but look for the comments some need _image to be kept. 
This file trains the model - for 99,000 training tiles it takes ~4 hours an epoch on the machine described in our paper. 

The save file function is written by Sam Valman in the CSC model but is written out in full here. You can use this or make sure the save weights function is set to save full model. 

### runTiramisu
Written by Sam Valman. 
This is where we can get predictions from the model and uses many of the same tiling, saving, F1 score functions as the CSC watermask model. 
