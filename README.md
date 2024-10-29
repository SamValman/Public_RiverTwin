# Public_RiverTwin
Public repository for code in article "Title: An AI approach to operationalize global daily PlanetScope satellite imagery for river water masking" Currently submitted to Remote Sensing of the Environment. As the paper describes the models were developed and run on a single Windows desktop with the scripts written in Python utalising Tensorflow. The scripts should be runnable in other operating systems with minor changes. 

<b> n.b. requirements.tx file includes packages not used here but gives the versions of tensorflow etc used in this study. </b>

## Runing the River Twin Water Masking model
* The model is a CNN Supervised Classification model designed based on Carbonneau et al., 2020 https://doi.org/10.1016/j.rse.2020.112107 \
* The model is designed for use with PlanetScope imagery and is designed to be used straight from the box and should provide good results. The model structure could be retrained from scratch with other imagery. However, other methods may be better in this regard. \
** The CSC method has two main benefits here:**\
  1. It retrains an ANN section of the model on each image individually which helps to overcome radiometric issues that result from attempting to combine 200 shoebox satellites every day in every weather condition.
  2. It requires very easy to create training data which means most users shouldn't struggle to fine tune it to their sites in question.
 
  ### Running the model straight from the box.
  As stated in the paper we found that the 20*20 model worked best for a variaty of reasons.
  But all models from this paper should be available from the ZZ_Models folder.
  M10,20 and 30 refer to the model tile size. In each there is a model folder which contains all the usable information required to use these models and is the directory that would need to be copied to run these models.

  ** RiverMaskCode ** contains the main scripts for developing this model and using it.\
  * Most of the code is packaged into functions in these python files and run in the files that are prefixed with dates. *
configuration issue: At the moment there is some hard coded file paths that need changing in the **RiverTwinWaterMask** and **TrainRiverTwinWaterMask** which require changing. These are  '''os.chdir('dir for RiverMaskCode')'''

As shown in **yyyy_mm_dd_RunANN.py** calling RiverTwinWaterMask as a function you can use with the model files already explained to gather predictions. The function is documented to show you how it can be used. 

The **testSuccess.py** file can be used to compare your results against test labels made in another format. We recomend a SAC again because manually you either misjudge the difficult pixels or ignore them, leading to incorrect or falsely sucessful F1 scores respectively. 

### Fine tuning the model 

The second benefit of the model is that it can be fine tuned to bias it to the type of river and research that you are conducting. 

** make new test data ** 
1. Download imagery from Planet explorer
2. Create shape layer
3. Create polygon ID 1 == Water | 2 == Land
4. Rasterise - set extent to full, x,y to raster x,y e.g. height/width
- if does not work try set extent to vector
5. Zoom to, then clip to layer (no data == 0)

This can be done very quickly (3-5 minutes) with very simple labels. See the supplimentary data for some examples of this. The model will balance classes so do not worry about including more of land or water but theoretically the more tiles available from your fine tuning the more bias to your site the results will be. The main body of original training would normally hold maintain the variety needed in the model to cope with various radiometric issues. 

** Fine Tuning ** 
At the moment this had to be done by retraining the model with the original data. However, we hope in future fine tuning methods will be able to make it easier to retrain the trained models which can also be done here although sometimes this was less effective for us. 
We cannot supply the training set from Planet Explorer because it is proprietry but a list of images and the various avenues to get PlanetScope data is available in the manuscript and suplementary material. 

### SAC Semi Automatic classifier
This information is available elsewhere but to help the reader we have also included it here: 
1. SAC semi automatic classifier
2. Load image into QGIS
3. Load SCP plugin
4. Tile image on left column - band set to image
5. Training input - new input
6. Draw polygon mc1 and 2 at least 8 for each
7. MCID 1 == water | 2 == Land
8. Band processing - classification, maximum liklihood, algorithm

## Retraining the model

The raw planetScope images and the label images need to have the same file names e.g., ..\label\river_1.tif and ..\im\river_2.tif 

Warning: There are still some instances of hard coding in the train file which need to be removed. This has been noted and will be completed as soon as possible. 

### Figures in the article
The scripts used to create these figures are available in the FigureScripts folder. This will be updated upon final acceptace of the article to provide the latest images. 

Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
