@ECHO off

:: EDIT these paths to your directories, respectively:

:: MaskRCNN folder (delete the word "rem" from the beginning of the line and set the path if you already have
:: a MaskRCNN folder - in this case, move the word "rem" from between the beginning of the last 2 lines):
rem set "pathToInsert=D:\Mask_RCNN\Mask_RCNN-2.1"

:: working directory where you downloaded the code and will have the output under ~\kaggle_workflow\outputs\maskrcnn:
set root_dir=%~dp0

:: directory of your images to pre-process:
set images_dir=%1
:: -----------------------------------------------------------------------------


:: --- DO NOT EDIT from here ---
rem run_image_preprocessing.bat %root_dir% %images_dir% %pathToInsert%
run_image_preprocessing.bat %root_dir% %images_dir%