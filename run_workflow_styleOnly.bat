@ECHO off
SETLOCAL ENABLEEXTENSIONS
SetLocal EnableDelayedExpansion
:: batch script to run prediction with postprocessing

:: parameters are coming
set root_dir=%1
set images_dir=%2
set pyVirtPath=%3
if "%4"=="" (
    echo "using default MaskRCNN folder: %root_dir%\Mask_RCNN-2.1"
    set "pathToInsert=%root_dir%\Mask_RCNN-2.1"
) else (
    set pathToInsert=%4
)

set workflow_root=%root_dir%\kaggle_workflow

set maskrcnn_scripts=%root_dir%\FinalModel
set maskrcnn=%workflow_root%\maskrcnn
set matlab_scripts=%root_dir%\matlab_scripts
set outputs_dir=%workflow_root%\outputs
set cellsize_est=%outputs_dir%\cellSizeEstimator

set input_images=%outputs_dir%\images
set input_images_2x=%outputs_dir%\2ximages\images
set original_data=%workflow_root%\inputs\original
set test1=%workflow_root%\inputs\test1
set test1_data=%outputs_dir%\test1
set validation=%outputs_dir%\validation
set validation_images_2x=%outputs_dir%\2xvalidation\images
set unet=%workflow_root%\unet
set unet_out=%outputs_dir%\unet_out
set unet_scripts=%root_dir%\UNet
set train_unet=%workflow_root%\inputs\train_unet
set mass_train_unet=%outputs_dir%\train_unet
set validation=%outputs_dir%\validation
set ensemble=%outputs_dir%\ensemble
set train_maskrcnn=%workflow_root%\inputs\train_maskrcnn
set mass_train_maskrcnn=%outputs_dir%\train_maskrcnn
set presegment=%outputs_dir%\presegment
set base_augmentations=%outputs_dir%\augmentations\base
set style_augmentations=%outputs_dir%\augmentations\style
set post_processing=%outputs_dir%\postprocessing

set pipeline_scripts=%root_dir%\biomag-kaggle\src
set cluster_config=%workflow_root%\inputs\clustering
set mask_db_dir=%cluster_config%\masks
set cluster_dir=%outputs_dir%\clusters
set style_inputs=%outputs_dir%\styleLearnInput
set synthetic_masks=%style_inputs%


:: check inputs
echo "root_dir: " %root_dir%
echo "workflow_root:" %workflow_root%
echo "maskrcnn_scripts: " %maskrcnn_scripts%
echo "maskrcnn: " %maskrcnn%
echo "matlab_scripts: " %matlab_scripts%
echo "outputs_dir: " %outputs_dir%
echo "cellsize_est: " %cellsize_est%
echo "images_dir: " %images_dir%

:: set pythonpath
rem Check if pathToInsert is not already in pythonpath
echo %pythonpath%|find /i "%pathToInsert%">nul  || set pythonpath=%pythonpath%;%pathToInsert%

echo "pythonpath: " %pythonpath%


:: ----------------- style transfer only ------------


::::::::::::::::::: STYLE TRANSFER :::::::::::::::::::
::python "%pipeline_scripts%\7_styaug\learn-styles2_win.py" --work_dir %style_inputs%\0 --cur_dir %pipeline_scripts%
::python "%pipeline_scripts%\7_styaug\learn-styles2_win.py" --work_dir %style_inputs%\1 --cur_dir %pipeline_scripts%
start cmd /k start_visdom.bat %pyVirtPath%
python "%pipeline_scripts%\7_styaug\apply-styles_win.py" --work_dir %style_inputs%\0 --cur_dir %pipeline_scripts%
python "%pipeline_scripts%\7_styaug\apply-styles_win.py" --work_dir %style_inputs%\1 --cur_dir %pipeline_scripts%
python "%pipeline_scripts%\7_styaug\gen-output.py" --work_dir %style_inputs%\0
python "%pipeline_scripts%\7_styaug\gen-output.py" --work_dir %style_inputs%\1
python "%pipeline_scripts%\7_styaug\collect-split-outputs.py" --splitted_dataset_dir %style_inputs% --out_dir %style_augmentations%

::::::::::::::::::: CLUSTERING END :::::::::::::::::::