@ECHO off
SETLOCAL ENABLEEXTENSIONS
SetLocal EnableDelayedExpansion
:: batch script to run prediction with postprocessing

:: parameters are coming
set root_dir=%1
set images_dir=%2
if "%3"=="" (
    echo "using default MaskRCNN folder: %root_dir%\Mask_RCNN-2.1"
    set "pathToInsert=%root_dir%\Mask_RCNN-2.1"
) else (
    set pathToInsert=%3
)

set workflow_root=%root_dir%\kaggle_workflow

set maskrcnn_scripts=%root_dir%\FinalModel
set maskrcnn=%workflow_root%\maskrcnn
set matlab_scripts=%root_dir%\matlab_scripts
set outputs_dir=%workflow_root%\outputs
set cellsize_est=%outputs_dir%\cellSizeEstimator

set input_images=%outputs_dir%\images
set input_images_2x=%outputs_dir%\2ximages\images
set unet=%workflow_root%\unet
set unet_out=%outputs_dir%\unet_out
set unet_scripts=%root_dir%\UNet
set mass_train_unet=%outputs_dir%\train_unet
set validation=%outputs_dir%\validation
set ensemble=%outputs_dir%\ensemble
set post_processing=%outputs_dir%\postprocessing


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

:: copy user test images to our expected images folder
echo "COPYING USER IMAGES TO %outputs_dir%\images:"
mkdir "%outputs_dir%/images"
copy %images_dir%\*.* %outputs_dir%\images\*
echo "COPYING DONE"

:: create dummy folders expected by unet prediction (are unused)
mkdir %mass_train_unet%\images
mkdir %unet_out%
::mkdir %validation%


::goto comment
:: run prediction only --- from run_workflow.sh ---

:: ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

::::::::::::::::::: MRCNN presegmentation :::::::::::::::::::
echo "PRESEGMENTATION (maskrcnn):"
python %maskrcnn_scripts%\\segmentation.py %maskrcnn%\config\predict\presegment.json
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: "Error during pre-segmentation"
    exit /B 1
)
echo "PRESEGMENTATION DONE"


echo "CELL SIZE ESTIMATION:"
matlab -noFigureWindows -nodesktop -nosplash -nojvm -minimize -wait -log -r "addpath('%matlab_scripts%\cellSizeEstimateForPrediction');cellSizeDataGenerate('%outputs_dir%\presegment\','%cellsize_est%\images\');exit;"
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: "Error during cell size estimation"
    exit /B 2
)
echo "CELL SIZE ESTIMATION DONE"


echo "2x IMAGES:"
matlab -noFigureWindows -nodesktop -nosplash -nojvm -minimize -wait -log -r "addpath('%matlab_scripts%\utils');resizeSet_onlyImages('%input_images%','%input_images_2x%\',2);exit;"
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: "Error during image 2x resize"
    exit /B 3
)
echo "2x IMAGES DONE"

:::comment


::::::::::::::::::: UNET segmentation :::::::::::::::::::
echo "SET UNET PARAMS:"
set unet_models=(UNet_sigma0.0_1 UNet_sigma0.0_2 UNet_sigma0.5_1 UNet_sigma0.5_2 UNet_sigma0.5_3 UNet_sigma1.0_1 UNet_sigma2.0_1)
set epochs=100
set batch=12


echo "UNET PREDICTION TO IMAGES:"
for %%m in %unet_models% do (
    set cur_model=%%m
    echo "PREDICT !cur_model! MODEL:"
    python %unet_scripts%\train_sh.py --results_folder=%unet_out% --train=%mass_train_unet% --val=%mass_train_unet% --test=%outputs_dir% --batch=1 --model_path=%unet%\!cur_model!\!cur_model! --model_name=!cur_model!
    IF %ERRORLEVEL% NEQ 0 (
        echo ERROR: "Error during unet prediction"
        exit /B 4
    )
    echo "PREDICT !cur_model! MODEL DONE"
)
echo "UNET PREDICTION TO IMAGES END"


echo "ENSEMBLE UNET RESULTS:"
matlab -noFigureWindows -nodesktop -nosplash -nojvm -minimize -wait -log -r "addpath('%matlab_scripts%\UNETensembling');ensembleProbFolders('%unet_out%\','%ensemble%\output\');exit;"
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: "Error during augmentation 1"
    exit /B 5
)
echo "ENSEMBLE UNET RESULTS DONE"





::::::::::::::::::: MRCNN SEGMENTATION :::::::::::::::::::
echo "SEGMENT: segmentation_params_2x.json"
python %maskrcnn_scripts%\\segmentation.py %maskrcnn%\config\predict\segmentation_params_2x.json
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: "Error during maskrcnn prediction"
    exit /B 6
)
echo "MaskRCNN SEGMENTATION DONE"


echo "SEGMENT: segmentation_params_4x.json"
python %maskrcnn_scripts%\\segmentation.py %maskrcnn%\config\predict\segmentation_params_4x.json
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: "Error during maskrcnn prediction"
    exit /B 6
)
echo "MaskRCNN SEGMENTATION DONE"
::::::::::::::::::: MRCNN SEGMENTATION :::::::::::::::::::


echo "POSTPROCESSING:"
matlab -noFigureWindows -nodesktop -nosplash -nojvm -minimize -wait -log -r "addpath('%matlab_scripts%\postProcess');postProcCodeRunnerFINAL('%outputs_dir%\maskrcnn\','%post_processing%\','2x\','4x\','','master','%ensemble%\output\','final',false,'%input_images%',[]);exit;"
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: "Error during postprocessing"
    exit /B 7
)
echo "POSTPROCESSING DONE"


:: --- delete dummy folders ---
rmdir %mass_train_unet%
::rmdir %validation%