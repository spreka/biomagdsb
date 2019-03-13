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
set outputs_dir=%workflow_root%\outputs

set input_images=%outputs_dir%\images


:: check inputs
echo "root_dir: " %root_dir%
echo "workflow_root:" %workflow_root%
echo "maskrcnn_scripts: " %maskrcnn_scripts%
echo "maskrcnn: " %maskrcnn%
echo "outputs_dir: " %outputs_dir%
echo "images_dir: " %images_dir%

:: set pythonpath
rem Check if pathToInsert is not already in pythonpath
if "!pythonpath:%pathToInsert%=!" equ "%pythonpath%" (
    setx pythonpath "%pythonpath%;%pathToInsert%"
)

echo "pythonpath: " %pythonpath%

:: copy user test images to our expected images folder
echo "COPYING USER IMAGES TO %outputs_dir%\images:"
mkdir "%outputs_dir%/images"
copy %images_dir%\*.* %outputs_dir%\images\*
echo "COPYING DONE"

:: ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

::::::::::::::::::: MRCNN presegmentation :::::::::::::::::::
echo "PRESEGMENTATION (maskrcnn):"
python %maskrcnn_scripts%\\segmentation.py %maskrcnn%\config\predict\presegment.json
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: "Error during pre-segmentation"
    exit /B 1
)
echo "PRESEGMENTATION DONE"
