@ECHO off
SETLOCAL ENABLEEXTENSIONS
SetLocal EnableDelayedExpansion
:: batch script to run prediction with postprocessing

:: parameters are coming
set root_dir=%1

set workflow_root=%root_dir%\kaggle_workflow
set matlab_scripts=%root_dir%\matlab_scripts
set outputs_dir=%workflow_root%\outputs
set input_images=%outputs_dir%\images
set pipeline_scripts=%root_dir%\biomag-kaggle\src

if "%2"=="images" (
    echo "using default image folder: %outputs_dir%\images"
    set orig_folder=%outputs_dir%
    set sub_folder=images
    set output_ext=".png"
) else if "%2"=="masks" (
    echo "using default mask folder: %workflow_root%\inputs\train_maskrcnn\masks"
    set orig_folder=%workflow_root%\inputs\train_maskrcnn
    set sub_folder=masks
    set output_ext=".tiff"
) else (
	echo "folder must be 'images' or 'masks'"
	exit /B 20
)


:: check inputs
echo "root_dir: " %root_dir%
echo "workflow_root:" %workflow_root%
echo "matlab_scripts: " %matlab_scripts%
echo "outputs_dir: " %outputs_dir%



:: preparing user test images for pre-processing

echo "MOVING USER IMAGES TO %orig_folder%\orig_!sub_folder!:"
mkdir "%orig_folder%\orig_!sub_folder!"
move %orig_folder%\!sub_folder!\*.* %orig_folder%\orig_!sub_folder!
echo "MOVING DONE"



echo "PRE-PROCESSING IMAGES TO 8-BIT 3 CHANNEL IMAGES:"
matlab -noFigureWindows -nodesktop -nosplash -nojvm -minimize -wait -log -r "addpath('%matlab_scripts%\utils'); src='%orig_folder%\orig_!sub_folder!\'; dest='%orig_folder%\!sub_folder!\'; extOut='!output_ext!'; run_preproc_16bit_image(src,dest,extOut); exit;"
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: "Error during image pre-processing"
    :: reorganize images to original folder
    echo "CLEANING UP INPUT IMAGE FOLDER"
    mkdir "%outputs_dir%\tmp"
    move %orig_folder%\!sub_folder!\\*.* %orig_folder%\tmp
    move %orig_folder%\orig_!sub_folder!\*.* %orig_folder%\!sub_folder!
    rmdir %orig_folder%\orig_!sub_folder!
    echo "CLEANING UP DONE"
    exit /B 1
)
echo "PRE-PROCESSING IMAGES TO 8-BIT 3 CHANNEL IMAGES DONE"
