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

if "%2"=="" (
    echo "using default mask folder: %outputs_dir%\presegment"
    set mask_dir=%outputs_dir%\presegment
) else (
    set mask_dir=%2
)


:: check inputs
echo "root_dir: " %root_dir%
echo "workflow_root:" %workflow_root%
echo "matlab_scripts: " %matlab_scripts%
echo "outputs_dir: " %outputs_dir%
echo "mask_dir: " %mask_dir%




set mergedImagesDir=%input_images%
set initialSegmentation=%mask_dir%
set clusterDir=%outputs_dir%\clusters
set styleTransTrainDir=%outputs_dir%\styleLearnInput
set splitOptionsFile=%workflow_root%\inputs\clustering\basicOptions_02.csv
:: default number of synthetic masks for style transfer:
set masks2generate=1000


echo "PREPARING STYLE TRANSFER INPUT FOR SINGLE EXPERIMENT:"
matlab -noFigureWindows -nodesktop -nosplash -nojvm -minimize -wait -log -r "addpath('%matlab_scripts%\clustering4singleExperiment'); addpath(genpath('%pipeline_scripts%')); kaggleStartMatlab; prepareClustering4singleClusterFcn('%mergedImagesDir%','%initialSegmentation%','%clusterDir%','%styleTransTrainDir%','%splitOptionsFile%',%masks2generate%); exit;"
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: "Error during style input preparation"
    exit /B 1
)
echo "PREPARING STYLE TRANSFER INPUT FOR SINGLE EXPERIMENT DONE"

:: remove dummy folder
rmdir "%styleTransTrainDir%\1"