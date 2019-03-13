@ECHO off
SETLOCAL ENABLEEXTENSIONS
SetLocal EnableDelayedExpansion
:: batch script to run prediction with postprocessing

:: parameters are coming
set root_dir=%1
set pyVirtPath=%2
if "%3"=="" (
    echo "using default MaskRCNN folder: %root_dir%\Mask_RCNN-2.1"
    set "pathToInsert=%root_dir%\Mask_RCNN-2.1"
) else (
    set pathToInsert=%3
)

set workflow_root=%root_dir%\kaggle_workflow

set matlab_scripts=%root_dir%\matlab_scripts
set outputs_dir=%workflow_root%\outputs


set input_images=%outputs_dir%\images

set presegment=%outputs_dir%\presegment
set style_augmentations=%outputs_dir%\augmentations\style

set pipeline_scripts=%root_dir%\biomag-kaggle\src
set cluster_config=%workflow_root%\inputs\clustering
set mask_db_dir=%cluster_config%\masks
set cluster_dir=%outputs_dir%\clusters
set style_inputs=%outputs_dir%\styleLearnInput
set synthetic_masks=%style_inputs%
set style_final_outputs=%style_inputs%\all


:: check inputs
echo "root_dir: " %root_dir%
echo "workflow_root:" %workflow_root%
echo "matlab_scripts: " %matlab_scripts%
echo "outputs_dir: " %outputs_dir%


:: set pythonpath
rem Check if pathToInsert is not already in pythonpath
if "!pythonpath:%pathToInsert%=!" equ "%pythonpath%" (
    setx pythonpath "%pythonpath%;%pathToInsert%"
)

echo "pythonpath: " %pythonpath%



::::::::::::::::::: CLUSTERING :::::::::::::::::::
echo "pipeline_scripts: %pipeline_scripts%"
echo "mask_db_dir: %mask_db_dir%"

echo "MATLAB CONFIG:"
matlab -noFigureWindows -nodesktop -nosplash -nojvm -minimize -wait -log -r "addpath('%pipeline_scripts%');kaggleStartMatlab; setUpConfigAndDB('%mask_db_dir%','%cluster_config%','%cluster_config%\pretrainedDistanceLearner.mat','%pipeline_scripts%'); exit;"
echo "MATLAB CONFIG DONE"


:: clustering needs jvm! do NOT add -nojvm option to matlab call
echo "CLUSTERING:"
matlab -noFigureWindows -nodesktop -nosplash -minimize -wait -log -r "addpath('%pipeline_scripts%');kaggleStartMatlab;mergedImagesDir='%input_images%';clusterDir='%cluster_dir%';sacFolder='%pipeline_scripts%\1_metalearning\matlab\sac\';clusteringType ='Kmeans-correlation-Best3Cluster';failCounter=0;canContinue=false;initialSegmentation='%presegment%';runfromKaggleToClusters(mergedImagesDir,clusterDir,clusteringType,initialSegmentation,sacFolder,failCounter,canContinue); exit;"
echo "CLUSTERING DONE"

::::::::::::::::::: MASK GENERATION :::::::::::::::::::
matlab -noFigureWindows -nodesktop -nosplash -nojvm -minimize -wait -log -r "addpath('%pipeline_scripts%'); kaggleStartMatlab; styleTransTrainDir='%style_inputs%'; clusterDir='%cluster_dir%';initialSegmentation='%presegment%'; splitOptionsFile='%cluster_config%\basicOptions_02.csv'; artificialMaskDir='%synthetic_masks%'; fromClustersToStyles; exit;"
echo "MASK GENERATION DONE"

:: clean up additional files after clustering
matlab -noFigureWindows -nodesktop -nosplash -nojvm -minimize -wait -log -r "addpath('%matlab_scripts%\utils');cleanUpAferClustering('%workflow_root%'); exit;"

::::::::::::::::::: STYLE TRANSFER :::::::::::::::::::
echo "STYLE TRANSFER:"
python "%pipeline_scripts%\7_styaug\learn-styles2_win.py" --work_dir %style_inputs%\0 --cur_dir %pipeline_scripts%
python "%pipeline_scripts%\7_styaug\learn-styles2_win.py" --work_dir %style_inputs%\1 --cur_dir %pipeline_scripts%
start cmd /k start_visdom.bat %pyVirtPath%
python "%pipeline_scripts%\7_styaug\apply-styles_win.py" --work_dir %style_inputs%\0 --cur_dir %pipeline_scripts%
python "%pipeline_scripts%\7_styaug\apply-styles_win.py" --work_dir %style_inputs%\1 --cur_dir %pipeline_scripts%
python "%pipeline_scripts%\7_styaug\gen-output.py" --work_dir %style_inputs%\0
python "%pipeline_scripts%\7_styaug\gen-output.py" --work_dir %style_inputs%\1
python "%pipeline_scripts%\7_styaug\collect-split-outputs.py" --splitted_dataset_dir %style_inputs% --out_dir %style_final_outputs%
echo "STYLE TRANSFER DONE:"
::::::::::::::::::: CLUSTERING END :::::::::::::::::::
