@ECHO off
SETLOCAL ENABLEEXTENSIONS
SetLocal EnableDelayedExpansion
:: batch script to run prediction with postprocessing

:: parameters are coming
set root_dir=%~dp0
set pyVirtPath=" "

:: default values
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


:: add input parameters for input and output folders customly
:loop
IF NOT "%1"=="" (
    IF "%1"=="--pyVirtPath" (
        SET pyVirtPath=%2
        SHIFT
    )
    IF "%1"=="--pathToInsert" (
    	SET mrcnnPathSet="true"
        SET pathToInsert=%2
        SHIFT
    )
    IF "%1"=="--input_images" (
        SET input_images=%2
        SHIFT
    )
    IF "%1"=="--input_masks" (
        SET presegment=%2
        SHIFT
    )
    IF "%1"=="--mask_db_dir" (
        SET mask_db_dir=%2
        SHIFT
    )
    IF "%1"=="--style_inputs" (
        SET style_inputs=%2
        SHIFT
    )
    IF "%1"=="--cluster_dir" (
        SET cluster_dir=%2
        SHIFT
    )
    SHIFT
    GOTO :loop
)
set synthetic_masks=%style_inputs%

IF NOT mrcnnPathSet=="true" (
	echo "using default MaskRCNN folder: %root_dir%\Mask_RCNN-2.1"
    set "pathToInsert=%root_dir%\Mask_RCNN-2.1"
)

echo "parameters are:"
echo "		root=%root_dir%"
echo "		pyVirtenv=%pyVirtPath%"
echo "		Mask R-CNN=%pathToInsert%"
echo "		input_images=%input_images%"
echo "		input_masks=%presegment%"
echo "		mask_db=%mask_db_dir%"
echo "		cluster=%cluster_dir%"
echo:


:: check inputs
rem echo "root_dir: " %root_dir%
rem echo "workflow_root:" %workflow_root%
echo "matlab_scripts: " %matlab_scripts%
rem echo "outputs_dir: " %outputs_dir%


:: set pythonpath
rem Check if pathToInsert is not already in pythonpath
echo %pythonpath%|find /i "%pathToInsert%">nul  || set pythonpath=%pythonpath%;%pathToInsert%

echo "pythonpath: " %pythonpath%


:: for testing the input structure, skip everything
skipEverything

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
:: check if the provided pyVirtPath exists and if so, perform style transfer
IF EXIST %pyVirtPath% (

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
) ELSE (
	rem :skipStyleTransfer
	echo "skipping STYLE TRANSFER"
)

:skipEverything