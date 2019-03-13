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
set style_final_outputs=%style_inputs%\all


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
if "!pythonpath:%pathToInsert%=!" equ "%pythonpath%" (
    setx pythonpath "%pythonpath%;%pathToInsert%"
)

echo "pythonpath: " %pythonpath%

:: copy user test images to our expected images folder
echo "COPYING USER IMAGES TO %outputs_dir%\images:"
mkdir "%outputs_dir%\images"
copy %images_dir%\*.* %outputs_dir%\images\*
echo "COPYING DONE"

:: create dummy folders expected by unet prediction (are unused)
mkdir %mass_train_unet%\images
mkdir %unet_out%

:: copy test1 images to our expected test1 folder
mkdir "%outputs_dir%\test1"
mkdir "%outputs_dir%\test1\images"
mkdir "%outputs_dir%\test1\masks"
copy %test1%\images\*.* %test1_data%\images\*
copy %test1%\masks\*.* %test1_data%\masks\*




:: ----- execution of pipeline ------

echo "Kaggle workflow path: %workflow_root%"
echo "Original data path: %original_data%"





echo "INIT UNET MASS TRAIN:"
mkdir "%mass_train_unet%"
mkdir "%mass_train_unet%\images"
mkdir "%mass_train_unet%\masks"
copy %train_unet%\images\*.* %mass_train_unet%\images\*
copy %train_unet%\masks\*.* %mass_train_unet%\masks\*
copy %test1_data%\images\*.* %mass_train_unet%\images\*
copy %test1_data%\masks\*.* %mass_train_unet%\masks\*
echo "INIT UNET MASS TRAIN DONE"


echo "INIT MRCNN MASS TRAIN:"
mkdir "%mass_train_maskrcnn%"
mkdir "%mass_train_maskrcnn%\images"
mkdir "%mass_train_maskrcnn%\masks"
copy %train_maskrcnn%\images\*.* %mass_train_maskrcnn%\images\*
copy %train_maskrcnn%\masks\*.* %mass_train_maskrcnn%\masks\*
echo "INIT MRCNN MASS TRAIN DONE"


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



echo "2x IMAGES:"
matlab -noFigureWindows -nodesktop -nosplash -nojvm -minimize -wait -log -r "addpath('%matlab_scripts%\utils');resizeSet_onlyImages('%input_images%','%input_images_2x%\',2);exit;"
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: "Error during image 2x resize"
    exit /B 3
)
echo "2x IMAGES DONE"


::::::::::::::::::: AUGMENTATION :::::::::::::::::::
echo "BASE AUGMENTATION (0):"
matlab -noFigureWindows -nodesktop -nosplash -nojvm -minimize -wait -log -r "addpath('%matlab_scripts%\augmentation');augmentTrainingSet('%mass_train_maskrcnn%','%base_augmentations%\',0);exit;"
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: "Error during base augmentation 0"
    exit /B 4
)
echo "BASE AUGMENTATION (0) DONE"


::'TODO: move after style transfer data is ready'
echo "BASE AUGMENTATION (1):"
matlab -noFigureWindows -nodesktop -nosplash -nojvm -minimize -wait -log -r "addpath('%matlab_scripts%\augmentation');augmentTrainingSet('%style_final_outputs%','%style_augmentations%\',1,5);exit;"
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: "Error during base augmentation 1"
    exit /B 5
)
echo "BASE AUGMENTATION (1) DONE"


echo "COPY AUGMENTATIONS TOGETHER:"
copy %base_augmentations%\images\*.* %mass_train_maskrcnn%\images\*
copy %base_augmentations%\masks\*.* %mass_train_maskrcnn%\masks\*
copy %style_augmentations%\images\*.* %mass_train_maskrcnn%\images\*
copy %style_augmentations%\masks\*.* %mass_train_maskrcnn%\masks\*
copy %style_augmentations%\images\*.* %mass_train_unet%\images\*
copy %style_augmentations%\masks\*.* %mass_train_unet%\masks\*
echo "COPY AUGMENTATIONS TOGETHER DONE"


set cont_mat_file="%matlab_scripts%\generateValidation\validationFileNames.mat"
echo "GENERATE validation DATA FROM mass_train_maskrcnn:"
matlab -noFigureWindows -nodesktop -nosplash -nojvm -minimize -wait -log -r "addpath('%matlab_scripts%\generateValidation');moveToValidation('%mass_train_maskrcnn%\','%validation%\','%cont_mat_file%');exit;"
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: "Error during generate validation maskrcnn"
    exit /B 6
)
echo "GENERATE validation DATA FROM mass_train_maskrcnn DONE"


echo "GENERATE validation DATA FROM mass_train_unet:"
matlab -noFigureWindows -nodesktop -nosplash -nojvm -minimize -wait -log -r "addpath('%matlab_scripts%\generateValidation');moveToValidation('%mass_train_unet%\','%validation%\','%cont_mat_file%');exit;"
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: "Error during generate validation unet"
    exit /B 7
)
echo "GENERATE validation DATA FROM mass_train_unet DONE"


echo "2x validation IMAGES:"
matlab -noFigureWindows -nodesktop -nosplash -nojvm -minimize -wait -log -r "addpath('%matlab_scripts%\utils');resizeSet_onlyImages('%validation%\images','%validation_images_2x%\',2);exit;"
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: "Error during generate validation image 2x resize"
    exit /B 8
)
echo "2x validation IMAGES DONE"


echo "CELL SIZE ESTIMATION ON validation:"
matlab -noFigureWindows -nodesktop -nosplash -nojvm -minimize -wait -log -r "addpath('%matlab_scripts%\cellSizeEstimateForPrediction');cellSizeDataGenerate('%validation%\masks\','%cellsize_est%\validation\');exit;"
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: "Error during cell size estimation (validation images)"
    exit /B 9
)
echo "CELL SIZE ESTIMATION ON validation DONE"


::::::::::::::::::: U-NET training :::::::::::::::::::
echo "INIT UNET ENVIRONMENT:"
::mkdir %unet%
::mkdir %unet_validation%
echo "INIT UNET ENVIRONMENT DONE"


echo "UNET TRAIN MODELS:"
set unet_models[0]=UNet_sigma0.0_1
set unet_models[1]=UNet_sigma0.0_2
set unet_models[2]=UNet_sigma0.5_1
set unet_models[3]=UNet_sigma0.5_2
set unet_models[4]=UNet_sigma0.5_3
set unet_models[5]=UNet_sigma1.0_1
set unet_models[6]=UNet_sigma2.0_1
set sigmas[0]=0.0
set sigmas[1]=0.0
set sigmas[2]=0.5
set sigmas[3]=0.5
set sigmas[4]=0.5
set sigmas[5]=1.0
set sigmas[6]=2.0

set epochs=100
set batch=12
set unet_counter=0

echo "UNET TRAIN MODELS:"
for /L %%m in (0,1,6) do (
    set cur_model=!unet_models[%%m]!
    set cur_sigma=!sigmas[%%m]!
    echo "PREDICT !cur_model! MODEL:"
    python %unet_scripts%\train_sh.py --results_folder=%unet% --train=%mass_train_unet% --val=%test1_data% --test=%outputs_dir% --epochs=%epochs% --batch=%batch% --model_name=!cur_model! --sigma=!cur_sigma!
    IF %ERRORLEVEL% NEQ 0 (
        echo ERROR: "Error during unet prediction"
        exit /B 10
    )
    echo "PREDICT !cur_model! MODEL DONE"
    set /a "unet_counter=!unet_counter!+1"
)
echo "UNET TRAIN MODELS END"



::::::::::::::::::: MRCNN training :::::::::::::::::::
echo "MaskRCNN TRAININGS:"

python %maskrcnn_scripts%\train.py %maskrcnn%\config\train\train.json
IF %ERRORLEVEL% NEQ 0 (
    echo ERROR: "Error during maskrcnn training"
    exit /B 11
)
echo "MaskRCNN TRAINING DONE"

