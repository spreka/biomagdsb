#!/bin/bash

if [ -z "$1" ]
  then
    echo "Usage: $0 settings_file"
    exit 1
fi

source $1


# vars from params.sh:
ROOT_DIR="$1"
IMAGES_DIR="$2"

if [ $# -eq 3 ]
then
    export PYTHONPATH=$PYTHONPATH:$3
else
    export PYTHONPATH=$PYTHONPATH:"Mask_RCNN-2.1"
fi

####################### SET inputs #######################

WORKFLOW_ROOT="$ROOT_DIR/kaggle_workflow"

MASKRCNN_SCRIPTS="$ROOT_DIR/FinalModel"
MASKRCNN="$WORKFLOW_ROOT/maskrcnn"
MATLAB_SCRIPTS="$ROOT_DIR/matlab_scripts"
OUTPUTS_DIR="$WORKFLOW_ROOT/outputs"
CELLSIZE_EST="$OUTPUTS_DIR/cellSizeEstimator"

INPUT_IMAGES="$OUTPUTS_DIR/images"
INPUT_IMAGES_2X="$OUTPUTS_DIR/2ximages/images"
ORIGINAL_DATA="$WORKFLOW_ROOT/inputs/original"
TEST1="$WORKFLOW_ROOT/inputs/test1"
TEST1_DATA="$OUTPUTS_DIR/test1"
VALIDATION="$OUTPUTS_DIR/validation"
VALIDATION_IMAGES_2X="$OUTPUTS_DIR/2xvalidation/images"
UNET="$WORKFLOW_ROOT/unet"
UNET_OUT="$OUTPUTS_DIR/unet_out"
UNET_SCRIPTS="$ROOT_DIR/UNet"
TRAIN_UNET="$WORKFLOW_ROOT/inputs/train_unet"
MASS_TRAIN_UNET="$OUTPUTS_DIR/train_unet"
ENSEMBLE="$OUTPUTS_DIR/ensemble"
TRAIN_MASKRCNN="$WORKFLOW_ROOT/inputs/train_maskrcnn"
MASS_TRAIN_MASKRCNN="$OUTPUTS_DIR/train_maskrcnn"
PRESEGMENT="$OUTPUTS_DIR/presegment"
BASE_AUGMENTATIONS="$OUTPUTS_DIR/augmentations/base"
STYLE_AUGMENTATIONS="$OUTPUTS_DIR/augmentations/style"
POST_PROCESSING="$OUTPUTS_DIR/postprocessing"

PIPELINE_SCRIPTS="$ROOT_DIR/biomag-kaggle/src"
CLUSTER_CONFIG="$WORKFLOW_ROOT/inputs/clustering"
MASK_DB_DIR="$CLUSTER_CONFIG/masks"
CLUSTER_DIR="$OUTPUTS_DIR/clusters"
#SYNTHETIC_MASKS="$OUTPUTS_DIR/syntheticMasks"
STYLE_INPUTS="$OUTPUTS_DIR/styleLearnInput"
SYNTHETIC_MASKS=$STYLE_INPUTS

# check inputs:
echo "ROOT_DIR: " $ROOT_DIR
echo "WORKFLOW_ROOT: " $WORKFLOW_ROOT
echo "MASKRCNN_SCRIPTS: " $MASKRCNN_SCRIPTS
echo "MASKRCNN: " $MASKRCNN
echo "MATLAB_SCRIPTS: " $MATLAB_SCRIPTS
echo "OUTPUTS_DIR: " $OUTPUTS_DIR
echo "CELLSIZE_EST: " $CELLSIZE_EST
echo "IMAGES_DIR: " $IMAGES_DIR

# copy user test images to our expected images folder
echo "COPYING USER IMAGES TO $OUTPUTS_DIR/images:"
mkdir -p "$OUTPUTS_DIR/images"
cp "$IMAGES_DIR/"*.* "$OUTPUTS_DIR/images"
echo "COPYING DONE"

# create dummy folders expected by unet prediction (are unused)
mkdir -p "$MASS_TRAIN_UNET/images"
mkdir -p "$UNET_OUT"




# ----- execution of pipeline ------

echo "Kaggle workflow path: $WORKFLOW_ROOT"
echo "Original data path: $ORIGINAL_DATA"





echo "INIT UNET MASS TRAIN:"
mkdir -p $MASS_TRAIN_UNET
cp -r $TRAIN_UNET/images $MASS_TRAIN_UNET
cp -r $TRAIN_UNET/masks $MASS_TRAIN_UNET
cp -r $TEST1_DATA/images $MASS_TRAIN_UNET
cp -r $TEST1_DATA/masks $MASS_TRAIN_UNET
echo "INIT UNET MASS TRAIN DONE"


echo "INIT MRCNN MASS TRAIN:"
mkdir -p $MASS_TRAIN_MASKRCNN
cp -r $TRAIN_MASKRCNN/images $MASS_TRAIN_MASKRCNN
cp -r $TRAIN_MASKRCNN/masks $MASS_TRAIN_MASKRCNN
echo "INIT MRCNN MASS TRAIN DONE"


####################### MRCNN presegmentation #######################
echo "PRESEGMENTATION (maskrcnn):"
python3 $MASKRCNN_SCRIPTS/segmentation.py $MASKRCNN/config/predict/presegment.json
if [ $? -ne 0 ]
then
    echo ERROR: "Error during pre-segmentation"
    exit 1
fi
echo "PRESEGMENTATION DONE"


echo "CELL SIZE ESTIMATION:"
matlab -nodisplay -nodesktop -nosplash -nojvm -r "addpath('${MATLAB_SCRIPTS}/cellSizeEstimateForPrediction');cellSizeDataGenerate('${OUTPUTS_DIR}/presegment/','${CELLSIZE_EST}/images/');exit;"
if [ $? -ne 0 ]
then
    echo ERROR: "Error during cell size estimation"
    exit 1
fi
echo "CELL SIZE ESTIMATION DONE"

####################### CLUSTERING #######################

# clustering has already been done by running the start_singleExperimentPreparation.sh

####################### STYLE TRANSFER #######################
python3 $PIPELINE_SCRIPTS/7_styaug/learn-styles2.py --work_dir "${STYLE_INPUTS}/0" --cur_dir $PIPELINE_SCRIPTS
python3 $PIPELINE_SCRIPTS/7_styaug/apply-styles.py --work_dir "${STYLE_INPUTS}/0" --cur_dir $PIPELINE_SCRIPTS
python3 $PIPELINE_SCRIPTS/7_styaug/gen-output.py --work_dir "${STYLE_INPUTS}/0"
python3 $PIPELINE_SCRIPTS/7_styaug/collect-split-outputs.py --splitted_dataset_dir ${STYLE_INPUTS} --out_dir ${STYLE_AUGMENTATIONS}

####################### CLUSTERING END #######################

echo "2x IMAGES:"
matlab -nodisplay -nodesktop -nosplash -nojvm -r "addpath('${MATLAB_SCRIPTS}/utils');resizeSet_onlyImages('${INPUT_IMAGES}','${INPUT_IMAGES_2X}/',2);exit;"
if [ $? -ne 0 ]
then
    echo ERROR: "Error during image 2x resize"
    exit 1
fi
echo "2x IMAGES DONE"

####################### AUGMENTATION #######################
echo "BASE AUGMENTATION (0):"
matlab -nodisplay -nodesktop -nosplash -nojvm -r "addpath('${MATLAB_SCRIPTS}/augmentation');augmentTrainingSet('${TEST1_DATA}','${BASE_AUGMENTATIONS}/',0);exit;"
if [ $? -ne 0 ]
then
    echo ERROR: "Error during augmentation 0"
    exit 1
fi
echo "BASE AUGMENTATION (0) DONE"


#'TODO: move after style transfer data is ready'
echo "BASE AUGMENTATION (1):"
matlab -nodisplay -nodesktop -nosplash -nojvm -r "addpath('${MATLAB_SCRIPTS}/augmentation');augmentTrainingSet('${STYLE_INPUTS}','${STYLE_AUGMENTATIONS}/',1,5);exit;"
if [ $? -ne 0 ]
then
    echo ERROR: "Error during augmentation 1"
    exit 1
fi
echo "BASE AUGMENTATION (1) DONE"


echo "COPY AUGMENTATIONS TOGETHER:"
cp -r $BASE_AUGMENTATIONS/images $MASS_TRAIN_MASKRCNN
cp -r $BASE_AUGMENTATIONS/masks $MASS_TRAIN_MASKRCNN
cp -r $STYLE_AUGMENTATIONS/images $MASS_TRAIN_MASKRCNN
cp -r $STYLE_AUGMENTATIONS/masks $MASS_TRAIN_MASKRCNN
cp -r $STYLE_AUGMENTATIONS/images $MASS_TRAIN_UNET
cp -r $STYLE_AUGMENTATIONS/masks $MASS_TRAIN_UNET
echo "COPY AUGMENTATIONS TOGETHER DONE"


CONT_MAT_FILE="$MATLAB_SCRIPTS/generateValidation/validationFileNames.mat"
echo "GENERATE VALIDATION DATA FROM MASS_TRAIN_MASKRCNN:"
matlab -nodisplay -nodesktop -nosplash -nojvm -r "addpath('${MATLAB_SCRIPTS}/generateValidation');moveToValidation('${MASS_TRAIN_MASKRCNN}/','${VALIDATION}/','${CONT_MAT_FILE}');exit;"
if [ $? -ne 0 ]
then
    echo ERROR: "Error during generate validation maskrcnn"
    exit 1
fi
echo "GENERATE VALIDATION DATA FROM MASS_TRAIN_MASKRCNN DONE"


echo "GENERATE VALIDATION DATA FROM MASS_TRAIN_UNET:"
matlab -nodisplay -nodesktop -nosplash -nojvm -r "addpath('${MATLAB_SCRIPTS}/generateValidation');moveToValidation('${MASS_TRAIN_UNET}/','${VALIDATION}/','${CONT_MAT_FILE}');exit;"
if [ $? -ne 0 ]
then
    echo ERROR: "Error during generate validation unet"
    exit 1
fi
echo "GENERATE VALIDATION DATA FROM MASS_TRAIN_UNET DONE"


echo "2x VALIDATION IMAGES:"
matlab -nodisplay -nodesktop -nosplash -nojvm -r "addpath('${MATLAB_SCRIPTS}/utils');resizeSet_onlyImages('${VALIDATION}/images','${VALIDATION_IMAGES_2X}/',2);exit;"
if [ $? -ne 0 ]
then
    echo ERROR: "Error during validation image 2x resize"
    exit 1
fi
echo "2x VALIDATION IMAGES DONE"


echo "CELL SIZE ESTIMATION ON VALIDATION:"
matlab -nodisplay -nodesktop -nosplash -nojvm -r "addpath('${MATLAB_SCRIPTS}/cellSizeEstimateForPrediction');cellSizeDataGenerate('${VALIDATION}/masks/','${CELLSIZE_EST}/validation/');exit;"
if [ $? -ne 0 ]
then
    echo ERROR: "Error during cell size estimation (validation images)"
    exit 1
fi
echo "CELL SIZE ESTIMATION ON VALIDATION DONE"


####################### U-NET training #######################
echo "INIT UNET ENVIRONMENT:"
#mkdir -p $UNET
#mkdir -p $UNET_VALIDATION
echo "INIT UNET ENVIRONMENT DONE"


echo "UNET TRAIN MODELS:"
declare -a unet_models=("UNet_sigma0.0_1" "UNet_sigma0.0_2" "UNet_sigma0.5_1" "UNet_sigma0.5_2" "UNet_sigma0.5_3" "UNet_sigma1.0_1" "UNet_sigma2.0_1")
declare -a sigmas=(0.0 0.0 0.5 0.5 0.5 1.0 2.0)
epochs=100
batch=12
unet_counter=0

echo "UNET TRAIN MODELS:"
for m in ${unet_models[@]}
do
    echo "PREDICT ${m} MODEL:"
    python3 $UNET_SCRIPTS/train_sh.py --results_folder=$UNET --train=$MASS_TRAIN_UNET --val=$TEST1_DATA --test=$OUTPUTS_DIR --epochs=$epochs --batch=$batch --model_name=${m} --sigma=${sigmas[$unet_counter]}
    if [ $? -ne 0 ]
    then
        echo ERROR: "Error during unet prediction"
        exit 1
    fi
    echo "PREDICT ${m} MODEL DONE"
    unet_counter=$unet_counter+1
done
echo "UNET TRAIN MODELS END"



####################### MRCNN training #######################
echo "MaskRCNN TRAININGS:"

python3 $MASKRCNN_SCRIPTS/train.py $MASKRCNN/config/train/train.json
if [ $? -ne 0 ]
then
    echo ERROR: "Error during maskrcnn training"
    exit 1
fi
echo "MaskRCNN TRAINING DONE"

