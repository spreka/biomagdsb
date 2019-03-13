#!/bin/bash

# vars from params.sh:
ROOT_DIR="$1"
IMAGES_DIR="$2"

if [ $# -eq 3 ]
then
    export PYTHONPATH=$PYTHONPATH:$3
else
    export PYTHONPATH=$PYTHONPATH:"Mask_RCNN-2.1"
fi


WORKFLOW_ROOT="$ROOT_DIR/kaggle_workflow"

MASKRCNN_SCRIPTS="$ROOT_DIR/FinalModel"
MASKRCNN="$WORKFLOW_ROOT/maskrcnn"
MATLAB_SCRIPTS="$ROOT_DIR/matlab_scripts"
OUTPUTS_DIR="$WORKFLOW_ROOT/outputs"
CELLSIZE_EST="$OUTPUTS_DIR/cellSizeEstimator"

INPUT_IMAGES="$OUTPUTS_DIR/images"
INPUT_IMAGES_2X="$OUTPUTS_DIR/2ximages/images"
UNET="$WORKFLOW_ROOT/unet"
UNET_OUT="$OUTPUTS_DIR/unet_out"
UNET_SCRIPTS="$ROOT_DIR/UNet"
MASS_TRAIN_UNET="$OUTPUTS_DIR/train_unet"
VALIDATION="$OUTPUTS_DIR/validation"
ENSEMBLE="$OUTPUTS_DIR/ensemble"
POST_PROCESSING="$OUTPUTS_DIR/postprocessing"

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

# run prediction only --- from run_workflow.sh ---

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


echo "2x IMAGES:"
matlab -nodisplay -nodesktop -nosplash -nojvm -r "addpath('${MATLAB_SCRIPTS}/utils');resizeSet_onlyImages('${INPUT_IMAGES}','${INPUT_IMAGES_2X}/',2);exit;"
if [ $? -ne 0 ]
then
    echo ERROR: "Error during image 2x resize"
    exit 1
fi
echo "2x IMAGES DONE"


####################### UNET segmentation #######################
echo "SET UNET PARAMS:"
declare -a unet_models=("UNet_sigma0.0_1" "UNet_sigma0.0_2" "UNet_sigma0.5_1" "UNet_sigma0.5_2" "UNet_sigma0.5_3" "UNet_sigma1.0_1" "UNet_sigma2.0_1")
epochs=100
batch=12


echo "UNET PREDICTION TO IMAGES:"
for m in ${unet_models[@]}
do
    echo "PREDICT ${m} MODEL:"
    python3 $UNET_SCRIPTS/train_sh.py --results_folder=$UNET_OUT --train=$MASS_TRAIN_UNET --val=$MASS_TRAIN_UNET --test=$OUTPUTS_DIR --batch=1 --model_path=$UNET/${m}/${m} --model_name=${m}
    if [ $? -ne 0 ]
    then
        echo ERROR: "Error during unet prediction"
        exit 1
    fi
    echo "PREDICT ${m} MODEL DONE"
done
echo "UNET PREDICTION TO IMAGES END"


echo "ENSEMBLE UNET RESULTS:"
matlab -nodisplay -nodesktop -nosplash -nojvm -r "addpath('${MATLAB_SCRIPTS}/UNETensembling');ensembleProbFolders('${UNET_OUT}/','${ENSEMBLE}/output/');exit;"
if [ $? -ne 0 ]
then
    echo ERROR: "Error during augmentation 1"
    exit 1
fi
echo "ENSEMBLE UNET RESULTS DONE"





####################### MRCNN SEGMENTATION #######################
echo "SEGMENT: segmentation_params2x.json"
python3 ${MASKRCNN_SCRIPTS}/segmentation.py ${MASKRCNN}/config/predict/segmentation_params_2x.json
if [ $? -ne 0 ]
then
   echo ERROR: "Error during maskrcnn prediction"
   exit 1
fi
echo "MaskRCNN SEGMENTATION DONE"


echo "SEGMENT: segmentation_params4x.json"
python3 ${MASKRCNN_SCRIPTS}/segmentation.py ${MASKRCNN}/config/predict/segmentation_params_4x.json
if [ $? -ne 0 ]
then
   echo ERROR: "Error during maskrcnn prediction"
   exit 1
fi
echo "MaskRCNN SEGMENTATION DONE"
####################### MRCNN SEGMENTATION #######################


echo "POSTPROCESSING:"
matlab -nodisplay -nodesktop -nosplash -nojvm -r "addpath('${MATLAB_SCRIPTS}/postProcess');postProcCodeRunnerFINAL('${OUTPUTS_DIR}/maskrcnn/','${POST_PROCESSING}/','2x/','4x/','','master','${ENSEMBLE}/output/','final',false,'${INPUT_IMAGES}',[]);exit;"
if [ $? -ne 0 ]
then
    echo ERROR: "Error during postprocessing"
    exit 1
fi
echo "POSTPROCESSING DONE"


# --- delete dummy folders ---
rm -r $MASS_TRAIN_UNET
