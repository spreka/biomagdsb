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

INPUT_IMAGES="$OUTPUTS_DIR/images"

# check inputs:
echo "ROOT_DIR: " $ROOT_DIR
echo "WORKFLOW_ROOT: " $WORKFLOW_ROOT
echo "MASKRCNN_SCRIPTS: " $MASKRCNN_SCRIPTS
echo "MASKRCNN: " $MASKRCNN
echo "MATLAB_SCRIPTS: " $MATLAB_SCRIPTS
echo "OUTPUTS_DIR: " $OUTPUTS_DIR
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
#python3 $MASKRCNN_SCRIPTS/segmentation.py $MASKRCNN/config/predict/presegment.json
python3 $MASKRCNN_SCRIPTS/segmentation_manualsize.py $MASKRCNN/config/predict/presegment_manualsize.json
if [ $? -ne 0 ]
then
    echo ERROR: "Error during pre-segmentation"
    exit 1
fi
echo "PRESEGMENTATION DONE"