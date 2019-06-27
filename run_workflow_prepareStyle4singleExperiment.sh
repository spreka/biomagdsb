#!/bin/bash

if [ -z "$1" ]
  then
    echo "Usage: $0 settings_file"
    exit 1
fi

source $1


# vars from params.sh:
ROOT_DIR="$1"

if [ $# -eq 2 ]
then
    MASK_DIR=$3
else
    echo "using default mask folder: $OUTPUTS_DIR/presegment"
    MASK_DIR="$OUTPUTS_DIR/presegment"
fi

####################### SET inputs #######################

WORKFLOW_ROOT="$ROOT_DIR/kaggle_workflow"
MATLAB_SCRIPTS="$ROOT_DIR/matlab_scripts"
OUTPUTS_DIR="$WORKFLOW_ROOT/outputs"
INPUT_IMAGES="$OUTPUTS_DIR/images"
PIPELINE_SCRIPTS="$ROOT_DIR/biomag-kaggle/src"


MERGED_IMAGES_DIR="$INPUT_IMAGES"
INITIAL_SEGMENTATION="$MASK_DIR"
CLUSTER_DIR="$OUTPUTS_DIR/clusters"
CLUSTER_CONFIG="$WORKFLOW_ROOT/inputs/clustering"
MASK_DB_DIR="$CLUSTER_CONFIG/masks"
STYLE_INPUTS="$OUTPUTS_DIR/styleLearnInput"
SPLIT_OPTIONS_FILE="$WORKFLOW_ROOT/inputs/clustering/basicOptions_02.csv"
# default number of synthetic masks for style transfer:
declare -a MASKS2GENERATE=1000
# object type to generate on masks
OBJ_TYPE="nuclei"


# check inputs:
echo "ROOT_DIR: " $ROOT_DIR
echo "WORKFLOW_ROOT: " $WORKFLOW_ROOT
echo "MATLAB_SCRIPTS: " $MATLAB_SCRIPTS
echo "OUTPUTS_DIR: " $OUTPUTS_DIR
echo "IMAGES_DIR: " $IMAGES_DIR


echo "PREPARING STYLE TRANSFER INPUT FOR SINGLE EXPERIMENT:"
matlab -nodisplay -nodesktop -nosplash -nojvm -r "addpath('${MATLAB_SCRIPTS}/clustering4singleExperiment'); addpath(genpath('${PIPELINE_SCRIPTS}')); kaggleStartMatlab; setUpConfigAndDB('${MASK_DB_DIR}','${CLUSTER_CONFIG}','${CLUSTER_CONFIG}/pretrainedDistanceLearner.mat','${PIPELINE_SCRIPTS}'); prepareClustering4singleClusterFcn('${MERGED_IMAGES_DIR}','${INITIAL_SEGMENTATION}','${CLUSTER_DIR}','${STYLE_INPUTS}','${SPLIT_OPTIONS_FILE}',${MASKS2GENERATE},'${OBJ_TYPE}'); exit;"
if [ $? -ne 0 ]
then
    echo ERROR: "Error during style input preparation"
    exit 1
fi
echo "PREPARING STYLE TRANSFER INPUT FOR SINGLE EXPERIMENT DONE"

# remove dummy folder
rm -r "${STYLE_INPUTS}/1"