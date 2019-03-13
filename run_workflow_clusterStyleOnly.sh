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
    export PYTHONPATH=$PYTHONPATH:$2
else
    export PYTHONPATH=$PYTHONPATH:"Mask_RCNN-2.1"
fi

####################### SET inputs #######################

WORKFLOW_ROOT="$ROOT_DIR/kaggle_workflow"

MATLAB_SCRIPTS="$ROOT_DIR/matlab_scripts"
OUTPUTS_DIR="$WORKFLOW_ROOT/outputs"

INPUT_IMAGES="$OUTPUTS_DIR/images"
PRESEGMENT="$OUTPUTS_DIR/presegment"
STYLE_AUGMENTATIONS="$OUTPUTS_DIR/augmentations/style"

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
echo "MATLAB_SCRIPTS: " $MATLAB_SCRIPTS
echo "OUTPUTS_DIR: " $OUTPUTS_DIR




####################### CLUSTERING #######################
echo "MATLAB CONFIG:"
matlab -nodisplay -nodesktop -nosplash -nojvm -r "addpath('${PIPELINE_SCRIPTS}');kaggleStartMatlab; setUpConfigAndDB('${MASK_DB_DIR}','${CLUSTER_CONFIG}','${CLUSTER_CONFIG}/pretrainedDistanceLearner.mat','${PIPELINE_SCRIPTS}'); exit;"
echo "MATLAB CONFIG DONE"


echo "CLUSTERING:"
matlab -nodisplay -nodesktop -nosplash -r "addpath('${PIPELINE_SCRIPTS}');kaggleStartMatlab;mergedImagesDir='${INPUT_IMAGES}';clusterDir='${CLUSTER_DIR}';sacFolder='${PIPELINE_SCRIPTS}/1_metalearning/matlab/sac/';clusteringType ='Kmeans-correlation-Best3Cluster';failCounter=0;canContinue=false;initialSegmentation='${PRESEGMENT}';runfromKaggleToClusters(mergedImagesDir,clusterDir,clusteringType,initialSegmentation,sacFolder,failCounter,canContinue); exit;"
echo "CLUSTERING DONE"

####################### MASK GENERATION #######################
matlab -nodisplay -nodesktop -nosplash -nojvm -r "addpath('${PIPELINE_SCRIPTS}'); kaggleStartMatlab; styleTransTrainDir='${STYLE_INPUTS}'; clusterDir='${CLUSTER_DIR}';initialSegmentation='${PRESEGMENT}'; splitOptionsFile='${CLUSTER_CONFIG}/basicOptions_02.csv'; artificialMaskDir='${SYNTHETIC_MASKS}'; fromClustersToStyles; exit;"
echo "MASK GENERATION DONE"

# clean up additional files after clustering
matlab -nodisplay -nodesktop -nosplash -r "addpath('${MATLAB_SCRIPTS}/utils');cleanUpAferClustering('${WORKFLOW_ROOT}'); exit;"

####################### STYLE TRANSFER #######################
python3 $PIPELINE_SCRIPTS/7_styaug/learn-styles2.py --work_dir "${STYLE_INPUTS}/0" --cur_dir $PIPELINE_SCRIPTS
python3 $PIPELINE_SCRIPTS/7_styaug/learn-styles2.py --work_dir "${STYLE_INPUTS}/1" --cur_dir $PIPELINE_SCRIPTS
python3 $PIPELINE_SCRIPTS/7_styaug/apply-styles.py --work_dir "${STYLE_INPUTS}/0" --cur_dir $PIPELINE_SCRIPTS
python3 $PIPELINE_SCRIPTS/7_styaug/apply-styles.py --work_dir "${STYLE_INPUTS}/1" --cur_dir $PIPELINE_SCRIPTS
python3 $PIPELINE_SCRIPTS/7_styaug/gen-output.py --work_dir "${STYLE_INPUTS}/0"
python3 $PIPELINE_SCRIPTS/7_styaug/gen-output.py --work_dir "${STYLE_INPUTS}/1"
python3 $PIPELINE_SCRIPTS/7_styaug/collect-split-outputs.py --splitted_dataset_dir ${STYLE_INPUTS} --out_dir ${STYLE_AUGMENTATIONS}

####################### CLUSTERING END #######################