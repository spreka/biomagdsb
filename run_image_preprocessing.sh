#!/bin/bash

if [ -z "$1" ]
  then
    echo "Usage: $0 settings_file"
    exit 1
fi

source $1


# vars from params.sh:
ROOT_DIR="$1"



####################### SET inputs #######################

WORKFLOW_ROOT="$ROOT_DIR/kaggle_workflow"
MATLAB_SCRIPTS="$ROOT_DIR/matlab_scripts"
OUTPUTS_DIR="$WORKFLOW_ROOT/outputs"
INPUT_IMAGES="$OUTPUTS_DIR/images"
PIPELINE_SCRIPTS="$ROOT_DIR/biomag-kaggle/src"

if [ $2 = "images" ]
then
    echo "using default mask folder: $OUTPUTS_DIR/images"
    ORIG_FOLDER="$outputs_dir"
    SUB_FOLDER="images"
    OUTPUT_EXT=".png"
elif [ $2 = "masks" ]
	echo "using default mask folder: $WORKFLOW_ROOT/inputs/train_maskrcnn/masks"
	ORIG_FOLDER="$WORKFLOW_ROOT/inputs/train_maskrcnn"
    SUB_FOLDER="masks"
    OUTPUT_EXT=".tiff"
else
    echo "folder must be 'images' or 'masks'"
	exit 20
fi



# check inputs:
echo "ROOT_DIR: " $ROOT_DIR
echo "WORKFLOW_ROOT: " $WORKFLOW_ROOT
echo "MATLAB_SCRIPTS: " $MATLAB_SCRIPTS
echo "OUTPUTS_DIR: " $OUTPUTS_DIR



# preparing user test images for pre-processing

echo "MOVING USER IMAGES TO ${ORIG_FOLDER}/orig_${SUB_FOLDER}:"
mkdir " ${ORIG_FOLDER}/orig_${SUB_FOLDER}"
mv  ${ORIG_FOLDER}/${SUB_FOLDER}/*.* ${ORIG_FOLDER}/orig_${SUB_FOLDER}
echo "MOVING DONE"



echo "PRE-PROCESSING IMAGES TO 8-BIT 3 CHANNEL IMAGES:"
matlab -nodisplay -nodesktop -nosplash -nojvm -r "addpath('${MATLAB_SCRIPTS}/utils'); src='${ORIG_FOLDER}/orig_${SUB_FOLDER}/'; dest='${ORIG_FOLDER}/${SUB_FOLDER}/'; extOut='$OUTPUT_EXT'; run_preproc_16bit_image(src,dest,extOut); exit;"
if [ $? -ne 0 ]
then
    echo ERROR: "Error during image pre-processing"
    :: reorganize images to original folder
    echo "CLEANING UP INPUT IMAGE FOLDER"
    mkdir "%outputs_dir%\tmp"
    mv ${ORIG_FOLDER}/${SUB_FOLDER}/*.* ${ORIG_FOLDER}/tmp
    mv ${ORIG_FOLDER}/orig_${SUB_FOLDER}/*.* ${ORIG_FOLDER}/${SUB_FOLDER}
    rmdir ${ORIG_FOLDER}/orig_${SUB_FOLDER}
    echo "CLEANING UP DONE"
    exit 1
fi
echo "PRE-PROCESSING IMAGES TO 8-BIT 3 CHANNEL IMAGES DONE"
