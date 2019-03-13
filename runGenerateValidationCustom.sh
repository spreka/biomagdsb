#!/bin/bash

ROOT_DIR="."

if [ -z "$1" ]
  then
    echo "Usage: $0 validation_folder"
    exit 1
fi 

IMAGES_DIR="$1"

if [ -z "$2" ]
then
    TARGET_DIR="$ROOT_DIR/matlab_scripts/"
else
    TARGET_DIR=$2
fi 

echo "GENERATING VALIDATION NAMES:"
matlab -nodisplay -nodesktop -nosplash -nojvm -r "addpath('${TARGET_DIR}/generateValidation');generateValidationCustom('${TARGET_DIR}','${IMAGES_DIR}');exit;"
if [ $? -ne 0 ]
then
    echo ERROR: "Error during validation generation"
    exit 1
fi
echo "GENERATING VALIDATION NAMES DONE"
