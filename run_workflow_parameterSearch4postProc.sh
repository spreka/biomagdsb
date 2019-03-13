#!/bin/bash

# vars from params.sh:
ROOT_DIR="$1"

MATLAB_SCRIPTS="$ROOT_DIR/matlab_scripts"

# check inputs:
echo "ROOT_DIR: " $ROOT_DIR
echo "MATLAB_SCRIPTS: " $MATLAB_SCRIPTS


echo "PARAMETER SEARCHING FOR POST PROCESSING:"
matlab -nodisplay -nodesktop -nosplash -nojvm -r "addpath(genpath(${MATLAB_SCRIPTS})); rootDir=%root_dir%; startPostProcParamSearch(${ROOT_DIR});exit"
if [ $? -ne 0 ]
then
    echo ERROR: "Error during cell size estimation"
    exit 1
fi
echo "PARAMETER SEARCHING FOR POST PROCESSING DONE" 