#!/bin/bash

export PYTHONPATH=$PYTHONPATH:~/deeplearning/Mask_RCNN

if [ -z "$1" ]
  then
    echo "Usage: $0 settings_file"
    exit 1
fi

#python3 train.py $1
if [ $? -ne 0 ]
then
    echo ERROR: "Error during training"
    exit 1
fi

python3 segmentation.py $1
if [ $? -ne 0 ]
then
    echo ERROR: "Error during segmentation"
    exit 1
fi

python3 eval_image.py $1
if [ $? -ne 0 ]
then
    echo ERROR: "Error during evaluation"
    exit 1
fi
