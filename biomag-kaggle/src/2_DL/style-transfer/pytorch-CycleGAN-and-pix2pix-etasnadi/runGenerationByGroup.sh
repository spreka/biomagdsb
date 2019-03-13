#!/bin/bash

for i in $(seq 1 1 16)
do
    python generate_images.py --model_name $i --test_set /home/biomag/szkabel/180330_MORE/group_$i
done
