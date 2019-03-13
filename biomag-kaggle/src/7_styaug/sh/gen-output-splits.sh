#!/bin/bash

. vars.sh $1

logs_dir=$WD_PARENT/logs_$1
if [ ! -d "$logs_dir" ]; then
	mkdir $logs_dir
fi


NID=`uname -n`
for SPLIT_ID in `seq $2 $3`
do
    CMD="python3 $CODE_DIR/src/7_styaug/gen-output.py --work_dir ${dataset_dir}/${SPLIT_ID}"
    echo "Executing command: instance=${NID}, split,gpu_id=${SPLIT_ID}, command=$CMD"
    $CMD &> $logs_dir/gen-output_${NID}_${SPLIT_ID}.log &
done

wait
