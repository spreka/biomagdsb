#!/bin/bash

. vars.sh $1

if [ "$#" -lt 3 ]; then
        echo "Usage: $0 dataset-name split-from spit-to"
else

	DATASET_NAME=$1

	NID=`uname -n`

	SPLIT_FROM=$2
	SPLIT_TO=$3

	logs_dir=$WD_PARENT/logs_${DATASET_NAME}
        if [ ! -d "$logs_dir" ]; then
                mkdir $logs_dir
        fi

	finished_file=$WD_PARENT/train_done_${DATASET_NAME}

	GPU_ID=0
	for SPLIT_ID in `seq $SPLIT_FROM $SPLIT_TO`
	do
	    CMD="python3 $CODE_DIR/src/7_styaug/learn-styles2.py --work_dir $dataset_dir/${SPLIT_ID} --gpu_ids ${GPU_ID}"
	    echo "Executing command: instance=${NID}, split,gpu_id=${SPLIT_ID},${GPU_ID}, command=$CMD"
	    $CMD &> $logs_dir/learn_styles_${NID}_${SPLIT_ID}.log &
	    let "GPU_ID+=1"
	done

	wait

	result_command="touch $finished_file"
	echo $result_command
	eval $result_command
fi

wait
