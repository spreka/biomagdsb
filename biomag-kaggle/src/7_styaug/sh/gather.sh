#!/bin/bash

. vars.sh $1

DATASET_NAME=$1

if [ "$#" -lt 1 ]; then
        echo "Usage: $0 dataset-name"
else
	echo "Gatgering the results and copying them to the bucket."
        dataset_dir="$WD_PARENT/$DATASET_NAME"

        echo "Dataset name: $DATASET_NAME"

        #gen_output_cmd="python3 $CODE_DIR/src/7_styaug/gen-output.py --work_dir $dataset_dir"
	#echo $gen_output_cmd
	#eval $gen_output_cmd

	NID=`uname -n`

	out_folder_name="${DATASET_NAME}_$NID"
	output_path="$dataset_dir/$out_folder_name"

	collect_splits_cmd="python3 $CODE_DIR/src/7_styaug/collect-split-outputs.py --splitted_dataset_dir $dataset_dir --out_dir $output_path"
	echo $collect_splits_cmd
	eval $collect_splits_cmd

	dat=`date +%s`
	zip_fname="${out_folder_name}_${dat}.zip"
	zip_cmd="zip -r $dataset_dir/$zip_fname $output_path"
	echo $zip_cmd
	eval $zip_cmd

	~/mount-bucket.sh
	cp_bucket_cmd="cp $dataset_dir/$zip_fname $BUCKET_MOUNT/style-transfer/$zip_fname" 
	echo $cp_bucket_cmd
	eval $cp_bucket_cmd
	sudo ~/umount-bucket.sh
fi
