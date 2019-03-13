#!/bin/bash

. vars.sh

DATASET_NAME=$1

if [ "$#" -lt 1 ]; then
        echo "Usage: $0 dataset-name"
else
	echo "Preparing the dataset to apply on the synthetic masks."

        bucket_zip_path="$BUCKET_MOUNT/style-transfer/${DATASET_NAME}_masks.zip"
        zip_fname=$(basename -- "$bucket_zip_path")
        zip_target="$WD_PARENT/$zip_fname"
        dataset_dir="$WD_PARENT/$DATASET_NAME"

        echo "Dataset name: $DATASET_NAME"
        echo "Bucket zip path: " $bucket_zip_path
        echo "Zip filename: " $zip_fname
        echo "Zip target: $zip_target"

        echo "---------------------------------"

        mount_command="~/mount-bucket.sh"
        echo $mount_command
        eval $mount_command

        cp_command="cp $bucket_zip_path $zip_target"
        echo $cp_command
        eval $cp_command

        unmount_command="sudo ~/umount-bucket.sh"
        echo $unmount_command
        eval $unmount_command

        unzip_command="unzip $zip_target -d $WD_PARENT"
        echo $unzip_command
        eval $unzip_command

	merge_styles_command="python3 ${CODE_DIR}/src/7_styaug/collect-synthetic-masks.py --synthetic_masks_dir ${WD_PARENT}/${DATASET_NAME}_masks --splitted_dataset_dir ${dataset_dir}"
	echo $merge_styles_command
	eval $merge_styles_command
	
fi
