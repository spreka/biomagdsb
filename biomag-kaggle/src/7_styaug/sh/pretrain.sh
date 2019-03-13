#!/bin/bash

DATASET_NAME=$1
FROM=$2
TO=$3

. vars.sh

if [ "$#" -lt 1 ]; then
	echo "Usage: $0 dataset-name"
else
	echo "Preparing the data for training..."

	bucket_zip_path="$BUCKET_MOUNT/style-transfer/${DATASET_NAME}.zip"
	zip_fname=$(basename -- "$bucket_zip_path")
	zip_target="$WD_PARENT/$zip_fname"

	echo "Dataset name: $DATASET_NAME"
	echo "Bucket zip path: " $bucket_zip_path
	echo "Zip filename: " $zip_fname
	echo "Zip target: $zip_target"

	echo "---------------------------------"

	mkdir_command="mkdir $WD_PARENT"
	echo $mkdir_command
	eval $mkdir_command

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
fi
