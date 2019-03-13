To run the framework:

Put the ${dataset_name}.zip into the bucket/style-transfer folder.
If the masks are ready, put the ${dataset_name}_masks.zip into the same folder.

In the zip there is a root folder with the same name as the zip filename without its extension.

gcloud compute --project "fimm-imaging" ssh --zone "europe-west1-b" "lassi_paavolainen@p2p-worker-4

0. $ cd biomag-kaggle/src/7_styaug/sh
1. $ ./pretrain.sh $dataset_name
2. $ ./train-splits.sh $dataset_name $split_from $split_to
3. $ ./preapply.sh $dataset_name
4. $ ./apply-styles-splits.sh $dataset_name $split_from $split_to
5. $ ./gatger.sh $dataset_name

You will find the results in the bucket/style-transfer folder.
