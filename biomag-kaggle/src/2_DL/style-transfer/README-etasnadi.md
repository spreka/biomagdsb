The code to launch the pipeline is in the deploy directory.

The deply/matlab dir is version controlled therefore its contents is in the BIOMAG's git repo biomag-kaggle somewhere in the 0_preprocessing directory.

The python scripts are placed in the deploy/python it is not currently version controll'd yet.

The original Pix2Pix is slightly modified in order to be able to generate simple output images. The modified python files has the _cus suffix and there are only 2 of them as I remember. So this modified project is in the deploy/pytorch-CycleGAN-and-pix2pix

There is a wrapper written by Tivadar that calls the pix2pix framework. The original one is version controlled in the biomags biomag-kaggle git repo in the 2_DL/pix2pix? directory. I had modified this one too that is not synced there yet so it is in the deploy/pix2pix project dir.

The central thing in the python script is the glob_vars.py that defines mostly the relative paths of the input/output of the pipeline elements. These variables usually have the _REL suffix. It should not be modified as these could be any random string but...

The point is that there are 2 important ones that should be set'd up: P2P_FRAMEWORK_DIR and P2P_PROJECT_DIR. The first one should point to the modified p2p fw dir, and tah latter one shou point to the modified p2p wrapper shit.

As it was mentioned above, the input and output of the steps is the work directory $WORK_DIR it is passed to the python scripts in the --work_dir argument usually.

The $WORK_DIR is currently /home/biomag/etasnadi/trial6/data/tissue/ on my computer or /home/biomag/etasnadi/trial6/data/not-tissue/ cause we have two very different dataset: for tissue and not-tissue. We should consider them as two distinct dataset and each of them has its own work_dir and in an ideal universe we should run the pipeline to process them sequentally.

So we have an 'input' folder with the raw images in the WORK_DIR! We are working in the WORK_DIR always that is for now still tha: /home/biomag/etasnadi/trial6/data/tissue/ and /home/biomag/etasnadi/trial6/data/tissue/ for the two distinct dataset. I'ill show how to process everything for the different sets simultaneously.

MATLAB
1. clustering with splitting (it should be done with the classifier but for now I defined the clusters by splpitting...):

splitimages('/home/biomag/etasnadi/trial6/data/tissue/input', '/home/biomag/etasnadi/trial6/data/tissue/input-clusters', 256, 256, '.png');
splitimages('/home/biomag/etasnadi/trial6/data/not-tissue/input', '/home/biomag/etasnadi/trial6/data/not-tissue/input-clusters', 256, 256, '.png');

In this tep we clustered the images by splitting them and the cluster names became the images the patches came from. The resuld should be put in the 'input-clusters' directory.

now we have the clustered dataset in the input-clusters directory, each images of the $CLUSTED_ID will be put in the input-clusters/$CLUSTER_ID folder

PYTHON
2. create the mrcnn dataset with python:
python3 prepare_mrcnn_input /home/biomag/etasnadi/trial6/data/not-tissue/
python3 prepare_mrcnn_input /home/biomag/etasnadi/trial6/data/tissue/

This call collects the clustered images from the input-clusters folder and puts into the 'mrcnn-input/images' folder in the work directory.

SHELL
3. run mR-CNN segmentation:

The resuld should be placed in the 'mrcnn-preds' folder:

source ~/biomag/kriston/bin/activate
export PYTHONPATH=/home/biomag/etasnadi/dev/Mask_RCNN
python3 batch_segmentation.py /home/biomag/etasnadi/trial6/data/not-tissue/mrcnn-input /home/biomag/etasnadi/trial6/data/not-tissue/mrcnn-preds /home/biomag/etasnadi/dev/models/mask_rcnn_best.h5 true true false 0.05 0 true 0.5

This takes the images folder from the $WORK_DIR/mrcnn-input and predicts them as tiff format masks into the $WORK_DIR/mrcnn-preds folder.

PYTHON
4. Reorganise the predictions we got in the same format as the input-clusters are.

python3 organise_predictions.py /home/biomag/etasnadi/trial6/data/not-tissue/
python3 organise_predictions.py /home/biomag/etasnadi/trial6/data/tissue/

So from the flat format $WORK_DIR/mrcnn-preds will be reorganised to the $WORK_DIR/preds-clusters.

[Optional: generate the probability maps]
5.

mrcnn-input -> unet-out

The mrcnn has the same input folder as the UNet has.

python3 predict_sh.py --model /home/biomag/etasnadi/UNet_predict/models/UNet_not_tissue --dataset /home/biomag/etasnadi/trial6/data/not-tissue/mrcnn-input --output /home/biomag/etasnadi/trial6/data/not-tissue/unet-out

python3 predict_sh.py --model /home/biomag/etasnadi/UNet_predict/models/UNet_tissue --dataset /home/biomag/etasnadi/trial6/data/tissue/mrcnn-input --output /home/biomag/etasnadi/trial6/data/tissue/unet-out


5.5 Post-process the generated UNet files

Collects the predictions from the 'unet-out' folder to 'unet-preds'

python3 unet_postprocess.py /home/biomag/etasnadi/UNet_predict/models/UNet_not_tissue
python3 unet_postprocess.py /home/biomag/etasnadi/UNet_predict/models/tissue

[/optional]

MATLAB
6. Generate pix2pix training set:

genpix2pixtrain('/home/biomag/etasnadi/trial6/data/tissue/preds-clusters', '/home/biomag/etasnadi/trial6/data/tissue/input-clusters', '/home/biomag/etasnadi/trial6/data/tissue/p2ptrain');
genpix2pixtrain('/home/biomag/etasnadi/trial6/data/not-tissue/preds-clusters', '/home/biomag/etasnadi/trial6/data/not-tissue/input-clusters', '/home/biomag/etasnadi/trial6/data/not-tissue/p2ptrain');

Basically, the images in the 'preds-clusters' and 'input-clusters' directory will be used to generate the masks in the 'p2ptrain' directory keeping the original directory structure.

MATLAB
7. Generate synthetic masks:

generateMasksForCluster('/home/biomag/etasnadi/trial6/data/not-tissue/preds-clusters', '/home/biomag/etasnadi/trial6/data/not-tissue/generated', '.tiff', 20);
generateMasksForCluster('/home/biomag/etasnadi/trial6/data/tissue/preds-clusters', '/home/biomag/etasnadi/trial6/data/tissue/generated', '.tiff', 20);

The masks are in the $WORK_DIR/preds-cluster directory while the synthetic masks are placed in the $WORK_DIR/generated.

PYTHON
8. Apply masks:
This call takes the $WORK_DIR/generated folder (512*512 masks), and applies the learned styles in the $WORK_DIR/checkpoints folder to these masks. The synthetic images will be placed in the $WORK_DIR/p2psynthetic folder.
python3 apply-styles.py --work_dir /home/biomag/etasnadi/trial6/data/tissue
python3 apply-styles.py --work_dir /home/biomag/etasnadi/trial6/data/not-tissue

PYTHON
9.

Collect the generated synthetic images and their corresponding gound truth.

copies the contents of $WORK_DIR/generated/*/grayscale/* to out/masks/
and $WORK_DIR/p2psynthetic/*/* to out/images

python3 gen-output.py --work_dir /home/biomag/etasnadi/trial6/data/not-tissue

10. Additional scripts for handling splitted datasets:
collect-split-outputs.py --splitted_dataset_dir '...' --out_dir
Implements: 

cp splitted_dataset_dir/$SPLIT_ID/out/masks/* out/masks/*
cp splitted_dataset_dir/$SPLIT_ID/out/images/* out/images/*

collect-synthetic-masks.py --synthetic-masks-dir --splitted_dataset_dir

This implements the following: cp splitted-gen-images/$ID/generated/* splitted-dataset/$ID/generated/*

Workflow for style transfer using the google cloud.

0. prepare the splitted dataset:
splitted-dataset/
    $SPLIT_ID/
        input-preds/
            ...
        p2ptrain/
            ...
            
Zip the whole dataset.
zip -r splitted-dataset.zip ./splitted-dataset

Then we got a splitted-dataset.zip file containing the whole splitted dataset.
Upload it to the google cloud bucket named 'biomag-shared' into the style-transfer folder.

Launch the instances, and then
mount the biomag-shared bucket: 
~/scripts/mount-bucket.sh

copy the contents to the local home: 
cp ~/shared/splitted-dataset.zip ~

extract it into the ~/style-transfer-data folder: 
unzip ~/splitted-dataset.zip -d ~/style-transfer-data

launch the training on the first eight split (calls the learn-styles.py for every split):
$BIOMAG_KAGGLE/src/7_styaug/sh/splitted-train.sh ~/style-transfer-data/splitted-dataset 0 7

It takes a long time but the step 1 can be launched while the training runs.

1. Launch the synthetic mask generation, and split it into parts!
Then we get a directory structure like this:
splitted-synthetic-masks/
    $SPLIT_ID/
        generated/
            ...

zip the synthetic masks.
zip -r splitted-synthetic-masks.zip ./splitted-synthetic-masks

Upload it to the google cloud bucket named 'biomag-shared' into the style transfer folder.
On the instances: launch it if they are stopped and mount the 'biomag-shared' bucket.

copy the synthetic masks into the home folder:
cp ~/shared/splitted-synthetic-masks.zip ~

extract the contents into the ~/style transfer-data:
unzip ~/splitted-synthetic-masks.zip -d ~/style-transfer-data

merge the splitted-dataset with the synthetic masks:
python3 collect-synthetic-masks.py --synthetic_masks_dir ~/style-transfer-data/splitted-synthetic-masks --splitted_dataset_dir ~/style-transfer-data/splitted-dataset

apply the styles on the splits (calls the apply-styles.py for every split):
$BIOMAG_KAGGLE/src/7_styaug/sh/splitted-apply-styles.sh ~/style-transfer-data/splitted-dataset 0 7

Create the result in every split (calls the gen-output.py for every split):
$BIOMAG_KAGGLE/src/7_styaug/sh/splitted-gen-output.sh ~/style-transfer-data/splitted-dataset 0 7

Grab the results:
collect-split-outputs.py --splitted-dataset-dir ~/style-transfer-data/splitted-dataset --out_dir ~/style-transfer-data/result_`uname -n`

Archive it and copy to the shared bucket
zip -r ~/result.zip ~/style-transfer-data/result_`uname -n` && cp ~/result.zip ~/shared/style-transfer/result_`uname -n`.zip

Download the zips from the bucket.

for ZN in `ls`; do unzip $ZN; done

Merge the results
collec-split-outputs.py --splitted-dataset-dir . --out_dir ~/result


