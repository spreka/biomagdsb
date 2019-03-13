import os
import argparse
import shutil
import glob_vars

parser = argparse.ArgumentParser()
parser.add_argument('--synthetic_masks_dir', required=True, type=str)
parser.add_argument('--splitted_dataset_dir', required=True, type=str)
args = parser.parse_args()

# Collects the generated mask splits:

'''

Dir structures:
synthetic_masks_dir/
    $SPLIT_ID/
        generated/
            $CLUS_ID/
                grayscale/
                    ...
                test/
                    ...
        
splitted_dataset_dir/
    $SPLIT_ID
        generated/
            $CLUS_ID/
                grascale/
                    ...
                test/
                    ...
        input-preds/
        ...

The script grabs the contents of the synthetic_masks_dir/$SPLIT_ID/generated/$CLUS_ID tree and puts it into the splitted_dataset_dir/$SPLIT_ID/generated.
        
'''

# This implements the following: cp synthetic_masks_dir/$ID/generated/* splitted-dataset/$ID/generated/*

for split_id in os.listdir(args.synthetic_masks_dir):
    # splitted_generated_images_dir/$ID/generated
    current_split_path = os.path.join(args.synthetic_masks_dir, split_id, glob_vars.P2P_TEST_MASKS_REL_DIR)
    # generated_images_dir/$ID
    for clus_id in os.listdir(current_split_path):
        # splitted_generated_images_dir/$SPLIT_ID/generated/$CLUS_ID
        src_dir = os.path.join(current_split_path, clus_id)
        # splitted_dataset/$SPLIT_ID/generated/$CLUS_ID
        dst_dir = os.path.join(args.splitted_dataset_dir, split_id, glob_vars.P2P_TEST_MASKS_REL_DIR, clus_id)

	# if the target directory exists, delete it. This case could happend when we want to replace the generated data with a new one.
        if os.path.isdir(dst_dir):
            shutil.rmtree(dst_dir)

        print('copytree {} {}'.format(src_dir, dst_dir))
        shutil.copytree(src_dir, dst_dir)
