import os
import argparse
import glob_vars
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', required=True, type=str)
parser.add_argument('--synt_masks_rel_dir',required=False, type=str, default=glob_vars.P2P_TEST_MASKS_REL_DIR)  #default=generated
parser.add_argument('--synt_images_rel_dir', required=False, type=str, default=glob_vars.P2P_RESULT_REL_DIR)    #default=p2psynthetic
parser.add_argument('--out_rel_dir', required=False, type=str, default=glob_vars.P2P_FINAl_OUTPUT_REL_DIR)
args = parser.parse_args()


def generate_output():
    # Where to put the results...
    output_dir = os.path.join(args.work_dir, args.out_rel_dir)    #$WORK_DIR/out
    output_dir_images = os.path.join(output_dir, 'images')                          #$WORK_DIR/out/images
    output_dir_masks = os.path.join(output_dir, 'masks')                            #$WORK_DIR/out/masks
    print('mkdir {}'.format(output_dir_images))
    print('mkdir {}'.format(output_dir_masks))
    os.makedirs(output_dir_masks, exist_ok=True)
    os.makedirs(output_dir_images, exist_ok=True)

    # Copy the generated masks into the output
    synthetic_masks_dir = os.path.join(args.work_dir, args.synt_masks_rel_dir)      #$WORK_DIR/generated
    
    if os.path.isdir(synthetic_masks_dir):
        clusters = os.listdir(synthetic_masks_dir)
        print(clusters)
    else:
        print('no clusters found')
        return
    
    for clus_id in clusters:
        cluster_dir = os.path.join(synthetic_masks_dir, clus_id, 'grayscale')       #$WORK_DIR/generated/grayscale
        for mask_file_name in os.listdir(cluster_dir):
            mask_path = os.path.join(cluster_dir, mask_file_name)
            mask_target_path = os.path.join(output_dir_masks, mask_file_name)
            print('cp {} {}'.format(mask_path, mask_target_path))
            shutil.copyfile(mask_path, mask_target_path)

    # Copy the generated synthetic images into the output
    synthetic_path = os.path.join(args.work_dir, args.synt_images_rel_dir)          #$WORK_DIR/p2psynthetic
    for clus_id in os.listdir(synthetic_path):
        cluster_dir = os.path.join(synthetic_path, clus_id)
        for im_file_name in os.listdir(cluster_dir):
            im_path = os.path.join(cluster_dir, im_file_name)
            im_target_path = os.path.join(output_dir_images, im_file_name)
            print('cp {} {}'.format(im_path, im_target_path))
            shutil.copyfile(im_path, im_target_path)

generate_output()
