import os
import argparse
import shutil
import glob_vars

parser = argparse.ArgumentParser()
parser.add_argument('--splitted_dataset_dir', required=True, type=str)
parser.add_argument('--out_dir', required=True, type=str)
parser.add_argument('--split_out_rel_dir', required=False, type=str, default=glob_vars.P2P_FINAl_OUTPUT_REL_DIR)
args = parser.parse_args()

# This implements:
# cp splitted_dataset_dir/$SPLIT_ID/out/masks/* out/masks/*
# cp splitted_dataset_dir/$SPLIT_ID/out/images/* out/images/*


def copy_contents(srcdir, dstdir):
    for f in os.listdir(srcdir):
        src = os.path.join(srcdir, f)
        dst = os.path.join(dstdir, f)
        print('cp {} {}'.format(src, dst))
        shutil.copyfile(src, dst)


for split_id in os.listdir(args.splitted_dataset_dir):
    srcdir_masks = os.path.join(args.splitted_dataset_dir, split_id, args.split_out_rel_dir, 'masks')
    dstdir_masks = os.path.join(args.out_dir, 'masks')
    if not os.path.isdir(srcdir_masks):
       print('{} does not exist!'.format(srcdir_masks))
       continue

    print('mkdir {}'.format(dstdir_masks))
    os.makedirs(dstdir_masks, exist_ok=True)
    print('cpcontents {} {}'.format(srcdir_masks, dstdir_masks))
    copy_contents(srcdir_masks, dstdir_masks)

    srcdir_images = os.path.join(args.splitted_dataset_dir, split_id, args.split_out_rel_dir, 'images')
    dstdir_images = os.path.join(args.out_dir, 'images')
    print('mkdir {}'.format(dstdir_images))
    os.makedirs(dstdir_images, exist_ok=True)
    print('cpcontents {} {}'.format(srcdir_images, dstdir_images))
    copy_contents(srcdir_images, dstdir_images)
