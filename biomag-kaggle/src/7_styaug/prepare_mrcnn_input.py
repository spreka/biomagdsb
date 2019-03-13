import dbtools
import argparse
import glob_vars
import os.path
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', required=True, type=str)
args = parser.parse_args()

def prepare_mrcnn_input():
    print('Preparing mR-CNN training input...')
    db = dbtools.read_db(args.work_dir)
    mrcnn_input_path = os.path.join(args.work_dir, glob_vars.MRCNN_TRAIN_INPUT)
    print('mkdir {}'.format(mrcnn_input_path))
    os.makedirs(mrcnn_input_path, exist_ok=True)
    for clus_id in db.keys():
        for image_file_name in db[clus_id]:
            src = os.path.join(args.work_dir, glob_vars.IMAGE_CLUSTERS_REL_DIR, clus_id, image_file_name)
            dst = os.path.join(mrcnn_input_path, image_file_name)
            print('cp {} {}'.format(src, dst))
            shutil.copyfile(src, dst)


prepare_mrcnn_input()