import dbtools
import argparse
import glob_vars
import os.path
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', required=True, type=str)
args = parser.parse_args()

def organise_predictions():
    db = dbtools.read_db(args.work_dir)
    for clus_id in db.keys():
        for image_file_name in db[clus_id]:
            image_file_name_wo_ext = os.path.splitext(image_file_name)[0]
            mrcnn_mask_file_name = '{}.{}'.format(image_file_name_wo_ext, 'tiff')

            cluster_path = os.path.join(args.work_dir, glob_vars.PREDICTIONS_CLUSTERS_REL_DIR, clus_id)
            print('mkdir {}'.format(cluster_path))
            os.makedirs(cluster_path, exist_ok=True)
            src = os.path.join(args.work_dir, glob_vars.MRCNN_TRAIN_OUTPUT, mrcnn_mask_file_name)
            dst = os.path.join(cluster_path, mrcnn_mask_file_name)
            print('cp {} {}'.format(src, dst))
            shutil.copyfile(src, dst)
    pass


organise_predictions()