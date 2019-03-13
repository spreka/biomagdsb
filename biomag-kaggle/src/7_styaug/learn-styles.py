import os
import argparse
import shutil
import dbtools
import glob_vars

parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', required=True, type=str)
parser.add_argument('--gpu_ids', required=False, type=str, default='0')
args = parser.parse_args()

def prepare_style_learn_input(db, work_dir):
    styles_path = os.path.join(work_dir, glob_vars.P2P_TRAIN_REL_DIR) #target_dir/styles
    os.makedirs(styles_path, exist_ok=True)
    for clus_id in db.keys():
        clus_style_path = os.path.join(styles_path, clus_id, 'train') #target_dir/styles/clus1
        os.makedirs(clus_style_path)
        for image_id in db[clus_id]:
            src = os.path.join(work_dir, glob_vars.IMAGE_CLUSTERS_REL_DIR, clus_id, image_id)
            dst = os.path.join(work_dir, clus_style_path, image_id)
            shutil.copyfile(src, dst)


def learn_styles(n_iter, styles_dir, models_dir):
    command = """
    python3 %s/learn_styles.py \
    --n_iter %s \
    --style_root %s \
    --ppdir '%s' \
    --checkpoints_dir %s \
    --HTMLresults_dir %s \
    --gpu_ids %s \
    """ % (
        glob_vars.P2P_PROJECT_DIR,
        n_iter,
        styles_dir,
        glob_vars.P2P_FRAMEWORK_DIR,
        models_dir,
        os.path.join(args.work_dir, glob_vars.P2P_HTML_RESULT_REL_DIR),
        args.gpu_ids
    )

    print(command)
    os.system(command)


db = dbtools.read_db(args.work_dir)
print(db)
#prepare_style_learn_input(db, args.work_dir)
learn_styles(10000,
             os.path.join(args.work_dir, glob_vars.P2P_TRAIN_REL_DIR),
             os.path.join(args.work_dir, glob_vars.P2P_MODELS_REL_DIR))
