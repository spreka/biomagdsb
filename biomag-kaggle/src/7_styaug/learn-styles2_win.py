import os
import argparse
import shutil
import dbtools
import glob_vars

parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', required=True, type=str)
parser.add_argument('--gpu_ids', required=False, default='0', type=str)
parser.add_argument('--cur_dir', required=False, type=str)
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

'''
Directory structure:

p2ptrain/
    $CLUS_ID/
        train/
            img...

The $CLUS_ID is the style.

'''
def learn_styles(n_iter, styles_train_dir, models_dir):
    for style in os.listdir(styles_train_dir):
        style_path = os.path.join(styles_train_dir, style)
        style_train_path = os.path.join(style_path, 'train')

        n_files = len(os.listdir(style_train_path))
        print('Train directory for style: {}. Num of files: {}.'.format(style_train_path, n_files))

        num_of_iters = n_iter // n_files

        model_name = style
        command = """
        python %s/train.py \
        --dataroot %s \
        --name %s \
        --model pix2pix \
        --which_model_netG unet_256 \
        --which_direction BtoA \
        --lambda_A 100 \
        --dataset_mode aligned \
        --no_lsgan \
        --norm batch \
        --pool_size 0 \
        --save_epoch_freq %d \
        --niter %d \
        --checkpoints_dir %s \
        --gpu_ids %s \
        --display_id 0 \
        """ % (
            os.path.join(args.cur_dir,glob_vars.P2P_FRAMEWORK_DIR),
            style_path, #dataroot
            model_name, #model_name
            num_of_iters+1000, #save_epoch_freq
            num_of_iters, #niter
            models_dir, #checkpoints_dir
            args.gpu_ids #gpu_ids
        )

        print(command)
        os.system(command)


#db = dbtools.read_db(args.work_dir)
#print(db)
#prepare_style_learn_input(db, args.work_dir)
learn_styles(5000,
             os.path.join(args.work_dir, glob_vars.P2P_TRAIN_REL_DIR),
             os.path.join(args.work_dir, glob_vars.P2P_MODELS_REL_DIR))
