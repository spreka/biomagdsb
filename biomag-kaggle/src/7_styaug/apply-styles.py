import os
import argparse
import glob_vars

parser = argparse.ArgumentParser()
parser.add_argument('--work_dir', required=True, type=str)
parser.add_argument('--fine_size', required=False, type=str, default='512')
parser.add_argument('--gpu_ids', required=False, type=str, default='0')
parser.add_argument('--synt_masks_rel_dir', required=False, type=str, default=glob_vars.P2P_TEST_MASKS_REL_DIR)
parser.add_argument('--synt_images_rel_dir', required=False, type=str, default=glob_vars.P2P_RESULT_REL_DIR)
parser.add_argument('--cur_dir', required=False, type=str)
args = parser.parse_args()

# Iterates through the synthetic masks directory

# Applies the style to the set of images
def apply_styles_dataset(work_dir):
    result_path = os.path.join(work_dir, glob_vars.P2P_RESULT_REL_DIR)
    os.makedirs(result_path, exist_ok=True)
    # Iterates through the synthetic masks
    synthetic_masks_dir = os.path.join(work_dir, args.synt_masks_rel_dir)
    synthetic_images_dir = os.path.join(work_dir, args.synt_images_rel_dir)
    checkpoints_dir =  os.path.join(work_dir, glob_vars.P2P_MODELS_REL_DIR)
    print('Checking directory: {}'.format(synthetic_masks_dir))
    if os.path.isdir(synthetic_masks_dir):
        clusters = os.listdir(synthetic_masks_dir)
        print(clusters)
    else:
        print('no clusters found')
        return
    
    for clus_id in clusters:
        os.makedirs(os.path.join(synthetic_images_dir, clus_id), exist_ok=True)
        print('Cluster: {}'.format(clus_id))
        command = """
        python3 %s/test_cus.py \
            --dataroot '%s' \
            --name '%s' \
            --model pix2pix \
            --which_model_netG unet_256 \
            --which_direction BtoA \
            --dataset_mode aligned \
            --norm batch \
            --checkpoints_dir %s \
            --output_dir %s \
            --fineSize %s \
            --nThreads 1 \
            --gpu_ids %s
        """ % (
            os.path.join(args.cur_dir,glob_vars.P2P_FRAMEWORK_DIR),
            os.path.join(synthetic_masks_dir, clus_id),
            clus_id,
            checkpoints_dir,
            os.path.join(synthetic_images_dir, clus_id),
            args.fine_size,
            args.gpu_ids)

        print('Executing command for model: {}: {}.'.format(clus_id, command))
        os.system(command)


apply_styles_dataset(args.work_dir)
