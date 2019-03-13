import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test_set', required=True, type=str)
parser.add_argument('--output_dir', required=True, type=str)
parser.add_argument('--ppdir', required=True, type=str)
args = parser.parse_args()

# Iterates through the models
image_ids = os.listdir(args.test_set)
print(image_ids)
for im_id in image_ids:
    #model_name = im_id!
    masks_path = os.path.join(args.test_set, im_id)
    command = """
    python %s/test_cus.py \
        --dataroot '%s' \
        --name '%s' \
        --model pix2pix \
        --which_model_netG unet_256 \
        --which_direction BtoA \
        --dataset_mode aligned \
        --norm batch \
	--output_dir %s
    """ % (args.ppdir, masks_path, im_id, args.output_dir)

    print('Executing command for model: {}: {}.'.format(im_id, command))
    os.system(command)
