import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', nargs='+', required=True, type=str)
parser.add_argument('--test_set', required=True, type=str)
args = parser.parse_args()

n_images = len(os.listdir(os.path.join(args.test_set, 'test')))

for model_name in args.model_name:
    command = """
    python test.py \
        --dataroot %s \
        --name %s \
        --model pix2pix \
        --which_model_netG unet_256 \
        --which_direction BtoA \
        --dataset_mode aligned \
        --norm batch \
        --max_dataset_size %d \
        --how_many %d
    """ % (args.test_set, model_name, n_images, n_images)

    os.system(command)
