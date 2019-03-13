import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', nargs='+', required=True, type=str)
parser.add_argument('--test_set', required=True, type=str)
parser.add_argument('--ppdir', required=True, type=str)
args = parser.parse_args()

# Iterates through the models
for model_name in args.model_name:
    command = """
    python %s/test.py \
        --dataroot '%s' \
        --name '%s' \
        --model pix2pix \
        --which_model_netG unet_256 \
        --which_direction BtoA \
        --dataset_mode aligned \
        --norm batch
    """ % (args.ppdir, args.test_set, model_name)

    print('Executing command for model: {}: {}.'.format(model_name, command))
    os.system(command)
~                        

