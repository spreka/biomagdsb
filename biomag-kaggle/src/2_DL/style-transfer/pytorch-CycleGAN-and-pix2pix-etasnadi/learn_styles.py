import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n_iter', default=1000, type=int)
parser.add_argument('--style_root', required=True, type=str)
parser.add_argument('--checkpoints_dir', required=True, type=str)
args = parser.parse_args()

# style_root = '/home/biomag/tivadar/pytorch-CycleGAN-and-pix2pix/datasets/csaba/styles'

for style in os.listdir(args.style_root):
    style_path = os.path.join(args.style_root, style)
    model_name = style
    command = """
    python train.py \
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
    --checkpoints_dir %s
    """ % (style_path, model_name, args.n_iter + 99,args.n_iter, args.checkpoints_dir)

    os.system(command)
