import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n_iter', default=1000, type=int)
parser.add_argument('--style_root', required=True, type=str)
parser.add_argument('--ppdir', required=True, type=str)
parser.add_argument('--checkpoints_dir', required=True, type=str)
parser.add_argument('--gpu_ids', required=True, type=str)
parser.add_argument('--HTMLresults_dir', required=False, type=str, default='./HTMLresults')
args = parser.parse_args()

# style_root = '/home/biomag/tivadar/pytorch-CycleGAN-and-pix2pix/datasets/csaba/styles'

for style in os.listdir(args.style_root):
    style_path = os.path.join(args.style_root, style)

    train_dir = os.path.join(style_path, 'train')
    n_files = len(os.listdir(train_dir))
    print('Train dir for style: {}. Num of files: {}.'.format(train_dir, n_files))    

    model_name = style
    command = """
    python3 %s/train.py \
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
    --gpu_ids=%s \
    """ % (args.ppdir, style_path, model_name, args.n_iter + 99,args.n_iter//n_files, args.checkpoints_dir, args.gpu_ids)

    print('Executing command: {}'.format(command))
    os.system(command)
