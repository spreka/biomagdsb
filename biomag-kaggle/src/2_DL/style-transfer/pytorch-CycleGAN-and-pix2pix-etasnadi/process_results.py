import os
import argparse
from shutil import copyfile

parser = argparse.ArgumentParser()
parser.add_argument('--pix2pix_results', default='/home/biomag/tivadar/pytorch-CycleGAN-and-pix2pix/results', type=str)
parser.add_argument('--output_folder', default='/home/biomag/tivadar/pytorch-CycleGAN-and-pix2pix/results_kaggle_format', type=str)
args = parser.parse_args()

results_folder = args.pix2pix_results
output_folder = args.output_folder

for style in os.listdir(results_folder):
    images_path = os.path.join(results_folder, style, 'test_latest', 'images')
    for image in os.listdir(images_path):
        if image[-10:] == 'fake_B.png':
            image_filename = image[:-10] + style
            image_new_path = os.path.join(output_folder, image_filename, 'images')
            os.makedirs(image_new_path)
            copyfile(
                src=os.path.join(images_path, image),
                dst=os.path.join(image_new_path, image_filename)
            )
            
