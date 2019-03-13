import os
import numpy as np
import pandas as pd
import skimage.io as io
from skimage.morphology import dilation, square
from skimage import img_as_ubyte
from scipy.ndimage import binary_fill_holes
from dataset import chk_mkdir


def concatenate_images_and_masks(in_path, out_path, random_mix_path=False):
    chk_mkdir(out_path)
    for image_name in os.listdir(in_path):
        if os.path.isdir(os.path.join(in_path, image_name)):
            images_folder = os.path.join(in_path, image_name, 'images')
            masks_folder = os.path.join(in_path, image_name, 'masks')
            for image in os.listdir(images_folder):
                mask_patch = io.imread(os.path.join(masks_folder, image))
                mask_patch = mask_patch.reshape(mask_patch.shape[0], mask_patch.shape[1], 1)
                mask_patch = np.concatenate([mask_patch, mask_patch, mask_patch], axis=2)

                if not random_mix_path:
                    image_patch = io.imread(os.path.join(images_folder, image))[:, :, :3]
                else:
                    random_image_path = np.random.choice(os.listdir(random_mix_path))
                    image_patch = io.imread(os.path.join(random_mix_path, random_image_path))[:, :, :3]

                concat_im = np.concatenate([image_patch, mask_patch], axis=1)

                io.imsave(os.path.join(out_path, image), concat_im)


def UNet_multilabel_mask_proc(masks_folder):
    for mask_name in os.listdir(masks_folder):
        mask_im = io.imread(os.path.join(masks_folder, mask_name))
        nuclei_mask = binary_fill_holes(mask_im[:, :, 1] + mask_im[:, :, 2])
        io.imsave(os.path.join(masks_folder, mask_name), img_as_ubyte(nuclei_mask))


def Peter_style_csv(style_csv_path, test_folder, mask_folder, output_folder):
    styles = pd.read_csv(style_csv_path, header=[0])
    chk_mkdir(output_folder)
    for image in styles.iterrows():
        style_folder = os.path.join(output_folder, str(image[1]['Style']), 'train')
        chk_mkdir(style_folder)

        real_image = io.imread(os.path.join(test_folder, image[1]['Name'][:-4], 'images', image[1]['Name']))[:, :, :3]
        mask_image = io.imread(os.path.join(mask_folder, image[1]['Name']))
         mask_image = img_as_ubyte(
            dilation((mask_image[:, :, 1]>255/3), square(5)).reshape(mask_image.shape[0], mask_image.shape[1], 1))
        mask_image = np.concatenate([mask_image, mask_image, mask_image], axis=2)

        pix2pix_style_image = np.concatenate([real_image, mask_image], axis=1)

        io.imsave(os.path.join(style_folder, image[1]['Name']), pix2pix_style_image)


if __name__ == '__main__':
    style_csv_path = '/home/namazu/Data/Kaggle/styles.csv'
    test_folder = '/home/namazu/Data/Kaggle/stage1_test'
    mask_folder = '/home/namazu/Data/Kaggle/result/UNet_classwise'
    output_folder = '/home/namazu/Data/Kaggle/pix2pix/styles'
    Peter_style_csv(style_csv_path, test_folder, mask_folder, output_folder)




