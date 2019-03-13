import os
import numpy as np
import pandas as pd
import skimage.io as io
import matplotlib.pyplot as plt

from skimage.morphology import erosion, dilation
from skimage.util import invert
from skimage import img_as_ubyte, img_as_float
from shutil import copyfile, copytree, move
from itertools import product
from copy import deepcopy
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity


def chk_mkdir(*args):
    for path in args:
        if not os.path.exists(path):
            os.makedirs(path)


def weight_map(mask_dir, erode=1, weight=2):
    dilate = 2*erode

    # getting dimensions
    mask_dim = io.imread(os.path.join(mask_dir, os.listdir(mask_dir)[0])).shape
    merged_mask_im = np.zeros(shape=(mask_dim[0], mask_dim[1]), dtype='uint16')
    dilated_im = np.zeros(shape=(mask_dim[0], mask_dim[1]), dtype='uint16')

    for im_idx, mask_im_filename in enumerate(os.listdir(mask_dir)):
        mask_im = io.imread(os.path.join(mask_dir, mask_im_filename))/255
        dilated_im = np.maximum.reduce([dilated_im, mask_im])
        if erode:
            assert 1 <= erode, 'erosion must be positive'
            for _ in range(int(erode)):
                mask_im = erosion(mask_im)
        merged_mask_im = np.maximum.reduce([merged_mask_im, mask_im])

    # dilation
    dilated_im = deepcopy(merged_mask_im)
    for _ in range(dilate):
        dilated_im = dilation(dilated_im)

    return weight*dilated_im + (1-weight)*merged_mask_im


def boundary_map(mask_dir, width=1, with_weights=False):
    # getting dimensions
    mask_dim = io.imread(os.path.join(mask_dir, os.listdir(mask_dir)[0])).shape
    merged_mask_im = np.zeros(shape=(mask_dim[0], mask_dim[1]), dtype='uint16')
    eroded_mask_im = np.zeros(shape=(mask_dim[0], mask_dim[1]), dtype='uint16')
    dilated_mask_im = np.zeros(shape=(mask_dim[0], mask_dim[1]), dtype='uint16')

    for im_idx, mask_im_filename in enumerate(os.listdir(mask_dir)):
        mask_im = io.imread(os.path.join(mask_dir, mask_im_filename))/255
        merged_mask_im = np.maximum.reduce([merged_mask_im, mask_im])

        assert 1 <= width, 'erosion must be positive'
        for _ in range(int(width)):
            mask_im = erosion(mask_im)
        eroded_mask_im = np.maximum.reduce([eroded_mask_im, mask_im])

        mask_im = io.imread(os.path.join(mask_dir, mask_im_filename)) / 255
        for _ in range(2 * width):
            mask_im = dilation(mask_im)
        dilated_mask_im = np.maximum.reduce([dilated_mask_im, mask_im])

    border = dilated_mask_im - eroded_mask_im
    multilabel_mask = 2*border + merged_mask_im - border*merged_mask_im

    if with_weights:
        w_background = 1
        w_border = np.sum(1-dilated_mask_im)/np.sum(border)
        w_interior = np.sum(eroded_mask_im)/np.sum(border)
        weight_im = border + w_interior*eroded_mask_im + w_background*(1 - dilated_mask_im)

        return multilabel_mask, weight_im

    return multilabel_mask #/np.max(multilabel_mask)


def UNet_weight_map(mask_dir, erode=1, sigma=1):
    # getting dimensions
    mask_dim = io.imread(os.path.join(mask_dir, os.listdir(mask_dir)[0])).shape
    merged_mask_im = np.zeros(shape=(mask_dim[0], mask_dim[1]), dtype='uint8')
    weight_im = np.zeros(shape=(mask_dim[0], mask_dim[1]), dtype='float')
    border_im = np.zeros(shape=(mask_dim[0], mask_dim[1]), dtype='float')

    for im_idx, mask_im_filename in enumerate(os.listdir(mask_dir)):
        mask_im = io.imread(os.path.join(mask_dir, mask_im_filename))/255
        merged_mask_im = np.maximum.reduce([merged_mask_im, mask_im])

    weight_ratio = np.sum(merged_mask_im) / (mask_dim[0] * mask_dim[1] - np.sum(merged_mask_im))
    weight_im += weight_ratio * (1 - merged_mask_im) + merged_mask_im

    for im_idx, mask_im_filename in enumerate(os.listdir(mask_dir)):
        mask_im = io.imread(os.path.join(mask_dir, mask_im_filename))/255
        eroded_temp = io.imread(os.path.join(mask_dir, mask_im_filename))/255
        dilated_temp = io.imread(os.path.join(mask_dir, mask_im_filename))/255
        if erode:
            assert 1 <= erode, 'erosion must be positive'
            for d in range(int(erode)):
                eroded_temp = erosion(eroded_temp)
                dilated_temp = dilation(dilated_temp)
                border_im += np.exp(-d / sigma) * (dilated_temp - eroded_temp)

    weight_im += border_im

    return weight_im


def copy_images(names, source_path, out_path, images_only=False):
    chk_mkdir(out_path)
    if not images_only:
        for image_folder in os.listdir(source_path):
            if image_folder in names:
                copytree(
                    src=os.path.join(source_path, image_folder),
                    dst=os.path.join(out_path, image_folder)
                )
    else:
        for image_name in names:
            if image_name != 'loc.csv':
                copyfile(
                    src=os.path.join(source_path, image_name, 'images', image_name + '.png'),
                    dst=os.path.join(out_path, image_name + '.png')
                )


def merge_masks(
        data_path, output_path, erode=False,
        database=True, multilabel_masks=False, balanced_weights=False
):
    """
    Takes the path to the data folder as downloaded from Kaggle,
    then merges the masks and restructures the folders. 
    """

    for image_dir in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, image_dir)):
            # creating the new folders
            chk_mkdir(output_path)
            new_image_root = os.path.join(output_path, image_dir)
            new_image_folder = os.path.join(new_image_root, 'images')
            new_mask_folder = os.path.join(new_image_root, 'masks')
            chk_mkdir(new_image_root, new_image_folder, new_mask_folder)

            if not multilabel_masks:
                # get dimensions of the image
                for im_filename in os.listdir(os.path.join(data_path, image_dir, 'images')):
                    im = io.imread(os.path.join(data_path, image_dir, 'images', im_filename))
                    # copying the image
                    io.imsave(os.path.join(new_image_folder, image_dir+'.png'), im)

                merged_mask_im = np.zeros(shape=(im.shape[0], im.shape[1]), dtype='uint16')

                for mask_im_filename in os.listdir(os.path.join(data_path, image_dir, 'masks')):
                    mask_im = io.imread(os.path.join(data_path, image_dir, 'masks', mask_im_filename))
                    if erode:
                        assert 0 <= erode, 'erosion must be nonnegative'
                        for _ in range(int(erode)):
                            mask_im = erosion(mask_im)
                    merged_mask_im = np.maximum.reduce([merged_mask_im, mask_im])

                io.imsave(os.path.join(new_mask_folder, image_dir + '.png'), merged_mask_im.astype('uint8'))

            else:
                for im_filename in os.listdir(os.path.join(data_path, image_dir, 'images')):
                    im = io.imread(os.path.join(data_path, image_dir, 'images', im_filename))
                    # copying the image
                    io.imsave(os.path.join(new_image_folder, image_dir+'.png'), im)

                if not balanced_weights:
                    merged_mask_im = img_as_ubyte(boundary_map(
                        os.path.join(data_path, image_dir, 'masks'), width=1,
                        with_weights=False
                    ).astype('uint8'))
                    io.imsave(os.path.join(new_mask_folder, image_dir + '.png'), merged_mask_im)
                else:
                    merged_mask_im, balanced_weights = boundary_map(
                        os.path.join(data_path, image_dir, 'masks'), width=1,
                        with_weights=True
                    )
                    new_weights_folder = os.path.join(new_image_root, 'weight')
                    io.imsave(os.path.join(new_mask_folder, image_dir + '.png'), merged_mask_im.astype('uint8'))
                    io.imsave(os.path.join(new_weights_folder, image_dir + '.png'), balanced_weights)

    if database:
        make_database(output_path)


def make_patches(data_path, output_path, patch_height, patch_width, database=False):
    """
    Takes the data folder CONTAINING MERGED MASKS and slices the
    images and masks into patches.
    """
    # make output directories
    chk_mkdir(output_path)

    for image_id in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, image_id)):
            # checking for weights folder to see if weights are there
            with_weights = os.path.exists(os.path.join(data_path, image_id, 'weight'))

            # reading images
            im = io.imread(os.path.join(data_path, image_id, 'images', image_id + '.png'))
            masked_im = io.imread(os.path.join(data_path, image_id, 'masks', image_id + '.png'))
            # make new folders
            patched_images_root = os.path.join(output_path, image_id)
            patched_images_folder = os.path.join(patched_images_root, 'images')
            patched_masks_folder = os.path.join(patched_images_root, 'masks')
            if with_weights:
                patched_weights_folder = os.path.join(patched_images_root, 'weight')
                chk_mkdir(patched_weights_folder)
                weight_im = io.imread(os.path.join(data_path, image_id, 'weight', image_id + '.png'))

            chk_mkdir(patched_images_root, patched_images_folder, patched_masks_folder)

            x_start = list()
            y_start = list()

            for x_idx in range(0, im.shape[0]-patch_height+1, patch_height//2):
                x_start.append(x_idx)

            if im.shape[0]-patch_height-1 > 0:
                x_start.append(im.shape[0]-patch_height-1)

            for y_idx in range(0, im.shape[1]-patch_width+1, patch_height//2):
                y_start.append(y_idx)

            if im.shape[1]-patch_width-1 > 0:
                y_start.append(im.shape[1]-patch_width-1)

            for num, (x_idx, y_idx) in enumerate(product(x_start, y_start)):
                image_filename = image_id + '_%d.png' % num
                # saving a patch of the original image
                io.imsave(
                    os.path.join(patched_images_folder, image_filename),
                    im[x_idx:x_idx + patch_height, y_idx:y_idx + patch_width, :]
                )
                # saving the corresponding patch of the mask
                io.imsave(
                    os.path.join(patched_masks_folder, image_filename),
                    masked_im[x_idx:x_idx + patch_height, y_idx:y_idx + patch_width]
                )
                # if weights are present, weights are also saved as well
                if with_weights:
                    io.imsave(
                        os.path.join(patched_weights_folder, image_filename),
                        weight_im[x_idx:x_idx + patch_height, y_idx:y_idx + patch_width]
                    )

    if database:
        make_database(output_path)


def make_database(data_path):
    database_export_path = os.path.join(data_path, 'loc.csv')

    # checking if weights are there
    with_weights = os.path.exists(os.path.join(data_path, os.listdir(data_path)[0], 'weight'))

    if with_weights:
        columns = ['name', 'image_path', 'mask_path', 'weight_path']
    else:
        columns = ['name', 'image_path', 'mask_path']

    path_csv = pd.DataFrame(columns=columns)
    for image_id_root in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, image_id_root)):
            for image in os.listdir(os.path.join(data_path, image_id_root, 'images')):
                image_path = os.path.join(data_path, image_id_root, 'images', image)
                mask_path = os.path.join(data_path, image_id_root, 'masks', image)
                if with_weights:
                    weight_folder = os.path.join(data_path, image_id_root, 'weight')
                    weight_file = os.listdir(weight_folder)[0]
                    weight_path = os.path.join(weight_folder, weight_file)
                    image_record = [[image[:-4], image_path, mask_path, weight_path]]
                else:
                    image_record = [[image[:-4], image_path, mask_path]]
                path_csv = pd.concat(
                    (path_csv, pd.DataFrame(
                        image_record,
                        columns=columns
                    )), ignore_index=True
                )

    path_csv.to_csv(database_export_path)


def resize(data_path, output_path):

    chk_mkdir(output_path)

    for image_dir in os.listdir(data_path):
        # padding images

        for image_file in os.listdir(os.path.join(data_path, image_dir, 'images')):
            im = io.imread(os.path.join(data_path, image_dir, 'images', image_file))
            if im.shape[0] < 256 or im.shape[1] < 256:
                im_x = np.max([im.shape[0], 256])
                im_y = np.max([im.shape[1], 256])

                padded_im = np.zeros(shape=(im_x, im_y, im.shape[2]), dtype='uint8')
                padded_im[:im.shape[0], :im.shape[1], :] = im
                if im.shape[2] > 3:
                    padded_im[:, :, 3] = 255

                new_image_path = os.path.join(output_path, image_dir, 'images')
                chk_mkdir(new_image_path)

                io.imsave(os.path.join(new_image_path, image_file), padded_im)

                print(image_file + ' %d %d, new: %d %d ' % (im.shape[0], im.shape[1], im_x, im_y))

                # padding masks
                for mask_idx, mask_file in enumerate(os.listdir(os.path.join(data_path, image_dir, 'masks'))):
                    im = io.imread(os.path.join(data_path, image_dir, 'masks', mask_file))
                    if im.shape[0] < 256 or im.shape[1] < 256:

                        im_x = np.max([im.shape[0], 256])
                        im_y = np.max([im.shape[1], 256])

                        padded_im = np.zeros(shape=(im_x, im_y), dtype='uint8')
                        padded_im[:im.shape[0], :im.shape[1]] = im

                        new_image_path = os.path.join(output_path, image_dir, 'masks')
                        chk_mkdir(new_image_path)

                        io.imsave(os.path.join(new_image_path, '%d.png' % mask_idx), padded_im)

            else:
                copytree(
                    src=os.path.join(data_path, image_dir),
                    dst=os.path.join(output_path, image_dir)
                )


def collect_images(data_path, out_path, invert_mask=False, with_masks=True):
    images_path = os.path.join(out_path, 'images')
    masks_path = os.path.join(out_path, 'masks')
    chk_mkdir(images_path, masks_path)

    for image_dir in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, image_dir)):
            # copy images
            for image in os.listdir(os.path.join(data_path, image_dir, 'images')):
                copyfile(
                    src=os.path.join(data_path, image_dir, 'images', image),
                    dst=os.path.join(images_path, image)
                )

            if with_masks:
                for mask_image in os.listdir(os.path.join(data_path, image_dir, 'masks')):
                    if not invert_mask:
                        copyfile(
                            src=os.path.join(data_path, image_dir, 'masks', mask_image),
                            dst=os.path.join(masks_path, mask_image)
                        )
                    else:
                        mask_im = io.imread(os.path.join(data_path, image_dir, 'masks', mask_image))
                        mask_im = invert(mask_im)
                        io.imsave(os.path.join(masks_path, mask_image), mask_im)


def balance(features_path):
    # features
    features = pd.read_csv(features_path, header=0)
    features['ImageName'] = features['ImageName'].apply(lambda x: x[:-4])
    features = features.rename(columns={'ImageName': 'name'}).sort_values(by='name')

    # PCA on features
    val = features.loc[:, features.columns != 'name'].values
    val_pca = PCA(n_components=2).fit_transform(val)

    # KDE on PCA
    kde = KernelDensity(kernel='gaussian', bandwidth=0.25)
    kde.fit(val_pca)
    image_density = np.exp(kde.score_samples(val_pca))


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


def beside(data_path, out_path):
    chk_mkdir(out_path)

    for image_folder_path in os.listdir(data_path):
        image_file_path = os.path.join(data_path, image_folder_path, 'images', image_folder_path + '.png')
        mask_file_path = os.path.join(data_path, image_folder_path, 'masks', image_folder_path + '.png')

        mask_im = io.imread(mask_file_path)
        mask_im = mask_im.reshape(mask_im.shape[0], mask_im.shape[1], 1)
        mask_im = np.concatenate([mask_im, mask_im, mask_im], axis=2)
        real_im = io.imread(image_file_path)

        pix2pix_im = np.concatenate([real_im, mask_im], axis=1)
        io.imsave(os.path.join(out_path, image_folder_path + '.png'), pix2pix_im)


def train_test_split(input_folder, output_folder, p_test=0.1):
    train_folder = os.path.join(output_folder, 'train')
    test_folder = os.path.join(output_folder, 'test')
    chk_mkdir(train_folder, test_folder)
    for subdir in os.listdir(input_folder):
        if os.path.isdir(os.path.join(input_folder, subdir)):
            if np.random.rand() < p_test:
                dst = test_folder
            else:
                dst = train_folder
            copytree(
                src=os.path.join(input_folder, subdir),
                dst=os.path.join(dst, subdir)
            )


def three_channelize(input_folder, output_folder):
    for image_dir in os.listdir(input_folder):
        new_image_folder = os.path.join(output_folder, image_dir, 'images')
        new_mask_folder = os.path.join(output_folder, image_dir, 'masks')

        image_path = os.path.join(os.path.join(input_folder, image_dir, 'images'))
        mask_path = os.path.join(os.path.join(input_folder, image_dir, 'masks'))
        if os.path.isdir(image_path):
            image_name = os.listdir(image_path)[0]
            image = io.imread(os.path.join(image_path, image_name))

            chk_mkdir(new_mask_folder, new_image_folder)

            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=2)
                image = np.concatenate([image, image, image], axis=2)
                io.imsave(os.path.join(new_image_folder, image_name), image)
            else:
                io.imsave(os.path.join(new_image_folder, image_name), image)

            mask_name = os.listdir(mask_path)[0]
            copyfile(
                src=os.path.join(mask_path, mask_name),
                dst=os.path.join(new_mask_folder, mask_name)
            )


# ================================= #
# MaskRCNN preprocessing functions  #
# ================================= #


def merge_masks_RCNN(data_path, output_path):
    for image_dir in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, image_dir)):
            mask_folder = os.path.join(data_path, image_dir, 'masks')
            # getting the dim
            try:
                temp = io.imread(os.path.join(mask_folder, os.listdir(mask_folder)[0]))
            except:
                continue
            merged_mask_im = np.zeros(shape=temp.shape, dtype='uint8')
            for mask_name in os.listdir(mask_folder):
                mask_im = io.imread(os.path.join(mask_folder, mask_name))
                merged_mask_im = np.maximum.reduce([merged_mask_im, mask_im])

            # saving merged image
            merged_mask_path = os.path.join(output_path, image_dir, 'RCNN-masks')
            chk_mkdir(merged_mask_path)
            io.imsave(os.path.join(merged_mask_path, image_dir+'.png'), merged_mask_im)


def make_database_RCNN(data_path):
    database_export_path = os.path.join(data_path, 'loc.csv')

    columns = ['name', 'image_path', 'mask_path', 'RCNN_mask']

    path_csv = pd.DataFrame(columns=columns)
    for image_id_root in os.listdir(data_path):
        if os.path.isdir(os.path.join(data_path, image_id_root)):
            for image in os.listdir(os.path.join(data_path, image_id_root, 'images')):
                image_path = os.path.join(data_path, image_id_root, 'images', image)
                mask_path = os.path.join(data_path, image_id_root, 'masks', image)
                rcnn_mask_path = os.path.join(data_path, image_id_root, 'RCNN-masks', image)
                image_record = [[image[:-4], image_path, mask_path, rcnn_mask_path]]
                path_csv = pd.concat(
                    (path_csv, pd.DataFrame(
                        image_record,
                        columns=columns
                    )), ignore_index=True
                )

    path_csv.to_csv(database_export_path)


def MaskRCNN_to_UNet(rcnn_output_path, gt_path, output_path):
    for image_dir in os.listdir(rcnn_output_path):
        if os.path.isdir(os.path.join(rcnn_output_path, image_dir)):
            gt_mask_folder = os.path.join(gt_path, image_dir, 'masks')
            rcnn_mask_folder = os.path.join(rcnn_output_path, image_dir, 'masks')

            # merge masks
            rcnn_merged_mask_im = 0
            for mask_im in os.listdir(rcnn_mask_folder):
                rcnn_merged_mask_im = np.maximum(io.imread(os.path.join(rcnn_mask_folder, mask_im)), rcnn_merged_mask_im)

            gt_merged_mask_im = 0
            for mask_im in os.listdir(gt_mask_folder):
                gt_merged_mask_im = np.maximum(io.imread(os.path.join(gt_mask_folder, mask_im)), gt_merged_mask_im)

            if len(os.listdir(rcnn_mask_folder)) == 0:
                rcnn_merged_mask_im = np.zeros(shape=gt_merged_mask_im.shape)

            multilabel_mask = MaskRCNN_GTMask_merge(rcnn_merged_mask_im, gt_merged_mask_im)

            new_rcnn_mask_folder = os.path.join(output_path, image_dir, 'RCNN-masks')
            new_gt_image_folder = os.path.join(output_path, image_dir, 'images')
            new_gt_mask_folder = os.path.join(output_path, image_dir, 'masks')
            chk_mkdir(new_gt_image_folder, new_gt_mask_folder, new_rcnn_mask_folder)
            copyfile(
                src=os.path.join(gt_path, image_dir, 'images', image_dir + '.png'),
                dst=os.path.join(new_gt_image_folder, image_dir + '.png')
            )
            io.imsave(os.path.join(new_rcnn_mask_folder, image_dir+'.png'), img_as_ubyte(rcnn_merged_mask_im))
            io.imsave(os.path.join(new_gt_mask_folder, image_dir+'.png'), multilabel_mask)

    make_database_RCNN(output_path)


def MaskRCNN_GTMask_merge(rcnn_mask, gt_mask):
    false_positive = np.maximum(img_as_float(rcnn_mask) - img_as_float(gt_mask), 0).astype('uint8')
    false_negative = np.maximum(img_as_float(gt_mask) - img_as_float(rcnn_mask), 0).astype('uint8')
    true_positive = (img_as_float(rcnn_mask)*img_as_float(gt_mask)).astype('uint8')

    multilabel_im = (true_positive + 2*false_negative + 3*false_positive).astype('uint8')

    return multilabel_im


if __name__ == '__main__':
    data_path = '/media/tdanka/B8703F33703EF828/tdanka/data/GOLD_merged'
    output = '/media/tdanka/B8703F33703EF828/tdanka/data/GOLD_split'

    train_test_split(data_path, output, p_test=0.1)
    make_database(os.path.join(output, 'train'))
    make_database(os.path.join(output, 'test'))

