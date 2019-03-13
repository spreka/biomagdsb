import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from shutil import copyfile
from skimage import io, img_as_ubyte
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F
import pandas as pd
from random import shuffle


def visualize_segmentation(X, y_mask, y_pred, export_path=None):
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(15, 5))
        ax = plt.subplot(1, 3, 1)
        ax.imshow(X)
        ax.axis('off')
        ax.set_title('Image')
        ax = plt.subplot(1, 3, 2)
        ax.imshow(1-y_mask, cmap='Greys')
        ax.axis('off')
        ax.set_title('Ground truth')
        ax = plt.subplot(1, 3, 3)
        ax.imshow(1-y_pred, cmap='Greys')
        ax.axis('off')
        ax.set_title('Result')

        if export_path:
            plt.savefig(export_path)
        else:
            plt.show()
        plt.close('all')


def chk_mkdir(*args):
    for path in args:
        if not os.path.exists(path):
            os.makedirs(path)


def pad_to_shape(array, new_shape):
    old_shape = array.shape
    assert len(old_shape) == len(new_shape), 'new shape must have the same number of dimensions than the old one'
    pad_params = tuple((0, np.max([0, new_shape[i] - old_shape[i]])) for i in range(len(new_shape)))
    return np.pad(array, pad_params, 'constant', constant_values=0)


def merge_masks(data_path, output_path):
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

            # get dimensions of the image
            for im_filename in os.listdir(os.path.join(data_path, image_dir, 'images')):
                im = io.imread(os.path.join(data_path, image_dir, 'images', im_filename))
                # copying the image
                io.imsave(os.path.join(new_image_folder, image_dir+'.png'), im)

            merged_mask_im = np.zeros(shape=(im.shape[0], im.shape[1]), dtype='uint16')

            for mask_im_filename in os.listdir(os.path.join(data_path, image_dir, 'masks')):
                mask_im = io.imread(os.path.join(data_path, image_dir, 'masks', mask_im_filename))
                merged_mask_im = np.maximum.reduce([merged_mask_im, mask_im])

            io.imsave(os.path.join(new_mask_folder, image_dir + '.png'), merged_mask_im.astype('uint8'))


def train_val_split(data_path, output_path, p_test=0.1):
    new_train_images_folder = os.path.join(output_path, 'train', 'images')
    new_train_masks_folder = os.path.join(output_path, 'train', 'masks')
    #new_train_bboxes_folder = os.path.join(output_path, 'train', 'bboxes')
    new_val_images_folder = os.path.join(output_path, 'val', 'images')
    new_val_masks_folder = os.path.join(output_path, 'val', 'masks')
    #new_val_bboxes_folder = os.path.join(output_path, 'val', 'bboxes')
    chk_mkdir(new_train_images_folder, new_train_masks_folder, new_val_images_folder, new_val_masks_folder)
    #chk_mkdir(new_train_images_folder, new_train_masks_folder, new_val_images_folder, new_val_masks_folder, new_train_bboxes_folder, new_val_bboxes_folder)
    for image_idx, image_filename in enumerate(os.listdir(os.path.join(data_path, 'images'))):
        print('copying image no. %d/%d' % (image_idx, len(os.listdir(os.path.join(data_path, 'images')))))
        image_name = image_filename[:-4]
        if np.random.rand() < p_test:
            # copy image
            copyfile(
                src=os.path.join(data_path, 'images', image_filename),
                dst=os.path.join(new_val_images_folder, image_filename)
            )
            # copy mask
            copyfile(
                src=os.path.join(data_path, 'masks', image_name + '.tiff'),
                dst=os.path.join(new_val_masks_folder, image_name + '.tiff')
            )
            # copy bbox
            #copyfile(
            #    src=os.path.join(data_path, 'bboxes', image_name + '.csv'),
            #    dst=os.path.join(new_val_bboxes_folder, image_name + '.csv')
            #)
        else:
            # copy image
            copyfile(
                src=os.path.join(data_path, 'images', image_filename),
                dst=os.path.join(new_train_images_folder, image_filename)
            )
            # copy mask
            copyfile(
                src=os.path.join(data_path, 'masks', image_name + '.tiff'),
                dst=os.path.join(new_train_masks_folder, image_name + '.tiff')
            )
            # copy bboxes
            #copyfile(
            #    src=os.path.join(data_path, 'bboxes', image_name + '.csv'),
            #    dst=os.path.join(new_train_bboxes_folder, image_name + '.csv')
            #)


def make_augmented_transform(
        size=(256, 256), p_flip=0.5, color_jitter_params=(0.5, 0.5, 0.5, 0.5),
        long_mask=False, random_resize=False, normalize=False
):
    if color_jitter_params:
        color_tf = T.ColorJitter(*color_jitter_params)
    else:
        color_tf = None

    if random_resize:
        assert len(random_resize) == 2, 'random_resize must be a tuple of (x_min, x_max) magnification range'
        assert random_resize[1] > random_resize[0], 'magnification range max must be larger than min'

    if normalize:
        tf_normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(1, 1, 1))

    def joint_transform(image, mask):
        # random magnification
        if random_resize:
            magnification_ratio = (random_resize[1] - random_resize[0]) * np.random.rand() + random_resize[0]
            new_shape = (int(magnification_ratio * image.shape[0]), int(magnification_ratio * image.shape[1]))
            image = img_as_ubyte(resize(image, new_shape))
            mask = resize(mask, new_shape)
            mask = img_as_ubyte(mask > 0.5)

        # resizing
        if image.shape[0] < size[0] or image.shape[1] < size[1]:
            new_im_shape = np.max([image.shape[0], size[0]]), np.max([image.shape[1], size[1]]), 3
            new_mask_shape = np.max([image.shape[0], size[0]]), np.max([image.shape[1], size[1]]), 1
            image = pad_to_shape(image, new_im_shape)
            mask = pad_to_shape(mask, new_mask_shape)

        # transforming to PIL image
        image, mask = F.to_pil_image(image), F.to_pil_image(mask)

        # random crop
        i, j, h, w = T.RandomCrop.get_params(image, size)
        image, mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w)
        if np.random.rand() < p_flip:
            image, mask = F.hflip(image), F.hflip(mask)

        # color transforms || ONLY ON IMAGE
        if color_tf is not None:
            image = color_tf(image)

        # transforming to tensor
        image = F.to_tensor(image)
        if not long_mask:
            mask = F.to_tensor(mask)
        else:
            mask = to_long_tensor(mask)

        # normalizing image
        if normalize:
            image = tf_normalize(image)

        return image, mask

    def single_transform(image):
        # resizing
        if image.shape[0] < size[0] or image.shape[1] < size[1]:
            new_im_shape = np.max([image.shape[0], size[0]]), np.max([image.shape[1], size[1]]), 3
            image = resize(image,new_im_shape, preserve_range=True).astype(np.uint8)#pad_to_shape(image, new_im_shape)

        # transforming to PIL image
        image = F.to_pil_image(image)

        # transforming to tensor
        image = F.to_tensor(image)

        # normalizing image
        if normalize:
            image = tf_normalize(image)

        return image

    #return single_transform
    return joint_transform


class JointlyTransformedDataset(Dataset):
    def __init__(self, dataset_path, transform=None, sigma=1.0):
        self.images_path = os.path.join(dataset_path, 'images')
        self.masks_path = os.path.join(dataset_path, 'masks')
        self.images = os.listdir(self.images_path)
        shuffle(self.images)
        self.sigma = sigma
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # getting the image and the mask
        image_name = self.images[idx][:-4]
        image_path = os.path.join(self.images_path, '%s.png' % image_name)
        mask_path = os.path.join(self.masks_path, '%s.tiff' % image_name)

        image = io.imread(image_path)[:, :, :3]

        # get mask
        mask = io.imread(mask_path)
        mask[mask > 0.5] = 1
        #gauss smooth mask
        mask = mask * 255
        mask[mask == 0] = 20
        mask[mask == 255] = 235
        if self.sigma:
            mask = gaussian_filter(mask, (self.sigma,self.sigma))
        mask[np.unravel_index(np.argmax(mask, axis=None), mask.shape)] = 255
        mask[np.unravel_index(np.argmin(mask, axis=None), mask.shape)] = 0
        #mask = img_as_ubyte(mask > 0.5)
        mask = img_as_ubyte(mask)
        mask = np.expand_dims(mask, axis=2)

        if self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask, image_name


class TestDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.images_path = os.path.join(dataset_path, 'images')
        self.images = os.listdir(self.images_path)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # getting the image and the mask
        image_name = self.images[idx][:-4]
        image_path = os.path.join(self.images_path, '%s.png' % image_name)

        image = io.imread(image_path)[:, :, :3]

        if self.transform is not None:
            image = self.transform(image)

        return image, image_name


class BoundingBoxDataset(Dataset):
    def __init__(self, dataset_path, transform=None, padding=2, minlen=5):
        self.images_path = os.path.join(dataset_path, 'images')
        self.masks_path = os.path.join(dataset_path, 'masks')
        self.bboxes_path = os.path.join(dataset_path, 'bboxes')
        self.transform = transform
        if padding != 0:
            assert isinstance(padding, int), 'padding must be integer'
        self.padding = padding
        self.minlen = minlen
        self.images = self.collectBBs(self.bboxes_path, self.images_path)

    def collectBBs(self, path, imgpath):
        images = pd.DataFrame(columns=['image_name', 'shape', 'x1', 'y1', 'x2', 'y2'])
        bbox_files = os.listdir(path)
        for bb_file in bbox_files:
            img = io.imread(os.path.join(imgpath,bb_file[:-4]+'.png'))
            df = pd.read_csv(os.path.join(path, bb_file), sep=';') #prediction ;
            df.loc[:, 'image_name'] = pd.Series([bb_file[:-4] for i in range(df.shape[0])], index=df.index)
            df.loc[:, 'shape'] = pd.Series([img.shape for i in range(df.shape[0])], index=df.index)
            images = images.append(df)

        # apply padding
        flb = lambda x: max(x - self.padding, 0)
        fxub = lambda x: min(x['x2'] + self.padding, x['shape'][1])
        fyub = lambda x: min(x['y2'] + self.padding, x['shape'][0])
        images['x1'] = images['x1'].apply(flb)
        images['y1'] = images['y1'].apply(flb)
        images['x2'] = images.apply(fxub, axis=1)
        images['y2'] = images.apply(fyub, axis=1)

        # remove too small bounding boxes
        images = images[(images['x1'] + self.minlen < images['x2']) & (images['y1'] + self.minlen < images['y2'])]
        return images

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        row = self.images.iloc[idx]
        image_name = row['image_name']
        fullimg = io.imread(os.path.join(self.images_path, image_name + '.png'))[:, :, :3]
        maskimg = io.imread(os.path.join(self.masks_path, image_name + '.tiff'))
        x1, x2, y1, y2 = row['x1'], row['x2'], row['y1'], row['y2']
        image = fullimg[y1:y2, x1:x2, :3]
        image_mask = maskimg[y1:y2, x1:x2]
        image_mask[image_mask > 0] = 255
        image_mask = image_mask.astype(np.uint8)
        image_mask = np.expand_dims(image_mask, axis=2)

        if self.transform is not None:
            #image, image_mask = self.transform(image, image_mask)
            image = self.transform(image)
            image_mask = self.transform(image_mask)

        return image, image_mask, image_name


class TestBoundingBoxDataset(BoundingBoxDataset):
    def __init__(self, dataset_path, transform=None, padding=2, minlen=5):
        BoundingBoxDataset.__init__(self, dataset_path, transform, padding, minlen)

    def __getitem__(self, idx):
        row = self.images.iloc[idx]
        image_name = row['image_name']
        fullimg = io.imread(os.path.join(self.images_path, image_name + '.png'))[:, :, :3]
        x1, x2, y1, y2 = row['x1'], row['x2'], row['y1'], row['y2']

        image = fullimg[y1:y2, x1:x2, :]
        if self.transform is not None:
            image = self.transform(image)

        return image, image_name


class JaccardLoss(nn.Module):
    def __init__(self):
        super(JaccardLoss, self).__init__()

    def forward(self, x, y):
        return -((x*y).sum() + 1e-40)/((x + y - x*y).sum() + 1e-40)


if __name__ == '__main__':
    data_path = '/media/tdanka/B8703F33703EF828/lassi/data/newest_data_merged'
    output_path = '/media/tdanka/B8703F33703EF828/lassi/data/newest_data_merged_split'
    train_val_split(data_path, output_path)