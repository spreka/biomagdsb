import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from functools import partial
from skimage import io, img_as_ubyte
from skimage.transform import resize
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_to_shape(array, new_shape):
    old_shape = array.shape
    assert len(old_shape) == len(new_shape), 'new shape must have the same number of dimensions than the old one'
    pad_params = tuple((0, np.max([0, new_shape[i] - old_shape[i]])) for i in range(len(new_shape)))
    return np.pad(array, pad_params, 'constant', constant_values=0)


def to_long_tensor(pic):
    # handle numpy array
    img = torch.from_numpy(np.array(pic, np.uint8))
    # backward compatibility
    return img.long()


def make_single_transform(size=(256, 256), p_flip=0.5, color_jitter_params=(0.5, 0.5, 0.5, 0.5)):

    if color_jitter_params is not None:
        color_tf = T.ColorJitter(*color_jitter_params)
    else:
        color_tf = None

    def transform(image):
        # transforming to PIL image
        image = F.to_pil_image(image)
        # random crop
        i, j, h, w = T.RandomCrop.get_params(image, size)
        image = F.crop(image, i, j, h, w)
        if np.random.rand() < p_flip:
            image = F.hflip(image)

        # color transforms || ONLY ON IMAGE
        if color_tf is not None:
            image = color_tf(image)

        # transforming to tensor
        image = F.to_tensor(image)

        return image

    return transform


def make_transform(
        size=(256, 256), p_flip=0.5, color_jitter_params=(0.5, 0.5, 0.5, 0.5),
        long_mask=False, random_resize=False, normalize=False
):

    if color_jitter_params is not None:
        color_tf = T.ColorJitter(*color_jitter_params)
    else:
        color_tf = None

    if random_resize is not None:
        assert len(random_resize) == 2, 'random_resize must be a tuple of (x_min, x_max) magnification range'
        assert random_resize[1] > random_resize[0], 'magnification range max must be larger than min'

    if normalize:
        tf_normalize = T.Normalize(mean=(0.5, 0.5, 0.5), std=(1, 1, 1))

    def joint_transform(image, mask):
        # random magnification
        if random_resize is not None:
            magnification_ratio = (random_resize[1] - random_resize[0])*np.random.rand() + random_resize[0]
            new_shape = (int(magnification_ratio*image.shape[0]), int(magnification_ratio*image.shape[1]))
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

    return joint_transform


def make_transform_with_weight(
        size=(256, 256), p_flip=0.5, color_jitter_params=(0.2, 0.2, 0.2, 0.2), long_mask=False,
):
    if color_jitter_params is not None:
        color_tf = T.ColorJitter(*color_jitter_params)
    else:
        color_tf = None

    def joint_transform(image, mask, weight):
        # transforming to PIL image
        image, mask, weight = F.to_pil_image(image), F.to_pil_image(mask), F.to_pil_image(weight)

        # random crop
        i, j, h, w = T.RandomCrop.get_params(image, size)
        image, mask, weight = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w), F.crop(weight, i, j, h, w)
        if np.random.rand() < p_flip:
            image, mask, weight = F.hflip(image), F.hflip(mask), F.hflip(weight)

        # color transforms || ONLY ON IMAGE
        if color_tf is not None:
            image = color_tf(image)

        # transforming to tensor
        image, weight = F.to_tensor(image), F.to_tensor(weight)
        if not long_mask:
            mask = F.to_tensor(mask)
        else:
            mask = to_long_tensor(mask)

        return image, mask, weight

    return joint_transform


def make_transform_RCNN(size=(256, 256), p_flip=0.5, long_mask=False):
    def joint_transform(image, mask, rcnn_mask):
        # resizing
        if image.shape[0] < size[0] or image.shape[1] < size[1]:
            new_im_shape = np.max([image.shape[0], size[0]]), np.max([image.shape[1], size[1]]), 3
            new_mask_shape = np.max([image.shape[0], size[0]]), np.max([image.shape[1], size[1]]), 1
            image = pad_to_shape(image, new_im_shape)
            mask, rcnn_mask = pad_to_shape(mask, new_mask_shape), pad_to_shape(rcnn_mask, new_mask_shape)

        # transforming to PIL image
        image, mask, rcnn_mask = list(map(F.to_pil_image, [image, mask, rcnn_mask]))

        # random crop
        i, j, h, w = T.RandomCrop.get_params(image, size)
        image, mask, rcnn_mask = F.crop(image, i, j, h, w), F.crop(mask, i, j, h, w), F.crop(rcnn_mask, i, j, h, w)

        # random flip
        if np.random.rand() < p_flip:
            image, mask, rcnn_mask = list(map(F.hflip, [image, mask, rcnn_mask]))
        if np.random.rand() < p_flip:
            image, mask, rcnn_mask = list(map(F.vflip, [image, mask, rcnn_mask]))

        # transforming to tensor
        image, rcnn_mask = list(map(F.to_tensor, [image, rcnn_mask]))
        if not long_mask:
            mask = F.to_tensor(mask)
        else:
            mask = to_long_tensor(mask)

        return image, mask, rcnn_mask

    return joint_transform


class JointlyTransformedDataset(Dataset):
    def __init__(self, im_paths, transform=None, remove_alpha=False, class_weights=False):
        self.im_paths = pd.read_csv(im_paths, index_col=[0])
        self.transform = transform
        self.class_weights = class_weights
        self.remove_alpha = remove_alpha

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        # getting the image and the mask
        image = io.imread(self.im_paths.loc[idx, 'image_path'])
        if self.remove_alpha:
            image = image[:, :, :3]

        # get mask
        mask = io.imread(self.im_paths.loc[idx, 'mask_path'])
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)

        # getting the filename
        name = self.im_paths.loc[idx, 'name']

        if self.transform:
            image, mask = self.transform(image, mask)

        if self.class_weights:
            class_weights = np.bincount(np.unique(mask, return_inverse=True)[1], minlength=3)
            if len(class_weights) != 3:
                print(class_weights)
            class_weights = torch.from_numpy(class_weights[0]/(class_weights+1e-10)).float()
            return image, mask, class_weights, name

        return image, mask, name


class JointlyTransformedWeightedDataset(Dataset):
    def __init__(self, im_paths, transform=None, remove_alpha=False):
        self.im_paths = pd.read_csv(im_paths, index_col=[0])
        self.transform = transform
        self.remove_alpha = remove_alpha

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        # getting the image and the mask
        image = io.imread(self.im_paths.loc[idx, 'image_path'])
        if self.remove_alpha:
            image = image[:, :, :3]

        # get mask
        mask = io.imread(self.im_paths.loc[idx, 'mask_path'])
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)

        # get weight
        weight = io.imread(self.im_paths.loc[idx, 'weight_path'])
        weight = weight.reshape(weight.shape[0], weight.shape[1], 1)

        # getting the filename
        name = self.im_paths.loc[idx, 'name']

        if self.transform:
            image, mask, weight = self.transform(image, mask, weight)

        return image, mask, weight, name


class TrainFromFolder(Dataset):

    def __init__(self, im_paths, transform=None, remove_alpha=False, one_hot_mask=False, weighted=False):
        self.im_paths = pd.read_csv(im_paths, index_col=[0])
        self.transform = transform
        self.remove_alpha = remove_alpha
        self.one_hot_mask = one_hot_mask
        self.weighted = weighted

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        # getting the image and the mask
        image = io.imread(self.im_paths.loc[idx, 'image_path'])
        if self.remove_alpha:
            image = image[:, :, :3]

        # get mask
        mask = io.imread(self.im_paths.loc[idx, 'mask_path'])
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)

        # getting the filename
        name = self.im_paths.loc[idx, 'name']

        # getting the weights
        if self.weighted:
            weight = io.imread(self.im_paths.loc[idx, 'weight_path'])
            weight = weight.reshape(weight.shape[0], weight.shape[1], 1)

        if self.transform:
            image, mask = self.transform(image), self.transform(mask)
            if self.weighted:
                weight = self.transform(weight)

        if self.one_hot_mask:
            assert self.one_hot_mask >= 0, 'one_hot_mask must be nonnegative'
            one_hot_mask = torch.FloatTensor(self.one_hot_mask, mask.shape[1], mask.shape[2]).zero_()
            one_hot_mask.scatter_(0, mask.long(), 1)

            if self.weighted:
                return image, one_hot_mask, weight, name
            else:
                return image, one_hot_mask, name

        if self.weighted:
            return image, mask, weight, name
        else:
            return image, mask, name


class TestFromFolder(Dataset):
    def __init__(self, im_paths, transform=None, remove_alpha=False):
        self.im_paths = pd.read_csv(im_paths, index_col=[0])
        self.transform = transform
        self.remove_alpha = remove_alpha

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        # getting the image and the mask
        image = io.imread(self.im_paths.loc[idx, 'image_path'])
        if self.remove_alpha:
            image = image[:, :, :3]

        # getting the filename
        name = self.im_paths.loc[idx, 'name']

        if self.transform:
            return self.transform(image), name
        else:
            return image, name


class ImageClassifierGenerator(Dataset):
    def __init__(self, im_paths, classes, transform=None):
        image_paths = pd.read_csv(im_paths, index_col=[0])
        classes_csv = pd.read_csv(classes, index_col=[0])
        self.data = classes_csv.merge(image_paths[['name', 'image_path']], on='name')
        self.class_labels = ['FluorNormal', 'FluorLarge', 'FluorSmall',
                             'TissueSmall', 'TissueLarge', 'Brightfield']

        self.transform = transform

    def __len__(self):
        return len(im_paths)

    def __getitem__(self, idx):
        image = io.imread(self.data.loc[idx, 'image_path'])[:, :, :3]
        name = self.data.loc[idx, 'name']
        class_prob = self.data.loc[idx, self.class_labels].values.astype('float32')
        class_label = np.array(np.sum(class_prob*np.arange(0, len(class_prob)))).reshape(-1,1)

        return self.transform(image), torch.from_numpy(class_prob), torch.from_numpy(class_label).long(), name


class TrainWithRCNNMask(Dataset):
    def __init__(self, im_paths, transform=None):
        self.im_paths = pd.read_csv(im_paths, index_col=[0])
        self.transform = transform

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        # getting the image and the mask
        image = io.imread(self.im_paths.loc[idx, 'image_path'])
        try:
            image = image[:, :, :3]
        except:
            if len(image.shape) == 2:
                image = image.reshape(image.shape[0], image.shape[1], 1)
                image = np.concatenate([image, image, image], axis=2)
        # get mask
        mask = io.imread(self.im_paths.loc[idx, 'mask_path'])
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
        # get RCNN_prediction
        rcnn_mask = io.imread(self.im_paths.loc[idx, 'RCNN_mask'])
        rcnn_mask = rcnn_mask.reshape(rcnn_mask.shape[0], rcnn_mask.shape[1], 1)

        # getting the filename
        name = self.im_paths.loc[idx, 'name']

        if self.transform:
            image, mask, rcnn_mask = self.transform(image, mask, rcnn_mask)

        return image, mask, rcnn_mask, name


def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def confusion(y_true, y_pred, labels):
    confusion_matrix = np.zeros(shape=(len(labels), len(labels)))
    for i_idx, i in enumerate(labels):
        i_true = (y_true == i).int()
        n_i = i_true.float().sum()
        for j_idx, j in enumerate(labels):
            j_predicted = (y_pred == j).int()
            confusion_matrix[i_idx, j_idx] = (i_true*j_predicted).sum()/i_true.sum()

    return confusion_matrix


"""
------------------------------------
Visualization for segmentation masks
------------------------------------
"""


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


def visualize_train_mask(model, dataset, export_path, use_gpu=False, with_ground_truth=False, n_inst=None):
    if not os.path.isdir(export_path):
        os.makedirs(export_path)

    for batch_idx, (X_batch, y_batch, name) in enumerate(DataLoader(dataset, batch_size=1)):
        if n_inst is None:
            n_inst = len(dataset)
        else:
            assert n_inst >= 0, 'n_inst must be nonnegative'
        if use_gpu:
            X_batch = Variable(X_batch.cuda())
            y_out = model(X_batch).cpu().data.numpy()
            X_im = X_batch[0].data.cpu().numpy().transpose((1, 2, 0))
        else:
            X_batch = Variable(X_batch)
            y_out = model(X_batch).data.numpy()
            X_im = X_batch[0].data.numpy().transpose((1, 2, 0))

        if with_ground_truth:
            visualize_segmentation(
                X=X_im,
                y_mask=y_batch[0, 0, :, :].numpy(),
                y_pred=y_out[0, 1, :, :],
                export_path=os.path.join(export_path, name[0] + '.png')
            )
        else:
            io.imsave(os.path.join(export_path, name[0] + '.png'), y_out[0, 1, :, :])

        if batch_idx > n_inst:
            break


def visualize_test_mask(model, dataset, export_path, use_gpu=False):
    if not os.path.isdir(export_path):
        os.makedirs(export_path)

    for X_batch, name in DataLoader(dataset, batch_size=1):
        if use_gpu:
            X_batch = Variable(X_batch.cuda())
            y_out = model(X_batch).cpu().data.numpy()
        else:
            X_batch = Variable(X_batch)
            y_out = model(X_batch).data.numpy()

        io.imsave(os.path.join(export_path, name[0] + '.png'), y_out[0, 1, :, :])


def visualize_everything_mask(
        model, train_original_dataset, test_dataset,
        comparison_path, train_pmap_path, test_pmap_path,
        use_gpu=True
):
    # result visualization
    visualize_train_mask(
        model, train_original_dataset, comparison_path,
        use_gpu=use_gpu, with_ground_truth=True, n_inst=20
    )

    # TRAIN probability map generation
    visualize_train_mask(
        model, train_original_dataset, train_pmap_path,
        use_gpu=use_gpu, with_ground_truth=False
    )

    # TEST probability map generation
    visualize_test_mask(model, test_dataset, test_pmap_path, use_gpu=use_gpu)


def write_to_log(model, logger, y_true, y_out, training_loss, batch_idx, class_labels):
    # compute accuracy
    _, pred_labels = torch.max(y_out, 1)
    accuracy = (y_true == pred_labels.float()).float().mean()

    confusion_matrix = confusion(y_true, pred_labels, labels=class_labels)

    # (1) Log the scalar values
    info = {
        'loss': training_loss.data[0],
        'accuracy': accuracy.data[0],
        'background accuracy': confusion_matrix[0, 0],
        'nuclei accuracy': confusion_matrix[1, 1]
    }

    for tag, value in info.items():
        logger.scalar_summary(tag, value, batch_idx + 1)

    # (2) Log values and gradients of the parameters (histogram)
    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, to_np(value), batch_idx + 1)
        logger.histo_summary(tag + '/grad', to_np(value.grad), batch_idx + 1)


if __name__ == '__main__':
    rcnn_mask_path = '/media/tdanka/B8703F33703EF828/tdanka/UNet_correction/stage1_test/train/loc.csv'
    transform = make_transform_RCNN()
    dataset = TrainWithRCNNMask(rcnn_mask_path, transform=transform)

    x = next(iter(dataset))
