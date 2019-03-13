import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.autograd import Variable
from torch.utils.data import DataLoader

# local imports
from utils import *
from image.dataset import chk_mkdir
# models
from models.unet import UNet
from models.blocks import CrossEntropyLoss2d


class GANv2:
    def __init__(
            self, s, s_loss, s_optim,
            g, g_optim, s_scheduler=None, g_scheduler=None
    ):
        self.s = s
        self.s_loss = s_loss
        self.s_optim = s_optim
        self.s_scheduler = s_scheduler

        self.g = g
        self.g_optim = g_optim
        self.g_scheduler = g_scheduler

    def train_segmenter(self, dataset, n_epochs=1, n_batch=1, use_gpu=True, verbose=False):
        print('----- Training the segmenter network on the dataset -----')
        self.s.train(True)
        self.g.train(False)
        for epoch_idx in range(n_epochs):
            s_running_loss = 0
            for batch_idx, (X_batch, y_batch, y_weight, name) in enumerate(
                    DataLoader(dataset, batch_size=n_batch, shuffle=True)):

                if use_gpu:
                    X_batch, y_batch = Variable(X_batch.cuda()), Variable(y_batch.long()[:, 1, :, :].cuda())
                else:
                    X_batch, y_batch = Variable(X_batch), Variable(y_batch.long()[:, 1, :, :])

                self.s_optim.zero_grad()
                y_mask = self.s(X_batch)
                s_training_loss = self.s_loss(y_mask, y_batch)
                s_training_loss.backward()
                s_optim.step()

                s_running_loss += s_training_loss.data[0]

                if verbose:
                    print('(Segmenter training, epoch no. %d, batch no. %d) loss = %f'
                          % (epoch_idx, batch_idx, s_training_loss.data[0]))

            s_epoch_loss = s_running_loss/(batch_idx + 1)

            # stepping the scheduler
            if self.s_scheduler is not None:
                self.s_scheduler.step(s_epoch_loss)

            print('(Segmenter training, epoch no. %d) loss = %f'
                  % (epoch_idx, s_epoch_loss))

        return s_epoch_loss

    def train_generator(self, dataset, n_epochs=1, n_batch=1, use_gpu=True, verbose=False):
        print('----- Training the generator on the segmenter -----')
        self.s.train(False)
        self.g.train(True)
        for epoch_idx in range(n_epochs):
            g_running_loss = 0
            for batch_idx, (X_batch, y_batch, name) in enumerate(
                    DataLoader(dataset, batch_size=n_batch, shuffle=True)):

                if use_gpu:
                    X_batch, y_batch = Variable(X_batch.cuda()), Variable(y_batch.long()[:, 1, :, :].cuda())
                else:
                    X_batch, y_batch = Variable(X_batch), Variable(y_batch.long()[:, 1, :, :])

                self.g_optim.zero_grad()
                X_fake = self.g(X_batch)

    def visualize_segmenter(self, dataset, export_path, use_gpu=False, with_ground_truth=False, n_inst=None):
        chk_mkdir(export_path)
        self.s.train(False)
        for batch_idx, (X_batch, y_batch, name) in enumerate(DataLoader(dataset, batch_size=1)):
            if n_inst is None:
                n_inst = len(dataset)
            else:
                assert n_inst >= 0, 'n_inst must be nonnegative'
            if use_gpu:
                X_batch = Variable(X_batch.cuda())
                y_out = self.s(X_batch).cpu().data.numpy()
                X_im = X_batch[0].data.cpu().numpy().transpose((1, 2, 0))
            else:
                X_batch = Variable(X_batch)
                y_out = self.s(X_batch).data.numpy()
                X_im = X_batch[0].data.numpy().transpose((1, 2, 0))

            if with_ground_truth:
                visualize_segmentation(
                    X=X_im,
                    y_mask=y_batch[0, 1, :, :].numpy(),
                    y_pred=y_out[0, 1, :, :],
                    export_path=os.path.join(export_path, name[0] + '.png')
                )
            else:
                io.imsave(os.path.join(export_path, name[0] + '.png'), y_out[0, 1, :, :])

            if batch_idx > n_inst:
                break

use_gpu = torch.cuda.is_available()

# paths
model_name = 'UNet_GANv2_test'
result_root_path = '/media/tdanka/B8703F33703EF828/tdanka'
all_datasets = '/home/tdanka/Data/new'
results_path = os.path.join(result_root_path, model_name)
s_comparison_path = os.path.join(results_path, 's_comparison')
# creating folders
chk_mkdir(result_root_path, results_path)
# datasets
train_dataset = TrainFromFolder(
    os.path.join(all_datasets, '/home/tdanka/Data/new/stage1_train_tissue_weighted_patch=256/loc.csv'),
    transform=T.ToTensor(), remove_alpha=True, one_hot_mask=2, weighted=True
)
x = next(iter(train_dataset))
train_original_dataset = TrainFromFolder(
    os.path.join(all_datasets, 'stage1_train_merged/loc.csv'),
    transform=T.ToTensor(), remove_alpha=True, one_hot_mask=2
)
test_dataset = TestFromFolder(
    os.path.join(all_datasets, 'stage1_test/loc.csv'),
    transform=T.ToTensor(), remove_alpha=True
)

"""
-----------------
----- Model -----
-----------------
"""

s = UNet(3, 2)
t = UNet(3, 3)
if use_gpu:
    s.cuda()
    t.cuda()

# lr = 0.001 seems to work WITHOUT PRETRAINING
s_optim = optim.Adam(s.parameters(), lr=0.1)
t_optim = optim.Adam(t.parameters(), lr=0.1)
s_scheduler = torch.optim.lr_scheduler.StepLR(s_optim, step_size=10)
t_scheduler = torch.optim.lr_scheduler.StepLR(t_optim, step_size=10)

gan = GANv2(
    s=s, s_optim=s_optim, s_loss=CrossEntropyLoss2d().cuda(), s_scheduler=s_scheduler,
    g=t, g_optim=t_optim
)

gan.train_segmenter(train_dataset, n_epochs=20, n_batch=4, use_gpu=use_gpu, verbose=False)
gan.visualize_segmenter(train_dataset, s_comparison_path, use_gpu=use_gpu, with_ground_truth=True, n_inst=10)
