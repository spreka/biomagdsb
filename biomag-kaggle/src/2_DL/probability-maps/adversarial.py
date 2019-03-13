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
from models.discriminator import Discriminator, SimpleDiscriminator

# TODO: add original image to the input of the generator


class GAN:
    def __init__(
            self, g, d,
            g_optim, d_optim,
            g_loss, d_loss,
            g_scheduler=None, d_scheduler=None
    ):
        self.g = g
        self.d = d
        self.g_optim = g_optim
        self.d_optim = d_optim
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.g_scheduler = g_scheduler
        self.d_scheduler = d_scheduler

    def train_generator(self, dataset, n_epochs, n_batch=1, use_gpu=True, verbose=True):
        print('--------- Generator training on real data ---------')
        self.g.train(True)
        g_epoch_loss = 0
        for epoch_idx in range(n_epochs):
            g_running_loss = 0
            for batch_idx, (X_batch, y_batch, name) in enumerate(
                    DataLoader(dataset, batch_size=n_batch, shuffle=False)):

                if use_gpu:
                    X_batch, y_batch = Variable(X_batch.cuda()), Variable(y_batch.cuda())
                else:
                    X_batch, y_batch = Variable(X_batch), Variable(y_batch)

                self.g_optim.zero_grad()
                y_out = self.g(X_batch)
                g_training_loss = self.g_loss(y_out, y_batch)
                g_training_loss.backward()
                self.g_optim.step()

                g_running_loss += g_training_loss.data[0]

                if verbose:
                    print('(Epoch no. %d, batch no. %d) loss = %f'
                          % (epoch_idx, batch_idx, g_training_loss.data[0]))

            g_epoch_loss = g_running_loss/(batch_idx+1)

            print('(Epoch no. %d) total loss = %f'
                  % (epoch_idx, g_epoch_loss))

            if self.g_scheduler is not None:
                self.g_scheduler.step(g_epoch_loss)

        return g_epoch_loss

    def train_generator_on_discriminator(self, dataset, n_epochs, n_batch=1, use_gpu=True, verbose=True):
        print('--------- Generator training on discriminator ---------')
        self.g.train(True)
        self.d.train(False)
        for epoch_idx in range(n_epochs):
            g_running_loss = 0
            for batch_idx, (X_batch, y_batch, name) in enumerate(
                    DataLoader(dataset, batch_size=n_batch, shuffle=False)):

                if use_gpu:
                    X_batch = Variable(X_batch.cuda())
                else:
                    X_batch = Variable(X_batch)

                # train g, DO NOT TRAIN d
                self.g_optim.zero_grad()
                y_fake_masks = self.g(X_batch)
                g_fake_decision = self.d(torch.cat([X_batch, y_fake_masks], dim=1))

                if use_gpu:
                    y_fake_labels = Variable(torch.ones(y_batch.shape[0]).cuda())
                else:
                    y_fake_labels = Variable(torch.ones(y_batch.shape[0]))

                g_learned_loss = self.d_loss(g_fake_decision, y_fake_labels)
                g_learned_loss.backward()
                self.g_optim.step()

                g_running_loss += g_learned_loss.data[0]

                if verbose:
                    print('(Epoch no. %d, batch no. %d) loss: %f'
                          % (epoch_idx, batch_idx, g_running_loss / (batch_idx + 1)))

            print('G mean = %f, sum = %f' % (
            sum([x.mean() for x in list(self.g.parameters())]), sum([x.sum() for x in list(self.g.parameters())])))

            g_epoch_loss = g_running_loss / (batch_idx + 1)
            if self.g_scheduler is not None:
                self.g_scheduler.step(g_epoch_loss)
            print('(Epoch no. %d) total loss: %f'
                  % (epoch_idx, g_epoch_loss))

        return g_epoch_loss

    def train_discriminator(self, dataset, n_epochs, n_batch=1, use_gpu=True, verbose=True):
        """
        Class labels: 0 = FAKE, 1 = REAL
        """
        print('--------- Discriminator training ---------')
        self.g.train(False)
        self.d.train(True)
        for epoch_idx in range(n_epochs):
            running_loss = 0
            for batch_idx, (X_batch, y_batch, name) in enumerate(
                    DataLoader(dataset, batch_size=n_batch, shuffle=False)):

                if use_gpu:
                    X_batch, y_batch = Variable(X_batch.cuda()), Variable(y_batch.cuda())
                else:
                    X_batch, y_batch = Variable(X_batch), Variable(y_batch)

                # 1. training on REAL data
                self.d_optim.zero_grad()
                d_real_decision = self.d(torch.cat([X_batch, y_batch], dim=1))
                if use_gpu:
                    y_real = Variable(torch.ones(y_batch.shape[0]).cuda())
                else:
                    y_real = Variable(torch.ones(y_batch.shape[0]))

                d_real_loss = self.d_loss(d_real_decision, y_real)
                d_real_loss.backward()
                self.d_optim.step()

                # 2. training on FAKE data, do not train g
                self.d_optim.zero_grad()
                y_fake = self.g(X_batch).detach()   # detaching from g to avoid training it
                d_fake_decision = self.d(torch.cat([X_batch, y_fake], dim=1))
                if use_gpu:
                    y_fake = Variable(torch.zeros(y_batch.shape[0]).cuda())
                else:
                    y_fake = Variable(torch.zeros(y_batch.shape[0]))
                d_fake_loss = self.d_loss(d_fake_decision, y_fake)
                d_fake_loss.backward()
                self.d_optim.step()

                # update running loss
                running_loss += d_real_loss.data[0] + d_fake_loss.data[0]

                if verbose:
                    print('(Epoch no. %d, batch no. %d) real loss = %f \t fake loss = %f'
                          % (epoch_idx, batch_idx, d_real_loss.data[0], d_fake_loss.data[0]))

            d_epoch_loss = running_loss/(2*batch_idx + 2)
            if self.d_scheduler is not None:
                self.d_scheduler.step(d_epoch_loss)
            print('(Epoch no. %d) total loss = %f' % (epoch_idx, d_epoch_loss))

            print('D mean = %f, sum = %f' % (sum([x.mean() for x in list(self.d.parameters())]), sum([x.sum() for x in list(self.d.parameters())])))

        return d_epoch_loss

    def visualize_train(self, dataset, export_path, use_gpu=False, with_ground_truth=False, n_inst=None):
        if not os.path.isdir(export_path):
            os.makedirs(export_path)

        for batch_idx, (X_batch, y_batch, name) in enumerate(DataLoader(dataset, batch_size=1)):
            if n_inst is None:
                n_inst = len(dataset)
            else:
                assert n_inst >= 0, 'n_inst must be nonnegative'
            if use_gpu:
                X_batch = Variable(X_batch.cuda())
                y_out = self.g(X_batch).cpu().data.numpy()
                X_im = X_batch[0].data.cpu().numpy().transpose((1, 2, 0))
            else:
                X_batch = Variable(X_batch)
                y_out = self.g(X_batch).data.numpy()
                X_im = X_batch[0].data.numpy().transpose((1, 2, 0))

            if with_ground_truth:
                visualize_segmentation(
                    X=X_im,
                    y_mask=y_batch[0, 0, :, :].numpy(),
                    y_pred=y_out[0, 0, :, :],
                    export_path=os.path.join(export_path, name[0] + '.png')
                )
            else:
                io.imsave(os.path.join(export_path, name[0] + '.png'), y_out[0, 0, :, :])

            if batch_idx > n_inst:
                break

    def visualize_test(self, dataset, export_path, use_gpu=False):
        if not os.path.isdir(export_path):
            os.makedirs(export_path)

        for X_batch, name in DataLoader(dataset, batch_size=1):
            if use_gpu:
                X_batch = Variable(X_batch.cuda())
                y_out = self.g(X_batch).cpu().data.numpy()
                X_im = X_batch[0].data.cpu().numpy().transpose((1, 2, 0))
            else:
                X_batch = Variable(X_batch)
                y_out = self.g(X_batch).data.numpy()
                X_im = X_batch[0].data.numpy().transpose((1, 2, 0))

            io.imsave(os.path.join(export_path, name[0] + '.png'), y_out[0, 0, :, :])

    def visualize_everything(
            self, train_original_dataset, test_dataset,
            comparison_path, train_pmap_path, test_pmap_path,
            use_gpu=True
    ):
        # result visualization
        self.visualize_train(
            train_original_dataset, comparison_path,
            use_gpu=use_gpu, with_ground_truth=True, n_inst=20
        )

        # TRAIN probability map generation
        self.visualize_train(
            train_original_dataset, train_pmap_path,
            use_gpu=use_gpu, with_ground_truth=False
        )

        # TEST probability map generation
        self.visualize_test(test_dataset, test_pmap_path, use_gpu=use_gpu)

use_gpu = torch.cuda.is_available()

# paths
model_name = 'UNet_GAN_2017_03_01'
result_root_path = '/media/tdanka/B8703F33703EF828/tdanka/results'
all_datasets = '/home/tdanka/Data/new'
results_path = os.path.join(result_root_path, model_name)
# creating folders
chk_mkdir(result_root_path, results_path)
# datasets
train_dataset = TrainFromFolder(os.path.join(all_datasets, 'stage1_train_merged_patch=256/loc.csv'), transform=T.ToTensor(), remove_alpha=True)
train_original_dataset = TrainFromFolder(os.path.join(all_datasets, 'stage1_train_merged/loc.csv'), transform=T.ToTensor(), remove_alpha=True)
test_dataset = TestFromFolder(os.path.join(all_datasets, 'stage1_test/loc.csv'), transform=T.ToTensor(), remove_alpha=True)

"""
-----------------
----- Model -----
-----------------
"""

generator = UNet(3, 1)
discriminator = Discriminator(4, 1)
generator.cuda()
discriminator.cuda()
# lr = 0.001 seems to work WITHOUT PRETRAINING
g_optim = optim.Adam(generator.parameters(), lr=0.001)
d_optim = optim.Adam(discriminator.parameters(), lr=0.001)
#g_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(g_optim, factor=0.1, verbose=True, patience=5)
#d_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(d_optim, factor=0.1, verbose=True, patience=5)

gan = GAN(
    g=generator, d=discriminator,
    g_optim=g_optim, d_optim=d_optim,
    g_loss=nn.MSELoss().cuda(), d_loss=nn.MSELoss().cuda(),
    #g_scheduler=g_scheduler, d_scheduler=d_scheduler
)

# pretraining generator
"""
gan.train_generator(train_dataset, n_epochs=1, n_batch=4, use_gpu=use_gpu, verbose=False)
gan.visualize_everything(
    train_original_dataset, test_dataset,
    os.path.join(results_path, 'pre', 'comparison'),
    os.path.join(results_path, 'pre', 'train_pmap'),
    os.path.join(results_path, 'pre', 'test_pmap')
)
torch.save(gan.g, os.path.join(results_path, 'pre', 'generator_pretrained'))
"""

# adversarial training
n_rounds = 300
for round_idx in range(n_rounds):
    print('******************************************')
    print('************** Round no. %d **************' % round_idx)
    print('******************************************')
    d_loss = gan.train_discriminator(train_dataset, n_epochs=5, n_batch=4, use_gpu=use_gpu, verbose=False)
    g_loss = gan.train_generator_on_discriminator(train_dataset, n_epochs=1, n_batch=8, use_gpu=use_gpu, verbose=False)

    if round_idx % 10 == 0:
        # saving models
        torch.save(gan.g, os.path.join(results_path, 'generator'))
        torch.save(gan.d, os.path.join(results_path, 'discriminator'))

        gan.visualize_everything(
            train_original_dataset, test_dataset,
            os.path.join(results_path, 'round_%d' % round_idx, 'comparison'),
            os.path.join(results_path, 'round_%d' % round_idx, 'train_pmap'),
            os.path.join(results_path, 'round_%d' % round_idx, 'test_pmap')
        )