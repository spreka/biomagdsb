import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.autograd import Variable
from torch.utils.data import DataLoader

# local imports
from utils import *
from image.dataset import chk_mkdir

# model imports
from models.unet import UNet
from models.blocks import CrossEntropyLoss2d
from models.discriminator import Discriminator, SimpleDiscriminator


class GAN:
    def __init__(
            self, g, g_optim, d, d_optim, d_loss, model_name, results_root_path,
            g_scheduler=None, d_scheduler=None,
    ):
        self.g = g
        self.d = d
        self.g_optim = g_optim
        self.d_optim = d_optim
        self.d_loss = d_loss
        self.g_scheduler = g_scheduler
        self.d_scheduler = d_scheduler
        self.model_name = model_name
        self.model_results_path = os.path.join(results_root_path, model_name)
        self.use_gpu = torch.cuda.is_available()

        chk_mkdir(self.model_results_path)

        if self.use_gpu:
            self.g.cuda()
            self.d.cuda()
            self.d_loss.cuda()

    def train_discriminator(self, dataset, n_epochs, n_batch=1, verbose=False):
        self.d.train(True)
        self.g.train(False)
        print('----- Training the discriminator -----')

        total_running_loss = 0
        for epoch_idx in range(n_epochs):

            epoch_running_loss = 0
            for batch_idx, (X_batch, y_batch, name) in enumerate(DataLoader(dataset, batch_size=n_batch, shuffle=True)):

                if self.use_gpu:
                    X_batch, y_batch = Variable(X_batch.cuda()), Variable(y_batch.cuda())
                    d_real = Variable(torch.ones(X_batch.shape[0]).cuda())
                    d_fake = Variable(torch.zeros(X_batch.shape[0]).cuda())
                else:
                    X_batch, y_batch = Variable(X_batch), Variable(y_batch)
                    d_real = Variable(torch.ones(X_batch.shape[0]))
                    d_fake = Variable(torch.zeros(X_batch.shape[0]))

                # training on real data
                self.d_optim.zero_grad()
                real_labels = self.d(torch.cat([X_batch, y_batch], dim=1))
                real_training_loss = self.d_loss(real_labels, d_real)   # 1 == REAL

                # training on fake data
                self.d_optim.zero_grad()
                fake_masks = self.g(X_batch).detach()
                fake_labels = self.d(torch.cat([X_batch, fake_masks], dim=1))
                fake_training_loss = self.d_loss(fake_labels, d_fake)   # 0 == FAKE

                training_loss = real_training_loss + fake_training_loss
                training_loss.backward()
                self.d_optim.step()
                epoch_running_loss += training_loss.data[0]

                if verbose:
                    print('(Epoch no. %d, batch no. %d) loss: %f' % (epoch_idx, batch_idx, training_loss.data[0]))

            torch.save(self.d, os.path.join(self.model_results_path, 'd'))

            total_running_loss += epoch_running_loss / (2*batch_idx + 2)
            print('(Epoch no. %d) loss: %f' % (epoch_idx, epoch_running_loss / (2*batch_idx + 2)))

            if self.d_scheduler is not None:
                self.d_scheduler.step(epoch_running_loss / (2*batch_idx + 2))

        self.d.train(False)

        return total_running_loss / n_batch

    def train_generator(self, dataset, n_epochs, n_batch=1, verbose=False):
        self.d.train(False)
        self.g.train(True)
        print('----- Training the generator -----')

        total_running_loss = 0
        for epoch_idx in range(n_epochs):

            epoch_running_loss = 0
            for batch_idx, (X_batch, y_batch, name) in enumerate(DataLoader(dataset, batch_size=n_batch, shuffle=True)):

                if self.use_gpu:
                    X_batch, y_batch = Variable(X_batch.cuda()), Variable(y_batch.cuda())
                    y_fake_labels = Variable(torch.ones(y_batch.shape[0]).cuda())
                else:
                    X_batch, y_batch = Variable(X_batch), Variable(y_batch)
                    y_fake_labels = Variable(torch.ones(y_batch.shape[0]))

                # training on real data
                self.g_optim.zero_grad()
                g_fake_masks = self.g(X_batch)
                d_fake_decision = self.d(torch.cat([X_batch, g_fake_masks], dim=1))
                training_loss = self.d_loss(d_fake_decision, y_fake_labels)
                training_loss.backward()
                self.g_optim.step()

                epoch_running_loss += training_loss.data[0]

                if verbose:
                    print('(Epoch no. %d, batch no. %d) loss: %f' % (epoch_idx, batch_idx, training_loss.data[0]))

            torch.save(self.g, os.path.join(self.model_results_path, 'g'))

            total_running_loss += epoch_running_loss / (2 * batch_idx + 2)
            print('(Epoch no. %d) loss: %f' % (epoch_idx, epoch_running_loss / (2 * batch_idx + 2)))

            if self.g_scheduler is not None:
                self.g_scheduler.step(epoch_running_loss / (2 * batch_idx + 2))

        self.d.train(False)

        return total_running_loss / n_batch

    def visualize(self, dataset, n_inst=20, folder_name='comparison'):
        self.g.train(False)
        export_path = os.path.join(self.model_results_path, folder_name)
        chk_mkdir(export_path)
        for batch_idx, (X_batch, y_batch, name) in enumerate(DataLoader(dataset, batch_size=1)):
            if n_inst is None:
                n_inst = len(dataset)
            else:
                assert n_inst >= 0, 'n_inst must be nonnegative'
            if self.use_gpu:
                X_batch = Variable(X_batch.cuda())
                y_out = self.g(X_batch).cpu().data.numpy()
                X_im = X_batch[0].data.cpu().numpy().transpose((1, 2, 0))
            else:
                X_batch = Variable(X_batch)
                y_out = self.g(X_batch).data.numpy()
                X_im = X_batch[0].data.numpy().transpose((1, 2, 0))

            visualize_segmentation(
                X=X_im,
                y_mask=y_batch[0, 0, :, :].numpy(),
                y_pred=y_out[0, 0],
                export_path=os.path.join(export_path, name[0] + '.png')
            )

            if batch_idx > n_inst:
                break

    def predict(self, dataset, folder_name='prediction'):
        self.g.train(False)
        export_path = os.path.join(self.model_results_path, folder_name)
        chk_mkdir(export_path)

        for batch_idx, (X_batch, name) in enumerate(DataLoader(dataset, batch_size=1)):
            if self.use_gpu:
                X_batch = Variable(X_batch.cuda())
                y_out = self.g(X_batch).cpu().data.numpy()
            else:
                X_batch = Variable(X_batch)
                y_out = self.g(X_batch).data.numpy()

            io.imsave(os.path.join(export_path, name[0] + '.png'), y_out[0, 0, :, :])


class GATrainer:
    def __init__(
            self, g, g_optim, g_loss, t, t_optim,
            model_name, results_root_path, g_scheduler=None, t_scheduler=None
    ):
        self.model_name = model_name
        self.model_results_path = os.path.join(results_root_path, model_name)
        self.use_gpu = torch.cuda.is_available()

        chk_mkdir(self.model_results_path)

        if self.use_gpu:
            self.g.cuda()
            self.g_loss.cuda()
            self.t.cuda()


class Model:
    def __init__(
            self, model, loss, optimizer,
            model_name, results_root_path,
            scheduler=None, validation_loss=None
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_name = model_name
        self.model_results_path = os.path.join(results_root_path, model_name)
        self.use_gpu = torch.cuda.is_available()
        if validation_loss is None:
            self.validation_loss = loss
        else:
            self.validation_loss = validation_loss

        if self.use_gpu:
            self.model.cuda()
            self.loss.cuda()
            #self.validation_loss.cuda()

        chk_mkdir(self.model_results_path)

    def train_model(self, dataset, n_epochs, n_batch=1, verbose=False, validation_dataset=None):
        self.model.train(True)
        min_loss = np.inf
        total_running_loss = 0
        for epoch_idx in range(n_epochs):

            epoch_running_loss = 0
            for batch_idx, (X_batch, y_batch, name) in enumerate(DataLoader(dataset, batch_size=n_batch, shuffle=True)):

                if self.use_gpu:
                    X_batch, y_batch = Variable(X_batch.cuda()), Variable(y_batch.cuda())
                else:
                    X_batch, y_batch = Variable(X_batch), Variable(y_batch)

                # training
                self.optimizer.zero_grad()
                y_out = self.model(X_batch)
                training_loss = self.loss(y_out, y_batch)
                training_loss.backward()
                self.optimizer.step()

                epoch_running_loss += training_loss.data[0]

                if verbose:
                    print('(Epoch no. %d, batch no. %d) loss: %f' % (epoch_idx, batch_idx, training_loss.data[0]))

            total_running_loss += epoch_running_loss/(batch_idx + 1)
            print('(Epoch no. %d) loss: %f' % (epoch_idx, epoch_running_loss/(batch_idx + 1)))

            if validation_dataset is not None:
                validation_error = self.validate(validation_dataset, n_batch=1)
                if validation_error < min_loss:
                    torch.save(self.model, os.path.join(self.model_results_path, self.model_name))
                    print('Validation loss improved from %f to %f, model saved' % (min_loss, validation_error))
                    min_loss = validation_error

            else:
                validation_error = None
                if epoch_running_loss/(batch_idx + 1) < min_loss:
                    torch.save(self.model, os.path.join(self.model_results_path, self.model_name))
                    print('Training loss improved from %f to %f, model saved' % (min_loss, epoch_running_loss / (batch_idx + 1)))
                    min_loss = epoch_running_loss / (batch_idx + 1)

            if self.scheduler is not None:
                if validation_error:
                    self.scheduler.step(validation_error)
                else:
                    self.scheduler.step(epoch_running_loss/(batch_idx + 1))

        self.model.train(False)

        del X_batch, y_batch

        return total_running_loss/n_batch

    def validate(self, dataset, n_batch=1):
        self.model.train(False)

        total_running_loss = 0
        for batch_idx, (X_batch, y_batch, name) in enumerate(DataLoader(dataset, batch_size=n_batch, shuffle=False)):

            if self.use_gpu:
                X_batch, y_batch = Variable(X_batch.cuda(), volatile=True), Variable(y_batch.cuda(), volatile=True)
            else:
                X_batch, y_batch = Variable(X_batch, volatile=True), Variable(y_batch, volatile=True)

            # training
            y_out = self.model(X_batch)
            training_loss = self.validation_loss(y_out, y_batch)

            total_running_loss += training_loss.data[0]

        print('Validation loss: %f' % (total_running_loss / (batch_idx + 1)))
        self.model.train(True)

        del X_batch, y_batch

        return total_running_loss/(batch_idx + 1)

    def visualize(self, dataset, n_inst=20, folder_name='comparison'):
        self.model.train(False)
        export_path = os.path.join(self.model_results_path, folder_name)
        chk_mkdir(export_path)
        for batch_idx, (X_batch, y_batch, *rest) in enumerate(DataLoader(dataset, batch_size=1)):
            name = rest[-1]
            if n_inst is None:
                n_inst = len(dataset)
            else:
                assert n_inst >= 0, 'n_inst must be nonnegative'
            if self.use_gpu:
                X_batch = Variable(X_batch.cuda())
                y_out = self.model(X_batch).cpu().data.numpy()
                X_im = X_batch[0].data.cpu().numpy().transpose((1, 2, 0))
            else:
                X_batch = Variable(X_batch)
                y_out = self.model(X_batch).data.numpy()
                X_im = X_batch[0].data.numpy().transpose((1, 2, 0))

            visualize_segmentation(
                X=X_im,
                y_mask=y_batch[0, 0, :, :].numpy(),
                y_pred=y_out[0, 0],
                export_path=os.path.join(export_path, name[0] + '.png')
            )

            if batch_idx > n_inst:
                break

    def predict(self, dataset, folder_name='prediction'):
        self.model.train(False)
        export_path = os.path.join(self.model_results_path, folder_name)
        chk_mkdir(export_path)

        for batch_idx, (X_batch, *rest) in enumerate(DataLoader(dataset, batch_size=1)):
            name = rest[-1]
            if self.use_gpu:
                X_batch = Variable(X_batch.cuda())
                y_out = self.model(X_batch).cpu().data.numpy()
            else:
                X_batch = Variable(X_batch)
                y_out = self.model(X_batch).data.numpy()

            io.imsave(os.path.join(export_path, name[0] + '.png'), y_out[0, 0, :, :])


class RCNNCorrection:
    def __init__(
            self, model, loss, optimizer,
            model_name, results_root_path,
            scheduler=None
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_name = model_name
        self.model_results_path = os.path.join(results_root_path, model_name)
        self.use_gpu = torch.cuda.is_available()

        if self.use_gpu:
            self.model.cuda()
            self.loss.cuda()

        chk_mkdir(self.model_results_path)

    def train_model(self, dataset, n_epochs, n_batch=1, verbose=False, validation_dataset=None):
        self.model.train(True)

        total_running_loss = 0
        for epoch_idx in range(n_epochs):

            epoch_running_loss = 0
            for batch_idx, (X_batch, y_mask_batch, y_rcnn_batch, name) in enumerate(DataLoader(dataset, batch_size=n_batch, shuffle=True)):

                if self.use_gpu:
                    X_batch, y_mask_batch, y_rcnn_batch = Variable(X_batch.cuda()), Variable(y_mask_batch.cuda()), Variable(y_rcnn_batch.cuda())
                else:
                    X_batch, y_mask_batch, y_rcnn_batch = Variable(X_batch), Variable(y_mask_batch), Variable(y_rcnn_batch)

                # training
                self.optimizer.zero_grad()
                y_out = self.model(torch.cat([X_batch, y_rcnn_batch], dim=1))
                training_loss = self.loss(y_out, y_mask_batch)
                training_loss.backward()
                self.optimizer.step()

                epoch_running_loss += training_loss.data[0]

                if verbose:
                    print('(Epoch no. %d, batch no. %d) loss: %f' % (epoch_idx, batch_idx, training_loss.data[0]))

            torch.save(self.model, os.path.join(self.model_results_path, self.model_name))

            total_running_loss += epoch_running_loss/(batch_idx + 1)
            print('(Epoch no. %d) loss: %f' % (epoch_idx, epoch_running_loss/(batch_idx + 1)))

            if validation_dataset is not None:
                validation_error = self.validate(validation_dataset, n_batch=1)
            else:
                validation_error = None

            if self.scheduler is not None:
                if validation_error:
                    self.scheduler.step(validation_error)
                else:
                    self.scheduler.step(epoch_running_loss/(batch_idx + 1))

        self.model.train(False)

        return total_running_loss/n_batch

    def validate(self, dataset, n_batch=1):
        self.model.train(False)

        total_running_loss = 0
        for batch_idx, (X_batch, y_mask_batch, y_rcnn_batch, name) in enumerate(
                DataLoader(dataset, batch_size=n_batch, shuffle=True)):

            if self.use_gpu:
                X_batch, y_mask_batch, y_rcnn_batch = Variable(X_batch.cuda()), Variable(
                    y_mask_batch.cuda()), Variable(y_rcnn_batch.cuda())
            else:
                X_batch, y_mask_batch, y_rcnn_batch = Variable(X_batch), Variable(y_mask_batch), Variable(
                    y_rcnn_batch)

            # training
            y_out = self.model(torch.cat([X_batch, y_rcnn_batch], dim=1))
            training_loss = self.loss(y_out, y_mask_batch)

            total_running_loss += training_loss.data[0]

        print('Validation loss: %f' % (total_running_loss / (batch_idx + 1)))
        self.model.train(True)

        return total_running_loss

    def predict(self, dataset, folder_name='prediction'):
        self.model.train(False)
        export_path = os.path.join(self.model_results_path, folder_name)
        chk_mkdir(export_path)

        for batch_idx, (X_batch, y_mask_batch, y_rcnn_batch, name) in enumerate(DataLoader(dataset, batch_size=1)):
            if self.use_gpu:
                X_batch, y_rcnn_batch = Variable(X_batch.cuda()), Variable(y_rcnn_batch.cuda())
                y_out = self.model(torch.cat([X_batch, y_rcnn_batch], dim=1)).cpu().data.numpy()
                y_rcnn_batch = y_rcnn_batch.data.cpu().numpy()
                X_im = X_batch[0].data.cpu().numpy().transpose((1, 2, 0))
            else:
                X_batch, y_rcnn_batch = Variable(X_batch), Variable(y_rcnn_batch)
                y_out = self.model(torch.cat([X_batch, y_rcnn_batch], dim=1)).data.numpy()
                y_rcnn_batch = y_rcnn_batch.data.numpy()
                X_im = X_batch[0].data.numpy().transpose((1, 2, 0))

            io.imsave(os.path.join(export_path, name[0] + '.png'), y_out[0].transpose((1, 2, 0)))

    def visualize(self, dataset, folder_name='visualize', n_inst=None):
        self.model.train(False)
        export_path = os.path.join(self.model_results_path, folder_name)
        chk_mkdir(export_path)
        for batch_idx, (X_batch, y_mask_batch, y_rcnn_batch, name) in enumerate(DataLoader(dataset, batch_size=1)):
            if n_inst is None:
                n_inst = len(dataset)
            else:
                assert n_inst >= 0, 'n_inst must be nonnegative'
            if self.use_gpu:
                X_batch, y_rcnn_batch = Variable(X_batch.cuda()), Variable(y_rcnn_batch.cuda())
                y_out = self.model(torch.cat([X_batch, y_rcnn_batch], dim=1)).cpu().data.numpy()
                y_rcnn_batch = y_rcnn_batch.data.cpu().numpy()
                X_im = X_batch[0].data.cpu().numpy().transpose((1, 2, 0))
            else:
                X_batch, y_rcnn_batch = Variable(X_batch), Variable(y_rcnn_batch)
                y_out = self.model(torch.cat([X_batch, y_rcnn_batch], dim=1)).data.numpy()
                y_rcnn_batch = y_rcnn_batch.data.numpy()
                X_im = X_batch[0].data.numpy().transpose((1, 2, 0))

            visualize_segmentation(
                X=X_im,
                y_mask=1 - y_out[0, 1:].transpose((1, 2, 0)),
                y_pred=1 - y_out[0, 1:].transpose((1, 2, 0)),
                export_path=os.path.join(export_path, name[0] + '.png')
            )

            if batch_idx > n_inst:
                break


class MultilabelModel:
    def __init__(
            self, model, loss, optimizer,
            model_name, results_root_path,
            scheduler=None
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_name = model_name
        self.model_results_path = os.path.join(results_root_path, model_name)
        self.use_gpu = torch.cuda.is_available()

        if self.use_gpu:
            self.model.cuda()
            self.loss.cuda()

        chk_mkdir(self.model_results_path)

    def train_model(self, dataset, n_epochs, n_batch=1, verbose=False):
        self.model.train(True)

        total_running_loss = 0
        for epoch_idx in range(n_epochs):

            epoch_running_loss = 0
            for batch_idx, (X_batch, y_batch, name) in enumerate(DataLoader(dataset, batch_size=n_batch, shuffle=True)):

                if self.use_gpu:
                    X_batch, y_batch = Variable(X_batch.cuda()), Variable(y_batch.cuda())
                else:
                    X_batch, y_batch = Variable(X_batch), Variable(y_batch)

                # training
                self.optimizer.zero_grad()
                y_out = self.model(X_batch)
                training_loss = self.loss(y_out, y_batch)
                training_loss.backward()
                self.optimizer.step()

                epoch_running_loss += training_loss.data[0]

                if verbose:
                    print('(Epoch no. %d, batch no. %d) loss: %f' % (epoch_idx, batch_idx, training_loss.data[0]))

            torch.save(self.model, os.path.join(self.model_results_path, self.model_name))

            total_running_loss += epoch_running_loss/(batch_idx + 1)
            print('(Epoch no. %d) loss: %f' % (epoch_idx, epoch_running_loss/(batch_idx + 1)))

            if self.scheduler is not None:
                self.scheduler.step(epoch_running_loss/(batch_idx + 1))

        self.model.train(False)

        return total_running_loss/n_batch

    def visualize(self, dataset, n_inst=20, folder_name='comparison'):
        self.model.train(False)
        export_path = os.path.join(self.model_results_path, folder_name)
        chk_mkdir(export_path)
        for batch_idx, (X_batch, y_batch, name) in enumerate(DataLoader(dataset, batch_size=1)):
            if n_inst is None:
                n_inst = len(dataset)
            else:
                assert n_inst >= 0, 'n_inst must be nonnegative'
            if self.use_gpu:
                X_batch = Variable(X_batch.cuda())
                y_out = self.model(X_batch).cpu().data.numpy()
                X_im = X_batch[0].data.cpu().numpy().transpose((1, 2, 0))
            else:
                X_batch = Variable(X_batch)
                y_out = self.model(X_batch).data.numpy()
                X_im = X_batch[0].data.numpy().transpose((1, 2, 0))

            visualize_segmentation(
                X=X_im,
                y_mask=y_batch[0].numpy(),
                y_pred=y_out[0].transpose((1, 2, 0)),
                export_path=os.path.join(export_path, name[0] + '.png')
            )

            if batch_idx > n_inst:
                break

    def predict(self, dataset, folder_name='prediction'):
        self.model.train(False)
        export_path = os.path.join(self.model_results_path, folder_name)
        chk_mkdir(export_path)

        for batch_idx, (X_batch, name) in enumerate(DataLoader(dataset, batch_size=1)):
            if self.use_gpu:
                X_batch = Variable(X_batch.cuda())
                y_out = self.model(X_batch).cpu().data.numpy()
            else:
                X_batch = Variable(X_batch)
                y_out = self.model(X_batch).data.numpy()

            io.imsave(os.path.join(export_path, name[0] + '.png'), y_out[0].transpose((1, 2, 0)))


class WeightedBCEModel:
    def __init__(
            self, model, optimizer,
            model_name, results_root_path,
            scheduler=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_name = model_name
        self.model_results_path = os.path.join(results_root_path, model_name)
        self.use_gpu = torch.cuda.is_available()

        if self.use_gpu:
            self.model.cuda()

        chk_mkdir(self.model_results_path)

    def train_model(self, dataset, n_epochs, n_batch=1, verbose=False):
        self.model.train(True)

        total_running_loss = 0
        for epoch_idx in range(n_epochs):

            epoch_running_loss = 0
            for batch_idx, (X_batch, y_batch, y_weight, name) in enumerate(DataLoader(dataset, batch_size=n_batch, shuffle=True)):

                if self.use_gpu:
                    X_batch, y_batch, y_weight = Variable(X_batch.cuda()), Variable(y_batch.cuda()), Variable(y_weight.cuda())
                else:
                    X_batch, y_batch, y_weight = Variable(X_batch), Variable(y_batch), Variable(y_weight)

                # training
                self.optimizer.zero_grad()
                loss = nn.BCELoss(weight=10*y_weight).cuda()
                y_out = self.model(X_batch)
                training_loss = loss(y_out, y_batch)
                training_loss.backward()
                self.optimizer.step()

                epoch_running_loss += training_loss.data[0]

                if verbose:
                    print('(Epoch no. %d, batch no. %d) loss: %f' % (epoch_idx, batch_idx, training_loss.data[0]))

            torch.save(self.model, os.path.join(self.model_results_path, self.model_name))

            total_running_loss += epoch_running_loss/(batch_idx + 1)
            print('(Epoch no. %d) loss: %f' % (epoch_idx, epoch_running_loss/(batch_idx + 1)))

            if self.scheduler is not None:
                self.scheduler.step(epoch_running_loss/(batch_idx + 1))

        self.model.train(False)

        return total_running_loss/n_batch

    def visualize(self, dataset, n_inst=20, folder_name='comparison'):
        self.model.train(False)
        export_path = os.path.join(self.model_results_path, folder_name)
        chk_mkdir(export_path)
        for batch_idx, (X_batch, y_batch, y_weight, name) in enumerate(DataLoader(dataset, batch_size=1)):
            if n_inst is None:
                n_inst = len(dataset)
            else:
                assert n_inst >= 0, 'n_inst must be nonnegative'
            if self.use_gpu:
                X_batch = Variable(X_batch.cuda())
                y_out = self.model(X_batch).cpu().data.numpy()
                X_im = X_batch[0].data.cpu().numpy().transpose((1, 2, 0))
            else:
                X_batch = Variable(X_batch)
                y_out = self.model(X_batch).data.numpy()
                X_im = X_batch[0].data.numpy().transpose((1, 2, 0))

            visualize_segmentation(
                X=X_im,
                y_mask=y_batch[0, 0, :, :].numpy(),
                y_pred=y_out[0, 0],
                export_path=os.path.join(export_path, name[0] + '.png')
            )

            if batch_idx > n_inst:
                break

    def predict(self, dataset, folder_name='prediction'):
        self.model.train(False)
        export_path = os.path.join(self.model_results_path, folder_name)
        chk_mkdir(export_path)

        for batch_idx, (X_batch, name) in enumerate(DataLoader(dataset, batch_size=1)):
            if self.use_gpu:
                X_batch = Variable(X_batch.cuda())
                y_out = self.model(X_batch).cpu().data.numpy()
            else:
                X_batch = Variable(X_batch)
                y_out = self.model(X_batch).data.numpy()

            io.imsave(os.path.join(export_path, name[0] + '.png'), y_out[0, 0, :, :])


class WeightedMultilabelModel:
    def __init__(
            self, model, optimizer,
            model_name, results_root_path,
            scheduler=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_name = model_name
        self.model_results_path = os.path.join(results_root_path, model_name)
        self.use_gpu = torch.cuda.is_available()

        if self.use_gpu:
            self.model.cuda()

        chk_mkdir(self.model_results_path)

    def train_model(self, dataset, n_epochs, n_batch=1, verbose=False):
        self.model.train(True)

        total_running_loss = 0
        for epoch_idx in range(n_epochs):

            epoch_running_loss = 0
            for batch_idx, (X_batch, y_batch, y_weight, name) in enumerate(DataLoader(dataset, batch_size=n_batch, shuffle=True)):
                if self.use_gpu:
                    X_batch, y_batch, y_weight = Variable(X_batch.cuda()), Variable(y_batch.cuda()), Variable(y_weight.cuda())
                else:
                    X_batch, y_batch, y_weight = Variable(X_batch), Variable(y_batch), Variable(y_weight)

                # training
                self.optimizer.zero_grad()
                y_out = self.model(X_batch)
                training_loss = CrossEntropyLoss2d(weight=y_weight.mean(dim=0))(y_out, y_batch)
                training_loss.backward()
                self.optimizer.step()

                epoch_running_loss += training_loss.data[0]

                if verbose:
                    print('(Epoch no. %d, batch no. %d) loss: %f' % (epoch_idx, batch_idx, training_loss.data[0]))

            torch.save(self.model, os.path.join(self.model_results_path, self.model_name))

            total_running_loss += epoch_running_loss/(batch_idx + 1)
            print('(Epoch no. %d) loss: %f' % (epoch_idx, epoch_running_loss/(batch_idx + 1)))

            if self.scheduler is not None:
                self.scheduler.step(epoch_running_loss/(batch_idx + 1))

        self.model.train(False)

        return total_running_loss/n_batch

    def visualize(self, dataset, n_inst=20, folder_name='comparison'):
        self.model.train(False)
        export_path = os.path.join(self.model_results_path, folder_name)
        chk_mkdir(export_path)
        for batch_idx, (X_batch, y_batch, y_weight, name) in enumerate(DataLoader(dataset, batch_size=1)):
            if n_inst is None:
                n_inst = len(dataset)
            else:
                assert n_inst >= 0, 'n_inst must be nonnegative'
            if self.use_gpu:
                X_batch = Variable(X_batch.cuda())
                y_out = self.model(X_batch).cpu().data.numpy()
                X_im = X_batch[0].data.cpu().numpy().transpose((1, 2, 0))
            else:
                X_batch = Variable(X_batch)
                y_out = self.model(X_batch).data.numpy()
                X_im = X_batch[0].data.numpy().transpose((1, 2, 0))

            visualize_segmentation(
                X=X_im,
                y_mask=y_batch[0].numpy(),
                y_pred=y_out[0].transpose((1, 2, 0)),
                export_path=os.path.join(export_path, name[0] + '.png')
            )

            if batch_idx > n_inst:
                break

    def predict(self, dataset, folder_name='prediction'):
        self.model.train(False)
        export_path = os.path.join(self.model_results_path, folder_name)
        chk_mkdir(export_path)

        for batch_idx, (X_batch, name) in enumerate(DataLoader(dataset, batch_size=1)):
            if self.use_gpu:
                X_batch = Variable(X_batch.cuda())
                y_out = self.model(X_batch).cpu().data.numpy()
            else:
                X_batch = Variable(X_batch)
                y_out = self.model(X_batch).data.numpy()

            io.imsave(os.path.join(export_path, name[0] + '.png'), y_out[0].transpose((1, 2, 0)))


class DimReduce:
    def __init__(
            self, model, loss, optimizer,
            model_name, results_root_path,
            scheduler=None
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_name = model_name
        self.model_results_path = os.path.join(results_root_path, model_name)
        self.use_gpu = torch.cuda.is_available()

        if self.use_gpu:
            self.model.cuda()
            self.loss.cuda()

        chk_mkdir(self.model_results_path)

    def train_model(self, dataset, n_epochs, n_batch=1, verbose=False):
        self.model.train(True)

        total_running_loss = 0
        for epoch_idx in range(n_epochs):

            epoch_running_loss = 0
            for batch_idx, (X_batch, name) in enumerate(DataLoader(dataset, batch_size=n_batch, shuffle=True)):

                if self.use_gpu:
                    X_batch = Variable(X_batch.cuda())
                else:
                    X_batch = Variable(X_batch)

                # training
                self.optimizer.zero_grad()
                X_out = self.model(X_batch)
                training_loss = self.loss(X_out, X_batch)
                training_loss.backward()
                self.optimizer.step()

                epoch_running_loss += training_loss.data[0]

                if verbose:
                    print('(Epoch no. %d, batch no. %d) loss: %f' % (epoch_idx, batch_idx, training_loss.data[0]))

            torch.save(self.model, os.path.join(self.model_results_path, self.model_name))

            total_running_loss += epoch_running_loss/(batch_idx + 1)
            print('(Epoch no. %d) loss: %f' % (epoch_idx, epoch_running_loss/(batch_idx + 1)))

            if self.scheduler is not None:
                self.scheduler.step(epoch_running_loss/(batch_idx + 1))

        self.model.train(False)

        return total_running_loss/n_batch

    def visualize(self, dataset, n_inst=20, folder_name='comparison'):
        self.model.train(False)
        export_path = os.path.join(self.model_results_path, folder_name)
        chk_mkdir(export_path)
        for batch_idx, (X_batch, name) in enumerate(DataLoader(dataset, batch_size=1)):
            if n_inst is None:
                n_inst = len(dataset)
            else:
                assert n_inst >= 0, 'n_inst must be nonnegative'
            if self.use_gpu:
                X_batch = Variable(X_batch.cuda())
                X_out = self.model(X_batch).cpu().data.numpy()
                X_im = X_batch[0].data.cpu().numpy().transpose((1, 2, 0))
            else:
                X_batch = Variable(X_batch)
                X_out = self.model(X_batch).data.numpy()
                X_im = X_batch[0].data.numpy().transpose((1, 2, 0))

            if sum([abs(X_out.shape[k] - X_batch.shape[k]) for k in range(len(X_out.shape))]) != 0:
                print(X_out.shape)

            visualize_segmentation(
                X=X_im,
                y_mask=1-X_im,
                y_pred=1-X_out[0].transpose((1, 2, 0)),
                export_path=os.path.join(export_path, name[0] + '.png')
            )

            if batch_idx > n_inst:
                break

    def predict(self, dataset, folder_name='prediction'):
        self.model.train(False)
        export_path = os.path.join(self.model_results_path, folder_name)
        chk_mkdir(export_path)

        for batch_idx, (X_batch, name) in enumerate(DataLoader(dataset, batch_size=1)):
            if self.use_gpu:
                X_batch = Variable(X_batch.cuda())
                X_out = self.model(X_batch).cpu().data.numpy()
            else:
                X_batch = Variable(X_batch)
                X_out = self.model(X_batch).data.numpy()

            io.imsave(os.path.join(export_path, name[0] + '.png'), X_out[0].transpose((1, 2, 0)))

    def encode(self, dataset):
        self.model.train(False)

        encoded = list()

        for batch_idx, (X_batch, name) in enumerate(DataLoader(dataset, batch_size=1)):
            if self.use_gpu:
                X_batch = Variable(X_batch.cuda())
                X_out = self.model.encode(X_batch).cpu()
            else:
                X_batch = Variable(X_batch)
                X_out = self.model.encode(X_batch)

            X_out = X_out.view(X_out.size()[0], -1).data.numpy()
            encoded.append((X_out, name))

        return encoded


class TypeClassifier:
    def __init__(
            self, model, loss, optimizer,
            model_name, results_root_path, class_labels,
            scheduler=None
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.model_name = model_name
        self.model_results_path = os.path.join(results_root_path, model_name)
        self.class_labels = class_labels
        self.use_gpu = torch.cuda.is_available()

        if self.use_gpu:
            self.model.cuda()
            self.loss.cuda()

        chk_mkdir(self.model_results_path)

    def train_model(self, dataset, n_epochs, n_batch=1, verbose=False):
        self.model.train(True)

        total_running_loss = 0
        for epoch_idx in range(n_epochs):

            epoch_running_loss = 0
            for batch_idx, (X_batch, y_prob, y_label, name) in enumerate(DataLoader(dataset, batch_size=n_batch, shuffle=True)):

                if self.use_gpu:
                    X_batch, y_prob, y_label = Variable(X_batch.cuda()), Variable(y_prob.cuda()), Variable(y_label.cuda())
                else:
                    X_batch, y_prob, y_label = Variable(X_batch), Variable(y_prob), Variable(y_label)

                # training
                self.optimizer.zero_grad()
                y_out = self.model(X_batch)
                training_loss = self.loss(y_out, y_label)
                training_loss.backward()
                self.optimizer.step()

                epoch_running_loss += training_loss.data[0]

                if verbose:
                    print('(Epoch no. %d, batch no. %d) loss: %f' % (epoch_idx, batch_idx, training_loss.data[0]))

            torch.save(self.model, os.path.join(self.model_results_path, self.model_name))

            total_running_loss += epoch_running_loss/(batch_idx + 1)
            print('(Epoch no. %d) loss: %f' % (epoch_idx, epoch_running_loss/(batch_idx + 1)))

            if self.scheduler is not None:
                self.scheduler.step(epoch_running_loss/(batch_idx + 1))

        self.model.train(False)

        return total_running_loss/n_batch

    def visualize(self, dataset, folder_name):
        self.model.train(False)
        export_path = os.path.join(self.model_results_path, folder_name)
        chk_mkdir(export_path)

        for batch_idx, (X_image, y_prob, y_label, name) in enumerate(DataLoader(dataset, batch_size=1)):
            if self.use_gpu:
                X_image = Variable(X_image.cuda())
                X_pred = self.model(X_image).data.cpu().numpy()
            else:
                X_image = Variable(X_image)
                X_pred = self.model(X_image).data.numpy()
