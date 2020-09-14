import os
import numpy as np
from skimage.measure import label
import skimage.transform
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

# local imports
from utils import *


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
            for batch_idx, (X_batch, y_batch, *rest) in enumerate(DataLoader(dataset, batch_size=n_batch, shuffle=True)):
                if self.use_gpu:
                    X_batch, y_batch = Variable(X_batch.cuda()), Variable(y_batch.cuda())
                else:
                    X_batch, y_batch = Variable(X_batch), Variable(y_batch)

                # training
                self.optimizer.zero_grad()
                y_out = self.model(X_batch)
                training_loss = self.loss(y_out, y_batch)
                epoch_running_loss += training_loss.data
                training_loss.backward()
                self.optimizer.step()

                if verbose:
                    print('(Epoch no. %d, batch no. %d) loss: %f' % (epoch_idx, batch_idx, training_loss.data))

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
        for batch_idx, (X_batch, y_batch, *rest) in enumerate(DataLoader(dataset, batch_size=n_batch, shuffle=False)):

            if self.use_gpu:
                X_batch, y_batch = Variable(X_batch.cuda(), volatile=True), Variable(y_batch.cuda(), volatile=True)
            else:
                X_batch, y_batch = Variable(X_batch, volatile=True), Variable(y_batch, volatile=True)

            # training
            y_out = self.model(X_batch)
            training_loss = self.validation_loss(y_out, y_batch)

            total_running_loss += training_loss.data

        print('Validation loss: %f' % (total_running_loss / (batch_idx + 1)))
        self.model.train(True)

        del X_batch, y_batch, training_loss, y_out

        return total_running_loss/(batch_idx + 1)

    def instancewise_stats(self, dataset):
        self.model.train(False)

        stats = {'losses': list(), 'names': list()}
        for batch_idx, (X_batch, y_batch, *rest) in enumerate(DataLoader(dataset, batch_size=1, shuffle=False)):

            name = rest[-1]
            if self.use_gpu:
                X_batch, y_batch = Variable(X_batch.cuda(), volatile=True), Variable(y_batch.cuda(), volatile=True)
            else:
                X_batch, y_batch = Variable(X_batch, volatile=True), Variable(y_batch, volatile=True)

            # training
            y_out = self.model(X_batch)
            training_loss = self.validation_loss(y_out, y_batch)
            stats['losses'].append(training_loss.data)
            stats['names'].append(name)

        self.model.train(True)

        del X_batch, y_batch, training_loss, y_out

        return stats

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

    def predictbb(self, dataset, folder_name='prediction'):
        self.model.train(False)
        export_path = os.path.join(self.model_results_path, folder_name)
        chk_mkdir(export_path)

        masks = {}
        for batch_idx, (X_batch, *rest) in enumerate(DataLoader(dataset, batch_size=1)):
            name = rest[-1]
            if self.use_gpu:
                X_batch = Variable(X_batch.cuda())
                y_out = self.model(X_batch).cpu().data.numpy()
            else:
                X_batch = Variable(X_batch)
                y_out = self.model(X_batch).data.numpy()
            curmasks = masks.get(name[0],[])
            curmasks.append((batch_idx, y_out))
            masks[name[0]] = curmasks

        for name in masks.keys():
            img = np.zeros(dataset.images.iloc[masks[name][0][0]]['shape'][:2], dtype=np.uint8)
            for i,mask in enumerate(masks[name]):
                info = dataset.images.iloc[masks[name][i][0]]
                fullmask = np.zeros(dataset.images.iloc[masks[name][0][0]]['shape'][:2], dtype=np.uint16)
                lmask = mask[1][0,0,:,:]
                lmask[lmask > 0.5] = 1
                lmask[lmask != 1] = 0
                #lmask = label(lmask)
                #cval = lmask[int(lmask.shape[0]/2),int(lmask.shape[1]/2)]
                #lmask[~cval] = 0
                #lmask[lmask > 0] = 1
                lmask = skimage.transform.resize(lmask, (info['y2']-info['y1'], info['x2']-info['x1']))
                fullmask[info['y1']:info['y2'], info['x1']:info['x2']] = lmask
                img[fullmask == 1] = 255#i+1
            io.imsave(os.path.join(export_path, name + '.tiff'), img)