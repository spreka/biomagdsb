import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

# local imports
from utils import *
from trainers import RCNNCorrection

# architecture imports
from models.unet import UNet
from models.blocks import CrossEntropyLoss2d


def ApplyToAll(*args):
    resize_convert = T.Compose([T.ToPILImage(), T.Resize(size=(256, 256)), T.ToTensor()])
    return [resize_convert(x) for x in args]


model_name = 'UNet_MaskRCNN_correction'
results_root_path = '/media/tdanka/B8703F33703EF828/tdanka/results'
train_dataset_loc = '/media/tdanka/B8703F33703EF828/tdanka/UNet_correction/TIVADAR/rcnn2unet_split/train/loc.csv'
validate_dataset_loc = '/media/tdanka/B8703F33703EF828/tdanka/UNet_correction/TIVADAR/rcnn2unet_split/test/loc.csv'
class_weights = [1.0, 5.0, 25.0, 25.0]

tf = make_transform_RCNN(size=(128, 128), p_flip=0.5, long_mask=True)
train_dataset = TrainWithRCNNMask(train_dataset_loc, transform=tf)
validate_dataset = TrainWithRCNNMask(validate_dataset_loc, transform=tf)
test_dataset = TrainFromFolder(train_dataset_loc, transform=T.ToTensor(), remove_alpha=True)

net = torch.load('/media/tdanka/B8703F33703EF828/tdanka/results/UNet_MaskRCNN_correction/UNet_MaskRCNN_correction')#UNet(4, 4, softmax=True)
loss = CrossEntropyLoss2d(weight=torch.from_numpy(np.array(class_weights)).float())
n_epochs = 20
lr_milestones = [int(p*n_epochs) for p in [0.3, 0.7, 0.9]]
optimizer = optim.Adam(net.parameters(), lr=1e-2)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_milestones)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, verbose=True)

model = RCNNCorrection(
    model=net, loss=loss, optimizer=optimizer, scheduler=scheduler,
    model_name=model_name, results_root_path=results_root_path
)

#model.train_model(train_dataset, n_epochs=n_epochs, n_batch=16, verbose=False, validation_dataset=validate_dataset)
model.visualize(train_dataset, n_inst=None)

