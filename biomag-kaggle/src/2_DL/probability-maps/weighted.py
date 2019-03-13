import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

# local imports
from utils import *
from trainers import WeightedBCEModel

# architecture imports
from models.unet import UNet

model_name = 'UNet_weighted'
all_datasets_path = '/media/tdanka/B8703F33703EF828/tdanka/data'
train_dataset_loc = os.path.join(all_datasets_path, 'stage1_train_merged_weight/loc.csv')
test_dataset_loc = os.path.join(all_datasets_path, 'stage1_test/loc.csv')
results_root_path = '/media/tdanka/B8703F33703EF828/tdanka/results'

tf = make_transform_with_weight(size=(256, 256), p_flip=0.5, color_jitter_params=(0, 0, 0, 0))
train_dataset = JointlyTransformedWeightedDataset(train_dataset_loc, transform=tf, remove_alpha=True)
test_dataset = TestFromFolder(test_dataset_loc, transform=T.ToTensor(), remove_alpha=True)
train_original_dataset = TestFromFolder(train_dataset_loc, transform=T.ToTensor(), remove_alpha=True)

net = torch.load('/media/tdanka/B8703F33703EF828/tdanka/results/UNet_weighted/UNet_weighted')#UNet(3, 1)
optimizer = optim.Adam(net.parameters(), lr=1e-2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1, verbose=True)

model = WeightedBCEModel(
    model=net, optimizer=optimizer, scheduler=scheduler,
    model_name=model_name, results_root_path=results_root_path
)

#model.train_model(train_dataset, n_epochs=100, n_batch=16, verbose=False)
model.visualize(train_dataset)
model.predict(test_dataset, 'test')
model.predict(train_original_dataset, 'train')