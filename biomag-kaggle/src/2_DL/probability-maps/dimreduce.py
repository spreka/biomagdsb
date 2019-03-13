import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torchvision.transforms import functional as F

# local imports
from utils import *
from trainers import DimReduce

# architecture imports
from models.stacked_autoencoder import StackedAutoencoder

model_name = 'autoencoder_dimreduce'
all_datasets_path = '/media/tdanka/B8703F33703EF828/tdanka/data'
train_dataset_loc = os.path.join(all_datasets_path, 'stage1_train_merged/loc.csv')
test_dataset_loc = os.path.join(all_datasets_path, 'stage1_test/loc.csv')
results_root_path = '/media/tdanka/B8703F33703EF828/tdanka/results'

tf = T.Compose([T.ToPILImage(), T.RandomCrop((256, 256)), T.ToTensor()])
train_dataset = TestFromFolder(train_dataset_loc, transform=tf, remove_alpha=True)

test_dataset = TestFromFolder(test_dataset_loc, transform=T.ToTensor(), remove_alpha=True)
train_original_dataset = TestFromFolder(train_dataset_loc, transform=T.ToTensor(), remove_alpha=True)

"""
net = StackedAutoencoder(
    encoder=nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1), nn.MaxPool2d(2), nn.ReLU(),
        nn.Conv2d(8, 16, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(16),
        nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.MaxPool2d(2), nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
        nn.MaxPool2d(4)
    ),
    decoder=nn.Sequential(
        nn.ConvTranspose2d(32, 32, kernel_size=4, stride=4), nn.ReLU(), nn.BatchNorm2d(32),
        nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(),
        nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2), nn.ReLU(), nn.BatchNorm2d(8),
        nn.Conv2d(8, 4, kernel_size=3, padding=1), nn.ReLU(),
        nn.ConvTranspose2d(4, 3, kernel_size=2, stride=2), nn.ReLU(),
    )
)
"""

net = torch.load('/media/tdanka/B8703F33703EF828/tdanka/results/autoencoder_dimreduce/autoencoder_dimreduce')

loss = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1, verbose=True)

model = DimReduce(
    model=net, loss=loss, optimizer=optimizer, scheduler=scheduler,
    model_name=model_name, results_root_path=results_root_path
)

encoded = model.encode(train_dataset)

exit()

model.train_model(train_dataset, n_epochs=100, n_batch=4, verbose=False)
model.visualize(train_dataset, n_inst=50)