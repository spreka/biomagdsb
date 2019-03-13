import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

# local imports
from utils import *
from trainers import GAN

# architecture imports
from models.unet import UNet
from models.discriminator import Discriminator

all_datasets_path = '/media/tdanka/B8703F33703EF828/tdanka/data'
train_dataset_loc = os.path.join(all_datasets_path, 'stage1_train_merged/loc.csv')
test_dataset_loc = os.path.join(all_datasets_path, 'stage1_test/loc.csv')
results_root_path = '/media/tdanka/B8703F33703EF828/tdanka/results'

tf = make_transform(size=(128, 128), p_flip=0.5, color_jitter_params=(0, 0, 0, 0))
train_dataset = JointlyTransformedDataset(train_dataset_loc, transform=tf, remove_alpha=True)
test_dataset = TestFromFolder(test_dataset_loc, transform=T.ToTensor(), remove_alpha=True)
train_original_dataset = TestFromFolder(train_dataset_loc, transform=T.ToTensor(), remove_alpha=True)

model_name = 'GAN_overnight_2018_03_06'
g = UNet(3, 1)
g_optimizer = optim.Adam(g.parameters(), lr=1e-4)
d = Discriminator(4, 1)
d_loss = nn.BCELoss()
d_optimizer = optim.Adam(d.parameters(), lr=1e-4)

gan = GAN(
    g=g, g_optim=g_optimizer, d=d, d_loss=d_loss, d_optim=d_optimizer,
    model_name=model_name, results_root_path=results_root_path
)

n_rounds = 1000
for round_idx in range(n_rounds):
    print('***** Round no. %d *****' % round_idx)
    gan.train_discriminator(train_dataset, n_epochs=10, n_batch=16, verbose=False)
    gan.train_generator(train_dataset, n_epochs=10, n_batch=16, verbose=False)

    if round_idx % 10 == 0:
        gan.visualize(train_dataset, folder_name='compare_round_%d' % round_idx)
        gan.predict(test_dataset, folder_name='test_round_%d' % round_idx)