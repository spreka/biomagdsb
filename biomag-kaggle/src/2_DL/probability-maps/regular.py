import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

# local imports
from utils import *
from trainers import Model

# architecture imports
from models.unet import UNet


def AllToTensor(*args):
    return [T.ToTensor()(x) for x in args]


def normalize(image, mask):
    image, mask = T.ToTensor()(image), T.ToTensor()(mask)
    image = T.Normalize(mean=(0.5, 0.5, 0.5), std=(1, 1, 1))(image)
    return image, mask


def Jaccard(x, y):
    return -(x*y).sum()/(x + y - x*y).sum()


class JaccardLoss(nn.Module):
    def __init__(self):
        super(JaccardLoss, self).__init__()

    def forward(self, x, y):
        return -(x*y).sum()/(x + y - x*y).sum()


model_name = 'UNet_MRCNN_correction_Jaccard_normalized'
all_datasets_path = '/media/tdanka/B8703F33703EF828/tdanka/data'
train_dataset_loc = '/media/tdanka/B8703F33703EF828/tdanka/data/GOLD_split/train/loc.csv'
test_dataset_loc = '/media/tdanka/B8703F33703EF828/tdanka/data/GOLD_split/test/loc.csv'
results_root_path = '/media/tdanka/B8703F33703EF828/tdanka/results'

tf_train = make_transform(
    size=(128, 128), p_flip=0.5, color_jitter_params=(0.2, 0.2, 0.2, 0.2),
    random_resize=[0.5, 1.5]
)
train_dataset = JointlyTransformedDataset(train_dataset_loc, transform=tf_train, remove_alpha=True)
test_dataset = JointlyTransformedDataset(test_dataset_loc, transform=normalize, remove_alpha=True)

test_transform = T.Compose([T.ToTensor(), T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
stage1_test = TestFromFolder('/media/tdanka/B8703F33703EF828/tdanka/data/stage1_test/loc.csv', transform=test_transform, remove_alpha=True)

net = torch.load(os.path.join(results_root_path, model_name, model_name))
loss = JaccardLoss()#nn.BCELoss()
n_epochs = 100
lr_milestones = [int(p*n_epochs) for p in [0.7, 0.9]]
optimizer = optim.Adam(net.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_milestones)
#scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1, verbose=True)

model = Model(
    model=net, loss=loss, optimizer=optimizer, scheduler=scheduler,
    model_name=model_name, results_root_path=results_root_path,
    #validation_loss=JaccardLoss()
)

#model.train_model(train_dataset, n_epochs=n_epochs, n_batch=8, verbose=False, validation_dataset=test_dataset)
#model.visualize(test_dataset)
model.predict(stage1_test, 'stage1_test')

