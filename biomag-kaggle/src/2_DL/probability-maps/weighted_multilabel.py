import os
import torch.optim as optim
import torchvision.transforms as T

# local imports
from utils import *
from trainers import WeightedMultilabelModel

# architecture imports
from models.unet import UNet
from models.blocks import CrossEntropyLoss2d

model_name = 'UNet_tissue_weighted_multilabel'
all_datasets_path = '/media/tdanka/B8703F33703EF828/tdanka/data'
train_dataset_loc = os.path.join(all_datasets_path, 'stage1_train_tissue_multilabel/loc.csv')
test_dataset_loc = os.path.join(all_datasets_path, 'stage1_test/loc.csv')
results_root_path = '/media/tdanka/B8703F33703EF828/tdanka/results'

tf = make_transform(size=(256, 256), p_flip=0.5, color_jitter_params=(0.5, 0.5, 0.5, 0.5), long_mask=True)
train_dataset = JointlyTransformedDataset(train_dataset_loc, transform=tf, remove_alpha=True, class_weights=True)
test_dataset = TestFromFolder(test_dataset_loc, transform=T.ToTensor(), remove_alpha=True)
train_original_dataset = TestFromFolder(train_dataset_loc, transform=T.ToTensor(), remove_alpha=True)

net = torch.load(os.path.join(results_root_path, model_name, model_name))#UNet(3, 3)
optimizer = optim.Adam(net.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, verbose=True)

model = WeightedMultilabelModel(
    model=net, optimizer=optimizer, scheduler=scheduler,
    model_name=model_name, results_root_path=results_root_path
)

n_rounds = 30
for round_idx in range(n_rounds):
    model.train_model(train_dataset, n_epochs=100, n_batch=16, verbose=False)
    model.visualize(train_dataset, n_inst=20, folder_name='compare_round%d' % round_idx)
    model.predict(test_dataset, 'test_round%d' % round_idx)
    model.predict(train_original_dataset, 'train_round%d' % round_idx)