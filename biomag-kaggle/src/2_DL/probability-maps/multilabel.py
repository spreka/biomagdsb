import os
import torch.optim as optim
import torchvision.transforms as T

# local imports
from utils import *
from trainers import MultilabelModel

# architecture imports
from models.unet import UNet
from models.blocks import CrossEntropyLoss2d

patience = 20
class_weights = [1.0, 5.0, 25.0]
jitter = 0.25
model_name = 'UNet_fluo_fix_weight=%s_patience=%s_jitter=%s_test' % (str(class_weights), str(patience), str(jitter))
all_datasets_path = '/media/tdanka/B8703F33703EF828/tdanka/data'
train_dataset_loc = os.path.join(all_datasets_path, 'stage1_train_fluo_multilabel/loc.csv')
train_original_loc = '/media/tdanka/B8703F33703EF828/tdanka/data/stage1_train_merged/loc.csv'
test_dataset_loc = os.path.join(all_datasets_path, 'stage1_test/loc.csv')
results_root_path = '/media/tdanka/B8703F33703EF828/tdanka/results'

tf = make_transform(size=(256, 256), p_flip=0.5, color_jitter_params=(jitter, jitter, jitter, jitter), long_mask=True)
train_dataset = JointlyTransformedDataset(train_dataset_loc, transform=tf, remove_alpha=True)
test_dataset = TestFromFolder(test_dataset_loc, transform=T.ToTensor(), remove_alpha=True)
train_original_dataset = TestFromFolder(train_original_loc, transform=T.ToTensor(), remove_alpha=True)

net = UNet(3, 3)
loss = CrossEntropyLoss2d(weight=torch.from_numpy(np.array(class_weights)).float())
optimizer = optim.Adam(net.parameters(), lr=1e-2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=0.1, verbose=True)

model = MultilabelModel(
    model=net, loss=loss, optimizer=optimizer, scheduler=scheduler,
    model_name=model_name, results_root_path=results_root_path
)

n_rounds = 10
for round_idx in range(n_rounds):
    model.train_model(train_dataset, n_epochs=50, n_batch=16, verbose=True)
    model.visualize(train_dataset, n_inst=20, folder_name='compare_round%d' % round_idx)
    model.predict(test_dataset, 'test_round%d' % round_idx)
    model.predict(train_original_dataset, 'train_round%d' % round_idx)