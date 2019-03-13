import torch
import torch.optim as optim
import torchvision.transforms as T

# local imports
from utils import *
from wrappers import Model

# architecture imports
from models.unet import UNet

# specifying paths, names, etc
model_name = 'UNet_forLassi'
#train_dataset_path = '/media/tdanka/B8703F33703EF828/lassi/data/NoColorShift_split/train'
#val_dataset_path = '/media/tdanka/B8703F33703EF828/lassi/data/NoColorShift_split/val'
#stage1_test_path = '/media/tdanka/B8703F33703EF828/lassi/data/stage1_test_collected/scaled'
train_dataset_path = '/media/tdanka/B8703F33703EF828/lassi/data/forLassi_split/train'
val_dataset_path = '/media/tdanka/B8703F33703EF828/lassi/data/forLassi_split/val'
stage1_test_path = '/media/tdanka/B8703F33703EF828/lassi/data/stage1_test_collected'
#stage1_test_path = '/media/tdanka/B8703F33703EF828/lassi/data/stage1_test_bboxes'
results_root_path = '/media/tdanka/B8703F33703EF828/lassi/results/new'

# constructing transforms
tf_train = make_augmented_transform(
    size=(256, 256), p_flip=0.5, color_jitter_params=(0.2, 0.2, 0.2, 0.2),
    random_resize=None, normalize=False
)

tf_predict = T.ToTensor()
# constructing datasets
train_dataset = JointlyTransformedDataset(train_dataset_path, transform=tf_train, sigma=0.5)
#train_dataset = BoundingBoxDataset(train_dataset_path, transform=tf_train, padding=12, minlen=32)
val_dataset = JointlyTransformedDataset(val_dataset_path, transform=tf_train, sigma=0.5)
#val_dataset = BoundingBoxDataset(val_dataset_path, transform=tf_train, padding=12, minlen=32)
stage1_test_dataset = TestDataset(stage1_test_path, transform=tf_predict)
#stage1_test_dataset = TestBoundingBoxDataset(stage1_test_path, transform=tf_predict, padding=16, minlen=32)

# defining the network
net = UNet(3, 1)
#net = torch.load('/media/tdanka/B8703F33703EF828/lassi/results/UNet_scaled_images/UNet_scaled_images')#UNet(3, 1)
#net = torch.load('/media/tdanka/B8703F33703EF828/lassi/results/UNet_bboxes_1/UNet_bboxes_1')
#net = torch.load('/media/tdanka/B8703F33703EF828/lassi/results/UNet_GOLD_gauss_3/UNet_GOLD_gauss_3')#UNet(3, 1)
loss = JaccardLoss()
n_epochs = 200
optimizer = optim.Adam(net.parameters(), lr=1e-3)
milestones = [int(p*n_epochs) for p in [0.5, 0.7, 0.9]]
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones)

# wrapping it into the wrappers.Model object for simplicity
model = Model(
    model=net, loss=loss, optimizer=optimizer, scheduler=scheduler,
    model_name=model_name, results_root_path=results_root_path,
)

# training, visualizing results, predicting the test set
model.train_model(train_dataset, n_epochs=n_epochs, n_batch=12, verbose=False, validation_dataset=val_dataset)
model.visualize(val_dataset, folder_name='val_visualization', n_inst=50)
model.predict(stage1_test_dataset, folder_name='stage1_test')
#model.predictbb(stage1_test_dataset, folder_name='stage1_test')
