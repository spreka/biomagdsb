import argparse
import torch
import torch.optim as optim
import torchvision.transforms as T

# local imports
from utils import *
from wrappers import Model

# architecture imports
from models.unet import UNet

parser = argparse.ArgumentParser()
parser.add_argument('--results_folder', type=str, required=True)
parser.add_argument('--train', type=str, required=True)
parser.add_argument('--val', type=str, default=None)
parser.add_argument('--test', type=str, default=None)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--model_path', type=str, default=None)
parser.add_argument('--model_name', type=str, default='UNet_model')
parser.add_argument('--verbose', type=bool, default=False)
parser.add_argument('--batch', type=int, default=1)
parser.add_argument('--sigma', type=float, default=0.0)
args = parser.parse_args()


# constructing transforms
tf_train = make_augmented_transform(
    size=(256, 256), p_flip=0.5, color_jitter_params=(0.2, 0.2, 0.2, 0.2),
    random_resize=None, normalize=False
)
tf_predict = T.ToTensor()

print(args.results_folder)
print(args.train)
print(args.val)
print(args.test)

if args.model_path is not None:
    net = torch.load(args.model_path)
else:
    net = UNet(3, 1)

loss = JaccardLoss()
lr_milestones = [int(p*args.epochs) for p in [0.5, 0.7, 0.9]]
optimizer = optim.Adam(net.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, lr_milestones)

train_dataset = JointlyTransformedDataset(args.train, transform=tf_train, sigma=args.sigma)
val_dataset = JointlyTransformedDataset(args.val, transform=tf_train, sigma=args.sigma)
stage1_test_dataset = TestDataset(args.test, transform=tf_predict)

model = Model(
    model=net, loss=loss, optimizer=optimizer, scheduler=scheduler,
    model_name=args.model_name, results_root_path=args.results_folder,
)
if args.model_path is None:
	model.train_model(train_dataset, n_epochs=args.epochs, n_batch=args.batch, verbose=args.verbose, validation_dataset=val_dataset)
model.predict(stage1_test_dataset, folder_name='images')
