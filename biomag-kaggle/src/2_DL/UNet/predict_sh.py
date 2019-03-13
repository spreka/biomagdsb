import os
import torch
import argparse
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# local imports
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--output', type=str, required=True)
args = parser.parse_args()

dataset = DataLoader(TestDataset(args.dataset, transform=ToTensor()), batch_size=1)

use_gpu = torch.cuda.is_available()
model = torch.load(args.model)
model.train(False)

if use_gpu:
    model.cuda()
else:
    model.cpu()

for image, name in dataset:
    pmap_path = os.path.join(args.output, name[0], 'masks')
    chk_mkdir(pmap_path)

    if use_gpu:
        X_in = Variable(image.cuda())

    else:
        X_in = Variable(image)

    y_out = model(X_in)

    if use_gpu:
        y_out = y_out.data.cpu().numpy()[0, 0, :, :]
    else:
        y_out = y_out.data.numpy()[0, 0, :, :]

    io.imsave(os.path.join(pmap_path, name[0] + '.png'), y_out)