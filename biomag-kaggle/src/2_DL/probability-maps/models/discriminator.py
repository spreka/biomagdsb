# HACK
# NEVER DO THIS
import sys
sys.path.append('./..')

import torch
import torch.nn as nn
from torch.autograd import Variable

from models.unet import First, Encoder


class Classifier(nn.Module):
    def __init__(self, in_nodes, middle_nodes, n_classes):
        super(Classifier, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_nodes, middle_nodes), nn.ELU(),
            nn.Linear(middle_nodes, n_classes), nn.Sigmoid()
        )

    def forward(self, x):
        return self.classifier(x.view(x.size()[0], -1))


# TODO: simplify Discriminator architecture
class Discriminator(nn.Module):
    def __init__(self, in_channels, out_classes):
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            First(in_channels, 64, 64),
            Encoder(64, 128, 128, downsample_kernel=4),
            Encoder(128, 256, 256, downsample_kernel=4),
            Encoder(256, 512, 512, downsample_kernel=4),
            Encoder(512, 1024, 1024, downsample_kernel=2),
            Classifier(1024, 1024, out_classes),
        )

    def forward(self, x):
        return self.discriminator(x)


class SimpleDiscriminator(nn.Module):
    def __init__(self, in_channels, out_classes):
        super(SimpleDiscriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, padding=2), nn.BatchNorm2d(16), nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(16, 32, kernel_size=5, padding=2), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(32, 128, kernel_size=5, padding=2), nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=4),
        )
        self.linear = nn.Sequential(
            nn.Linear(2048, 1024), nn.Dropout(p=0.9), nn.ELU(),
            nn.Linear(1024, out_classes), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        return self.linear(x.view(x.size()[0], -1))

if __name__ == '__main__':
    classifier = Classifier(128, 32, 2)
    x = Variable(torch.Tensor(5, 128, 1, 1))
    # print(classifier(x).shape)

    discriminator = Discriminator(2, 2)
    x = Variable(torch.Tensor(10, 2, 256, 256))
    #print(discriminator(x).shape)

    simple_discriminator = SimpleDiscriminator(5, 2)
    x = Variable(torch.Tensor(10, 5, 256, 256))
    print(simple_discriminator(x).shape)
