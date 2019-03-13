import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class First(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, dropout=False):
        super(First, self).__init__()

        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.first = nn.Sequential(*layers)

    def forward(self, x):
        return self.first(x)


class Encoder(nn.Module):
    def __init__(
            self, in_channels, middle_channels, out_channels,
            dropout=False, downsample_kernel=2
    ):
        super(Encoder, self).__init__()

        layers = [
            nn.MaxPool2d(kernel_size=downsample_kernel),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)


class Center(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Center, self).__init__()

        layers = [
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.center = nn.Sequential(*layers)

    def forward(self, x):
        return self.center(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, deconv_channels, dropout=False):
        super(Decoder, self).__init__()

        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, deconv_channels, kernel_size=2, stride=2)
        ]

        if dropout:
            assert 0 <= dropout <= 1, 'dropout must be between 0 and 1'
            layers.append(nn.Dropout2d(p=dropout))

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(x)


class Last(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, softmax=False):
        super(Last, self).__init__()

        layers = [
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        ]

        if softmax:
            layers.append(nn.Softmax2d())

        self.first = nn.Sequential(*layers)

    def forward(self, x):
        return self.first(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, softmax=False):
        super(UNet, self).__init__()
        self.first = First(in_channels, 64, 64)
        self.encoder_1 = Encoder(64, 128, 128)
        self.encoder_2 = Encoder(128, 256, 256)
        self.encoder_3 = Encoder(256, 512, 512)
        self.center = Center(512, 1024, 1024, 512)
        self.decoder_3 = Decoder(1024, 512, 512, 256)
        self.decoder_2 = Decoder(512, 256, 256, 128)
        self.decoder_1 = Decoder(256, 128, 128, 64)
        self.last = Last(128, 64, out_channels, softmax=softmax)

    def forward(self, x):
        x_first = self.first(x)
        x_enc_1 = self.encoder_1(x_first)
        x_enc_2 = self.encoder_2(x_enc_1)
        x_enc_3 = self.encoder_3(x_enc_2)
        x_cent = self.center(x_enc_3)
        x_dec_3 = self.decoder_3(torch.cat([pad_to_shape(x_cent, x_enc_3.shape), x_enc_3], dim=1))
        x_dec_2 = self.decoder_2(torch.cat([pad_to_shape(x_dec_3, x_enc_2.shape), x_enc_2], dim=1))
        x_dec_1 = self.decoder_1(torch.cat([pad_to_shape(x_dec_2, x_enc_1.shape), x_enc_1], dim=1))
        return self.last(torch.cat([pad_to_shape(x_dec_1, x_first.shape), x_first], dim=1))


def pad_to_shape(this, shp):
    """
    Not a very safe function.
    """
    return F.pad(this, (0, shp[3] - this.shape[3], 0, shp[2] - this.shape[2]))


if __name__ == '__main__':
    print("----- component test -----")
    first = First(4, 6, 8)
    encoder = Encoder(8, 10, 12)
    center = Center(12, 14, 16, 18)
    decoder = Decoder(18, 16, 14, 12)
    last = Last(12, 10, 8)

    x = Variable(torch.ones(1, 4, 256, 256))
    print(x.shape)

    for layer in [first, encoder, center, decoder, last]:
        x = layer(x)
        print(layer.__class__.__name__, x.shape)

    print("----- unet test -----")
    unet = UNet(4, 2)
    x = Variable(torch.ones(1, 4, 256, 256))
    print(x.shape)
    x = unet(x)
    print(x.shape)
