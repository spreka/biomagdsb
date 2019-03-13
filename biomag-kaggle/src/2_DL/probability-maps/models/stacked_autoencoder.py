import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Binarize(nn.Module):
    def __init__(self, lower, upper):
        super(Binarize, self).__init__()
        self.lower = lower
        self.upper = upper
        self.threshold = Parameter(torch.Tensor(1))
        self.reset_parameters()

    def forward(self, input):
        # TODO: implement the corresponding functional
        pass

    def reset_parameters(self):
        self.threshold.data.uniform_(self.lower, self.upper)

    def __repr__(self):
        return self.__class__.__name__ + '(upper=%f, lower=%f, threshold=%f)' % (self.upper, self.lower, self.threshold)


class StackedAutoencoder(nn.Module):
    def __init__(self, encoder, decoder):
        """

        Parameters
        ----------
        encoder: list of encoding layers and activating functions

        decoder: list of decoding layers and activating functions
        """
        super(StackedAutoencoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        for encoding_layer in self.encoder:
            x = encoding_layer(x)

        return x

    def decode(self, x):
        for decoding_layer in self.decoder:
            x = decoding_layer(x)

        return x

    def forward(self, x):
        return self.decode(self.encode(x))

    def train_batch(self, X, y, optimizer, loss):
        optimizer.zero_grad()
        out = self.forward(X)
        training_loss = loss(out, y)
        training_loss.backward()
        optimizer.step()

        return training_loss.data[0]

    def test(self, X, y, loss):
        pass

    def stack(self, stacked_autoencoder):
        assert isinstance(stacked_autoencoder, StackedAutoencoder), 'argument must be a StackedEncoder'
        self.encoder = nn.ModuleList([*self.encoder, *stacked_autoencoder.encoder])
        self.decoder = nn.ModuleList([*stacked_autoencoder.decoder, *self.decoder])

ae = StackedAutoencoder(
    encoder=nn.ModuleList(
        [nn.Conv2d(in_channels=4, out_channels=48, kernel_size=(3, 3), padding=1),
         nn.ReLU(),
         nn.MaxPool2d(kernel_size=(2, 2)),
         nn.ReLU(),
         nn.Conv2d(in_channels=48, out_channels=128, kernel_size=(3, 3), padding=1),
         nn.ReLU(),
         nn.MaxPool2d(kernel_size=(2, 2))]
    ),
    decoder=nn.ModuleList(
        [nn.Conv2d(in_channels=128, out_channels=48, kernel_size=(3, 3), padding=1),
         nn.ReLU(),
         nn.Upsample(scale_factor=2),
         nn.Conv2d(in_channels=48, out_channels=16, kernel_size=(3, 3), padding=1),
         nn.ReLU(),
         nn.Upsample(scale_factor=2),
         nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(3, 3), padding=1),
         nn.Sigmoid()]
    )
)

ae_dropout = StackedAutoencoder(
    encoder=nn.ModuleList(
        [nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(3, 3), padding=1), nn.ReLU(),
         nn.MaxPool2d(kernel_size=(2, 2)),
         nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1), nn.ReLU(),
         nn.MaxPool2d(kernel_size=(2, 2)),
         nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1), nn.ReLU(),
         nn.Dropout2d(p=0.2),
         nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1), nn.ReLU(),
         nn.MaxPool2d(kernel_size=(2, 2))
         ]
    ),
    decoder=nn.ModuleList(
        [nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(3, 3), padding=1), nn.ReLU(),
         nn.Dropout2d(p=0.2),
         nn.Upsample(scale_factor=2),
         nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1), nn.ReLU(),
         nn.Upsample(scale_factor=2),
         nn.Conv2d(in_channels=128, out_channels=16, kernel_size=(3, 3), padding=1), nn.ReLU(),
         nn.Upsample(scale_factor=2),
         nn.Conv2d(in_channels=16, out_channels=1, kernel_size=(3, 3), padding=1), nn.Sigmoid()]
    )
)


if __name__ == '__main__':
    pass