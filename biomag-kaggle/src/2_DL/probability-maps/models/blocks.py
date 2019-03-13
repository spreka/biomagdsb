import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-100):
        super(CrossEntropyLoss2d, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(self.log_softmax(inputs), targets)


class CrossEntropyLoss2d_v2(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=-100):
        super(CrossEntropyLoss2d, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(self.log_softmax(inputs), targets)