import torch.nn as nn
from tnn.layers.resnet_layers import ResidualLayer, AntiSymmetricResidualLayer, HamiltonianResidualLayer


class ResNet(nn.Module):

    def __init__(self, width, depth, h, activation=None, bias=True):
        super(ResNet, self).__init__()
        self.width = width
        self.depth = depth
        self.h = h
        self.layers = nn.Sequential(*[ResidualLayer(width, h=h, activation=activation, bias=bias)
                                      for _ in range(depth)])

    def forward(self, x):
        x = self.layers(x)
        return x


class AntiSymmetricResNet(nn.Module):

    def __init__(self, width, depth, h, activation=None, bias=True):
        super(AntiSymmetricResNet, self).__init__()
        self.width = width
        self.depth = depth
        self.h = h
        self.layers = nn.Sequential(*[AntiSymmetricResidualLayer(width, h=h, activation=activation, bias=bias)
                                      for _ in range(depth)])

    def forward(self, x):
        x = self.layers(x)
        return x


class HamiltonianResNet(nn.Module):

    def __init__(self, in_features, width, depth, h, activation=None, bias=True):
        super(HamiltonianResNet, self).__init__()
        self.in_features = in_features
        self.width = width
        self.depth = depth
        self.h = h

        self.layers = nn.Sequential(*[HamiltonianResidualLayer(in_features, width,
                                                               h=h, activation=activation, bias=bias)
                                      for _ in range(depth)])

    def forward(self, x, z=None):
        for layer in self.layers:
            x, z = layer(x, z)
        return x
    