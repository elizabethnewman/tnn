import torch
import torch.nn as nn
from networks.single_layers import LinearLayer, AntiSymmetricLayer, HamiltonianLayer


class ResidualLayer(nn.Module):

    def __init__(self, width, h=1.0, activation=None, bias=True):
        super(ResidualLayer, self).__init__()
        self.width = width
        self.h = h
        self.layer = LinearLayer(width, width, bias=bias, activation=activation)

    def forward(self, x):
        x = x + self.h * self.layer(x)
        return x


class AntiSymmetricResidualLayer(nn.Module):

    def __init__(self, width, h, activation=None, bias=True, gamma=1e-4):
        super(AntiSymmetricResidualLayer, self).__init__()
        self.width = width
        self.h = h
        self.layer = AntiSymmetricLayer(width, bias=bias, activation=activation, gamma=gamma)

    def forward(self, x):
        x = x + self.h * self.layer(x)
        return x


class HamiltonianResidualLayer(nn.Module):

    def __init__(self, in_features, width, h=1.0, activation=None, bias=True):
        super(HamiltonianResidualLayer, self).__init__()
        self.in_features = in_features
        self.width = width
        self.h = h
        self.layer = HamiltonianLayer(in_features, width, h=self.h, activation=activation, bias=bias)

    def forward(self, x, z=None):
        x, z = self.layer(x, z)
        return x, z

