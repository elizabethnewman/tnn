import torch.nn as nn
from tnn.layers import tResidualLayer, tAntiSymmetricResidualLayer, tHamiltonianResidualLayer


class tResNet(nn.Module):

    def __init__(self, width, dim3, M, depth, h, activation=None, bias=True):
        super(tResNet, self).__init__()
        self.width = width
        self.M = M
        self.depth = depth
        self.h = h
        self.layers = nn.Sequential(*[tResidualLayer(width, dim3, h=h, activation=activation, bias=bias)
                                      for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x, M=self.M)
        return x


class tAntisymmetricResNet(nn.Module):

    def __init__(self, width, dim3, M, depth, h=1.0, activation=None, bias=True, gamma=1e-4):
        super(tAntisymmetricResNet, self).__init__()
        self.width = width
        self.M = M
        self.depth = depth
        self.h = h
        self.layers = nn.Sequential(*[tAntiSymmetricResidualLayer(width, dim3, h=h, activation=activation,
                                                                  bias=bias, gamma=gamma)
                                      for _ in range(depth)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x, M=self.M)
        return x


class tHamiltonianResNet(nn.Module):

    def __init__(self, in_features, width, dim3, M, depth, h, activation=None, bias=True):
        super(tHamiltonianResNet, self).__init__()
        self.in_features = in_features
        self.width = width
        self.M = M
        self.depth = depth
        self.h = h

        self.layers = nn.Sequential(*[tHamiltonianResidualLayer(in_features, width, dim3,
                                                                h=h, activation=activation, bias=bias)
                                      for _ in range(depth)])

    def forward(self, x, z=None):
        for layer in self.layers:
            x, z = layer(x, z, M=self.M)
        return x