import torch.nn as nn
from tnn.layers import tLinearLayer, tAntiSymmetricLayer, tHamiltonianLayer


class tResidualLayer(nn.Module):

    def __init__(self, width, dim3, h=1.0, activation=None, bias=True, M=None):
        super(tResidualLayer, self).__init__()
        self.width = width
        self.M = M
        self.h = h
        self.layer = tLinearLayer(width, width, dim3, bias=bias, activation=activation)

    def forward(self, x, M=None):
        if M is None:
            M = self.M
        x = x + self.h * self.layer(x, M=M)
        return x


class tAntiSymmetricResidualLayer(nn.Module):

    def __init__(self, width, dim3, h=1.0, activation=None, bias=True, gamma=1e-4, M=None):
        super(tAntiSymmetricResidualLayer, self).__init__()
        self.width = width
        self.dim3 = dim3
        self.M = M
        self.h = h
        self.layer = tAntiSymmetricLayer(width, dim3, bias=bias, activation=activation, gamma=gamma)

    def forward(self, x, M=None):
        if M is None:
            M = self.M
        x = x + self.h * self.layer(x, M=M)
        return x


class tHamiltonianResidualLayer(nn.Module):

    def __init__(self, in_features, width, dim3, h=1.0, activation=None, bias=True, M=None):
        super(tHamiltonianResidualLayer, self).__init__()
        self.in_features = in_features
        self.width = width
        self.dim3 = dim3
        self.M = M
        self.h = h
        self.activation = activation
        self.layer = tHamiltonianLayer(in_features, width, dim3, h=self.h, activation=activation, bias=bias)

    def forward(self, x, z=None, M=None):
        if M is None:
            M = self.M
        x, z = self.layer(x, z, M=M)
        return x, z
