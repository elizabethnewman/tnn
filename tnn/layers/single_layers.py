import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
import math


class LinearLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True, activation=None):
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.activation = activation

        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = F.linear(x, self.weight, self.bias)

        if self.activation is not None:
            x = self.activation(x)

        return x

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class AntiSymmetricLayer(nn.Module):

    def __init__(self, in_features, bias=True, gamma=1e-4, activation=None):
        super(AntiSymmetricLayer, self).__init__()
        self.in_features = in_features
        self.gamma = gamma
        self.bias = bias
        self.activation = activation
        self.weight = Parameter(torch.Tensor(in_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(in_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = F.linear(x, self.weight - self.weight.t()
                     - self.gamma * torch.eye(self.in_features, dtype=x.dtype, device=x.device), self.bias)

        if self.activation is not None:
            x = self.activation(x)

        return x

    def extra_repr(self) -> str:
        return 'in_features={}, bias={}'.format(
            self.in_features, self.bias is not None
        )


class HamiltonianLayer(nn.Module):

    def __init__(self, in_features, width, h=1.0, activation=None, bias=True):
        super(HamiltonianLayer, self).__init__()
        self.in_features = in_features
        self.width = width
        self.h = h
        self.activation = activation
        self.bias = bias
        self.weight = Parameter(torch.Tensor(in_features, width))
        if bias:
            self.bias = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, z):

        # update z
        if z is None:
            z = torch.zeros(1)

        dz = F.linear(x, self.weight.t(), self.bias)
        if self.activation is not None:
            dz = self.activation(dz)
        z = z - self.h * dz

        # update x
        dx = F.linear(z, self.weight, self.bias)
        if self.activation is not None:
            dx = self.activation(dx)
        x = x + self.h * dx

        return x, z

    def extra_repr(self) -> str:
        return 'in_features={}, width={}, bias={}'.format(
            self.in_features, self.width, self.bias is not None
        )
