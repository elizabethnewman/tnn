import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.parameter import Parameter
import math

from tnn.tensor_utils import mprod, mtran, t_eye


class tLinearLayer(nn.Module):
    # only supports third-order tensors currently
    def __init__(self, in_features, out_features, dim3, bias=True, activation=None, M=None):
        super(tLinearLayer, self).__init__()

        self.in_features = in_features
        self.dim3 = dim3
        self.M = M
        self.out_features = out_features
        self.bias = bias
        self.activation = activation

        self.weight = Parameter(torch.Tensor(out_features, in_features, dim3))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features, 1, dim3))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, M=None):
        if M is None:
            M = self.M
        x = mprod(self.weight, x, M)
        if self.bias is not None:
            x += self.bias

        if self.activation is not None:
            x = self.activation(x)

        return x

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, dim3={},, bias={}, activation={}'.format(
            self.in_features, self.out_features,  self.dim3, self.bias is not None, self.activation
        )


class tAntiSymmetricLayer(nn.Module):
    # only supports third-order tensors currently
    def __init__(self, in_features, dim3, bias=True, gamma=1e-4, activation=None, M=None):
        super(tAntiSymmetricLayer, self).__init__()

        self.in_features = in_features
        self.dim3 = dim3
        self.M = M
        self.bias = bias
        self.gamma = gamma
        self.activation = activation

        self.weight = Parameter(torch.Tensor(in_features, in_features, dim3))
        if bias:
            self.bias = Parameter(torch.Tensor(in_features, 1, dim3))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, M=None):
        if M is None:
            M = self.M
        x = mprod(self.weight + mtran(self.weight) - self.gamma * t_eye(self.in_features, M), x, M)
        if self.bias is not None:
            x += self.bias

        if self.activation is not None:
            x = self.activation(x)

        return x

    def extra_repr(self) -> str:
        return 'in_features={},  bias={}, activation={}'.format(
            self.in_features, self.bias is not None, self.activation
        )


class tHamiltonianLayer(nn.Module):

    def __init__(self, in_features, width, dim3, bias=True, h=1.0, activation=None, M=None):
        super(tHamiltonianLayer, self).__init__()
        self.in_features = in_features
        self.width = width
        self.M = M
        self.dim3 = dim3
        self.h = h
        self.activation = activation
        self.bias = bias

        self.weight = Parameter(torch.Tensor(in_features, width, dim3))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, dim3))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x, z=None, M=None):
        if M is None:
            M = self.M

        # update z
        if z is None:
            z = torch.zeros(1, dtype=x.dtype, device=x.device)

        dz = mprod(mtran(self.weight), x, M)
        if self.bias is not None:
            dz += self.bias

        if self.activation is not None:
            dz = self.activation(dz)

        z = z - self.h * dz

        # update x
        dx = mprod(self.weight, z, M)
        if self.bias is not None:
            dx += self.bias

        if self.activation is not None:
            dx = self.activation(dx)

        x = x + self.h * dx

        return x, z

    def extra_repr(self) -> str:
        return 'in_features={}, width={}, dim3={}, bias={}, activation={}'.format(
            self.in_features, self.width, self.dim3, self.bias is not None, self.activation
        )
