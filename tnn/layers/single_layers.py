import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
import math
from tnn.tensor_utils import modek_product


class LinearLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True, activation=None, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.activation = activation

        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
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

    def __init__(self, in_features, bias=True, gamma=1e-4, activation=None, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AntiSymmetricLayer, self).__init__()
        self.in_features = in_features
        self.gamma = gamma
        self.bias = bias
        self.activation = activation
        self.weight = Parameter(torch.empty((in_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(in_features, **factory_kwargs))
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

    def __init__(self, in_features, width, h=1.0, activation=None, bias=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(HamiltonianLayer, self).__init__()
        self.in_features = in_features
        self.width = width
        self.h = h
        self.activation = activation
        self.bias = bias
        self.weight = Parameter(torch.empty((in_features, width), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(1, **factory_kwargs))
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
            z = torch.zeros(1, device=x.device, dtype=x.dtype)

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


class ModeKLayer(nn.Module):

    def __init__(self, in_features, out_features, k, bias=True, activation=None, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(ModeKLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k                      # axis to which to apply the matrix
        self.bias = bias
        self.activation = activation

        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
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
        x = modek_product(x, self.weight, k=self.k)

        if self.bias:
            # ensure we add to the proper dimension
            x = x + self.bias.unsqueeze((self.k + 1) % 3).unsqueeze((self.k + 2) % 3)

        if self.activation is not None:
            x = self.activation(x)

        return x

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

