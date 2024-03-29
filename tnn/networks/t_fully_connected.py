import torch.nn as nn
from tnn.layers import tLinearLayer


class tFullyConnected(nn.Module):

    def __init__(self, layer_widths, dim3, M, activation=None, bias=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(tFullyConnected, self).__init__()

        self.M = M
        self.depth = len(layer_widths)

        self.layers = nn.Sequential(*[tLinearLayer(layer_widths[i], layer_widths[i + 1], dim3,
                                                   activation=activation, bias=bias, **factory_kwargs)
                                      for i in range(self.depth - 1)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x, self.M)
        return x
