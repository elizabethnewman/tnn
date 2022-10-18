import torch.nn as nn
from tnn.layers import LinearLayer


class FullyConnected(nn.Sequential):

    def __init__(self, layer_widths, activation=None, bias=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(FullyConnected, self).__init__()

        self.depth = len(layer_widths)

        self.layers = nn.Sequential(*[LinearLayer(layer_widths[i], layer_widths[i + 1],
                                                  activation=activation, bias=bias, **factory_kwargs)
                                      for i in range(self.depth - 1)])

    def forward(self, x):
        x = self.layers(x)
        return x
