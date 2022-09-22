import torch.nn as nn
from tnn.layers.t_single_layers import tLinearLayer


class tFullyConnected(nn.Module):

    def __init__(self, layer_widths, dim3, M, activation=None, bias=True):
        super(tFullyConnected, self).__init__()

        self.M = M
        self.depth = len(layer_widths)

        self.layers = nn.Sequential(*[tLinearLayer(layer_widths[i], layer_widths[i + 1], dim3,
                                                   activation=activation, bias=bias)
                                      for i in range(self.depth - 1)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x, self.M)
        return x
