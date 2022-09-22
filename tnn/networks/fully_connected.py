import torch.nn as nn
from tnn.layers.single_layers import LinearLayer


class FullyConnected(nn.Sequential):

    def __init__(self, layer_widths, activation=None, bias=True):
        super(FullyConnected, self).__init__()

        self.depth = len(layer_widths)

        self.layers = nn.Sequential(*[LinearLayer(layer_widths[i], layer_widths[i + 1],
                                                  activation=activation, bias=bias)
                                      for i in range(self.depth - 1)])

    def forward(self, x):
        x = self.layers(x)
        return x
