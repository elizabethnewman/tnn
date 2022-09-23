# tNN Networks

## Quick Start

You can create a network from a sequence of layers:
```python
import torch
import torch.nn as nn
import tnn.layers as lay

# setup data
in_features = 3
out_features = 4
n_samples = 10
x = torch.randn(n_samples, in_features)

# create a layer
widths = [10, 15, 7]
net = nn.Sequential(
         lay.LinearLayer(in_features, widths[0], activation=nn.Tanh()), 
         lay.ResidualLayer(widths[0], h=0.1, activation=nn.Tanh()), 
         lay.LinearLayer(widths[0], widths[1], activation=nn.Sigmoid()), 
         lay.LinearLayer(widths[1], widths[2], activation=nn.ReLU()),
         lay.LinearLayer(widths[2], out_features, activation=nn.LeakyReLU())
         )

out = net(x)
```

## Building Blocks

* `FullyConnected` or `tFullyConnected` is a sequence of `LinearLayer` or `tLinearLayer`, respectively.
* `ResNet` is a sequence of `ResidualLayer`
* and so on
