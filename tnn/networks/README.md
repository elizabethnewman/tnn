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

We can similarly create a fully-connected tNN as 
```python
import torch
import torch.nn as nn
import tnn.layers as lay
import tnn.tensor_utils as t_utils

# setup data
in_features = 3
out_features = 4
dim3 = 5
n_samples = 10
x = torch.randn(in_features, n_samples, dim3) # note: the second dimension is the sample dimension

# create orthogonal transformation matrix
M = t_utils.random_orthogonal(dim3)

# create a tlayer
widths = [10, 15, 7]
net = nn.Sequential(
         lay.tLinearLayer(in_features, widths[0], dim3, M=M, activation=nn.Tanh()), 
         lay.tResidualLayer(widths[0], dim3, M=M, h=0.1, activation=nn.Tanh()), 
         lay.tLinearLayer(widths[0], widths[1], dim3, M=M, activation=nn.Sigmoid()), 
         lay.tLinearLayer(widths[1], widths[2], dim3, M=M, activation=nn.ReLU()),
         lay.tLinearLayer(widths[2], out_features, dim3, M=M, activation=nn.LeakyReLU())
         )

out = net(x)
```


We also include some built-in networks, which can also be concatenated.

## Building Blocks

* `FullyConnected` or `tFullyConnected` is a sequence of `LinearLayer` or `tLinearLayer`, respectively.
* `ResNet` is a sequence of `ResidualLayer`
* and so on
