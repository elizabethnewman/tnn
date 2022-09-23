# tNN Layers

## Quick Start

Creating a fully-connected layer is simple: 
```python
import tnn.nn as nn
import tnn.layers as lay

# setup data
in_features = 3
out_features = 4
n_samples = 10
x = torch.randn(n_samples, in_features)

# create a layer
layer = lay.LinearLayer(in_features, out_features, activation=nn.Tanh())

out = layer(x)
```

Creating a tensor layer is also simple, but requires a few more parameters, namely the size of the third dimension and an orthogonal transformation matrix
```python
import tnn.nn as nn
import tnn.layers as lay
import tnn.tensor_utils as t_utils

# setup data
in_features = 3
out_features = 4
dim3 = 5
n_samples = 10
x = torch.randn(in_features, n_samples, dim3) # note: the second dimension is the sample dimension

# create network from la
layer = lay.tLinearLayer(in_features, out_features, activation=nn.Tanh())

out = layer(x)
```
In accordance with the tensor-tensor product notation, the second dimension of the data is the sample dimension.


## Building Blocks

### Single Layers

* `LinearLayer` of the form $\qquad \sigma(Ky + b)$
* `AntiSymmetricLayer` of the form $\sigma((K - K^\top - \gamma I) y + b)$
* `HamiltonianLayer` of the form $z \gets z - h\sigma(K^\top y + b)$ and $y \gets  y + h \sigma(K z + b)$

### Residual Layers
All residual layers are of the form $y_{j+1} \gets y + h \text{layer}(y)$ (with the exception of Hamiltonian Residual layers)

### t-Single Layers
Similar to above, but we replace $Ky$ with $\mathcal{K} * \vec{\mathcal{Y}}$ where $*$ is a tensor-tensor product.

