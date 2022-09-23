# tNN Layers

## Quick Start

```python
import tnn.nn as nn
import tnn.layers as lay

# setup data
in_features = 3
out_features = 4
n_samples = 10
x = torch.randn(n_samples, in_features)

# create network from la
layer = lay.LinearLayer(in_features, out_features, activation=nn.Tanh())

out = layer(x)
```


## Building Blocks

### Single Layers

* `LinearLayer` of the form $\qquad \sigma(Ky + b)$
* `AntiSymmetricLayer` of the form $\sigma((K - K^\top - \gamma I) y + b)$
* `HamiltonianLayer` of the form $z \gets z - h\sigma(K^\top y + b)$ and $y \gets  y + h \sigma(K z + b)$

### Residual Layers
All residual layers are of the form $y_{j+1} \gets y + h \text{layer}(y)$ (with the exception of Hamiltonian Residual layers)

### t-Single Layers
Similar to above, but we replace $Ky$ with $\mathcal{K} * \vec{\mathcal{Y}}$ where $*$ is a tensor-tensor product.

