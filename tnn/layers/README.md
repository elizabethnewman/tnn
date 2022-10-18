# tNN Layers

## Quick Start

Creating a fully-connected layer is simple: 
```python
import torch
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

Creating a tensor layer is also simple, but requires a few more parameters, namely the size of the third dimension and an orthogonal transformation matrix. In accordance with the tensor-tensor product notation, the second dimension of the data is the sample dimension.
```python
import torch
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
layer = lay.tLinearLayer(in_features, out_features, dim3, activation=nn.Tanh(), M=M)

out = layer(x) # or out = layer(x, M) if M is not an attribute of the layer
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

## Special Layers

* `Unfold` and `Fold`: transforms an $n_0 \times n_1\times n_2$ tensor into an $n_0n_2 \times n_1$ matrix by vectorizing the lateral slices. For example,
```python
import torch
x = torch.arange(3 * 4 * 2).reshape(2, 3, 4)
print(x.shape)
print(x)
```
```
torch.Size([2, 3, 4])
tensor([[[ 0,  1,  2,  3],
         [ 4,  5,  6,  7],
         [ 8,  9, 10, 11]],
         
        [[12, 13, 14, 15],
         [16, 17, 18, 19],
         [20, 21, 22, 23]]])
```
Note that ```[0, 1, 2, 3]``` is the $(1,1)$-tube in this case.  The first matrix starting with $0$ and ending with $11$ corresponds to the first horizontal slice.

```python
from tnn.layers import Unfold

lay = Unfold()
y = lay(x)
print(y.shape)
print(y)
```

```
torch.Size([8, 3])
tensor([[ 0,  4,  8],
        [ 1,  5,  9],
        [ 2,  6, 10],
        [ 3,  7, 11],
        [12, 16, 20],
        [13, 17, 21],
        [14, 18, 22],
        [15, 19, 23]])
```


* `ModeKLayer`: a generalization of a `LinearLayer` to apply a matrix along a particular dimension of a tensor.  

```python
import torch
from tnn.layers import LinearLayer, ModeKLayer
from copy import deepcopy

lay1 = LinearLayer(3, 4, bias=False)
lay2 = ModeKLayer(3, 4, k=0, bias=False)
lay2.weight = deepcopy(lay1.weight)

x = torch.randn(3, 7)
y1 = lay1(x)
y2 = lay2(x)
print(torch.norm(y1 - y2))

```


