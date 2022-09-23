# tNN Layers

## Single Layers

* `LinearLayer` of the form $\qquad \sigma(Ky + b)$
* `AntiSymmetricLayer` of the form $\sigma((K - K^\top - \gamma I) y + b)$
* `HamiltonianLayer` of the form $z \gets z - h\sigma(K^\top y + b)$ and $y \gets  y + h \sigma(K z + b)$

## Residual Layers
All residual layers are of the form $y_{j+1} \gets y + h \text{layer}(y)$ (with the exception of Hamiltonian Residual layers)

## t-Single Layers
Similar to above, but we replace $Ky$ with $\mathcal{K} * \vec{\mathcal{Y}}$ where $*$ is a tensor-tensor product.

