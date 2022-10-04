# Matrix-Mimetic Cross Entropy Loss

Suppose $\vec{\mathcal{X}} \in \mathbb{R}^{c\times 1 \times n}$ are the output features of a network for one sample 
where $c$ is the number of classes and $n$ is the number of features in the third dimension.  

Let $\sigma: \mathbb{R}^c \to \mathbb{R}^c$ be the softmax function.  We compute $\sigma(\vec{\mathcal{X}})$ as follows:

$$
\sigma(\vec{\mathcal{X}}) = \hat{\sigma}(\vec{\mathcal{X}} \times_3 \mathbf{M}) \times_3 \mathbf{M}^{-1}
$$

where $\hat{\sigma}$ is the softmax function applied to each frontal slice in the transform domain.  
We can interpret the output $\sigma(\vec{\mathcal{X}})$ as a vector of tubal probabilities;
that is $\sum_{i} \vec{\mathcal{X}}_{i,1,:} = \mathbf{e}$ where $\mathbf{e}$ is the identity tube 
and each tube $\vec{\mathcal{X}}_{i,1,:}$ contains nonnegative entries in the transform domain.

We determine the predicted class as the tube with the largest overall norm; that is,

$$
j^* = \text{argmax}_i \|\vec{\mathcal{X}}_{i,1,:}\|_F^2
$$


Note: the code is different than this description because we compute a log-softmax for numerical stability. 
