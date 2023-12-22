
import torch
from tnn.loss import t_softmax
import matplotlib.pyplot as plt
import imageio


tmp = imageio.v3.imread('elephant.jpeg')

img = torch.tensor(tmp).float() / 255.0

M = 1.0 * torch.tensor([[-1, 1, 1], [0, 1, 1], [0, 0, 1]])
M_inv = 1.0 * torch.tensor([[-1, 1, 0], [0, 1, -1], [0, 0, 1]])

f = [lambda x: torch.abs(x), lambda x: torch.relu(x), lambda x: torch.sign(x), lambda x: torch.tanh(x)]

g = lambda f, x: f(x @ M.T) @ M_inv.T


def rescale(x):
    x = x - x.min()
    x = x / x.max()
    return x


plt.imshow(img)
plt.axis('off')
plt.savefig('elephant_img/orig.png', bbox_inches='tight')

# show channels
for i, c in enumerate(['red', 'green', 'blue']):
    z = torch.zeros_like(img)
    z[:, :, i] = img[:, :, i]
    plt.imshow(z)
    plt.axis('off')
    plt.savefig('elephant_img/' + c + '.png', bbox_inches='tight')


# plt.imshow(f[0](img))
for i in range(len(f)):
    plt.imshow(rescale(f[i](img)))
    plt.savefig(f'elephant_img/pointwise_{i}.png', bbox_inches='tight')

    plt.imshow(rescale(g(f[i], img)))
    plt.savefig(f'elephant_img/tubal_{i}.png', bbox_inches='tight')

