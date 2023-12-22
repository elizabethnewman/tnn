
import os
import torch
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl

load_dir = '/Users/elizabethnewman/Desktop/tnn_results/cifar10/tensor/'

subdir_names = []
for subdir_name in os.listdir(load_dir):
    if subdir_name == '.DS_Store':
        continue
    subdir_names.append(subdir_name)

subdir_names.sort()

results_tensor = []
weights_tensor = []
for subdir_name in subdir_names:
    print(subdir_name)
    results_tensor.append(pickle.load(open(load_dir + subdir_name + '/results.pkl', "rb")))
    weights_tensor.append(torch.load(load_dir + subdir_name + '/best_val_acc_net.pt', map_location=torch.device('cpu')))


load_dir = '/Users/elizabethnewman/Desktop/tnn_results/cifar10/matrix/'

subdir_names = []
for subdir_name in os.listdir(load_dir):
    if subdir_name == '.DS_Store':
        continue
    subdir_names.append(subdir_name)

subdir_names.sort()

results_matrix = []
weights_matrix = []
for subdir_name in subdir_names:
    print(subdir_name)
    results_matrix.append(pickle.load(open(load_dir + subdir_name + '/results.pkl', "rb")))
    weights_matrix.append(torch.load(load_dir + subdir_name + '/best_val_acc_net.pt', map_location=torch.device('cpu')))


#%%
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['lines.linewidth'] = 4
mpl.rcParams['font.size'] = 10

exp_idx = [0, 1]

for i in exp_idx:
    idx = results_tensor[i]['str'].index('loss') + 2
    p = plt.semilogy(results_tensor[i]['val'][:, idx] / results_tensor[i]['val'][0, idx], label='tensor')
    p = plt.semilogy(results_matrix[i]['val'][:, idx] / results_matrix[i]['val'][0, idx], '--', label='matrix', color=p[0].get_color())
    plt.legend()

# plt.ylim([0, 100])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['lines.linewidth'] = 4
mpl.rcParams['font.size'] = 10


for i in exp_idx:
    idx = results_tensor[i]['str'].index('acc') + 4
    p = plt.plot(results_tensor[i + 2]['val'][:, idx], label='tensor')
    p = plt.plot(results_matrix[i]['val'][:, idx], '--', label='matrix', color=p[0].get_color())
    plt.legend()

# plt.ylim([0, 100])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

#%% Look at features
from tnn.layers import Permute, View, LinearLayer, tLinearLayer
from tnn.networks import tFullyConnected, tHamiltonianResNet, HamiltonianResNet
from tnn.loss import tCrossEntropyLoss
from tnn.utils import seed_everything, number_network_weights, get_logger, makedirs, setup_parser, matrix_match_tensor_single_layer
import os
from tnn.tensor_utils import dct_matrix, random_orthogonal
from tnn.training import test

os.chdir('/Users/elizabethnewman/Library/CloudStorage/OneDrive-EmoryUniversity/[000] Code/tnn/examples')

print(os.getcwd())
from cifar10.setup_cifar10 import setup_cifar10

net_idx = 0


# -------------------------------------------------------------------------------------------------------------------- #
# MATRIX
args = results_matrix[net_idx]['args']

train_loader, val_loader, test_loader = setup_cifar10(args.n_train, args.n_val, args.n_test, args.batch_size,
                                                      args.data_dir)

# form matrix network
if args.opening_layer:
    net_matrix = torch.nn.Sequential(View((-1, 32 * 32 * 3)),
                                     LinearLayer(32 * 32 * 3, args.width, activation=torch.nn.Tanh()),
                                     HamiltonianResNet(args.width, width=args.width + args.add_width_hamiltonian,
                                                       depth=args.depth, h=args.h_step, activation=torch.nn.Tanh()),
                                     LinearLayer(args.width, 10, activation=None)
                                     )
else:
    w = 32 * 32 * 3
    net_matrix = torch.nn.Sequential(View((-1, 32 * 32 * 3)),
                                     HamiltonianResNet(w, width=w + args.add_width_hamiltonian,
                                                       depth=args.depth, h=args.h_step, activation=torch.nn.Tanh()),
                                     LinearLayer(w, 10, activation=None)
                                     )

loss_matrix = torch.nn.CrossEntropyLoss()
net_matrix.load_state_dict(weights_matrix[net_idx])

test_out = test(net_matrix, loss_matrix, test_loader)
print('Test performance: MATRIX')
print('loss = {:<15.4e}'.format(test_out[0]))
print('accuracy = {:<15.4f}'.format(test_out[1]))

# -------------------------------------------------------------------------------------------------------------------- #
# TENSOR
args = results_tensor[net_idx]['args']

# create transformation matrix
dim3 = 32
if args.M == 'dct':
    M = dct_matrix(dim3, dtype=torch.float32)
elif args.M == 'random':
    M = random_orthogonal(dim3, dtype=torch.float32)
else:
    M = torch.eye(dim3, dtype=torch.float32)

# form network and choose loss function
if args.opening_layer:
    if args.loss == 't_cross_entropy':
        net_tensor = torch.nn.Sequential(View((-1, 32 * 3, 32)),
                                         Permute((1, 0, 2)),
                                         tLinearLayer(32 * 3, args.width, dim3, M=M, activation=torch.nn.Tanh()),
                                         tHamiltonianResNet(args.width, args.width + args.add_width_hamiltonian, dim3, M,
                                                            depth=args.depth, h=args.h_step, activation=torch.nn.Tanh()),
                                         tLinearLayer(args.width, 10, dim3, M=M, activation=None)
                                         )
        loss_tensor = tCrossEntropyLoss(M=M)
    else:
        net_tensor = torch.nn.Sequential(View((-1, 32 * 3, 32)),
                                         Permute((1, 0, 2)),
                                         tLinearLayer(32 * 3, args.width, dim3, M=M, activation=torch.nn.Tanh()),
                                         tHamiltonianResNet(args.width, args.width + args.add_width_hamiltonian, dim3, M,
                                                            depth=args.depth, h=args.h_step, activation=torch.nn.Tanh()),
                                         Permute((1, 0, 2)),
                                         View((-1, args.width * dim3)),
                                         LinearLayer(args.width * dim3, 10, activation=None, bias=args.bias)
                                         )

        loss_tensor = torch.nn.CrossEntropyLoss()

else:
    w = 32 * 3

    if args.loss == 't_cross_entropy':
        net_tensor = torch.nn.Sequential(View((-1, w, 32)),
                                         Permute((1, 0, 2)),
                                         tHamiltonianResNet(w, w + args.add_width_hamiltonian, dim3, M,
                                                            depth=args.depth, h=args.h_step, activation=torch.nn.Tanh()),
                                         tLinearLayer(w, 10, dim3, M=M, activation=None)
                                         )
        loss_tensor = tCrossEntropyLoss(M=M)
    else:
        net_tensor = torch.nn.Sequential(View((-1, w, 32)),
                                         Permute((1, 0, 2)),
                                         tHamiltonianResNet(w, w + args.add_width_hamiltonian, dim3, M,
                                                            depth=args.depth, h=args.h_step, activation=torch.nn.Tanh()),
                                         Permute((1, 0, 2)),
                                         View((-1, w * dim3)),
                                         LinearLayer(w * dim3, 10, activation=None, bias=args.bias)
                                         )
        loss_tensor = torch.nn.CrossEntropyLoss()

net_tensor.load_state_dict(weights_tensor[net_idx])

test_out = test(net_tensor, loss_tensor, test_loader)
print('Test performance: TENSOR')
print('loss = {:<15.4e}'.format(test_out[0]))
print('accuracy = {:<15.4f}'.format(test_out[1]))

#%% FORWARD PROPAGATE
from copy import deepcopy

net_tensor.eval()

with torch.no_grad():
    x, y = next(iter(train_loader))

    x_list = [deepcopy(x)]

    x = net_tensor[1](net_tensor[0](x))

    z = None
    for layer in net_tensor[2].layers:
        x, z = layer(x, z, M)
        x_list.append(deepcopy(x.permute(1, 0, 2).reshape(32, 3, 32, 32)))


cmaps = [mpl.colormaps['Reds'], mpl.colormaps['Greens'], mpl.colormaps['Blues']]
for i in [2, 8, 19, 29]:
    # plt.subplot(4, 8, i + 1)
    for j in range(len(x_list)):
        for k in range(3):
            plt.clf()
            plt.imshow(x_list[j][i, :, :, :].permute(1, 2, 0)[:, :, k], cmap=cmaps[k])
            plt.axis('off')
            plt.savefig('tmp/tensor_{}_channel_{}_layer_{}.png'.format(i, k, j), bbox_inches='tight', pad_inches=0)


# plt.show()

#%% FORWARD PROPAGATE

from copy import deepcopy

net_matrix.eval()

with torch.no_grad():
    # x, y = next(iter(train_loader))
    x = deepcopy(x_list[0])
    y = deepcopy(y)

    x_list = [deepcopy(x)]

    x = net_matrix[0](x)

    z = None
    for layer in net_matrix[1].layers:
        x, z = layer(x, z)
        x_list.append(deepcopy(x.reshape(32, 3, 32, 32)))


cmaps = [mpl.colormaps['Reds'], mpl.colormaps['Greens'], mpl.colormaps['Blues']]
for i in [2, 8, 19, 29]:
    # plt.subplot(4, 8, i + 1)
    for j in range(len(x_list)):
        for k in range(3):
            plt.clf()
            plt.imshow(x_list[j][i, :, :, :].permute(1, 2, 0)[:, :, k], cmap=cmaps[k])
            plt.axis('off')
            plt.savefig('tmp/matrix_{}_channel_{}_layer_{}.png'.format(i, k, j), bbox_inches='tight', pad_inches=0)

#%% ORIGINAL
for i in [2, 8, 19, 29]:
    plt.clf()
    plt.imshow(train_loader.dataset.data[i])
    plt.axis('off')
    plt.savefig('tmp/orig_{}.png'.format(i), bbox_inches='tight', pad_inches=0)

#%%
import numpy as np

idx = 29
tmp = (255 * (x[idx] - x[idx].min() / (x[idx].max() - x[idx].min())))
tmp = train_loader.dataset.data[idx]
plt.imshow(tmp)
plt.show()
