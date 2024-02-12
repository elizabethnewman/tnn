import os
import torch
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl

load_dir = '/Users/elizabethnewman/Desktop/tnn_results/autoencoder/tensor/'

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


load_dir = '/Users/elizabethnewman/Desktop/tnn_results/autoencoder/matrix/'

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

from tnn.layers import Permute, View, LinearLayer, tLinearLayer
from tnn.networks import tFullyConnected, FullyConnected
from tnn.loss import tCrossEntropyLoss
from tnn.utils import seed_everything, number_network_weights, get_logger, makedirs, setup_parser, matrix_match_tensor_single_layer
import os
from tnn.tensor_utils import dct_matrix, random_orthogonal
from autoencoder.autoencoder_batch_train import test

os.chdir('/Users/elizabethnewman/Library/CloudStorage/OneDrive-EmoryUniversity/[000] Code/tnn/examples')

from autoencoder.setup_mnist import setup_mnist


net_idx = 0

loss = torch.nn.MSELoss()

#%%
args = results_matrix[net_idx + 3]['args']

net_matrix = torch.nn.Sequential(View((-1, 784)),
                                 FullyConnected((784, args.auto_width, args.width), activation=torch.nn.Tanh(),
                                                bias=args.bias),
                                 FullyConnected((args.width, args.auto_width), activation=torch.nn.Tanh(),
                                                bias=args.bias),
                                 LinearLayer(args.auto_width, 784, activation=None, bias=args.bias),
                                 View((-1, 1, 28, 28))
                                 )

seed_everything(args.seed)

train_loader, val_loader, test_loader = setup_mnist(args.n_train, args.n_val, args.n_test, args.batch_size,
                                                    args.data_dir)

net_matrix.load_state_dict(weights_matrix[net_idx + 3])

test_out = test(net_matrix, loss, test_loader)

print('Test performance: MATRIX')
print('loss = {:<15.4e}'.format(test_out[0]))
print('accuracy = {:<15.4f}'.format(test_out[1]))

#%%
args = results_tensor[net_idx]['args']

# create transformation matrix
dim3 = 28
if args.M == 'dct':
    M = dct_matrix(dim3, dtype=torch.float32)
elif args.M == 'random':
    M = random_orthogonal(dim3, dtype=torch.float32)
elif args.M == 'data':
    # compute left singular vectors of mode-3 unfolding of data
    A = torch.empty(0)
    for data in train_loader:
        A = torch.cat((A, data[0]), dim=0)
    A = A.reshape(-1, A.shape[-1]).T
    M, _, _ = torch.linalg.svd(A, full_matrices=False)
    M = M.T.to(dtype=torch.float32)
    del A
else:
    M = torch.eye(dim3, dtype=torch.float32)

if args.auto_width is None or args.auto_width == 0:
    net_tensor = torch.nn.Sequential(View((-1, 28, 28)),
                              Permute((1, 0, 2)),
                              tLinearLayer(28, args.width, dim3, M=M,
                                           activation=torch.nn.Tanh(), bias=args.bias),
                              tLinearLayer(args.width, 28, dim3, M=M,
                                           activation=None, bias=args.bias),
                              Permute((1, 0, 2)),
                              View((-1, 1, 28, 28))
                              )
else:
    net_tensor = torch.nn.Sequential(View((-1, 28, 28)),
                              Permute((1, 0, 2)),
                              tFullyConnected((28, args.auto_width, args.width), dim3, M=M,
                                              activation=torch.nn.Tanh(), bias=args.bias),
                              tFullyConnected((args.width, args.auto_width), dim3, M=M,
                                              activation=torch.nn.Tanh(), bias=args.bias),
                              tLinearLayer(args.auto_width, 28, dim3, M=M,
                                           activation=None, bias=args.bias),
                              Permute((1, 0, 2)),
                              View((-1, 1, 28, 28))
                              )
seed_everything(args.seed)

net_tensor.load_state_dict(weights_tensor[net_idx])

test_out = test(net_tensor, loss, test_loader)

print('Test performance: TENSOR')
print('loss = {:<15.4e}'.format(test_out[0]))
print('accuracy = {:<15.4f}'.format(test_out[1]))


#%% approximations

import matplotlib.pyplot as plt
x, y = next(iter(test_loader))
z_mat = net_matrix(x).cpu().detach()
z_ten = net_tensor(x).cpu().detach()


vmax = x.max()
vmin = x.min()
vmax_diff = max((x - z_mat).abs().max(), (x - z_ten).abs().max())

plt.figure()
for i in range(x.shape[0]):
    plt.figure()
    plt.imshow(x[i].squeeze(), vmin=vmin, vmax=vmax, cmap='gray')
    plt.axis('off')
    plt.savefig('/Users/elizabethnewman/Desktop/tnn_results/autoencoder/approx/orig_' + str(i) + '.png',
                bbox_inches='tight', pad_inches=0.0)
    plt.close()

    plt.figure()
    plt.imshow(z_mat[i].squeeze(), vmin=vmin, vmax=vmax, cmap='gray')
    plt.axis('off')
    plt.savefig('/Users/elizabethnewman/Desktop/tnn_results/autoencoder/approx/matrix_21_280_approx_' + str(i) + '.png',
                bbox_inches='tight', pad_inches=0.0)
    plt.close()

    plt.figure()
    plt.imshow((x[i] - z_mat[i]).abs().squeeze(), vmin=0.0, vmax=vmax_diff, cmap='hot')
    plt.axis('off')
    plt.savefig('/Users/elizabethnewman/Desktop/tnn_results/autoencoder/approx/matrix_21_280_diff_' + str(i) + '.png',
                bbox_inches='tight', pad_inches=0.0)
    plt.close()

    plt.figure()
    plt.imshow(z_ten[i].squeeze(), vmin=vmin, vmax=vmax, cmap='gray')
    plt.axis('off')
    plt.savefig('/Users/elizabethnewman/Desktop/tnn_results/autoencoder/approx/tensor_approx_' + str(i) + '.png',
                bbox_inches='tight', pad_inches=0.0)
    plt.close()

    plt.figure()
    plt.imshow((x[i] - z_ten[i]).abs().squeeze(), vmin=0.0, vmax=vmax_diff, cmap='hot')
    plt.axis('off')
    plt.savefig('/Users/elizabethnewman/Desktop/tnn_results/autoencoder/approx/tensor_diff_' + str(i) + '.png',
                bbox_inches='tight', pad_inches=0.0)
    plt.close()


    # plt.subplot(3, 8, i + 1 + 1 * 8)
    # plt.imshow(z[i].squeeze(), vmin=vmin, vmax=vmax)
    # plt.axis('off')
    #
    # plt.subplot(3, 8, i + 1 + 2 * 8)
    # im = plt.imshow(abs(x[i] - z[i]).squeeze())
    # plt.axis('off')
    # plt.colorbar(im, fraction=0.046, pad=0.04)

# plt.savefig(sPath + '/train_approx.png')
