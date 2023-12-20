
import os
import torch
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl

load_dir = '/Users/elizabethnewman/Desktop/tnn_results/mnist/tensor/'

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


load_dir = '/Users/elizabethnewman/Desktop/tnn_results/mnist/matrix/'

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

exp_idx = [0, 1, 2, 3, 4]
# exp_idx = [5, 6, 7, 8, 9]


for i in exp_idx:
    idx = results_tensor[i]['str'].index('loss') + 2
    p = plt.semilogy(results_tensor[i]['val'][:, idx] / results_tensor[i]['val'][0, idx], label='tensor')
    p = plt.semilogy(results_matrix[i + 5]['val'][:, idx] / results_matrix[i]['val'][0, idx], '--', label='matrix', color=p[0].get_color())
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
    p = plt.plot(results_tensor[i]['val'][:, idx], label='tensor')
    p = plt.plot(results_matrix[i]['val'][:, idx], '--', label='matrix', color=p[0].get_color())
    plt.legend()

# plt.ylim([0, 100])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

#%% Look at features
from tnn.layers import Permute, View, LinearLayer
from tnn.networks import tFullyConnected
from tnn.loss import tCrossEntropyLoss
from tnn.utils import seed_everything, number_network_weights, get_logger, makedirs, setup_parser, matrix_match_tensor_single_layer
import os
from tnn.tensor_utils import dct_matrix, random_orthogonal
from tnn.training import test

os.chdir('/Users/elizabethnewman/Library/CloudStorage/OneDrive-EmoryUniversity/[000] Code/tnn/examples/mnist')

from mnist.setup_mnist import setup_mnist

net_idx = 0


# -------------------------------------------------------------------------------------------------------------------- #
# MATRIX
args = results_matrix[net_idx]['args']

train_loader, val_loader, test_loader = setup_mnist(args.n_train, args.n_val, args.n_test, args.batch_size,
                                                    args.data_dir)

# form matrix network
if args.width == 0:
    net_matrix = torch.nn.Sequential(View((-1, 784)),
                                     LinearLayer(784, 10, activation=None, bias=args.bias)
                                     )
else:
    w = args.width
    if args.matrix_match_tensor:
        w = matrix_match_tensor_single_layer(args.width, args.loss)

    net_matrix = torch.nn.Sequential(View((-1, 784)),
                                     LinearLayer(784, w, activation=torch.nn.Tanh(), bias=args.bias),
                                     LinearLayer(w, 10, activation=None, bias=args.bias)
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

if args.width == 0:
    net_tensor = torch.nn.Sequential(Permute((1, 0, 2)),
                                     tFullyConnected((28, 10), dim3, M=M, activation=None, bias=args.bias)
                                     )
    loss_tensor = tCrossEntropyLoss(M=M)
else:
    if args.loss == 't_cross_entropy':
        net_tensor = torch.nn.Sequential(Permute((1, 0, 2)),
                                         tFullyConnected((28, args.width), dim3, M=M,
                                                         activation=torch.nn.Tanh(), bias=args.bias),
                                         tFullyConnected((args.width, 10), dim3, M=M,
                                                         activation=None, bias=args.bias),
                                         )
        loss_tensor = tCrossEntropyLoss(M=M)

    else:
        net_tensor = torch.nn.Sequential(Permute((1, 0, 2)),
                                         tFullyConnected((28, args.width), dim3, M=M,
                                                         activation=torch.nn.Tanh(), bias=args.bias),
                                         Permute((1, 0, 2)),
                                         View((-1, args.width * dim3)),
                                         LinearLayer(args.width * dim3, 10, activation=None, bias=args.bias)
                                         )
        loss_tensor = torch.nn.CrossEntropyLoss()

net_tensor.load_state_dict(weights_tensor[net_idx])

test_out = test(net_tensor, loss_tensor, test_loader)
print('Test performance: TENSOR')
print('loss = {:<15.4e}'.format(test_out[0]))
print('accuracy = {:<15.4f}'.format(test_out[1]))

#%% FORWARD PROPAGATE
from copy import deepcopy

# os.mkdir('tmp/')

net_tensor.eval()

with torch.no_grad():
    x, y = next(iter(train_loader))

    x_list = [deepcopy(x)]

    x_list.append(net_tensor[1](net_tensor[0](x_list[-1])).permute(1, 0, 2))
    x_list.append(net_tensor[4](net_tensor[3](net_tensor[2](x_list[-1].permute(1, 0, 2)))))


# for i in range(x.shape[0]):
#     plt.subplot(4, 8, i + 1)
#     plt.imshow(x_list[0][i, 0, :, :], vmax=x_list[0].max(), vmin=x_list[0].min())
#     plt.axis('off')
#
# plt.show()


for i in range(x.shape[0]):
    # plt.subplot(4, 8, i + 1)
    plt.clf()
    plt.imshow(x_list[1][i, :, :], vmax=x_list[0].max(), vmin=x_list[0].min())
    plt.axis('off')
    plt.savefig('tmp/mnist_features_tensor_' + str(i) + '.png', bbox_inches='tight')


# plt.show()

#%% FORWARD PROPAGATE

from copy import deepcopy

args = results_matrix[-1]['args']

# form matrix network
if args.width == 0:
    net_matrix2 = torch.nn.Sequential(View((-1, 784)),
                                     LinearLayer(784, 10, activation=None, bias=args.bias)
                                     )
else:
    w = args.width
    if args.matrix_match_tensor:
        w = matrix_match_tensor_single_layer(args.width, args.loss)

    net_matrix2 = torch.nn.Sequential(View((-1, 784)),
                                     LinearLayer(784, w, activation=torch.nn.Tanh(), bias=args.bias),
                                     LinearLayer(w, 10, activation=None, bias=args.bias)
                                     )

net_matrix2.load_state_dict(weights_matrix[-1])

net_matrix2.eval()

with torch.no_grad():
    x, y = next(iter(train_loader))

    x_list = [deepcopy(x)]
    x_list.append(net_matrix2[1](net_matrix2[0](x_list[-1])).reshape(-1, 28, 28))


for i in range(x.shape[0]):
    # plt.subplot(4, 8, i + 1)
    plt.imshow(x_list[0][i, 0, :, :], vmax=x_list[0].max(), vmin=x_list[0].min())
    plt.axis('off')
    plt.savefig('tmp/mnist_input_' + str(i) + '.png', bbox_inches='tight')

# plt.show()


for i in range(x.shape[0]):
    # plt.subplot(4, 8, i + 1)
    plt.imshow(x_list[1][i, :, :], vmax=x_list[0].max(), vmin=x_list[0].min())
    plt.axis('off')
    plt.savefig('tmp/mnist_features_matrix_' + str(i) + '.png', bbox_inches='tight')


# plt.show()
