
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
for subdir_name in subdir_names:
    print(subdir_name)
    results_tensor.append(pickle.load(open(load_dir + subdir_name + '/results.pkl', "rb")))


load_dir = '/Users/elizabethnewman/Desktop/tnn_results/cifar10/matrix/'

subdir_names = []
for subdir_name in os.listdir(load_dir):
    if subdir_name == '.DS_Store':
        continue
    subdir_names.append(subdir_name)

subdir_names.sort()



results_matrix = []
for subdir_name in subdir_names:
    print(subdir_name)
    results_matrix.append(pickle.load(open(load_dir + subdir_name + '/results.pkl', "rb")))




#%%

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['lines.linewidth'] = 4
mpl.rcParams['font.size'] = 10


for i in [0, 1]:
# for i in [4]:
# for i in [0, 1, 2, 3, 4]:
# for i in [5, 6, 7, 8, 9]:
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


# for i in [0, 1, 2, 3, 4]:
# for i in [5, 6, 7, 8, 9]:
for i in [0, 1]:
    idx = results_tensor[i]['str'].index('acc') + 4
    p = plt.plot(results_tensor[i]['val'][:, idx], label='tensor')
    p = plt.plot(results_matrix[i]['val'][:, idx], '--', label='matrix', color=p[0].get_color())
    plt.legend()

# plt.ylim([0, 100])
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()

#%% test results
# import torch
# from tnn.layers import View, Permute, tLinearLayer, LinearLayer
# from tnn.networks import tHamiltonianResNet
# from tnn.loss import tCrossEntropyLoss
# from tnn.regularization import SmoothTimeRegularization, TikhonovRegularization, BlockRegularization
# from tnn.training.batch_train import train
# from tnn.utils import seed_everything, number_network_weights, get_logger, makedirs, setup_parser
# from tnn.tensor_utils import dct_matrix, random_orthogonal
# import os
# import datetime
# import time
# from copy import deepcopy
# import pickle
#
# import pandas as pd
# import os
#
# print(os.getcwd())
# os.chdir('/Users/elizabethnewman/Library/CloudStorage/OneDrive-EmoryUniversity/[000] Code/tnn/examples/mnist')
#
# print(os.getcwd())
#
# from cifar10.setup_cifar10 import setup_cifar10
#
# args = results_tensor[0]['args']
# # seed for reproducibility
# seed_everything(args.seed)
#
# # setup data
# train_loader, val_loader, test_loader = setup_cifar10(args.n_train, args.n_val, args.n_test, args.batch_size,
#                                                       args.data_dir)
#
#


