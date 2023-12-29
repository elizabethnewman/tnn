import os
import torch
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

load_dir = '/Users/elizabethnewman/Desktop/tnn_results/cifar10_timing/tensor/'

results_tensor = []
for subdir_name in ['depth_4', 'depth_8', 'depth_16', 'depth_32', 'depth_64', 'depth_128']:
    print(subdir_name)
    results_tensor.append(pd.read_csv(load_dir + subdir_name + '/cifar10_tensor.csv'))


load_dir = '/Users/elizabethnewman/Desktop/tnn_results/cifar10_timing/matrix/'

results_matrix = []
for subdir_name in ['depth_4', 'depth_8', 'depth_16', 'depth_32', 'depth_64', 'depth_128']:
    print(subdir_name)
    results_matrix.append(pd.read_csv(load_dir + subdir_name + '/cifar10_matrix.csv'))

#%%

import numpy as np
time_tensor = []
std_tensor = []
time_matrix = []
std_matrix = []
for i in range(len(results_tensor)):
    time_tensor.append(results_tensor[i]['time'].values[1:].mean())
    std_tensor.append(results_tensor[i]['time'].values[1:].std())
    time_matrix.append(results_matrix[i]['time'].values[1:].mean())
    std_matrix.append(results_matrix[i]['time'].values[1:].std())

time_tensor = np.array(time_tensor)
std_tensor = np.array(std_tensor)
time_matrix = np.array(time_matrix)
std_matrix = np.array(std_matrix)

plt.fill_between([4, 8, 16, 32, 64, 128], time_tensor - std_tensor, time_tensor + std_tensor)
plt.plot([4, 8, 16, 32, 64, 128], time_tensor, label='tensor')
plt.fill_between([4, 8, 16, 32, 64, 128], time_matrix - std_matrix, time_matrix + std_matrix)
plt.plot([4, 8, 16, 32, 64, 128], time_matrix, label='matrix')
plt.show()