import torch
from torchvision import datasets, transforms
from tnn.layers import Permute, View, LinearLayer
from tnn.networks import tFullyConnected
from tnn.loss import tCrossEntropyLoss
from tnn.training.batch_train import train
from tnn.tensor_utils import dct_matrix, random_orthogonal
from tnn.utils import seed_everything, number_network_weights, get_logger, makedirs, setup_parser
import os
import datetime
import argparse
import time
from copy import deepcopy
import pickle
from setup_mnist import setup_mnist
import pandas as pd


# setup parser
parser = setup_parser()
args = parser.parse_args()

args.loss = 't_cross_entropy'
args.n_train = 2000

# seed for reproducibility
seed_everything(args.seed)

# setup data
train_loader, val_loader, test_loader = setup_mnist(args.n_train, args.n_val, args.n_test, args.batch_size,
                                                    args.data_dir)

# get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create transformation matri
dim3 = 28
if args.M == 'dct':
    M = dct_matrix(dim3, dtype=torch.float32, device=device)
elif args.M == 'random':
    M = random_orthogonal(dim3, dtype=torch.float32, device=device)
elif args.M == 'data':
    # compute left singular vectors of mode-3 unfolding of data
    A = torch.empty(0)
    for data in train_loader:
        A = torch.cat((A, data[0]), dim=0)
    A = A.reshape(-1, A.shape[-1]).T
    M, _, _ = torch.linalg.svd(A, full_matrices=False)
    M = M.T.to(dtype=torch.float32, device=device)
    del A
else:
    M = torch.eye(dim3, dtype=torch.float32, device=device)

# form network and choose loss
if args.width == 0:
    net = torch.nn.Sequential(Permute((1, 0, 2)),
                              tFullyConnected((28, 10), dim3, M=M, activation=None, bias=args.bias),
                              ).to(device)
    loss = tCrossEntropyLoss(M=M)
else:
    if args.loss == 't_cross_entropy':
        net = torch.nn.Sequential(Permute((1, 0, 2)),
                                  tFullyConnected((28, args.width), dim3, M=M,
                                                  activation=torch.nn.Tanh(), bias=args.bias),
                                  tFullyConnected((args.width, 10), dim3, M=M,
                                                  activation=None, bias=args.bias),
                                  ).to(device)
        loss = tCrossEntropyLoss(M=M)

    else:
        net = torch.nn.Sequential(Permute((1, 0, 2)),
                                  tFullyConnected((28, args.width), dim3, M=M,
                                                  activation=torch.nn.Tanh(), bias=args.bias),
                                  Permute((1, 0, 2)),
                                  View((-1, args.width * dim3)),
                                  LinearLayer(args.width * dim3, 10, activation=None, bias=args.bias)
                                  ).to(device)
        loss = torch.nn.CrossEntropyLoss()

# choose optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

# -------------------------------------------------------------------------------------------------------------------- #
# create logger
# path to save results
start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
sPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'experiments', 'tensor', start_time)
filename = 'mnist_tensor.log'
makedirs(sPath)
logger = get_logger(logpath=os.path.join(sPath, filename), filepath=os.path.abspath(__file__), saving=True, mode="w")
logger.info(f'mnist_tensor')
logger.info(f'args: {args}')

logger.info("---------------------- Network ----------------------------")
logger.info(net)
logger.info("Number of trainable parameters: {}".format(number_network_weights(net)))
logger.info("--------------------------------------------------")
logger.info(str(optimizer))
logger.info(str(scheduler))
logger.info("dtype={:} device={:}".format(train_loader.dataset.data.dtype, device))
logger.info("epochs={:} ".format(args.max_epochs))
logger.info("saveLocation = {:}".format(sPath))
logger.info("--------------------------------------------------\n")

# -------------------------------------------------------------------------------------------------------------------- #

# train!
t0 = time.perf_counter()
results = train(net, loss, optimizer, train_loader, val_loader, test_loader, max_epochs=args.max_epochs,
                scheduler=scheduler, device=device, logger=logger, sPath=sPath)
t1 = time.perf_counter()

if torch.cuda.is_available():
    torch.cuda.synchronize()

logger.info('Total Training Time: {:.2f} seconds'.format(t1 - t0))

torch.save(net.state_dict(), sPath + '/last_net.pt')

pickle.dump(results, open(os.path.join(sPath, 'results.pkl'), 'wb'))
pd.DataFrame.to_csv(pd.DataFrame(results['val'], columns=results['str']), os.path.join(sPath, filename[:-4] + '.csv'))
