import torch
from tnn.layers import View, LinearLayer
from tnn.networks import FullyConnected, HamiltonianResNet
from tnn.regularization import SmoothTimeRegularization, TikhonovRegularization, BlockRegularization
from tnn.training.batch_train import train
from tnn.utils import seed_everything, number_network_weights, get_logger, makedirs, setup_parser, matrix_match_tensor_single_layer
import os
import datetime
import time
from copy import deepcopy
import pickle
from setup_cifar10 import setup_cifar10
import pandas as pd

# setup parser
parser = setup_parser()
args = parser.parse_args()

# seed for reproducibility
seed_everything(args.seed)

# setup data
train_loader, val_loader, test_loader = setup_cifar10(args.n_train, args.n_val, args.n_test, args.batch_size,
                                                      args.data_dir)


# get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# form network and regularizer
if args.opening_layer:
    net = torch.nn.Sequential(View((-1, 32 * 32 * 3)),
                              LinearLayer(32 * 32 * 3, args.width, activation=torch.nn.Tanh()),
                              HamiltonianResNet(args.width, width=args.width + args.add_width_hamiltonian,
                                                depth=args.depth, h=args.h_step, activation=torch.nn.Tanh()),
                              LinearLayer(args.width, 10, activation=None)
                              ).to(device)

    regularizer = BlockRegularization((None, TikhonovRegularization(alpha=args.alpha),
                                       SmoothTimeRegularization(alpha=args.alpha),
                                       TikhonovRegularization(alpha=args.alpha)))
else:
    w = 32 * 32 * 3
    net = torch.nn.Sequential(View((-1, 32 * 32 * 3)),
                              HamiltonianResNet(w, width=w + args.add_width_hamiltonian,
                                                depth=args.depth, h=args.h_step, activation=torch.nn.Tanh()),
                              LinearLayer(w, 10, activation=None)
                              ).to(device)

    regularizer = BlockRegularization((None,
                                       SmoothTimeRegularization(alpha=args.alpha),
                                       TikhonovRegularization(alpha=args.alpha)))


# choose loss function
loss = torch.nn.CrossEntropyLoss()

# choose optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)



# -------------------------------------------------------------------------------------------------------------------- #
# create logger
# path to save results
start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
sPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'experiments', 'matrix', start_time)
filename = 'cifar10_matrix.log'
makedirs(sPath)
logger = get_logger(logpath=os.path.join(sPath, filename), filepath=os.path.abspath(__file__), saving=True, mode="w")
logger.info(f'mnist_matrix')
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
                scheduler=scheduler, regularizer=regularizer, device=device, logger=logger)
t1 = time.perf_counter()


if torch.cuda.is_available():
    torch.cuda.synchronize()

logger.info('Total Training Time: {:.2f} seconds'.format(t1 - t0))

torch.save(net.state_dict(), sPath + '/last_net.pt')

pickle.dump(results, open(os.path.join(sPath, 'results.pkl'), 'wb'))
pd.DataFrame.to_csv(pd.DataFrame(results['val'], columns=results['str']), os.path.join(sPath, filename[:-4] + '.csv'))

