import torch
from tnn.layers import View, Permute, tLinearLayer, LinearLayer
from tnn.networks import tHamiltonianResNet
from tnn.loss import tCrossEntropyLoss
from tnn.regularization import SmoothTimeRegularization, TikhonovRegularization, BlockRegularization
from tnn.training.batch_train import train
from tnn.utils import seed_everything, number_network_weights, get_logger, makedirs, setup_parser
from tnn.tensor_utils import dct_matrix, random_orthogonal
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

# create transformation matrix
dim3 = 32
if args.M == 'dct':
    M = dct_matrix(dim3, dtype=torch.float32, device=device)
elif args.M == 'random':
    M = random_orthogonal(dim3, dtype=torch.float32, device=device)
else:
    M = torch.eye(dim3, dtype=torch.float32, device=device)

# form network and choose loss function and regularizer
if args.opening_layer:
    if args.loss == 't_cross_entropy':
        net = torch.nn.Sequential(View((-1, 32 * 3, 32)),
                                  Permute((1, 0, 2)),
                                  tLinearLayer(32 * 3, args.width, dim3, M=M, activation=torch.nn.Tanh()),
                                  tHamiltonianResNet(args.width, args.width + args.add_width_hamiltonian, dim3, M,
                                                     depth=args.depth, h=args.h_step, activation=torch.nn.Tanh()),
                                  tLinearLayer(args.width, 10, dim3, M=M, activation=None)
                                  ).to(device)
        loss = tCrossEntropyLoss(M=M)
    else:
        net = torch.nn.Sequential(View((-1, 32 * 3, 32)),
                                  Permute((1, 0, 2)),
                                  tLinearLayer(32 * 3, args.width, dim3, M=M, activation=torch.nn.Tanh()),
                                  tHamiltonianResNet(args.width, args.width + args.add_width_hamiltonian, dim3, M,
                                                     depth=args.depth, h=args.h_step, activation=torch.nn.Tanh()),
                                  Permute((1, 0, 2)),
                                  View((-1, args.width * dim3)),
                                  LinearLayer(args.width * dim3, 10, activation=None, bias=args.bias)
                                  ).to(device)

        loss = torch.nn.CrossEntropyLoss()

    regularizer = BlockRegularization((None, None, TikhonovRegularization(alpha=args.alpha),
                                       SmoothTimeRegularization(alpha=args.alpha),
                                       TikhonovRegularization(alpha=args.alpha)))

else:
    w = 32 * 3

    if args.loss == 't_cross_entropy':
        net = torch.nn.Sequential(View((-1, w, 32)),
                                  Permute((1, 0, 2)),
                                  tHamiltonianResNet(w, w + args.add_width_hamiltonian, dim3, M,
                                                     depth=args.depth, h=args.h_step, activation=torch.nn.Tanh()),
                                  tLinearLayer(w, 10, dim3, M=M, activation=None)
                                  ).to(device)
        loss = tCrossEntropyLoss(M=M)
    else:
        net = torch.nn.Sequential(View((-1, w, 32)),
                                  Permute((1, 0, 2)),
                                  tHamiltonianResNet(w, w + args.add_width_hamiltonian, dim3, M,
                                                     depth=args.depth, h=args.h_step, activation=torch.nn.Tanh()),
                                  Permute((1, 0, 2)),
                                  View((-1, w * dim3)),
                                  LinearLayer(w * dim3, 10, activation=None, bias=args.bias)
                                  ).to(device)
        loss = torch.nn.CrossEntropyLoss()

    regularizer = BlockRegularization((None, None,
                                       SmoothTimeRegularization(alpha=args.alpha),
                                       None, None,
                                       TikhonovRegularization(alpha=args.alpha)))


# choose optimizer and scheduler
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

# -------------------------------------------------------------------------------------------------------------------- #
# create logger
# path to save results
start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
sPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'experiments', 'tensor', start_time)
filename = 'cifar10_tensor.log'
makedirs(sPath)
logger = get_logger(logpath=os.path.join(sPath, filename), filepath=os.path.abspath(__file__), saving=True, mode="w")
logger.info(f'cifar10_tensor')
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
                scheduler=scheduler, regularizer=regularizer, device=device, logger=logger, sPath=sPath)
t1 = time.perf_counter()


if torch.cuda.is_available():
    torch.cuda.synchronize()

logger.info('Total Training Time: {:.2f} seconds'.format(t1 - t0))

torch.save(net.state_dict(), sPath + '/last_net.pt')

results['args'] = args
pickle.dump(results, open(os.path.join(sPath, 'results.pkl'), 'wb'))
pd.DataFrame.to_csv(pd.DataFrame(results['val'], columns=results['str']), os.path.join(sPath, filename[:-4] + '.csv'))

