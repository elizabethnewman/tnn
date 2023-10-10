import torch
from torchvision import datasets, transforms
from tnn.layers import Permute, View, LinearLayer
from tnn.networks import tFullyConnected
from tnn.loss import tCrossEntropyLoss
from tnn.training.batch_train import train
from tnn.tensor_utils import dct_matrix, random_orthogonal
from tnn.utils import seed_everything, number_network_weights, makedirs, get_logger
import os
import datetime
import argparse
import time
from copy import deepcopy
import pickle
from setup_mnist import setup_mnist, setup_parser


# setup parser
parser = setup_parser()
args = parser.parse_args()

# seed for reproducibility
seed_everything(args.seed)

# setup data
train_loader, val_loader, test_loader = setup_mnist(args.n_train, args.n_val, args.n_test, args.batch_size)

# get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create transformation matrix
dim3 = 28
if args.M == 'dct':
    M = dct_matrix(dim3, dtype=torch.float32, device=device)
elif args.M == 'random':
    M = random_orthogonal(dim3, dtype=torch.float32, device=device)
else:
    M = torch.eye(dim3, dtype=torch.float32, device=device)

# form network
net = torch.nn.Sequential(Permute((1, 0, 2)),
                          tFullyConnected((28, 10), dim3, M=M, activation=None, bias=False),
                          # Permute((1, 0, 2)),
                          # View((-1, 10 * 28)),
                          # LinearLayer(10 * 28, 10, activation=None, bias=False)
                          ).to(device)

# choose loss function
loss = tCrossEntropyLoss(M=M)
# loss = torch.nn.CrossEntropyLoss()

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
                scheduler=scheduler, device=device, logger=logger)
t1 = time.perf_counter()

if torch.cuda.is_available():
    torch.cuda.synchronize()

logger.info('Total Training Time: {:.2f} seconds'.format(t1 - t0))

results['last_net'] = deepcopy(net)

pickle.dump(results, open(os.path.join(sPath, 'results.pkl'), 'wb'))