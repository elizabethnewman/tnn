import torch
from tnn.layers import Permute, View, LinearLayer, tLinearLayer
from tnn.networks import tFullyConnected
# from tnn.training.batch_train import train
from autoencoder_batch_train import train
from tnn.tensor_utils import dct_matrix, random_orthogonal
from tnn.utils import seed_everything, number_network_weights, get_logger, makedirs, setup_parser
import os
import datetime
import time
import pickle
import pandas as pd
from setup_mnist import setup_mnist


# setup parser
parser = setup_parser()
args = parser.parse_args()

# seed for reproducibility
seed_everything(args.seed)

# setup data
train_loader, val_loader, test_loader = setup_mnist(args.n_train, args.n_val, args.n_test, args.batch_size,
                                                    args.data_dir)

# get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create transformation matrix
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

if args.auto_width is None or args.auto_width == 0:
    net = torch.nn.Sequential(View((-1, 28, 28)),
                              Permute((1, 0, 2)),
                              tLinearLayer(28, args.width, dim3, M=M,
                                           activation=torch.nn.Tanh(), bias=args.bias),
                              tLinearLayer(args.width, 28, dim3, M=M,
                                           activation=None, bias=args.bias),
                              Permute((1, 0, 2)),
                              View((-1, 1, 28, 28))
                              ).to(device)
else:
    net = torch.nn.Sequential(View((-1, 28, 28)),
                              Permute((1, 0, 2)),
                              tFullyConnected((28, args.auto_width, args.width), dim3, M=M,
                                              activation=torch.nn.Tanh(), bias=args.bias),
                              tFullyConnected((args.width, args.auto_width), dim3, M=M,
                                              activation=torch.nn.Tanh(), bias=args.bias),
                              tLinearLayer(args.auto_width, 28, dim3, M=M,
                                           activation=None, bias=args.bias),
                              Permute((1, 0, 2)),
                              View((-1, 1, 28, 28))
                              ).to(device)
loss = torch.nn.MSELoss()


# choose optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

# -------------------------------------------------------------------------------------------------------------------- #
# create logger
# path to save results
start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
sPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'experiments', 'tensor', start_time)
filename = 'autoencoder_tensor.log'
makedirs(sPath)
logger = get_logger(logpath=os.path.join(sPath, filename), filepath=os.path.abspath(__file__), saving=True, mode="w")
logger.info(f'autoencoder_tensor')
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

results['args'] = args
pickle.dump(results, open(os.path.join(sPath, 'results.pkl'), 'wb'))
pd.DataFrame.to_csv(pd.DataFrame(results['val'], columns=results['str']), os.path.join(sPath, filename[:-4] + '.csv'))

#%%
import matplotlib.pyplot as plt
x, y = next(iter(train_loader))
z = net(x.to(device)).cpu().detach()

vmax = x.max()
vmin = x.min()

plt.figure()
for i in range(8):
    plt.subplot(3, 8, i + 1 + 0 * 8)
    plt.imshow(x[i].squeeze(), vmin=vmin, vmax=vmax)
    plt.axis('off')

    plt.subplot(3, 8, i + 1 + 1 * 8)
    plt.imshow(z[i].squeeze(), vmin=vmin, vmax=vmax)
    plt.axis('off')

    plt.subplot(3, 8, i + 1 + 2 * 8)
    im = plt.imshow(abs(x[i] - z[i]).squeeze())
    plt.axis('off')
    # plt.colorbar(im, fraction=0.046, pad=0.04)

plt.savefig(sPath + '/train_approx.png')
# plt.show()

