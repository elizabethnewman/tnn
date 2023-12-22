
import torch
import matplotlib.pyplot as plt
from tnn.utils import seed_everything, setup_parser
from examples.tensor_trajectories.utils import setup_circle_3D, setup_circle_3D_grid, split_data, DummyData

# setup parser
parser = setup_parser()
args = parser.parse_args()

# hard code some parameters
args.n_train = 1200
args.n_val = 600
args.max_epochs = 50
args.batch_size = 10
args.lr = 1e-2
args.weight_decay = 0.0
args.alpha = 1e-4
args.h = 2.0
args.M = 'dct'
args.loss = 't_cross_entropy'

# seed for reproducibility
seed_everything(args.seed)

# get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# plot data to check classes
x, y = setup_circle_3D(n=2000)

# create training data
data_train, data_val, data_test = split_data(x, y, args.n_train, n_val=args.n_val, n_test=None, shuffle=True)

dummy_data_train = DummyData(*data_train)
train_loader = torch.utils.data.DataLoader(dummy_data_train, batch_size=args.batch_size, shuffle=True)

dummy_data_val = DummyData(*data_val)
val_loader = torch.utils.data.DataLoader(dummy_data_val, batch_size=args.batch_size, shuffle=False)

dummy_data_test = DummyData(*data_test)
test_loader = torch.utils.data.DataLoader(dummy_data_test, batch_size=args.batch_size, shuffle=False)

#%% create network
from tnn.tensor_utils import dct_matrix, random_orthogonal
from tnn.networks import tResNet, tHamiltonianResNet, ResNet, tAntisymmetricResNet
from tnn.layers import LinearLayer, tResidualLayer, tLinearLayer, tAntiSymmetricResidualLayer, View, Permute, tHamiltonianLayer
from tnn.regularization import SmoothTimeRegularization, TikhonovRegularization, BlockRegularization

# create transformation matrix
dim3 = 3
if args.M == 'dct':
    M = dct_matrix(dim3, dtype=torch.float32, device=device)
elif args.M == 'random':
    M = random_orthogonal(dim3, dtype=torch.float32, device=device)
else:
    M = torch.eye(dim3, dtype=torch.float32, device=device)


if args.loss == 't_cross_entropy':
    net = torch.nn.Sequential(
        View((-1, 1, 3)),
        Permute((1, 0, 2)),
        tResNet(1, dim3=dim3, M=M, depth=32, h=args.h, activation=torch.nn.Tanh()),
        tLinearLayer(1, 3, dim3=dim3, M=M, activation=None)
    )
else:
    net = torch.nn.Sequential(
        View((-1, 1, 3)),
        Permute((1, 0, 2)),
        tResNet(1, dim3=dim3, M=M, depth=32, h=args.h, activation=torch.nn.Tanh()),
        Permute((1, 0, 2)),
        View((-1, 3)),
        torch.nn.Linear(3, 3)
    )

# regularizer
regularizer = BlockRegularization((None, None,
                                   SmoothTimeRegularization(alpha=args.alpha / args.h),
                                   None, None,
                                   TikhonovRegularization(alpha=args.alpha))
                                  )


#%% create loss function
from tnn.loss import tCrossEntropyLoss

if args.loss == 't_cross_entropy':
    loss = tCrossEntropyLoss(M=M)
else:
    loss = torch.nn.CrossEntropyLoss()

#%% train network
from tnn.training.batch_train import train

optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

#%% create logger
import os
from tnn.utils import makedirs, get_logger, number_network_weights

# path to save results
sPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'experiments', 'euler')
filename = 'tmp.log'
makedirs(sPath)
logger = get_logger(logpath=os.path.join(sPath, filename), filepath=os.path.abspath(__file__), saving=True, mode="w")
logger.info(f'mnist_matrix')

logger.info("---------------------- Network ----------------------------")
logger.info(net)
logger.info("Number of trainable parameters: {}".format(number_network_weights(net)))
logger.info("--------------------------------------------------")
logger.info(str(optimizer))
logger.info("saveLocation = {:}".format(sPath))
logger.info("--------------------------------------------------\n")


results = train(net, loss, optimizer, train_loader, val_loader, test_loader, max_epochs=args.max_epochs,
                logger=logger, sPath=sPath,
                device=device, regularizer=regularizer)

#%% plot results
from copy import deepcopy

os.chdir(sPath)

# if directory doesnot exist, create it
if not os.path.exists('img'):
    os.mkdir('img')

with torch.no_grad():

    z = [deepcopy(net[1](net[0](data_train[0])))]

    for layer in net[2].layers:
        z.append(deepcopy(layer(z[-1], M)))


for i in range(len(z)):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(z[i][0, :, 0], z[i][0, :, 1].squeeze(), z[i][0, :, 2].squeeze(), c=data_train[1])
    # ax.set_xlim(z[0][0, :, 1].min(), z[0][0, :, 1].max())
    # ax.set_ylim(z[0][0, :, 1].min(), z[0][0, :, 1].max())
    # ax.set_zlim(z[0][0, :, 1].min(), z[0][0, :, 1].max())
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.xaxis.set_visible(False)
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.zaxis.set_ticks_position('none')
    ax.grid(True)

    # plt.show()
    plt.savefig(f'img/layer_{i}.png')

    # close figure
    plt.close(fig)
