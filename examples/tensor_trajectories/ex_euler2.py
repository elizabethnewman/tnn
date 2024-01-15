
import torch
import matplotlib.pyplot as plt
from tnn.utils import seed_everything
from examples.tensor_trajectories.utils import setup_circle_3D, setup_circle_3D_grid, split_data, DummyData

seed_everything(42)

# plot data to check classes
x, y = setup_circle_3D(n=2000)
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y)
# plt.show()

# create contour plot based on classes
# xx, yy, zz = setup_circle_3D_grid()
# plt.contourf(xx.numpy(), yy.numpy(), zz.numpy())
# plt.show()

# create training data

data_train, data_val, data_test = split_data(x, y, 1200, n_val=600, n_test=None, shuffle=True)

dummy_data_train = DummyData(*data_train)
train_loader = torch.utils.data.DataLoader(dummy_data_train, batch_size=10, shuffle=True)

dummy_data_val = DummyData(*data_val)
val_loader = torch.utils.data.DataLoader(dummy_data_val, batch_size=10, shuffle=False)

dummy_data_test = DummyData(*data_test)
test_loader = torch.utils.data.DataLoader(dummy_data_test, batch_size=10, shuffle=False)

#%% create network
from tnn.tensor_utils import dct_matrix, random_orthogonal
from tnn.networks import tResNet, tHamiltonianResNet, ResNet, tAntisymmetricResNet
from tnn.layers import LinearLayer, tResidualLayer, tLinearLayer, tAntiSymmetricResidualLayer, View, Permute, tHamiltonianLayer
from tnn.regularization import SmoothTimeRegularization, TikhonovRegularization, BlockRegularization

# create transformation matrix
dim3 = 3
# M = dct_matrix(dim3, dtype=torch.float32)
M = torch.eye(dim3, dtype=torch.float32)

h = 5.0
net = torch.nn.Sequential(
    View((-1, 1, 3)),
    Permute((1, 0, 2)),
    tResNet(1, dim3=dim3, M=M, depth=32, h=h, activation=torch.nn.Tanh()),
    Permute((1, 0, 2)),
    View((-1, 3)),
    torch.nn.Linear(3, 3)
)


alpha = 1e-4
regularizer = BlockRegularization((None, None,
                                   SmoothTimeRegularization(alpha=alpha / h),
                                   None, None,
                                   TikhonovRegularization(alpha=0.0)))

# regularizer = None

#%% create loss function
from tnn.loss import tCrossEntropyLoss

# loss = tCrossEntropyLoss(M=M)
loss = torch.nn.CrossEntropyLoss()

#%% train network
from tnn.training.batch_train import train

optimizer = torch.optim.SGD(net.parameters(), lr=1e-2, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

#%% create logger
# -------------------------------------------------------------------------------------------------------------------- #
# create logger
import os
from tnn.utils import makedirs, get_logger, number_network_weights

# get device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# path to save results
sPath = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'experiments', 'tmp')
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


results = train(net, loss, optimizer, train_loader, val_loader, test_loader, max_epochs=50, logger=logger, sPath=sPath,
                device=device, regularizer=regularizer)

#%% plot results
from copy import deepcopy

with torch.no_grad():

    z = [deepcopy(net[1](net[0](data_train[0])))]

    # z.append(net[3](net[2](z[-1])))

    for layer in net[2].layers:
        z.append(deepcopy(layer(z[-1], M)))


for i in [0, 1, 2, 8, 16, 32]:
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(z[i][0, :, 0], z[i][0, :, 1].squeeze(), z[i][0, :, 2].squeeze(), c=data_train[1])
    # ax.set_xlim(z[0][0, :, 1].min(), z[0][0, :, 1].max())
    # ax.set_ylim(z[0][0, :, 1].min(), z[0][0, :, 1].max())
    # ax.set_zlim(z[0][0, :, 1].min(), z[0][0, :, 1].max())
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 10)
    plt.show()

    # close figure
    plt.close(fig)
