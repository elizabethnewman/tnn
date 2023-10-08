
import unittest
import torch
import math
from tnn.regularization import TikhonovRegularization, SmoothTimeRegularization, BlockRegularization
from tnn.utils import extract_data, insert_data


class TikhonovRegularizerTest(unittest.TestCase):

    # setup tnn network
    def setup_network(self):
        torch.set_default_dtype(torch.float64)
        torch.manual_seed(42)

        # create data
        x = torch.randn(100, 2)
        y = torch.randn(100, 3)

        # define network
        net = torch.nn.Sequential(
            torch.nn.Linear(x.shape[1], 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 5),
            torch.nn.ReLU(),
            torch.nn.Linear(5, y.shape[1])
        )

        # define loss function
        loss_fn = torch.nn.MSELoss()

        # define optimizer
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)

        # return everything for testing
        return x, y, net, loss_fn, optimizer

    def test_regularizer_alpha_zero(self):
        # setup network and regularizer
        x, y, net, loss_fn, optimizer = self.setup_network()

        # extract network weights
        theta = extract_data(net, 'data')

        # define regularizer
        reg = TikhonovRegularization(alpha=0)

        # ------------------------------------------------------------ #
        # evaluate loss without regularization

        # zero gradient
        optimizer.zero_grad()

        # forward pass
        y_pred = net(x)

        # compute loss
        loss = loss_fn(y_pred, y)

        # compute gradient
        loss.backward()

        # extract gradient
        dtheta = extract_data(net, 'grad')

        # ------------------------------------------------------------ #
        # compete pass again with regularization
        # zero gradient
        optimizer.zero_grad()

        # forward pass
        y_pred = net(x)

        # compute loss + regularzation
        loss = loss_fn(y_pred, y) + reg(net)

        # compute gradient
        loss.backward()

        # extract gradient
        dtheta_reg = extract_data(net, 'grad')

        # ------------------------------------------------------------ #
        # check if gradient without reg is almost equal to gradient with reg
        self.assertTrue(torch.allclose(dtheta, dtheta_reg, rtol=1e-5, atol=1e-5))

    def test_regularizer(self):
        # setup network and regularizer
        x, y, net, loss_fn, optimizer = self.setup_network()

        # extract network weights
        theta = extract_data(net, 'data')

        # define regularizer
        reg = TikhonovRegularization(alpha=1e-2)

        # ------------------------------------------------------------ #
        # evaluate loss without regularization

        # zero gradient
        optimizer.zero_grad()

        # forward pass
        y_pred = net(x)

        # compute loss
        loss = loss_fn(y_pred, y)

        # compute gradient
        loss.backward()

        # extract gradient
        dtheta = extract_data(net, 'grad')

        # ------------------------------------------------------------ #
        # compete pass again with regularization
        # zero gradient
        optimizer.zero_grad()

        # forward pass
        y_pred = net(x)

        # compute loss + regularzation
        loss = loss_fn(y_pred, y) + reg(net)

        # compute gradient
        loss.backward()

        # extract gradient
        dtheta_reg = extract_data(net, 'grad')

        # ------------------------------------------------------------ #
        # check if gradient without reg is almost equal to gradient with reg
        self.assertTrue(torch.allclose(dtheta + reg.alpha * theta, dtheta_reg, rtol=1e-5, atol=1e-5))

