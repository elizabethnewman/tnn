from torch.nn.modules.loss import _Loss
import torch.nn
import torch.nn.functional as F
from tnn.tensor_utils import modek_product


class tLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(tLoss, self).__init__(size_average, reduce, reduction)


class tMSELoss(tLoss):

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(tMSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        input = input.permute(1, 0, 2)
        return F.mse_loss(input.reshape(input.shape[0], -1), target, reduction=self.reduction)


class tCrossEntropyLoss(tLoss):
    __constants__ = ['ignore_index', 'reduction']
    ignore_index: int

    def __init__(self, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', M=None):
        super(tCrossEntropyLoss, self).__init__(size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.M = M

    def forward(self, input, target, M=None):
        if M is None:
            M = self.M
        input_hat = t_log_softmax(input, M, return_spatial=False)

        # tube with the smallest average norm (in absolute value) should be the target
        neg_input_nrm = -torch.norm(input_hat, dim=2).t() / input_hat.shape[2]

        # val = F.nll_loss(neg_input_nrm, target, ignore_index=self.ignore_index, reduction=self.reduction)

        val = F.nll_loss(input_hat[..., 0].t(), target, ignore_index=self.ignore_index, reduction=self.reduction)
        for i in range(1, input_hat.shape[-1]):
            val = val + F.nll_loss(input_hat[..., i].t(), target, ignore_index=self.ignore_index, reduction=self.reduction)

        val = val / input_hat.shape[-1]
        return val, neg_input_nrm


class tLogSoftmax(tLoss):

    def __init__(self, return_spatial=False) -> None:
        super(tLogSoftmax, self).__init__()
        self.return_spatial = return_spatial

    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, 'dim'):
            self.dim = None

    def forward(self, input, M):

        return t_log_softmax(input, M, self.return_spatial)


def t_log_softmax(input, M, return_spatial=False):
    input_hat = modek_product(input, M)

    input_hat = F.log_softmax(input_hat, dim=0)

    if return_spatial:
        input_hat = modek_product(input, M.t())

    return input_hat


def t_softmax(input, M, return_spatial=False):
    input_hat = modek_product(input, M)

    input_hat = F.softmax(input_hat, dim=0)

    if return_spatial:
        input_hat = modek_product(input, M.t())

    return input_hat


if __name__ == "__main__":
    from tnn.layers import tLinearLayer, Permute
    din, dim3 = 2, 3
    batch_size, n_classes = 11, 10
    M = torch.eye(dim3)
    sample, target = torch.rand(batch_size, din, dim3), torch.randint(0, n_classes, (batch_size,))
    net = torch.nn.Sequential(Permute((1, 0, 2)),
                              tLinearLayer(din, n_classes, 3, M=M)
                              )

    loss = tCrossEntropyLoss()

    y = net(sample)
    # y = loss(net(sample), target)

