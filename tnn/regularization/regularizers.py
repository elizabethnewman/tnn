import torch
import torch.nn as nn
from copy import deepcopy


class BlockRegularization(nn.Sequential):
    def __init__(self, args):
        super(BlockRegularization, self).__init__(*args)


class TikhonovRegularization(nn.Module):
    def __init__(self, alpha=1e-4):
        super(TikhonovRegularization, self).__init__()
        self.alpha = alpha

    def forward(self, net: nn.Module):

        p0 = next(iter(net.parameters()))
        device, dtype = p0.device, p0.dtype

        reg = torch.zeros(1, requires_grad=True, device=device, dtype=dtype)
        for p in net.parameters():
            reg = reg + torch.norm(p) ** 2

        return 0.5 * self.alpha * reg


class SmoothTimeRegularization(nn.Module):
    def __init__(self, alpha=1e-4):
        super(SmoothTimeRegularization, self).__init__()
        self.alpha = alpha

    def forward(self, net: nn.Module, bias=True):

        p0 = next(iter(net.parameters()))
        device, dtype = p0.device, p0.dtype

        reg = torch.zeros(1, requires_grad=True, device=device, dtype=dtype)

        p_old_weight = torch.zeros(1, requires_grad=False, device=device, dtype=dtype)
        if bias:
            p_old_bias = torch.zeros(1, requires_grad=False, device=device, dtype=dtype)

        for i, p in enumerate(net.parameters()):
            if bias:
                if (i + 1) % 2:
                    reg = reg + torch.norm(p - p_old_weight)
                    p_old_weight = deepcopy(p)
                else:
                    reg = reg + torch.norm(p - p_old_bias)
                    p_old_bias = deepcopy(p)
            else:
                reg = reg + torch.norm(p - p_old_weight)
                p_old_weight = deepcopy(p)

        return 0.5 * self.alpha * reg
