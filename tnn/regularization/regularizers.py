import torch
import torch.nn as nn


class BlockRegularization(nn.Sequential):
    def __init__(self, blocks):
        super(BlockRegularization, self).__init__()
        self.blocks = blocks

    def evaluate(self, net):
        reg = torch.zeros(1, requires_grad=True)
        for i in range(len(net)):
            reg = reg + self.blocks[i].evaluate(net[i])

        return reg


class TikhonovRegularization:
    def __init__(self, alpha=1e-4):
        super(TikhonovRegularization, self).__init__()
        self.alpha = alpha

    def evaluate(self, net: nn.Module):
        reg = torch.zeros(1, requires_grad=True)
        for p in net.parameters():
            reg = reg + torch.norm(p) ** 2

        return 0.5 * self.alpha * reg


class SmoothTimeRegularization:
    def __init__(self, alpha=1e-4):
        super(SmoothTimeRegularization, self).__init__()
        self.alpha = alpha

    def evaluate(self, net: nn.Module, bias=True):
        reg = torch.zeros(1, requires_grad=True)

        p_old_weight = torch.zeros(1, requires_grad=False)
        if bias:
            p_old_bias = torch.zeros(1, requires_grad=False)

        for i, p in enumerate(net.parameters()):
            if bias:
                if (i + 1) % 2:
                    reg = reg + torch.norm(p - p_old_weight)
                    p_old_weight = p
                else:
                    reg = reg + torch.norm(p - p_old_bias)
                    p_old_bias = p
            else:
                reg = reg + torch.norm(p - p_old_weight)
                p_old_weight = p

        return 0.5 * self.alpha * reg