import torch.nn as nn


class tSequential(nn.Sequential):
    def __init__(self, *args, M=None):
        super(tSequential, self).__init__(*args)
        self.M = M

    def forward(self, input):
        for module in self:
            if hasattr(module, 'M'):
                input = module(input, self.M)
            else:
                input = module(input)
        return input

