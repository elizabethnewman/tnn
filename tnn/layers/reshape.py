import torch.nn as nn


class View(nn.Module):
    def __init__(self, shape_out):
        super(View, self).__init__()
        self.shape_out = shape_out

    def forward(self, x):
        return x.view(self.shape_out)


class Permute(nn.Module):
    def __init__(self, dims=(1, 0, 2)):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.squeeze().permute(*self.dims).contiguous()
