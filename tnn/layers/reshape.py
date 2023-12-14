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
        if x.ndim > len(self.dims):
            x = x.squeeze()
        return x.permute(*self.dims).contiguous()


class Unfold(nn.Module):
    """
    Unfold a tensor such that columns are vectorized lateral slices
    """
    def __init__(self):
        super(Unfold, self).__init__()

    def forward(self, x):

        if x.ndim < 3:
            return x
        else:
            # make samples last dimension
            x = x.permute(0, 2, 1).contiguous()

            return x.view(-1, x.shape[-1])


class Fold(nn.Module):
    """
    Unfold a tensor such that columns are vectorized lateral slices
    """
    def __init__(self, shape_out):
        super(Fold, self).__init__()
        self.shape_out = shape_out  # shape of original tensor

    def forward(self, x):

        x = x.view(self.shape_out[1], self.shape_out[2], self.shape_out[1])
        return x.permute(0, 2, 1).contiguous()
