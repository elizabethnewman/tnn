import torch


# create three-dimensional data with three classes based on distance to the origin
def setup_circle_3D(n=1000, R=5.5, r=3.5):
    x = 3 * torch.randn(n, 3)
    y = torch.zeros(n, dtype=torch.long)

    y[(x ** 2).sum(dim=1) < R ** 2] = 1
    y[(x ** 2).sum(dim=1) < r ** 2] = 2

    return x, y


def setup_circle_3D_grid(n=100, R=1.5, r=0.75, a=-3, b=3):
    xx, yy = torch.meshgrid(torch.linspace(a, b, n), torch.linspace(a, b, n), indexing='ij')
    zz = torch.zeros(xx.shape, dtype=torch.long)
    zz[(xx ** 2 + yy ** 2) < R ** 2] = 2
    zz[(xx ** 2 + yy ** 2) < r ** 2] = 1

    return xx, yy, zz


def split_data(x, y, n_train, n_val=None, n_test=None, shuffle=True):
    if shuffle:
        idx = torch.randperm(x.shape[0])
    else:
        idx = torch.arange(x.shape[0])

    x_val, y_val, x_test, y_test = None, None, None, None

    x_train, y_train = x[idx[:n_train]], y[idx[:n_train]]

    count = n_train
    if n_val is not None:
        x_val, y_val = x[idx[count:count + n_val]], y[idx[count:count + n_val]]

    count += n_val
    if n_test is not None:
        x_test, y_test = x[idx[count:count + n_test]], y[idx[count:count + n_test]]
    else:
        x_test, y_test = x[idx[count:]], y[idx[count:]]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


class DummyData(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

