import torch
from torchvision import datasets, transforms
import argparse


def setup_mnist(n_train=50000, n_val=10000, n_test=10000, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)

    # make the dataset smaller to test if code runs
    idx = torch.randperm(train_dataset.data.shape[0])

    train_dataset.data = train_dataset.data[idx[:n_train]]
    train_dataset.targets = train_dataset.targets[idx[:n_train]]

    val_dataset.data = val_dataset.data[idx[n_train:n_train + n_val]]
    val_dataset.targets = val_dataset.targets[idx[n_train:n_train + n_val]]

    test_dataset.data = test_dataset.data[:n_test]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_epochs', type=int, default=10, help='maximum number of epochs')
    parser.add_argument('--M', type=str, default='dct', help='transformation matrix')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--n_train', type=int, default=1000, help='number of training points')
    parser.add_argument('--n_val', type=int, default=100, help='number of validation points')
    parser.add_argument('--n_test', type=int, default=100, help='number of test points')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--gamma', type=float, default=1, help='decay rate for scheduler')
    parser.add_argument('--step_size', type=float, default=100, help='step size for scheduler')
    parser.add_argument('--width', type=int, default=0, help='width of network')
    parser.add_argument('--loss', type=str, default='t_cross_entropy',
                        help='loss function for tensor network (t_cross_entropy or cross_entropy)')
    parser.add_argument('--bias', type=bool, default=True, help='bias for network')
    parser.add_argument('--matrix_match_tensor', action='store_true',
                        help='match number of matrix weights to tensor based on width')
    return parser
