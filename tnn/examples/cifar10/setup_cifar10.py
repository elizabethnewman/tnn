import torch
from torchvision import datasets, transforms
import argparse


def setup_cifar10(n_train=40000, n_val=10000, n_test=10000, batch_size=32, data_dir='data'):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    # make the dataset smaller to test if code runs
    idx = torch.randperm(train_dataset.data.shape[0])

    train_dataset.data = train_dataset.data[idx[:n_train]]
    train_dataset.targets = list(train_dataset.targets[i] for i in idx[:n_train])

    val_dataset.data = val_dataset.data[idx[n_train:n_train + n_val]]
    val_dataset.targets = list(val_dataset.targets[i] for i in idx[n_train:n_train + n_val])

    test_dataset.data = test_dataset.data[:n_test]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader


# def setup_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--seed', type=int, default=42, help='random seed')
#     parser.add_argument('--max_epochs', type=int, default=10, help='maximum number of epochs')
#     parser.add_argument('--M', type=str, default='dct', help='transformation matrix')
#     parser.add_argument('--batch_size', type=int, default=32, help='batch size')
#     parser.add_argument('--n_train', type=int, default=1000, help='number of training points')
#     parser.add_argument('--n_val', type=int, default=100, help='number of validation points')
#     parser.add_argument('--n_test', type=int, default=100, help='number of test points')
#     parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
#     parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
#     parser.add_argument('--gamma', type=float, default=1, help='decay rate for scheduler')
#     parser.add_argument('--step_size', type=float, default=100, help='step size for scheduler')
#     parser.add_argument('--width', type=int, default=100, help='width of network')
#     parser.add_argument('--depth', type=int, default=4, help='depth of network')
#     parser.add_argument('--h_step', type=float, default=0.1, help='number of steps in Hamiltonian')
#     parser.add_argument('--alpha', type=float, default=0.0, help='regularization parameter')
#     parser.add_argument('--add_width_hamiltonian', type=int, default=0, help='additional width of Hamiltonian network')
#     parser.add_argument('--opening_layer', action='store_true', help='opening linear layer')
#     parser.add_argument('--loss', type=str, default='cross_entropy')
#     return parser
