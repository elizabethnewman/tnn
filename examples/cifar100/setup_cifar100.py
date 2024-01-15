import torch
from torchvision import datasets, transforms
import argparse


def setup_cifar100(n_train=50000, n_val=10000, n_test=10000, batch_size=64, data_dir='data'):
    # https://priyanshwarke2015-ndcs.medium.com/image-classification-with-cnn-model-in-cifar100-dataset-8d4122b75bad
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), inplace=True)])

    train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)

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


