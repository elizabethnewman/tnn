import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
import time
from tnn.training import optimizer_parameters, parameters_norm


def train(net, criterion, optimizer, scheduler, train_loader, test_loader,
          max_epochs=10, verbose=True):

    keys, opt_params = optimizer_parameters(optimizer)

    results = {'str': ('epoch',) + keys +
                      ('|params|', '|grad|', 'time') +
                      ('running_loss', 'running_acc', 'train_loss', 'train_acc', 'test_loss', 'test_acc'),
               'frmt': '{:<15d}' + len(keys) * '{:<15.4e}' + '{:<15.4e}{:<15.4e}{:<15.2f}' +
                       '{:<15.4e}{:<15.2f}{:<15.4e}{:<15.2f}{:<15.4e}{:<15.2f}',
               'val': torch.empty(0)}

    # initial evaluation
    train_out2 = test(net, criterion, train_loader)
    test_out = test(net, criterion, test_loader)
    his = [-1] + (len(results['str']) - 5) * [0] + [*train_out2] + [*test_out]
    results['val'] = torch.cat((results['val'], torch.tensor(his).view(1, -1)), dim=0)

    # print outs for training
    if verbose:
        print((len(results['str']) * '{:<15s}').format(*results['str']))
        print(results['frmt'].format(*his))

    total_start = time.time()
    for epoch in range(max_epochs):
        start = time.time()
        train_out = train_one_epoch(net, criterion, optimizer, train_loader)
        end = time.time()

        # get overall loss
        train_out2 = test(net, criterion, train_loader)
        test_out = test(net, criterion, test_loader)

        # norm of network weights
        param_norm, grad_norm = parameters_norm(net)

        # store results
        his = [epoch]
        _, opt_params = optimizer_parameters(optimizer)
        his += opt_params
        his += [param_norm, grad_norm, end - start]
        his += [*train_out] + [*train_out2] + [*test_out]
        results['val'] = torch.cat((results['val'], torch.tensor(his).view(1, -1)), dim=0)

        # print outs for training
        if verbose:
            print(results['frmt'].format(*his))

        # update learning rate
        scheduler.step()

    total_end = time.time()
    print('Total training time = ', total_end - total_start)

    return results


def train_one_epoch(model, criterion, optimizer, train_loader):
    model.train()
    running_loss = 0
    correct = 0
    num_samples = 0
    criterion.reduction = 'mean'

    for data, target in train_loader:

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, target)
        running_loss += data.shape[0] * loss.item()
        num_samples += data.shape[0]

        pred = output.argmax(dim=1, keepdim=True).squeeze()  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

        loss.backward()
        optimizer.step()

    return running_loss / num_samples, 100 * correct / num_samples


def test(model: Module, criterion: Module, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    num_samples = 0

    criterion.reduction = 'sum'
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            num_samples += data.shape[0]

            test_loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True).squeeze()  # get the index of the max log-probability

            correct += pred.eq(target.view_as(pred)).sum().item()

    return test_loss / num_samples, 100. * correct / num_samples
