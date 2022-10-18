import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
import time
from tnn.training import optimizer_parameters, parameters_norm
from tnn.loss import tLoss
from copy import deepcopy


def train(net, criterion, optimizer, train_loader, test_loader, scheduler=None, regularizer=None,
          max_epochs=10, verbose=True, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}

    keys, opt_params = optimizer_parameters(optimizer)
    param_norm, grad_norm = parameters_norm(net)

    results = {'headers': ('',) * (4 + len(keys)) + ('running', '') + ('train', '') + ('test', ''),
               'str': ('epoch',) + keys + ('|params|', '|grad|', 'time') +
                      ('loss', 'acc', 'loss', 'acc', 'loss', 'acc'),
               'frmt': '{:<15d}' + len(keys) * '{:<15.4e}' + '{:<15.4e}{:<15.4e}{:<15.2f}' +
                       '{:<15.4e}{:<15.2f}{:<15.4e}{:<15.2f}{:<15.4e}{:<15.2f}',
               'val': None,
               'best_val_loss': torch.tensor(float('inf')).item(),
               'best_val_loss_net': deepcopy(net),
               'best_val_acc': 0.0,
               'best_val_acc_net': deepcopy(net)}

    # initial evaluation
    train_out2 = test(net, criterion, train_loader)
    test_out = test(net, criterion, test_loader)
    his = [-1] + opt_params + [param_norm, grad_norm, 0, 0, 0] + [*train_out2] + [*test_out]
    results['val'] = torch.tensor(his).view(1, -1)

    # print outs for training
    if verbose:
        print((len(results['str']) * '{:<15s}').format(*results['headers']))
        print((len(results['str']) * '{:<15s}').format(*results['str']))
        print(results['frmt'].format(*his))

    total_start = time.time()
    for epoch in range(max_epochs):
        start = time.time()
        train_out = train_one_epoch(net, criterion, optimizer, train_loader, regularizer=regularizer, **factory_kwargs)
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

        if test_out[0] <= results['best_val_loss']:
            results['best_val_loss'] = deepcopy(test_out[0])
            results['best_val_loss_net'] = deepcopy(net)

        if test_out[1] >= results['best_val_acc']:
            results['best_val_acc'] = deepcopy(test_out[1])
            results['best_val_acc_net'] = deepcopy(net)

        # print outs for training
        if verbose:
            print(results['frmt'].format(*his))

        # update learning rate
        if scheduler is not None:
            scheduler.step()

    total_end = time.time()
    print('Total training time = ', total_end - total_start)

    return results


def train_one_epoch(model, criterion, optimizer, train_loader, regularizer=None, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    model.train()
    running_loss = 0
    correct = 0
    num_samples = 0
    criterion.reduction = 'mean'

    for data, target in train_loader:
        data, target = data.to(**factory_kwargs), target.to(**factory_kwargs)

        optimizer.zero_grad()
        output = model(data)

        if isinstance(criterion, tLoss):
            loss, target_pred = criterion(output, target)
            pred = target_pred.argmax(dim=1, keepdim=True).squeeze()
        else:
            loss = criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True).squeeze()

        running_loss += data.shape[0] * loss.item()
        num_samples += data.shape[0]

        if regularizer is not None:
            loss = loss + regularizer(model)

        correct += pred.eq(target.view_as(pred)).sum().item()

        loss.backward()
        optimizer.step()

    return running_loss / num_samples, 100 * correct / num_samples


def test(model: Module, criterion: Module, test_loader, device = None, dtype = None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    model.eval()
    test_loss = 0
    correct = 0
    num_samples = 0

    criterion.reduction = 'sum'
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(**factory_kwargs), target.to(**factory_kwargs)

            output = model(data)
            num_samples += data.shape[0]

            if isinstance(criterion, tLoss):
                loss, target_pred = criterion(output, target.to(**factory_kwargs))
                test_loss += loss.item()
                pred = target_pred.argmax(dim=1, keepdim=True).squeeze()
            else:
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True).squeeze()

            correct += pred.eq(target.view_as(pred)).sum().item()

    return test_loss / num_samples, 100. * correct / num_samples
