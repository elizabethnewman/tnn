import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
import time
from tnn.training import optimizer_parameters, parameters_norm
from tnn.loss import tLoss
from copy import deepcopy


def train(net, criterion, optimizer, train_loader, val_loader, test_loader, scheduler=None, regularizer=None,
          max_epochs=10, device=None, dtype=None, logger=None, sPath='tmp/'):
    factory_kwargs = {'device': device, 'dtype': dtype}

    keys, opt_params = optimizer_parameters(optimizer)
    param_norm, grad_norm = parameters_norm(net)

    results = {'headers': ('',) * (4 + len(keys)) + ('running', '', '') + ('train', '') + ('valid', ''),
               'str': ('epoch',) + keys + ('|params|', '|grad|', 'time') +
                      ('obj', 'loss', 'acc', 'loss', 'acc', 'loss', 'acc'),
               'frmt': '{:<15d}' + len(keys) * '{:<15.4e}' + '{:<15.4e}{:<15.4e}{:<15.2f}' +
                       '{:<15.4e}{:<15.4e}{:<15.2f}{:<15.4e}{:<15.2f}{:<15.4e}{:<15.2f}',
               'val': None,
               'best_val_loss': torch.tensor(float('inf')).item(),
               'best_val_loss_epoch': -1,
               'best_val_acc': 0.0,
               'best_val_acc_epoch': -1,
               'total_time': 0.0}
    torch.save(net.state_dict(), sPath + '/best_val_loss_net.pt')
    torch.save(net.state_dict(), sPath + '/best_val_acc_net.pt')

    # initial evaluation
    train_out2 = test(net, criterion, train_loader, **factory_kwargs)
    test_out = test(net, criterion, test_loader, **factory_kwargs)
    his = [-1] + opt_params + [param_norm, grad_norm, 0, 0, 0, 0] + [*train_out2] + [*test_out]
    results['val'] = torch.tensor(his).view(1, -1)

    # store printouts for training
    if logger is not None:
        logger.info((len(results['headers']) * '{:<15s}').format(*results['headers']))
        logger.info((len(results['str']) * '{:<15s}').format(*results['str']))
        logger.info(results['frmt'].format(*his))

    total_start = time.time()
    for epoch in range(max_epochs):
        start = time.time()
        train_out = train_one_epoch(net, criterion, optimizer, train_loader, regularizer=regularizer, **factory_kwargs)
        end = time.time()

        # get overall loss
        train_out2 = test(net, criterion, train_loader, **factory_kwargs)
        val_out = test(net, criterion, val_loader, **factory_kwargs)

        # norm of network weights
        param_norm, grad_norm = parameters_norm(net)

        # store results
        his = [epoch]
        _, opt_params = optimizer_parameters(optimizer)
        his += opt_params
        his += [param_norm, grad_norm, end - start]
        his += [*train_out] + [*train_out2] + [*val_out]
        results['val'] = torch.cat((results['val'], torch.tensor(his).view(1, -1)), dim=0)

        if val_out[0] <= results['best_val_loss']:
            results['best_val_loss'] = deepcopy(val_out[0])
            results['best_val_loss_epoch'] = epoch
            torch.save(net.state_dict(), sPath + '/best_val_loss_net.pt')

        if val_out[1] >= results['best_val_acc']:
            results['best_val_acc'] = deepcopy(val_out[1])
            results['best_val_acc_epoch'] = epoch
            torch.save(net.state_dict(), sPath + '/best_val_acc_net.pt')

        # print outs for training
        if logger is not None:
            logger.info(results['frmt'].format(*his))

        # update learning rate
        if scheduler is not None:
            scheduler.step()

    total_end = time.time()
    results['total_time'] = total_end - total_start

    # performance on test data
    test_out = test(net, criterion, test_loader, **factory_kwargs)
    logger.info('Test performance:')
    logger.info('loss = {:<15.4e}'.format(test_out[0]))
    logger.info('accuracy = {:<15.4f}'.format(test_out[1]))

    return results


def train_one_epoch(model, criterion, optimizer, train_loader, regularizer=None, device=None, dtype=None):
    factory_kwargs = {'device': device, 'dtype': dtype}
    model.train()
    running_loss = 0
    running_obj = 0
    correct = 0
    num_samples = 0
    criterion.reduction = 'mean'

    for data, target in train_loader:
        data, target = data.to(**factory_kwargs), target.to(**factory_kwargs)

        optimizer.zero_grad()
        output = model(data)

        loss = criterion(output, data)

        running_loss += data.shape[0] * loss.item()
        num_samples += data.shape[0]

        if regularizer is not None:
            loss = loss + regularizer(model)
        running_obj += data.shape[0] * loss.item()

        loss.backward()
        optimizer.step()

    return running_obj / num_samples, running_loss / num_samples, 100 * correct / num_samples


def test(model: Module, criterion: Module, test_loader, device=None, dtype=None):
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

            loss = criterion(output, data)
            test_loss += loss.item()

    return test_loss / num_samples, 100. * correct / num_samples
