import torch


def optimizer_keys(optimizer):
    keys = list(optimizer.param_groups[0].keys())
    keys = tuple(keys[1:])

    names = ()
    for name in keys:
        param = optimizer.param_groups[0][name]
        if isinstance(param, tuple):
            names += len(param) * (name,)
        else:
            names += (name,)

    return names


def optimizer_parameters(optimizer):
    keys = list(optimizer.param_groups[0].keys())
    keys = tuple(keys[1:])

    his = []
    names = ()
    for name in keys:
        param = optimizer.param_groups[0][name]
        if isinstance(param, tuple):
            names += len(param) * (name,)
            his += [*param]
        else:
            names += (name,)
            his += [param]

    return names, his


def parameters_norm(net: torch.nn.Module):
    with torch.no_grad():
        param_norm = torch.zeros(1)
        grad_norm = torch.zeros(1)
        for p in net.parameters():
            param_norm += torch.sum(p.data ** 2)
            grad_norm += torch.sum(p.grad ** 2)

        return torch.sqrt(param_norm).item(), torch.sqrt(grad_norm).item()
