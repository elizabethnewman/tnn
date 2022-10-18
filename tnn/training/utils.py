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
        if param is None:
            param = 0

        if isinstance(param, tuple):
            names += len(param) * (name,)
            his += [*param]
        else:
            names += (name,)
            his += [param]

    return names, his


def parameters_norm(net: torch.nn.Module):
    with torch.no_grad():
        p0 = next(iter(net.parameters()))
        device, dtype = p0.device, p0.dtype

        param_norm = torch.zeros(1, requires_grad=True, device=device, dtype=dtype)
        grad_norm = torch.zeros(1, requires_grad=True, device=device, dtype=dtype)
        for p in net.parameters():
            param_norm += torch.sum(p.data ** 2)

            if p.grad is not None:
                grad_norm += torch.sum(p.grad ** 2)

        if grad_norm is None:
            grad_norm = torch.zeros(1)

        return torch.sqrt(param_norm).item(), torch.sqrt(grad_norm).item()
