import torch
import numpy as np
import random


def seed_everything(seed):
    # option to add numpy, random, etc.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)


def number_network_weights(net):
    n = 0
    for p in net.parameters():
        n += p.numel()

    return n

import torch
from torch import Tensor
from torch.nn import Module
from typing import Tuple, List


def module_getattr(obj: Module, names: Tuple or List or str):
    r"""
    Get specific attribute of module at any level
    """
    if isinstance(names, str) or len(names) == 1:
        if len(names) == 1:
            names = names[0]

        return getattr(obj, names)
    else:
        return module_getattr(getattr(obj, names[0]), names[1:])


def module_setattr(obj: Module, names: Tuple or List, val: Tensor):
    r"""
    Set specific attribute of module at any level
    """
    if isinstance(names, str) or len(names) == 1:
        if len(names) == 1:
            names = names[0]

        return setattr(obj, names, val)
    else:
        return module_setattr(getattr(obj, names[0]), names[1:], val)


def extract_data(net: Module, attr: str = 'data') -> (Tensor, Tuple, Tuple):
    """
    Extract data stored in specific attribute and store as 1D array
    """
    theta = torch.empty(0)
    for name, w in net.named_parameters():
        if getattr(w, attr) is None:
            w = torch.zeros_like(w.data)
        else:
            w = getattr(w, attr)

        theta = torch.cat((theta, w.reshape(-1)))

    return theta


def insert_data(net: Module, theta: Tensor) -> None:
    """
    Insert 1D array of data into specific attribute
    """
    count = 0
    for name, w in net.named_parameters():
        name_split = name.split('.')
        n = w.numel()
        module_setattr(net, name_split + ['data'], theta[count:count + n].reshape(w.shape))
        count += n

