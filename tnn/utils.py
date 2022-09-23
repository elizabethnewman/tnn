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
