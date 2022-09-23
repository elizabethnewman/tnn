import torch
import numpy as np
import scipy.fftpack


def random_orthogonal(dim3, dtype=None, device=None):
    q, _ = torch.qr(torch.randn(dim3, dim3))
    q = q.to(dtype=dtype, device=device)
    return q


def dct_matrix(dim, dtype=None, device=None):
    """
    Form orthogonal dct matrix for transformations
    :param dim: size of transformation matrix
    :return: dim x dim orthogonal transformation matrix
    """

    C = scipy.fftpack.dct(np.eye(dim), norm="ortho")
    C = np.transpose(C)

    return torch.tensor(C, dtype=dtype, device=device)
