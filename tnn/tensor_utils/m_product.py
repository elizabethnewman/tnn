import torch
from tnn.tensor_utils import modek_product


def facewise(A, B):
    r"""
    Batch multiply frontal slices of tensors (first two dimensions)
    :param A: (n_1, p, n_3, ..., n_d) array
    :type A: torch.Tensor
    :param B: (p, n_2, n_3, ..., n_d) array
    :type B: torch.Tensor
    :return: (n_1, n_2, n_3, ..., n_d) array
    :rtype: torch.Tensor
    """
    # reorient for batch multiplication
    A = torch.moveaxis(A, (0, 1), (-2, -1))
    B = torch.moveaxis(B, (0, 1), (-2, -1))

    # batch matrix multiplication
    C = A @ B

    # return to original orientation
    C = torch.moveaxis(C, (-2, -1), (-2, -1))

    return C


def mprod(A, B, M, transpose=False, conjugate=False, inverse=False):
    modek_args = {'transpose': transpose, 'conjugate': conjugate, 'inverse': inverse}

    A = modek_product(A, M)
    B = modek_product(B, M)
    C = facewise(A, B)
    C = modek_product(C, M, **modek_args)
    return C


def mtran(A):
    return A.transpose(0, 1)


def t_eye(n, M, transpose=True, conjugate=False, inverse=False):
    modek_args = {'transpose': transpose, 'conjugate': conjugate, 'inverse': inverse}
    args = {'dtype': M.dtype, 'device': M.device}

    # faster on GPU
    I = torch.eye(n, **args).unsqueeze(-1) * torch.ones(1, 1, M.shape[1], **args)

    return modek_product(I, **modek_args)