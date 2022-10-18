import torch


def modek_product(A, M, k=None, transpose=False, conjugate=False, inverse=False):
    r"""
    Mode-k product
    :param A:
    :type A:
    :param M:
    :type M:
    :param k:
    :type k:
    :param transpose:
    :type transpose:
    :param conjugate:
    :type conjugate:
    :param inverse:
    :type inverse:
    :return:
    :rtype:
    """

    if M is None:
        return A
    else:
        if k is None:
            k = A.ndim - 1

        assert A.shape[k] == M.shape[1]

        # apply M to tubes of A (note that the transpose is reversed because we apply M on the right)
        if transpose or inverse:
            if inverse:
                A_hat = torch.moveaxis(torch.linalg.solve(M, torch.moveaxis(A, k, -2)), -2, -1)
            else:
                if conjugate:
                    A_hat = torch.moveaxis(A, k, -1) @ torch.conj(M)
                else:
                    A_hat = torch.moveaxis(A, k, -1) @ M
        else:
            A_hat = torch.moveaxis(A, k, -1) @ M.T

        # return to original orientation
        A_hat = torch.moveaxis(A_hat, -1, k)

        return A_hat


def modek_unfold(A, k):
    # A is a tensor
    A = torch.moveaxis(A, k, 0)
    return torch.reshape(A, (A.shape[0], -1))


def modek_fold(A, k, shape_A):
    # A is a matrix
    # shape_A is the shape of A before unfolding, as a tensor

    if isinstance(shape_A, list):
        shape_A = tuple(shape_A)

    A = torch.reshape(A, (-1,) + shape_A[:k] + shape_A[k+1:])

    return torch.moveaxis(A, 0, k)
