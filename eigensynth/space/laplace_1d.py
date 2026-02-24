# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
import numpy as np
from numpy.typing import NDArray

__all__ = ['laplace_1d_eigen']


def laplace_1d_eigen(x: NDArray | float, N: int, L: float = 1.):
    """
    Eigenfunctions and eigenvalues of the 1D Laplace operator ( d_xx u)
    on the interval [0,L] with zero Dirichlet boundary conditions, i.e.,

    e_k(x) such that d_xx e_k = lambda_k * e_k, k = 1, ..., N

    evaluated at x.

    These are given by

    e_k(x) = sqrt(L/2) * sin( k * pi / L * x),
    lambda_k = -(k * pi / L)**2

    where the factor sqrt(L/2) is chosen to normalize the eigenfunctions.

    :param x: Positions at which e_k are evaluated.
    :param N: Number of eigenvectors and eigenvalues
    :param L: Length of the domain
    :return: (e_k, lambda_k). e_k is an array with shape (x.size, N), lambda_k is a vector of length N
    """
    x = np.atleast_1d(x)
    assert x.ndim == 1
    k = np.linspace(1, N, N)
    arg = k * np.pi / L
    return np.sqrt(L / 2.) * np.sin(np.outer(x, arg)), -1. * arg ** 2
