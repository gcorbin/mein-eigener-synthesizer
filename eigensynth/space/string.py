# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
import numpy as np
from numpy.typing import NDArray

__all__ = ['String']

from eigensynth.space.linear_deformation import LinearDeformation


class String(LinearDeformation):
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

    def grid(self, Nx: int):
        return np.linspace(0., self.L, Nx)

    @property
    def wavenumbers(self):
        return np.linspace(1, self.N, self.N, dtype=int)

    def indices(self, wavenumbers):
        n = np.atleast_1d(wavenumbers)
        assert np.all(0 < n <= self.N)
        return n - 1

    @property
    def eigenvalues(self):
        arg = self.wavenumbers * np.pi / self.L
        return -1. * arg ** 2

    def eigenmodes(self, x):
        x = np.atleast_1d(x)
        assert x.ndim == 1
        arg = self.wavenumbers * np.pi / self.L
        return np.sqrt(self.L / 2.) * np.sin(np.outer(x, arg))
