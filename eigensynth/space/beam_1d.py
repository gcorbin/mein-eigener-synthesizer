# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar

__all__ = ['cantilevered_beam_eigen']

from eigensynth.space.linear_deformation import LinearDeformation


class Beam1D(LinearDeformation):
    """
    Modes of the cantilevered beam.
    Compute the eigenfunctions and eigenvalues of

        -d_xxxx u = 0

    on the interval [0,L],
    with BC u(0) = d_x u(0) = d_xx u(0) = d_xxx u(0) = 0,
    i.e. e_k such that -d_xxxx e_k = lambda_k * e_k, k = 1,...N.

    See https://en.wikipedia.org/wiki/Euler%E2%80%93Bernoulli_beam_theory#Dynamic_beam_equation
    for details.
    The modes e_k are orthogonal and normalized.

    :param x: Positions at which the modes are evaluated.
    :param N: Number of modes
    :param L: Length of the domain
    :return: Modes and eigenvalues (e_k, lambda_k). e_k is an array with shape (x.size, N), lambda_k is a vector of length N
    """
    def __init__(self, L, N):
        super().__init__(L,N)
        self._beta = _roots_cosh_cos_plus_1(self.N) / self.L

    def grid(self, Nx: int):
        return np.linspace(0., self.L, Nx)

    @property
    def wavenumbers(self):
        return np.linspace(1, self.N, self.N)

    @property
    def eigenvalues(self):
        return -1. * np.power(self._beta, 4.)

    def eigenmodes(self, x):
        x = np.atleast_1d(x)
        assert x.ndim == 1

        arg = np.outer(x, self._beta)
        betaL = self._beta * self.L
        return ((np.cosh(arg) - np.cos(arg))
                + (np.cos(betaL) + np.cosh(betaL)) / (np.sin(betaL) + np.sinh(betaL)) * (np.sin(arg) - np.sinh(arg)))


def cantilevered_beam_eigen(x: NDArray | float, N: int, L: float = 1.):
    beam = Beam1D(L, N)
    return beam.eigenmodes(x), beam.eigenvalues


def _roots_cosh_cos_plus_1(N):
    """
    Find the first N roots of cosh(x_k) * cos(x_k) + 1 = 0

    :param N:
    :return: Array containing the roots
    """

    def f(x):
        return np.cosh(x) * np.cos(x) + 1.

    # Use that, as k increases, x_k converges to ( k + 0.5 ) * pi
    # And x_0 is at roughly 0.6 pi
    # Initialize roots with our guess
    roots = (np.linspace(0, N - 1, N) + 0.5) * np.pi
    if N > 0:
        roots[0] = 0.6 * np.pi
    for k in range(N):
        guess = roots[k]
        sol = root_scalar(f, method='brentq', bracket=(guess - 0.05, guess + 0.05), x0=guess, rtol=1e-15)
        # print(f'k = {k}, root = {sol.root/np.pi} * pi, converged = {sol.converged}, iterations = {sol.iterations}, function_calls = {sol.function_calls}')
        assert sol.converged
        roots[k] = sol.root
        # Starting with this root, all roots are so close to the guess ( k + 0.5 ) * pi
        # that we just take the initial guess as solution
        if np.abs(sol.root - guess) <= 1e-15 * guess:
            break

    return roots
