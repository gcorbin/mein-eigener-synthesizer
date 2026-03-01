# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import root_scalar

__all__ = ['Beam']

from eigensynth.space.linear_deformation import LinearDeformation


class Beam(LinearDeformation):
    def __init__(self, L, N):
        """
        Modes of the (homogeneous) cantilevered beam, i.e. eigenvalues and eigenfunctions of

            -d_xxxx u = 0

        on the interval [0,L],
        with clamping boundary condition at x=0: u(0) = d_x u(0) = d_xx u(0) = d_xxx u(0) = 0.

        See https://en.wikipedia.org/wiki/Euler%E2%80%93Bernoulli_beam_theory#Dynamic_beam_equation
        for details.
        """
        super().__init__(L,N)
        self._beta = _roots_cosh_cos_plus_1(self.N) / self.L

    def grid(self, Nx: int):
        return np.linspace(0., self.L, Nx)

    @property
    def K(self):
        return self.N

    @property
    def wavenumbers(self):
        return np.linspace(1, self.N, self.N, dtype=int)

    def indices(self, wavenumbers):
        n = np.atleast_1d(wavenumbers)
        assert np.all(0 < n <= self.N)
        return n - 1

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


def _roots_cosh_cos_plus_1(N: int):
    """
    Find the first N roots of cosh(x_k) * cos(x_k) + 1 = 0

    :param N: Number of roots
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
