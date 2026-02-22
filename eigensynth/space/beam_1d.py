# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
import numpy as np
from scipy.optimize import root_scalar

__all__ = ['cantilevered_beam_eigen']


def cantilevered_beam_eigen(x, N, L=1.):
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
    :return: Modes and eigenvalues (e_k, lambda_k). e_k is an array with shape (N, x.size), lambda_k is a vector of length N
    """
    beta_k = _roots_cosh_cos_plus_1(N) / L
    lam_k = -1. * np.power(beta_k, 4.)

    arg = beta_k.reshape(-1, 1) * np.atleast_2d(x)
    betaL = beta_k.reshape(-1,1) * L
    e_k = ( np.cosh(arg) - np.cos(arg) ) + (np.cos(betaL) + np.cosh(betaL)) / (np.sin(betaL) + np.sinh(betaL)) * (np.sin(arg) - np.sinh(arg))

    return e_k, lam_k


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
    roots = (np.linspace(0, N-1, N) + 0.5 ) * np.pi
    if N > 0:
        roots[0] = 0.6 * np.pi
    for k in range(N):
        guess = roots[k]
        sol = root_scalar(f, method='brentq', bracket=(guess - 0.05, guess + 0.05), x0=guess, rtol=1e-15)
        #print(f'k = {k}, root = {sol.root/np.pi} * pi, converged = {sol.converged}, iterations = {sol.iterations}, function_calls = {sol.function_calls}')
        assert sol.converged
        roots[k] = sol.root
        # Starting with this root, all roots are so close to the guess ( k + 0.5 ) * pi
        # that we just take the initial guess as solution
        if np.abs(sol.root - guess) <= 1e-15 * guess:
            break

    return roots

