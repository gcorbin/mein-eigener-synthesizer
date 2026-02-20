# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
import numpy as np
from dataclasses import dataclass

from eigensynth.space import laplace_1d_eigen
from eigensynth.time import damped_oscillator_coefficients, damped_oscillator


def _hat_fun(x, L, x0):
    assert L > 0.
    assert x0 >= 0. and x0 <= L
    return np.where(x<x0,(L-x0)/L*x, x0*(L-x)/L)



@dataclass(init=True)
class StringOptions:
    base_frequency: float = 440  # Hz,
    halflife: float = 0.2  # seconds
    L: float = 1.  # meter
    Nk: int = 20
    Nx: int = 100
    pick_pos: float = 0.85


class String:
    def __init__(self,
                 options: StringOptions):
        self._options = options

        x = np.linspace(0., self.options.L, self.options.Nx)
        self.e_k, self.lam_k = self._eigen(x)

        U0 = _hat_fun(x, x0=self.options.pick_pos * self.options.L, L=self.options.L)  # Initial condition, evaluated at x
        self.u0_k = np.matmul(self.e_k, U0)  # Coefficients in expansion of initial condition U0 = sum_k u0_k * e_k(x)

    @property
    def options(self):
        return self._options

    def _eigen(self, x):
        return laplace_1d_eigen(x, N=self.options.Nk, L=self.options.L)

    def sound(self, t, x):
        ks, kd = damped_oscillator_coefficients(self.options.base_frequency, self.options.halflife)
        # Damped 1-D wave equation
        # u_tt   + kd * u_t   - c^2 * u_xx          = 0
        # Use Eigen-decomposition with d_xx u_k = lam u_k
        # u_k_tt + kd * u_k_t - c^2 * lam_k * u_k   = 0, k = 1,...,N
        c2 = ks / np.abs(self.lam_k[0])  # choose wave speed c such that the oscillator has the given base frequency
        u_k = damped_oscillator(
            np.atleast_2d(t).transpose(),
            -c2 * self.lam_k, kd,
            self.u0_k,
            0.)

        e_out, _ = self._eigen(x)
        return np.matmul(u_k, e_out)
