# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
from abc import ABC, abstractmethod

import numpy as np
from dataclasses import dataclass

from eigensynth.time import damped_oscillator_coefficients, damped_oscillator

__all__ = ['InstrumentOptions', 'Instrument']

@dataclass(init=True)
class InstrumentOptions:
    base_frequency: float = 440  # Hz,
    halflife: float = 0.2  # seconds
    L: float = 1.  # meter
    Nk: int = 20
    Nx: int = 100
    pick_pos: float = 0.8  # relative to length


class Instrument(ABC):
    def __init__(self,
                 options: InstrumentOptions):
        self._options = options

        self.x = np.linspace(0., self.options.L, self.options.Nx)
        self.e_k, self.lam_k = self.compute_eigen(self.x)
        self.u0_k = self._compute_initial_coefficients()  # Coefficients in expansion of initial condition U0 = sum_k u0_k * e_k(x)

    @property
    def options(self):
        return self._options

    @abstractmethod
    def compute_eigen(self, x):
        return np.zeros((self.options.Nk, x.size)), np.ones(self.options.Nk)

    def _compute_initial_coefficients(self):
        """
        Initial condition corresponding to a stationary loaded state

            -c^2 * Dx(u0) = q0
            Dx(u0) = - q0 / c^2 = delta(x - x0)

        with a point (dirac delta) force - c^2 * delta(x - x0)

        Use the composition of u0 into eigenfunctions

            Dx(u0)
            = Dx (sum u0_l e_l)
            = sum Dx(u0_l) e_l
            = sum lam_l u0_l e_l = delta(x - x0)

        Scalar product with e_k, using orthonormality of e_k gives

            lam_k u0_k = <delta(x - x0),e_k> = e_k(x0)
            u0_k = 1 / lam_k * e_k(x0)
        """
        x0 = np.array([self.options.pick_pos * self.options.L])
        e_k_L, lam_k = self.compute_eigen(x0)
        return 1. / lam_k * e_k_L.reshape((-1,))

    @property
    def initial_coefficients(self):
        return self.u0_k

    def sound(self, t, x):
        ks, kd = damped_oscillator_coefficients(self.options.base_frequency, self.options.halflife)
        # Damped 1-D wave equation
        # u_tt   + kd * u_t   - c^2 * Dx(u) = 0
        # Use Eigen-decomposition of Dx: Dx(u_k) = lam u_k
        # u_k_tt + kd * u_k_t - c^2 * lam_k * u_k  = 0, k = 1,...,N

        # choose wave speed c such that the oscillator has the given base frequency
        # assume eigenvalues are sorted in descending magnitude
        c2 = ks / np.abs(self.lam_k[0])
        u_k = damped_oscillator(
            t=t.reshape(-1, 1),
            k=-c2 * self.lam_k,
            d=kd,
            x0=self.u0_k,
            dx0=0.)

        e_out, _ = self.compute_eigen(x)
        return np.matmul(u_k, e_out)
