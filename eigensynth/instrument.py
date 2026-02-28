# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
import numpy as np

from eigensynth.space.linear_deformation import LinearDeformation
from eigensynth.time import damped_oscillator_coefficients, damped_oscillator, oscillator_frequency

__all__ = ['Instrument']


class Instrument:
    """
    Damped wave equation

        u_tt   + kd * u_t   - c^2 * Dx(u) = 0

    Use Eigen-decomposition of Dx: Dx(u_k) = lam_k u_k

        u_k_tt + kd * u_k_t - c^2 * lam_k * u_k  = 0

    """
    def __init__(self,
                 oscillator: LinearDeformation,
                 base_frequency: float = 440.,
                 halflife: float = 0.2):
        self.oscillator = oscillator
        self.base_frequency = base_frequency
        self.halflife = halflife

        # choose wave speed c such that the oscillator has the given base frequency
        eigenvalues = self.oscillator.eigenvalues
        ks, kd = damped_oscillator_coefficients(self.base_frequency, self.halflife)
        self._c2 = ks / np.min(np.abs(eigenvalues))
        self._stiffness = -1. * ks / np.min(np.abs(eigenvalues)) * eigenvalues
        self._damping = kd

    def solution(self, t, x, x0):
        t = np.atleast_1d(np.array(t))
        assert t.ndim == 1

        u_k = damped_oscillator(
            t=t,
            k=self._stiffness,
            d=self._damping,
            x0=self.initial_coefficients(x0),
            dx0=0.)

        return np.inner(u_k, self.oscillator.eigenmodes(x))

    def sound(self, t, x, x0):
        t = np.atleast_1d(np.array(t))
        return np.sum(self.solution(t, x, x0).reshape(t.size, -1), axis=-1)

    def initial_coefficients(self, x0):
        return self.oscillator.point_force_coefficients(x0)

    @property
    def frequencies(self):
        return oscillator_frequency(self._stiffness)
