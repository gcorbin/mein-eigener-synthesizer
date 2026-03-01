# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
import numpy as np

from eigensynth.space.linear_deformation import LinearDeformation
from eigensynth.time import damped_oscillator_coefficients, damped_oscillator, oscillator_frequency

__all__ = ['Instrument']


class Instrument:
    def __init__(self,
                 oscillator: LinearDeformation,
                 base_frequency: float = 440.,
                 halflife: float = 0.2):
        """
        Solve the damped wave equation

            d_tt u   + kd * d_t u - c^2 * Dx(u) = 0

        using the Eigen-decomposition of the differential operator Dx: Dx(u_k) = lam_k u_k

            d_tt u_k + kd *  d_t u_k - c^2 * lam_k * u_k  = 0

        The instrument is tuned to the base_frequency with the stiffness parameter c^2:
        c is chosen such that c * min(sqrt(-lam_k)) * 2 * pi is the base frequency.

        The damping coefficient kd is chosen such that the amplitude decays with a given halflife.

        :param oscillator: Provides eigenvalues and modes of the differential operator
        :param base_frequency: Tune the instrument such that this is the lowest frequency.
        :param halflife: Damp the instrument such that the amplitude decays with this halflife.
        """
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
        """
        Evaluate the solution of the wave equation at times t and positions x,
        starting from an initial loaded state where a point force is applied
        to the spacial operator at forcing position x0.

        :param t: 1D array of float, or a scalar float.
        :param x: Compatible as input for oscillator.eigenmodes(x)
        :param x0: Compatible as input for oscillator.point_force_coefficients(x0)
        :return:
        """
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
        """
        Evaluate the solution at all given coordinates x, and sum over coordinates.

        :param t: array of nT floats: sampling times
        :param x: Compatible as input for oscillator.eigenmodes(x)
        :param x0: Compatible as input for oscillator.point_force_coefficients(x0)
        :return: 1D array of nT floats.
        """
        t = np.atleast_1d(np.array(t))
        return np.sum(self.solution(t, x, x0).reshape(t.size, -1), axis=-1)

    def initial_coefficients(self, x0):
        """
        Initial coefficients u_k(t=0) for a loaded state with point force
        at x0.
        :param x0: Compatible as input for oscillator.point_force_coefficients(x0)
        :return: 1D array size K
        """
        return self.oscillator.point_force_coefficients(x0)

    @property
    def frequencies(self):
        """
        Frequencies of the instrument.
        :return: 1D array size K
        """
        return oscillator_frequency(self._stiffness)
