# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
import abc
from abc import ABC, abstractmethod

__all__ = ['LinearDeformation']


class LinearDeformation(ABC):
    def __init__(self, L, N):
        self._L = L
        self._N = N

    @property
    def N(self):
        return self._N

    @property
    def L(self):
        return self._L

    @abstractmethod
    def grid(self, Nx):
        pass

    @property
    @abc.abstractmethod
    def wavenumbers(self):
        pass

    @abc.abstractmethod
    def indices(self, wavenumbers):
        pass

    @property
    @abc.abstractmethod
    def eigenvalues(self):
        pass

    @abc.abstractmethod
    def eigenmodes(self, x):
        pass

    def point_force_coefficients(self, x0):
        """
        Coefficients corresponding to a stationary loaded state

            Dx(u) = q = delta(x - x0)

        with a point (dirac delta) force delta(x - x0)

        Use the composition of u into eigenfunctions

            Dx(u)
            = Dx (sum u_l e_l)
            = sum Dx(u_l) e_l
            = sum lam_l u_l e_l = delta(x - x0)

        Scalar product with e_k, using orthonormality of e_k gives

            lam_k u_k = <delta(x - x0),e_k> = e_k(x0)
            u_k = 1 / lam_k * e_k(x0)
        """
        return 1. / self.eigenvalues * self.eigenmodes(x0).ravel()


    """    def compute_coefficients(self, fun):
        u = fun(self.x)  # Initial condition, evaluated at x
        # This is integration of the piecewise-constant reconstructions of U0 and e_k
        integration_weights = np.ones_like(u) * (self.options.L / (self.options.Nx - 1))
        integration_weights[0] *= 0.25
        integration_weights[-1] *= 0.25
        return np.matmul(self.e_k.transpose(), u * integration_weights)"""