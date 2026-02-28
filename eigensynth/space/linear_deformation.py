# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
import abc
from abc import ABC, abstractmethod


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


