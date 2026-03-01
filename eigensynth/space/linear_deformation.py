# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
import abc
from abc import ABC, abstractmethod

__all__ = ['LinearDeformation']


class LinearDeformation(ABC):
    def __init__(self, L, N):
        """
        Represents a linear differential operator D(u) by its eigenvalues and eigenmodes.

        The operator acts on the spatial components of u.
        Eigenvalues and modes are defined by the operator D, its domain Omega, and boundary conditions.
        Eigenmodes must be pairwise orthogonal. All eigenvalues must be strictly negative:

        D(e_k) = lam_k e_k, k = 0,...,K

        lam_k < 0
        <e_k, e_k> = 1
        <e_k, e_l> = 0, if k != l

        The ordering of eigenvalues and modes is defined in 'wavenumbers' and 'indices'.

        :param L: Spatial parameters of the operator's domain. In 1D this is typically the upper bound of the interval Omega=[0,L]
        :param N: Order of the decomposition. The number of eigenfunctions and eigenvalues is in general not equal to N:
                  Multiple modes may exist for each order. For operators in more than one space dimension, N itself is a
                  multi-index.
        """
        self._L = L
        self._N = N

    @property
    def N(self):
        """
        Maximum order of the eigendecomposition.
        """
        return self._N

    @property
    def L(self):
        """
        Spatial domain parameters.
        """
        return self._L

    @property
    @abc.abstractmethod
    def K(self):
        """
        Number of eigenvalues and modes.
        """
        pass

    @abstractmethod
    def grid(self, Nx):
        pass

    @property
    @abc.abstractmethod
    def wavenumbers(self):
        """
        Each eigenmode is defined by its wavenumber, which is in general a multi-index (m,n,...)
        Eigenvalues are returned as flat arrays, indexed by a flat index k.

        This function returns the wavenumbers in the same flat order as the eigenvalues:
        wavenumbers[k,:] = (m,n,...).

        Note: The dimension of the wavenumber multi-index does not have to coincide with the space dimension.
              For instance, the wavenumber can have an additional component describing parity.

        :return: Array of int.
        """
        pass

    @abc.abstractmethod
    def indices(self, wavenumbers):
        """
        The inversion of wavenumbers, i.e. given an array of wavenumbers (m,n,...) as multi-indices,
        compute the flat indices k.

        Note: self.indices(self.wavenumbers) should therefore be equivalent to np.arange(K)

        :param wavenumbers: Array of int. Each row wavenumbers[i,:] is a wavenumber.
        :return: 1D Array of int.
        """
        pass

    @property
    @abc.abstractmethod
    def eigenvalues(self):
        """
        The eigenvalues lam_k, k = 0, ...., K-1 of the operator D
        :return: 1D array of float.
        """
        pass

    @abc.abstractmethod
    def eigenmodes(self, x):
        """
        The Eigenmodes e_k, k = 0,...,K-1 of the operator D, evaluated at x.

        :param x: Array of positions, shape (Nx, dimX)
        :return: Array e_k.ndim = dimX + 1. The last axis in the array is the flat index k.
        """
        pass

    def point_force_coefficients(self, x0):
        """
        Coefficients corresponding to a stationary loaded state

            D(u) = q = delta(x - x0)

        with a point (dirac delta) force delta(x - x0)

        Use the composition of u into eigenfunctions

            D(u)
            = D (sum u_l e_l)
            = sum D(u_l) e_l
            = sum lam_l u_l e_l = delta(x - x0)

        Scalar product with e_k, using orthonormality of e_k gives

            lam_k u_k = <delta(x - x0),e_k> = e_k(x0)
            u_k = 1 / lam_k * e_k(x0)

        :param x0: The position where the force is applied.
        :return: 1D Array of size K.
        """
        return 1. / self.eigenvalues * self.eigenmodes(x0).ravel()


    """    def compute_coefficients(self, fun):
        u = fun(self.x)  # Initial condition, evaluated at x
        # This is integration of the piecewise-constant reconstructions of U0 and e_k
        integration_weights = np.ones_like(u) * (self.options.L / (self.options.Nx - 1))
        integration_weights[0] *= 0.25
        integration_weights[-1] *= 0.25
        return np.matmul(self.e_k.transpose(), u * integration_weights)"""