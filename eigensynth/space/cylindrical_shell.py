# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0

import numpy as np
from numpy.typing import NDArray

from eigensynth.array import outer_product_nd

__all__ = ['CylindricalShell']

from eigensynth.space.linear_deformation import LinearDeformation


class CylindricalShell(LinearDeformation):
    """
        Compute eigenfunctions and eigenvalues of the cylindrical thin shell

        See https://doi.org/10.1121/2.0000945 for details.

        :param x: cylindrical coordinates Z, Theta as output of np.meshgrid
        :param N: Nz, Ntheta Number of axial and circumferential modes
        :param L: Length and radius of the cylinder
        :param shell_constant: A function of the Poisson ratio mu of the material and the shell thickness relative to the radius (h/a)
                               shell_constant = 12  (1-mu^2)/(h/a)^2
                               Since shell theory is only valid for thin walls (h/a) << 1, this is typically a large value
        :return: Modes and eigenvalues (e_k, gamma_k). For each wavenumber (m,n) there are two modes, thus there are 2 * Nz * Ntheta in total.
                 gamma_k is a vector length 2 * Nz * Ntheta.
                 The first Nz * Ntheta entries k = (m-1) * Ntheta + (n-1) belong to the even modes sqrt(2/pi/L) * sin(m*pi*z/L) * cos(n*theta) and
                 the next Nz * Ntheta entries k = Nz*Ntheta + (m-1) * Ntheta + (n-1) belong to the odd modes sqrt(2/pi/L) * sin(m*pi*z/L) * sin(n*theta).
                 e_k is a 3d array of the modes with shape (Z.shape,  2 * Nz * Ntheta)
        """

    def __init__(self, L: float|tuple[float, float], N: int|tuple[int, int], shell_constant: float):
        if isinstance(L, float):
            L = (L, 1.)
        if isinstance(N, int):
            N = (N, N)
        super().__init__(L, N)
        self._shell_constant = shell_constant

    def grid(self, Nx: int|tuple[int, int]):
        if isinstance(Nx, int):
            Nx = (Nx, Nx)
        L_z, a = self.L
        nx, ntheta = Nx
        z = L_z * np.linspace(0., 1., nx)
        phi = 2. * np.pi * np.linspace(0., 1., ntheta)
        Z, Phi = np.meshgrid(z, phi)
        return Z, Phi

    @property
    def _wavenumbers_even(self):
        """
        Wavenumbers W[k,:] = (m,n) for the even modes only. They are ordered
            k = (m-1) * N_theta + n, m = 1,...N_z, n = 0,...,N_theta

        :return: Array of wavenumbers of shape ( N_z*(N_theta + 1), 2 )
        """
        N_z = self.N[0]
        N_theta = self.N[1]
        # k = (m-1) * N_theta + (n-1)
        m = np.repeat(np.arange(1, N_z + 1, 1, dtype=int), N_theta + 1)  # m = 1,...N_z, slow index
        n = np.tile(np.arange(0, N_theta + 1, 1, dtype=int), N_z)  # n = 0,...,N_theta, fast index
        return np.stack([m, n], axis=-1)

    @property
    def wavenumbers(self):
        """
        Wavenumbers W[k,:] = (m,n,parity) for even and odd modes.
        First come the even modes with
            k = (m-1) * N_theta + n, m = 1,...N_z, n = 0,...,N_theta
        then the odd modes with
            k = N_z * Ntheta + (m-1) * N_theta + n, m = 1,...N_z, n = 0,...,N_theta
        :return: Array W of wavenumbers of shape ( 2 * N_z * (N_theta +1), 3). W[k,:] is (m,n,parity)
        """
        mn = self._wavenumbers_even
        mne = np.concatenate([mn, np.zeros(mn.shape[0], dtype=int)], axis=1)
        mno = np.concatenate([mn, np.ones(mn.shape[0], dtype=int)], axis=1)
        return np.concatenate([mne, mno], axis=0)

    @property
    def eigenvalues(self):
        L_z, a = self.L
        m = self._wavenumbers_even[:, 0]
        n = self._wavenumbers_even[:, 1]
        lam_k = -1. * (np.power(np.pi * m / L_z, 4) + 2. * np.power(np.pi / L_z / a * m * n, 2) + np.power(n / a, 4))
        kappa_k = np.power(m * np.pi / L_z, 2)
        gamma_k = lam_k + self._shell_constant / np.power(a, 4) * np.power(kappa_k, 2) / lam_k
        # Eigenvalues for even and odd functions are the same
        return np.tile(gamma_k, 2)

    def eigenmodes(self, x: float|NDArray|tuple[float, float]|tuple[NDArray, NDArray]):
        if isinstance(x, tuple):
            Z, Theta = x
            # Convert tuple of floats to 1x1 meshgrid
            if isinstance(Z, float):
                Z, Theta = np.meshgrid(np.atleast_1d(Z), np.atleast_1d(Theta))
        else:
            # Treat single float or array as (array of) Z coordinates,
            # assume Theta = 0
            Z, Theta = np.meshgrid(np.atleast_1d(x), np.array([0.]))
        assert Z.ndim == 2 and Z.shape == Theta.shape
        L_z, a = self.L
        m = self._wavenumbers_even[:, 0]
        n = self._wavenumbers_even[:, 1]
        # First come the even eigenmodes, then the odd eigenmodes
        eZ = np.sin(outer_product_nd(Z, m) * np.pi / L_z)
        arg_Theta = outer_product_nd(Theta, n)
        e_k = np.sqrt(2. / np.pi / L_z) * np.concatenate([
            eZ * np.cos(arg_Theta),
            eZ * np.sin(arg_Theta)
        ], axis=2)
        return e_k
