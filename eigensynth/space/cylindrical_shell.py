# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0

import numpy as np
from numpy.typing import NDArray

from eigensynth.array import outer_product_nd

__all__ = ['CylindricalShell']

from eigensynth.space.linear_deformation import LinearDeformation


class CylindricalShell(LinearDeformation):
    def __init__(self, L: float|tuple[float, float], N: int|tuple[int, int], shell_constant: float):
        """
        Modes and eigenvalues of the cylindrical thin shell with length L and radius a.
        According to thin shell theory in cylindrical coordinates,
        radial displacement w(x), and auxiliary stress F(x) are given by

            (E*h)/s * a^2 * Nabla^4 w + 1/a d_zz F = 0
            Nabla^4 F - (E*h)/a d_zz w = 0

        for x = (z,theta) in Omega = (0,L)x(0,2pi).

        The shell constant s = (12 (1-mu^2)) / (h/a)^2 depends on the Poisson-ratio of the material and the shell's
        thickness-to-radius ratio (h/a). Since shell theory is only valid for thin walls (h/a) << 1,
        this is typically a large value.
        The constant E is the Young's modulus (stiffness) of the material.

        Eigenfunctions e_mnp for the displacement w for wavenumber (m,n,p) in [1,M]x[0,N]x[0,1] are given by

                                                                       | cos(n * theta), p == 0
            e_mnp(z, theta) = sqrt(2 / pi / L) * sin(m * pi * z / L) * |
                                                                       | sin(n * theta), p == 1

        Eigenvalues gamma_mn0 = gamma_mn1 =: gamma_mn are

            gamma_mn = -1 * (E*h)/s * a^2 * ( lambda_mn + s/a^4 * kappa_mn^2 / lambda_mn )

        with

            lambda_mn = (m * pi / L)^4 + 2 * ( m * pi * n / L / a)^2 + ( n / a )^4
            kappa_mn = - ( m * pi / L )^2

        See https://doi.org/10.1121/2.0000945 for details.

        :param L: tuple (L, a): Length and radius of the cylinder.
                  A scalar value is interpreted as (L, 1.)
        :param N: tuple (M,N): Maximum axial and circumferential wavenumber. There are 2*M*(N+1) eigenvalues.
                  A scalar value is interpredet as (N,N)
        :param shell_constant: shell_constant = 12(1-mu^2)/(h/a)^2.
        """
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
    def K(self):
        M, N = self.N
        return 2 * M * (N+1)

    @property
    def _wavenumbers_even(self):
        """
        Wavenumbers W[k,:] = (m,n) for the even modes only.

        They are ordered

            k = (m-1) * N + n, m = 1,...M, n = 0,...,N

        :return: Array of wavenumbers of shape ( M*(N + 1), 2 )
        """
        M, N = self.N
        m = np.repeat(np.arange(1, M + 1, 1, dtype=int), N + 1)  # m = 1,...M, slow index
        n = np.tile(np.arange(0, N + 1, 1, dtype=int), M)  # n = 0,...,N, fast index
        return np.stack([m, n], axis=-1)

    @property
    def wavenumbers(self):
        """
        Wavenumbers W[k,:] = (m,n,parity) for even and odd modes.

        First come the even modes with

            k = (m-1) * N + n, m = 1,...M, n = 0,...,N

        then the odd modes with

            k = (M * N) + (m-1) * N + n, m = 1,...M, n = 0,...,N

        :return: Array W of wavenumbers of shape (K, 3). W[k,:] is (m,n,parity)
        """
        mn = self._wavenumbers_even
        mne = np.concatenate([mn, np.zeros((mn.shape[0],1), dtype=int)], axis=1)
        mno = np.concatenate([mn, np.ones((mn.shape[0],1), dtype=int)], axis=1)
        return np.concatenate([mne, mno], axis=0)

    def indices(self, wavenumbers):
        """
        Indices I[m,n,p] holds the index of wavenumber (m,n,p) in the eigenvalues/eigenmodes
        arrays.

        self.indices(self.wavenumbers) is equivalent to np.arange(K)

        :param wavenumbers: Array (K, 3), where wavenumbers[i, :] is a wavenumber tuple (m,n,p)
        :return: 1D Array size K, holding the indices corresponding to the wavenumbers.
        """
        M,N = self.N

        wavenumbers = np.atleast_2d(wavenumbers)
        m = wavenumbers[:, 0]
        n = wavenumbers[:, 1]
        p = wavenumbers[:, 2]
        assert np.all(0 <  m <= M)
        assert np.all(0 <= n <= N)
        assert np.all(0 <= p <= 1)

        return (M*(N+1))*p + (m - 1) * (N+1) + n

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
        """
        :param x: cylindrical coordinates Z, Theta, either
                  - (Z,Theta) as output of np.meshgrid
                  - (z,theta) as single coordinate pair
                  - Z as array of z-values, assume theta = 0
                  - z as single z-value, assume theta = 0

        :return: Modes evaluated at x. 3D array. The first two axes have the same shape as Z.
                 The last axis is the flat index of wavenumbers, length K.
        """
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
