# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0

import numpy as np
from numpy.typing import NDArray

from eigensynth.array import outer_product_nd

__all__ = ['cylindrical_shell_eigen']


def cylindrical_shell_eigen(x: tuple[NDArray, NDArray], N: tuple[int, int], L: tuple[float, float],
                            shell_constant=5.e5):
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
    Z, Theta = x
    assert Z.ndim == 2 and Z.shape == Theta.shape
    N_z, N_theta = N
    L_z, a = L
    # k = (m-1) * N_theta + (n-1)
    m = np.repeat(np.arange(1, N_z + 1, 1, dtype=int), N_theta)  # slow index
    n = np.tile(np.arange(1, N_theta + 1, 1, dtype=int), N_z)  # fast index

    lam_k = -1. * (np.power(np.pi * m / L_z, 4) + 2. * np.power(np.pi / L_z / a * m * n, 2) + np.power(n / a, 4))
    kappa_k = np.power(m * np.pi / L_z, 2)
    gamma_k = lam_k + shell_constant / np.power(a, 4) * np.power(kappa_k, 2) / lam_k

    # For each (m,n) there are two eigenfunctions: cos(n * theta ) and sin(n * theta)
    eZ = np.sin(outer_product_nd(Z, m) * np.pi / L_z)
    arg_Theta = outer_product_nd(Theta, n)
    e_k = np.sqrt(2. / np.pi / L_z) * np.concatenate([
        eZ * np.cos(arg_Theta),
        eZ * np.sin(arg_Theta)
    ], axis=2)
    return e_k, np.tile(gamma_k, 2)
