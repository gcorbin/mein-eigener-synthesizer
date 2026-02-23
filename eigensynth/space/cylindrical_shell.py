# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0

import numpy as np
from numpy.typing import NDArray


def cylindrical_shell_eigen(x: tuple[NDArray, NDArray], N: tuple[int, int], L: tuple[float, float], shell_constant=5.e5):
    """
    Compute eigenfunctions and eigenvalues of the cylindrical thin shell

    See https://doi.org/10.1121/2.0000945 for details.

    :param x: cylindrical coordinates Z, Theta as output of np.meshgrid
    :param N: Nz, Ntheta Number of axial and circumferential modes
    :param L: Length and radius of the cylinder
    :param shell_constant: A function of the Poisson ratio mu of the material and the shell thickness relative to the radius (h/a)
                           shell_constant = 12  (1-mu^2)/(h/a)^2
                           Since shell theory is only valid for thin walls (h/a) << 1, this is typically a large value
    :return:
    """
    N_z, N_theta = N
    L_z, a = L
    # k = m * N_theta + n
    m = np.repeat(np.arange(1, N_z+1, 1, dtype=int), N_theta)  # slow index
    n = np.tile(np.arange(1, N_theta+1, 1, dtype=int), N_z)  # fast index

    lam_k = -1. * ( np.power(np.pi * m / L_z, 4) + 2. + np.power(np.pi / L_z / a * m * n, 2) + np.power(n / a, 4) )
    kappa_k = np.power(m * np.pi / L_z, 2)
    gamma_k = lam_k + shell_constant * np.power(a, -4) * np.power(kappa_k, 2) * np.power(lam_k, -1)

    Z, Theta = x
    # (z,theta,k)
    e_k = np.sqrt(2. / np.pi / L_z) * np.sin(m.reshape(1,1,-1) * np.pi / L_z * np.atleast_3d(Z)) * np.cos(n.reshape(1,1,-1) * np.atleast_3d(Theta))
    return e_k, gamma_k

