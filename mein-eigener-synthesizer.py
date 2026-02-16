# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
from eigensynth import writer
from eigensynth.time import oscillator, damped_oscillator, damped_oscillator_coefficients, c2_from_frequency
from eigensynth.space import laplace_1d_eigen

import numpy as np
from matplotlib import pyplot as plt


def hat_fun(x, L, x0):
    assert L > 0.
    assert x0 >= 0. and x0 <= L
    return np.where(x<x0,(L-x0)/L*x, x0*(L-x)/L)


if __name__ == '__main__':
    samplerate = 44100  # Hz
    dur = 2  # seconds
    halflife = 0.2  # seconds
    base_freq = 880  # Hz
    L = 0.5  # m
    Nx = 101  # Discretize x with Nx values
    Nk = 20  # Use Nk eigenfunctions

    x = np.linspace(0., L, Nx)
    e_k, lam_k = laplace_1d_eigen(x, N=Nk, L=L)

    U0 = hat_fun(x, x0 = 0.85*L, L=L)  # Initial condition, evaluated at x
    u0_k = np.matmul(e_k, U0)  # Coefficients in expansion of initial condition U0 = sum_k u0_k * e_k(x)

    #freq = np.atleast_2d(np.array([base_freq, base_freq*np.pow(2., 3./12.), base_freq*np.pow(2., 7./12.)]))

    t = np.atleast_2d(np.arange(0, dur, 1./samplerate)).transpose()

    c2, d = damped_oscillator_coefficients(base_freq, halflife)
    u_k = damped_oscillator(t, -4. * L**2 * base_freq**2 * lam_k, d, u0_k, 0.)
    print(f'base c^2 @ 880 Hz = {c2}, d = {d}')
    print(f'actual c^2 = {-4. * L**2 * base_freq**2 * lam_k}')

    # We don't need to evaluate the full solution
    #U = np.matmul(u_k, e_k)

    x_out = 0.8*L
    e_out, _ = laplace_1d_eigen(x_out, N=Nk, L=L)
    U_x50 = np.matmul(u_k, e_out)


    N = 4000
    fig, ax = plt.subplots(2,1)
    #ax[0].plot(t[0:N], U[0:N, 50], label='U(t, x=0.5)')
    ax[0].plot(t[0:N], U_x50[0:N], label='U(t, x=0.5)')
    ax[0].legend()

    Uf = np.fft.rfft(U_x50[0:N], axis=0)
    f = np.fft.rfftfreq(N, d = 1./samplerate)
    ax[1].plot(f, np.abs(Uf), label='U, amplitude spectrum')
    ax[1].legend()
    plt.show()
    writer.write_soundfile('string', U_x50, samplerate)
