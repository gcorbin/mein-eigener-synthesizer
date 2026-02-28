# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import sounddevice
from matplotlib import cm
import numpy as np

from eigensynth.plot import plot_sound_signal, plot_sound_spectrum
from eigensynth.sounds import play_sound, normalize
from eigensynth.space.cylindrical_shell import cylindrical_shell_eigen, CylindricalShell
from eigensynth.instruments import Pipe, PipeOptions
from eigensynth.time import samples, damped_oscillator_coefficients, damped_oscillator, oscillator_frequency


def main():
    args = parse_command_line_args()

    visualize_initial_condition()
    return 0


def visualize_initial_condition():
    a = 1.
    L = 4*np.pi*a
    shell_constant = 1e2

    (m,n) = (12,12)

    forcing_point = np.meshgrid(np.array([0.7*L]), np.array([0.25 *  np.pi]))
    e0, gamma_k = cylindrical_shell_eigen(forcing_point, (m,n), (L,a), shell_constant)
    c0 = e0.reshape(-1) / gamma_k

    samplerate = 44100.
    base_frequency = 440.
    duration = 2.
    halflife = duration / 10.

    ks, kd = damped_oscillator_coefficients(base_frequency, halflife)
    tuned_gamma = -ks / np.min(np.abs(gamma_k)) * gamma_k

    # Show eigenvalues gamma_mn as (m,n) matrix
    print(f"ks = {ks}, kd = {kd}")
    print(f"base mode: {np.argmin(tuned_gamma)}, freq = {oscillator_frequency(np.min(tuned_gamma))}")

    plt.matshow(tuned_gamma[0:m*(n+1)].reshape((m,n+1)))
    plt.xlabel('n')
    plt.ylabel('m')
    plt.show()

    t = samples(samplerate, duration)

    u_k = damped_oscillator(
        t=t,
        k=tuned_gamma,
        d=kd,
        x0=c0,
        dx0=0.)

    sound = normalize(np.sum(np.inner(u_k, e0), axis=(1,2)), scale=4.)
    play_sound(sound, samplerate, blocking=False, mode='clip', device='pipewire')

    fig, (ax1, ax2) = plt.subplots(2,1)
    plot_samples = int(100. / base_frequency * samplerate)
    plot_sound_signal(ax1, t[0:plot_samples], sound[0:plot_samples], label='sound')
    plot_samples = int(1 * samplerate)
    plot_sound_spectrum(ax2, sound[0:plot_samples], samplerate, base_frequency=base_frequency, label='sound')
    plt.show()

    shell = CylindricalShell([L, a], [m, n], shell_constant)
    Z, Phi = shell.grid([101,101])

    e_k, gamma_k = cylindrical_shell_eigen((Z,Phi), (m,n), (L,a), shell_constant)
    w0 = np.sum(c0.reshape(1,1,-1) * e_k, axis=2)
    print(np.amax(w0))
    R = a * np.ones_like(Z) +  0.1 / np.amax(w0) * w0
    X = R * np.cos(Phi)
    Y = R * np.sin(Phi)

    # Plot the surface
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z, vmin=Z.min() * 2, facecolors=cm.Blues(R/np.amax(R)), rcount=100, ccount=100)
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    plt.show()



def parse_command_line_args():
    parser = ArgumentParser(description='Console app stub')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    return_code = main()
