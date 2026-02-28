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
from eigensynth.space.cylindrical_shell import CylindricalShell
from eigensynth.instrument import Instrument
from eigensynth.time import samples, damped_oscillator_coefficients, damped_oscillator, oscillator_frequency


def main():
    args = parse_command_line_args()

    base_frequency = 440.
    duration = 2.
    halflife = duration / 10.

    a = 1.
    L = 4 * np.pi * a
    shell_constant = 5e5

    (m, n) = (12, 12)
    shell = CylindricalShell((L, a), (m, n), shell_constant)
    instrument = Instrument(shell, base_frequency=base_frequency, halflife=halflife)

    visualize_initial_condition(instrument, args)
    show_sound(instrument, args)
    return 0


def visualize_initial_condition(instrument, args):
    L, a = instrument.oscillator.L
    forcing_point = np.meshgrid(np.array([0.7*L]), np.array([0.25 *  np.pi]))

    Z, Phi = instrument.oscillator.grid([51,51])
    w0 = instrument.solution(0, (Z,Phi), forcing_point)[0,:,:]

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


def show_sound(instrument, args):
    samplerate = 44100  # Hz
    t = samples(samplerate, instrument.halflife*20)
    # Show eigenvalues gamma_mn as (m,n) matrix
    #print(f"base mode: {np.argmin(instrument.frequencies)}, freq = {instrument.frequencies}")

    M,N = instrument.oscillator.N
    plt.matshow(instrument.frequencies[0:M * (N + 1)].reshape((M, N+1)))
    plt.xlabel('n')
    plt.ylabel('m')
    plt.show()

    L, a = instrument.oscillator.L
    forcing_point = np.meshgrid(np.array([0.7 * L]), np.array([0.25 * np.pi]))
    sound = instrument.sound(t, forcing_point, forcing_point)

    play_sound(normalize(sound, scale=0.8), samplerate, blocking=False, mode='clip', device=args.device)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    plot_samples = int(100. / instrument.base_frequency * samplerate)
    plot_sound_signal(ax1, t[0:plot_samples], sound[0:plot_samples], label='sound')
    plot_samples = int(1 * samplerate)
    plot_sound_spectrum(ax2, sound[0:plot_samples], samplerate, base_frequency=instrument.base_frequency, label='sound')
    plt.show()



def parse_command_line_args():
    parser = ArgumentParser(description='Visualize oscillations of a cylindrical shell.')
    parser.add_argument('--device', type=int_or_str,
                        help='output device (numeric ID or substring)')
    args = parser.parse_args()
    return args


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text

if __name__ == '__main__':
    return_code = main()
