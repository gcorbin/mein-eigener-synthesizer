# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
from matplotlib.ticker import FuncFormatter

from eigensynth.sounds import write_soundfile, play_sound, normalize
from eigensynth.instruments.string import String, StringOptions

import numpy as np
from matplotlib import pyplot as plt


def samples(samplerate, duration):
    return np.arange(0, duration, 1. / samplerate)


def minor_chord(base_frequency):
    return base_frequency * np.pow(2., np.array([0., 3./12., 7./12.]))


def major_chord(base_frequency):
    return base_frequency * np.pow(2., np.array([0., 4./12., 7./12.]))


if __name__ == '__main__':
    samplerate = 44100  # Hz
    duration = 2  # seconds

    delay_fan = 0.01  # seconds, emulate strumming with delay between two consecutive sounds in the chord

    L = 1.  # meter, Length of the string
    base_freq = 440.  # Hz
    opts = [StringOptions(base_frequency=f, L=L) for f in minor_chord(base_freq)]
    strings = [String(o) for o in opts]

    t = samples(samplerate, duration)

    # emulate the hole of an acoustic guitar by integrating the solution over this domain
    # this makes a muffled sound
    x_out = L * np.arange(0.6, 0.8, 0.02)

    # emulate a single pickup of an electric guitar by using a single point for evaluation
    # this makes a crisp sound
    #x_out = L * 0.7

    sounds = np.stack([
        np.sum(s.sound(t-i*delay_fan, x_out), axis=1)
        for i,s in enumerate(strings)])
    sound = normalize(np.sum(sounds, axis=0))

    play_sound(sound, samplerate)
    write_soundfile('string', sound, samplerate)

    N = 4000
    fig, ax = plt.subplots(2,1)
    #ax[0].plot(t[0:N], U[0:N, 50], label='U(t, x=0.5)')
    ax[0].plot(t[0:N], sound[0:N], label='U(t, x=0.5)')
    ax[0].legend()

    Uf = np.fft.rfft(sound[0:N], axis=0)[1:]  # Discard 0Hz (constant) component
    f = np.fft.rfftfreq(N, d = 1./samplerate)[1:] # Discard 0Hz (constant) component
    # Getting tick positions and labels right in an axis with log scale is too much hassle.
    # Instead, this plot uses a linear x scale with log-scaled x-values
    f_octave = np.log2(f / base_freq)  #  Convert frequencies to octaves above the base frequency
    ax[1].plot(f_octave, np.pow(np.abs(Uf), 2.), label='U, amplitude spectrum')
    ax[1].set_xscale("linear")
    ax[1].set_yscale("log", base=10)
    ax[1].legend()
    # Major ticks on all integers in the data range
    x_majorticks = np.arange(np.min(f_octave), np.max(f_octave), 1, dtype=int)
    # Subdivide each major tick interval into 3 minor intervals
    x_minorticks = (np.atleast_2d(x_majorticks).transpose() + np.array([1./3., 2./3.], ndmin=2)).flatten()
    ax[1].set_xticks(x_majorticks)
    ax[1].set_xticks(x_minorticks, labels=(), minor=True)
    ax[1].yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{int(10*np.log10(val))}"))
    ax[1].grid(which='major', alpha=0.5)
    ax[1].grid(which='minor', alpha=0.3, linestyle='--')
    ax[1].set_xlabel(f'log2(f/{int(base_freq)} Hz)')
    ax[1].set_ylabel('Power / dB')
    plt.show()