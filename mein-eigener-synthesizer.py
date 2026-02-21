# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
import time

from eigensynth import writer
from eigensynth.instruments.string import String, StringOptions

import numpy as np
from matplotlib import pyplot as plt
import sounddevice


def samples(samplerate, duration):
    return np.arange(0, duration, 1. / samplerate)


def normalize(a, axis=None):
    return a / np.max(a, axis)


def minor_chord(base_frequency):
    return base_frequency * np.pow(2., np.array([0., 3./12., 7./12.]))


def major_chord(base_frequency):
    return base_frequency * np.pow(2., np.array([0., 4./12., 7./12.]))


def play_sound(sound, samplerate, blocking=False):
    sounddevice.play((sound*32768).astype(np.int16), samplerate, blocking=blocking)


if __name__ == '__main__':
    samplerate = 44100  # Hz
    duration = 2  # seconds

    L = 1
    opts = [StringOptions(base_frequency=f, L=L) for f in minor_chord(440)]
    strings = [String(o) for o in opts]

    t = samples(samplerate, duration)
    x_out = L * np.arange(0.6, 0.8, 0.02)
    #x_out = L * 0.7

    sounds = np.stack([normalize(np.sum(s.sound(t, x_out), axis=1)) for s in strings])
    sound = normalize(np.sum(sounds, axis=0))

    play_sound(sound, samplerate)
    writer.write_soundfile('string', sound, samplerate)

    N = 4000
    fig, ax = plt.subplots(2,1)
    #ax[0].plot(t[0:N], U[0:N, 50], label='U(t, x=0.5)')
    ax[0].plot(t[0:N], sound[0:N], label='U(t, x=0.5)')
    ax[0].legend()

    Uf = np.fft.rfft(sound[0:N], axis=0)
    f = np.fft.rfftfreq(N, d = 1./samplerate)
    ax[1].plot(f, np.abs(Uf), label='U, amplitude spectrum')
    ax[1].legend()
    ax[1].loglog()
    plt.grid()
    plt.show()
    writer.write_soundfile('string', sound, samplerate)