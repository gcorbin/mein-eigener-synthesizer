# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
from eigensynth import writer
from eigensynth.instruments.string import String, StringOptions

import numpy as np
from matplotlib import pyplot as plt


def samples(samplerate, duration):
    return np.arange(0, duration, 1. / samplerate)


def normalize(a, axis=None):
    return a / np.max(a, axis)


if __name__ == '__main__':
    samplerate = 44100  # Hz
    duration = 2  # seconds

    opts = StringOptions(base_frequency=440)
    string = String(opts)

    t = samples(samplerate, duration)
    x_out = string.options.L * np.arange(0.6, 0.8, 0.02)
    #x_out = string.options.L * 0.7

    sound = normalize(np.sum(string.sound(t, x_out), axis=1))

    N = 4000
    fig, ax = plt.subplots(2,1)
    #ax[0].plot(t[0:N], U[0:N, 50], label='U(t, x=0.5)')
    ax[0].plot(t[0:N], sound[0:N], label='U(t, x=0.5)')
    ax[0].legend()

    Uf = np.fft.rfft(sound[0:N], axis=0)
    f = np.fft.rfftfreq(N, d = 1./samplerate)
    ax[1].plot(f, np.abs(Uf), label='U, amplitude spectrum')
    ax[1].legend()
    plt.show()
    writer.write_soundfile('string', sound, samplerate)
