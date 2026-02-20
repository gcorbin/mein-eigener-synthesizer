# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
from eigensynth import writer
from eigensynth.instruments.string import String, StringOptions

import numpy as np
from matplotlib import pyplot as plt


def samples(samplerate, duration):
    return np.arange(0, duration, 1. / samplerate)


if __name__ == '__main__':
    samplerate = 44100  # Hz
    duration = 2  # seconds

    opts = StringOptions(base_frequency=440)
    string = String(opts)

    t = samples(samplerate, duration)
    x_out = 0.8 * string.options.L

    sound = string.sound(t, x_out)

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
