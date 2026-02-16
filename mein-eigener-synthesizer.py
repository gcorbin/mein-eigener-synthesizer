# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
from eigensynth import writer
from eigensynth.time import oscillator, damped_oscillator, damped_oscillator_coefficients, c2_from_frequency

import numpy as np
from matplotlib import pyplot as plt


if __name__ == '__main__':
    samplerate = 44100  # Hz
    dur = 2  # seconds
    halflife = 0.2  # seconds
    freq = 880  # Hz

    t = np.arange(0, dur, 1./samplerate)
    c2 = c2_from_frequency(freq)
    x = oscillator(t, c2, 1., 0.)

    c2_dampened, d = damped_oscillator_coefficients(freq, halflife)
    xd = damped_oscillator(t, c2_dampened, d, 1., 0.)

    print(f'harmonic c^2 = {c2}, dampened c^2 = {c2_dampened}, dampened d = {d}')

    N = 1000
    plt.plot(t[0:N], x[0:N], label='harmonic oscillator')
    plt.plot(t[0:N], xd[0:N], label='dampened oscillator')
    plt.legend()
    plt.show()
    writer.write_soundfile('harmonic_880Hz', x, samplerate)
    writer.write_soundfile('dampened_880Hz', xd, samplerate)
