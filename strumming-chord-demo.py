# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0

import numpy as np
from matplotlib import pyplot as plt

from eigensynth.instrument import Instrument
from eigensynth.plot import plot_sound_signal, plot_sound_spectrum
from eigensynth.sounds import write_soundfile, play_sound, normalize
from eigensynth.space import String
from eigensynth.time import samples


def minor_chord(base_frequency):
    return base_frequency * np.pow(2., np.array([0., 3. / 12., 7. / 12.]))


def major_chord(base_frequency):
    return base_frequency * np.pow(2., np.array([0., 4. / 12., 7. / 12.]))


if __name__ == '__main__':
    samplerate = 44100  # Hz
    duration = 2  # seconds

    delay_fan = 0.01  # seconds, emulate strumming with delay between two consecutive sounds in the chord

    L = 1.  # meter, Length of the string
    base_freq = 440.  # Hz
    strings = [Instrument(String(L, N=10), base_frequency=f, halflife=duration / 10.) for f in minor_chord(base_freq)]

    t = samples(samplerate, duration)

    # emulate the hole of an acoustic guitar by integrating the solution over this domain
    # this makes a muffled sound
    x_out = L * np.arange(0.6, 0.8, 0.02)

    # emulate a single pickup of an electric guitar by using a single point for evaluation
    # this makes a crisp sound
    # x_out = L * 0.7

    sounds = np.stack([
        s.sound(t - i * delay_fan, x_out, x0=L * 0.8)
        for i, s in enumerate(strings)])
    sound = normalize(np.sum(sounds, axis=0))

    play_sound(sound, samplerate)
    write_soundfile('chord', sound, samplerate)

    plot_samples = int(0.1 * samplerate)
    fig, (ax1, ax2) = plt.subplots(2, 1)
    plot_sound_signal(ax1, t[0:plot_samples], sound[0:plot_samples], label='sound')
    plot_samples = int(1 * samplerate)
    plot_sound_spectrum(ax2, sound[0:plot_samples], samplerate, base_frequency=base_freq, label='sound')
    plt.show()
