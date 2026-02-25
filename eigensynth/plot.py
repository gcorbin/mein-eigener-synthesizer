# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
import numpy as np
from matplotlib.ticker import FuncFormatter


def plot_sound_signal(ax, t, sound, label='u'):
    ax.plot(t[:], sound[:], label=label)
    ax.set_xlabel(f'time / s')
    ax.set_ylabel('x')
    return ax

def plot_sound_spectrum(ax, sound, samplerate, base_frequency, label='u'):
    nsamples = sound.size
    Uf = np.fft.rfft(sound[:], axis=0)[1:]  # Discard 0Hz (constant) component
    f = np.fft.rfftfreq(nsamples, d =1. / samplerate)[1:] # Discard 0Hz (constant) component
    # Getting tick positions and labels right in an axis with log scale is too much hassle.
    # Instead, this plot uses a linear x scale with log-scaled x-values
    f_octave = np.log2(f / base_frequency)  # Convert frequencies to octaves above the base frequency
    ax.plot(f_octave, np.pow(np.abs(Uf), 2.), label=label)
    ax.set_xscale("linear")
    ax.set_yscale("log", base=10)
    # Major ticks on all integers in the data range
    x_majorticks = np.arange(np.min(f_octave), np.max(f_octave), 1, dtype=int)
    # Subdivide each major tick interval into 3 minor intervals
    x_minorticks = (np.atleast_2d(x_majorticks).transpose() + np.array([1./3., 2./3.], ndmin=2)).flatten()
    ax.set_xticks(x_majorticks)
    ax.set_xticks(x_minorticks, labels=(), minor=True)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda val, pos: f"{int(10*np.log10(val))}"))
    ax.grid(which='major', alpha=0.5)
    ax.grid(which='minor', alpha=0.3, linestyle='--')
    ax.set_xlabel(f'log2(f/{int(base_frequency)} Hz)')
    ax.set_ylabel('Power / dB')