# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
from argparse import ArgumentParser

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

from eigensynth.instrument import Instrument
from eigensynth.space import String, Beam
from eigensynth.plot import plot_sound_signal, plot_sound_spectrum
from eigensynth.sounds import play_sound
from eigensynth.time import samples


oscillators = {
    'string': String,
    'beam': Beam,
}


def main():
    args = parse_command_line_args()

    oscillator = oscillators[args.instrument](N=args.basis_functions, L=1.)
    instrument = Instrument(oscillator, base_frequency=args.base_frequency, halflife=args.duration/20.)

    animate_initial_condition(instrument, args)
    animate_vibration(instrument, args)
    show_sound(instrument, args)
    return 0


def show_sound(instrument, args):
    samplerate = 44100  # Hz
    t = samples(samplerate, args.duration)
    #x_out = instrument.options.L * np.arange(0.6, 0.8, 0.02)
    x0 = args.pick_pos * instrument.oscillator.L
    x_out = args.pickup_pos * instrument.oscillator.L
    sound = instrument.sound(t, x_out, x0)

    play_sound(sound, samplerate, mode='normalize', device=args.device)

    plot_samples = int(0.1 * samplerate)
    fig, (ax1, ax2) = plt.subplots(2,1)
    plot_sound_signal(ax1, t[0:plot_samples], sound[0:plot_samples], label='sound')
    plot_samples = int(1 * samplerate)
    plot_sound_spectrum(ax2, sound[0:plot_samples], samplerate, base_frequency=instrument.base_frequency, label='sound')
    plt.show()


def animate_initial_condition(instrument, args):
    x = instrument.oscillator.grid(100)
    x0 = args.pick_pos * instrument.oscillator.L
    U0 = instrument.solution(np.array([0.]), x, x0)

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # Plot the individual basis functions
    lines1 = []
    modes = instrument.oscillator.eigenmodes(x)
    for i, lam in enumerate(modes):
        line, = ax1.plot(x, np.zeros_like(x), label=f'e_{i}')
        lines1.append(line)
    ax1.set_xlabel('x / m')
    ax1.set_ylabel('u')
    plot_scale = 1.1 * np.max(np.abs(modes))
    ax1.set_ylim((-plot_scale, plot_scale))

    # Plot the sum of basis functions
    line2, = ax2.plot(x, np.zeros_like(x), label=f'U0 with {0} basis functions')
    ax2.plot(x, U0[0], label=f'U0 with {instrument.oscillator.N} basis functions')
    ax2.legend()
    ax2.axvline(x=x0, color="black", linestyle="--")
    ax2.set_xlabel('x / m')
    ax2.set_ylabel('u')
    plot_scale = 1.1 * np.max(np.abs(U0))
    ax2.set_ylim((-plot_scale, plot_scale))

    def init_animation():
        for k in range(len(lines1)):
            lines1[k].set_ydata(np.zeros_like(x))
        line2.set_ydata(np.zeros_like(x))
        return tuple(lines1) + (line2,)

    def animate(i):
        changed = []
        for k in range(min(i, len(lines1))):
            lines1[k].set_ydata(modes[:, k])
            changed.append(lines1[k])
        if i > 0:
            line2.set_ydata(np.inner(instrument.initial_coefficients(x0)[:i], modes[:, :i]))
            line2.set_label(f'U0 with {i} basis functions')
            legend = ax2.legend()
            changed.append(line2)
            changed.append(legend)
        return changed

    ani = animation.FuncAnimation(fig,
                                  animate,
                                  init_func=init_animation,
                                  interval=1000,
                                  frames=instrument.oscillator.N + 1,
                                  blit=True,
                                  repeat_delay=2000)

    plt.show()


def animate_vibration(instrument, args):
    samplerate = 44100  # Hz
    duration = 5. / instrument.base_frequency
    t = samples(samplerate, duration)
    x = instrument.oscillator.grid(100)
    x0 = args.pick_pos * instrument.oscillator.L
    U = instrument.solution(t, x, x0)

    fig, ax = plt.subplots()
    ax.set_xlabel('x / m')
    ax.set_ylabel('u')
    ax.axvline(x=x0, color="black", linestyle="--")
    plot_scale = np.max(np.abs(U[0, :]))
    ax.set_ylim((-1. * plot_scale, plot_scale))
    line, = ax.plot(x, U[0, :])

    def animate(i):
        line.set_ydata(U[i, :])  # update the data.
        return line,

    ani = animation.FuncAnimation(fig,
                                  animate,
                                  interval=50,
                                  blit=True,
                                  frames=t.size)
    plt.show()




def parse_command_line_args():
    parser = ArgumentParser(description='Plot oscillations of a cantilevered beam')
    parser.add_argument('--device', type=int_or_str,
                        help='output device (numeric ID or substring)')
    parser.add_argument('--pick-pos', type=float_in_interval(0., 1.), default='0.8')
    parser.add_argument('--pickup-pos', type=float_in_interval(0., 1.), default='0.7')
    parser.add_argument('--base-frequency', type=float, default=440.)
    parser.add_argument('--duration', type=float, default=2.)
    parser.add_argument('--instrument', default='string', choices=oscillators.keys(),
                        help='the instrument to play')
    parser.add_argument('--basis-functions', type=int, default=10,
                        help='the number of basis functions to plot')
    args = parser.parse_args()
    return args


def float_in_interval(lb: float, ub: float):
    def float_in_interval(f: str):
        number = float(f)
        if number < lb or number > ub:
            raise ValueError(f'{number} is outside the range [{lb},{ub}]')
        return number
    return float_in_interval


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


if __name__ == '__main__':
    return_code = main()
