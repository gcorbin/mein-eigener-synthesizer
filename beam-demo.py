# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
from argparse import ArgumentParser

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation

from eigensynth.instruments.beam import Beam, BeamOptions
from eigensynth.sounds import play_sound
from eigensynth.time import samples


def main():
    args = parse_command_line_args()

    samplerate = 44100  # Hz
    duration = 2  # seconds

    opts = BeamOptions(base_frequency=args.base_frequency, L=1., Nk=5, Nx=100, pick_pos=args.pick_pos)
    beam = Beam(opts)

    # Are the modes normalized and orthogonal to each other?
    print(f"lambda_k = {beam.lam_k}")
    print(f"<e_k, e_l>")
    integration_weights = beam.options.L / ( beam.options.Nx - 1 ) * np.ones((1, beam.options.Nx))
    integration_weights[:,0] *= 0.25
    integration_weights[:, -1] *= 0.25
    E = np.matmul(integration_weights * beam.e_k, beam.e_k.transpose())
    print(E)

    t = samples(samplerate, duration)
    U = beam.sound(t, beam.x)

    play_sound(beam.sound(t, beam.options.L), samplerate, mode='normalize', device=4)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    for i in range(beam.options.Nk):
        ax1.plot(beam.x, beam.u0_k[i] * beam.e_k[i, :], label=f'e_{i}')
    ax1.set_xlabel('x / m')
    ax1.set_ylabel('u')
    ax1.legend()

    print(beam.u0_k)
    ax2.plot(beam.x, np.sum(beam.u0_k.reshape(-1,1) * beam.e_k, axis=0), label='U0')
    ax2.plot(beam.x, U[0, :], label='U(t=0)')
    ax2.legend()
    ax2.set_xlabel('x / m')
    ax2.set_ylabel('u')

    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel('x / m')
    ax.set_ylabel('u')
    plot_scale = np.max(np.abs(U[0, :]))
    ax.set_ylim((-plot_scale, plot_scale))
    line, = ax.plot(beam.x, U[0, :])

    ani_speed = 1
    def animate(i):
        line.set_ydata(U[i*ani_speed, :])  # update the data.
        return line,

    ani = animation.FuncAnimation(fig, animate, interval=100, blit=True, save_count=1)
    plt.show()
    return 0


def parse_command_line_args():
    parser = ArgumentParser(description='Plot oscillations of a cantilevered beam')
    parser.add_argument('--pick-pos', type=float_in_interval(0., 1.), default='1.')
    parser.add_argument('--base-frequency', type=float, default=440.)
    args = parser.parse_args()
    return args


def float_in_interval(lb: float, ub: float):
    def float_in_interval(f: str):
        number = float(f)
        if number < lb or number > ub:
            raise ValueError(f'{number} is outside the range [{lb},{ub}]')
        return number
    return float_in_interval


if __name__ == '__main__':
    return_code = main()
