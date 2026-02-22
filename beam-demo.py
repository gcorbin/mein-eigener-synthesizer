# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
from argparse import ArgumentParser
from matplotlib import pyplot as plt

from eigensynth.instruments.beam import Beam, BeamOptions


def main():
    args = parse_command_line_args()

    samplerate = 44100  # Hz
    duration = 2  # seconds

    opts = BeamOptions(Nk=5)
    beam = Beam(opts)

    fig, ax1 = plt.subplots(1, 1)
    for i in range(beam.options.Nk):
        ax1.plot(beam.x, beam.e_k[i, :], label=f'e_{i}')
    ax1.set_xlabel('x / m')
    ax1.set_ylabel('u')
    ax1.legend()

    plt.show()
    return 0


def parse_command_line_args():
    parser = ArgumentParser(description='Plot oscillations of a cantilevered beam')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    return_code = main()
