# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
from argparse import ArgumentParser

from eigensynth.space import cantilevered_beam_eigen


def main():
    args = parse_command_line_args()

    samplerate = 44100  # Hz
    duration = 2  # seconds


    return 0


def parse_command_line_args():
    parser = ArgumentParser(description='Plot oscillations of a cantilevered beam')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    return_code = main()
