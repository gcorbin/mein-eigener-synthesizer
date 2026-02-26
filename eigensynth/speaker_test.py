# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
from argparse import ArgumentParser

import sounddevice as sd
import numpy as np

from eigensynth.time import samples
from sounds import play_sound


def main():
    args = parse_command_line_args()

    samplerate = 44100.
    base_frequency = 440.
    duration = 0.5
    halflife = duration / 10.

    t = samples(samplerate, duration)

    sine_sound = np.sin(2.*np.pi*base_frequency*t) *np.exp(-np.log(2.)/halflife*t)
    for dev in enumerate(sd.query_devices()):
        print(dev)
        play_sound(sound=sine_sound, samplerate=samplerate, blocking=True, device=dev)
    return 0


def parse_command_line_args():
    parser = ArgumentParser(description='Console app stub')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    return_code = main()
