# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
import numpy as np
import sounddevice
from .utils import normalize

__all__=['convert_for_sounddevice', 'play_sound']

def convert_for_sounddevice(sound, mode='clip'):
    if mode == 'clip':
        funnel = lambda a: np.clip(a, -1., 1.)
    elif mode == 'normalize':
        funnel = lambda a: normalize(a, scale=1.)
    else:
        raise RuntimeError(f"Mode '{mode}' not supported")
    return (32768*funnel(sound)).astype(np.int16)


def play_sound(sound, samplerate, mode='clip', blocking=False):
    sounddevice.play(convert_for_sounddevice(sound, mode),
                     samplerate,
                     blocking=blocking)
