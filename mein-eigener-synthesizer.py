# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
from argparse import ArgumentParser
from typing import DefaultDict

from pynput import keyboard
import numpy as np

from eigensynth.sounds import write_soundfile, play_sound, normalize
from eigensynth.instruments.string import String, StringOptions
from eigensynth.time import samples


def on_press(key, samplerate, key_to_sound):
    try:
        note, sound = key_to_sound[key.char]
        if note != '0':
            print(f'{note}')
        play_sound(sound, samplerate, mode='normalize', blocking=True)
    except AttributeError:
        #print('special key {0} pressed'.format(key))
        if key is keyboard.Key.enter:
            print('Stopping Synthesizer')
            return False


def main():
    parser = ArgumentParser(description="Play sounds on key presses")
    args = parser.parse_args()

    samplerate = 44100  # Hz
    damping_halflife = 0.05  # seconds
    duration = 20 * damping_halflife  # seconds
    base_frequency = 440.  # Hz

    num_strings = 13
    frequencies = base_frequency * np.pow(2., 1./12. * np.linspace(0, 12, num_strings))
    opts = [StringOptions(base_frequency=f) for f in frequencies]
    strings = [String(o) for o in opts]

    t = samples(samplerate, duration)

    # emulate the hole of an acoustic guitar by integrating the solution over this domain
    # this makes a muffled sound
    x_out = np.arange(0.6, 0.8, 0.02)

    # emulate a single pickup of an electric guitar by using a single point for evaluation
    # this makes a crisp sound
    #x_out = 0.7

    notes =['C',  'C#',  'D',   'D#',  'E',   'F',   'F#',  'G',   'G#',  'A',   'A#',  'B',   'C']
    keys = ['a', 'w', 's', 'e', 'd', 'f', 't', 'g', 'y', 'h', 'u', 'j', 'k']
    sounds = [np.sum(s.sound(t, x_out), axis=1) for i, s in enumerate(strings)]
    key_to_sound = DefaultDict(lambda: ('0', np.zeros((100,))), {k:(n,s) for k,n,s in zip(keys, notes, sounds)})

    print("Synthesizer ready. Press Enter to exit the program.")
    listener = keyboard.Listener(
        on_press=lambda key: on_press(key, samplerate, key_to_sound),
        suppress=True)

    listener.start()
    listener.join()


if __name__ == '__main__':
    main()