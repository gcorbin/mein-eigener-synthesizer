# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
import sys
from argparse import ArgumentParser
from typing import DefaultDict

from pynput import keyboard
import numpy as np
import sounddevice as sd

from eigensynth.sounds import write_soundfile, play_sound, normalize
from eigensynth.instruments.string import String, StringOptions
from eigensynth.sounds.play import convert_for_sounddevice
from eigensynth.time import samples


def main():
    args = parse_program_args()

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
    sounds = [
        convert_for_sounddevice(normalize(np.sum(s.sound(t, x_out), axis=1), scale=0.2), mode='clip')
        for i, s in enumerate(strings)]
    sound_lib = DefaultDict(lambda: ['0', np.zeros((100,)), 100], {k:[n,s,s.size] for k,n,s in zip(keys, notes, sounds)})

    def _fill_output_stream(outdata, req_frames, time, status):
        fill_output_stream(outdata, req_frames, time, status, sound_lib)

    with sd.OutputStream(samplerate=samplerate,
                         #device=args.device,
                         dtype=np.int16,
                         channels=1, #  mono for all
                         callback=_fill_output_stream):
        print("Synthesizer ready. Press Enter to exit the program.")
        listener = keyboard.Listener(
            on_press=lambda key: on_press(key, samplerate, sound_lib),
            suppress=True)
        listener.start()
        listener.join()


def parse_program_args():
    parser = ArgumentParser(description="Play sounds on key presses")
    parser.add_argument('--device')
    args = parser.parse_args()


def on_press(key, samplerate, sound_lib):
    try:
        entry = sound_lib[key.char]
        note, sound, pos = entry
        if note != '0':
            print(f'{note}')
            entry[2] = 0
            #play_sound(sound, samplerate, mode='normalize', blocking=True)
    except AttributeError:
        #print('special key {0} pressed'.format(key))
        if key is keyboard.Key.enter:
            print('Stopping Synthesizer')
            return False


def fill_output_stream(outdata, req_frames, time, status, sound_lib):
    if status:
        print(status, file=sys.stderr)

    buf = np.zeros(req_frames)
    for entry in sound_lib.values():
        note, sound, pos = entry
        frames = min(req_frames, sound.size - pos)
        buf[:frames] += sound[pos:pos+frames]
        entry[2] += frames
    outdata[:] = buf.reshape(-1, 1)


if __name__ == '__main__':
    main()