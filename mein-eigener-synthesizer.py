# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
import sys
from argparse import ArgumentParser
from dataclasses import dataclass, field
from typing import DefaultDict

from pynput import keyboard
import numpy as np
import sounddevice as sd

from eigensynth.sounds import normalize
from eigensynth.instruments.string import String, StringOptions
from eigensynth.sounds import convert_for_sounddevice
from eigensynth.time import samples


@dataclass
class Sound:
    name: str = '0'
    sound: np.typing.NDArray = field(default_factory=lambda: np.zeros(0, dtype=np.int16))
    playback_pos: int = field(default=0, init=False)

    def __post_init__(self):
        self.playback_pos = self.sound.size

    def is_empty(self):
        return self.sound.size == 0

    def advance(self, output, req_frames: int=0):
        frames = min(req_frames, self.sound.size - self.playback_pos)
        output[:frames] += self.sound[self.playback_pos:self.playback_pos + frames]
        self.playback_pos += frames
        return frames

    def reset(self):
        self.playback_pos = 0


def main():
    args = parse_program_args()

    samplerate = 44100  # Hz
    sound_lib = make_sound_lib(samplerate, args.base, args.octave)

    def _fill_output_stream(outdata, req_frames, time, status):
        fill_output_stream(outdata, req_frames, time, status, sound_lib)

    with sd.OutputStream(samplerate=samplerate,
                         #device=args.device, # default device for now
                         dtype=np.int16,
                         channels=1, #  mono for all
                         callback=_fill_output_stream):
        print("Synthesizer ready. Press Enter to exit the program.")
        listener = keyboard.Listener(
            on_press=lambda key: on_press(key, samplerate, sound_lib),
            suppress=True)
        listener.start()
        listener.join()


keybindings = ('a', 'w', 's', 'e', 'd', 'f', 't', 'g', 'y', 'h', 'u', 'j', 'k', 'o', 'l', 'p')
notes = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
note_indices = {n: i for i,n in enumerate(notes)}


def parse_program_args():
    parser = ArgumentParser(description="Play sounds on key presses")
    parser.add_argument('--device')
    parser.add_argument('--base', default='C', choices=notes)
    parser.add_argument('--octave', type=int, default=4)
    args = parser.parse_args()
    return args


def make_sound_lib(samplerate, note: str='C', octave: int=4):
    num_sounds = len(keybindings)

    damping_halflife = 0.05  # seconds
    duration = 20 * damping_halflife  # seconds

    A4 = 440.  # Hz
    C4 = A4 * np.pow(2., -9./12.)
    shift = note_indices[note] + 12 * (octave - 4)

    t = samples(samplerate, duration)

    # emulate the hole of an acoustic guitar by integrating the solution over this domain
    # this makes a muffled sound
    x_out = np.arange(0.6, 0.8, 0.02)

    # emulate a single pickup of an electric guitar by using a single point for evaluation
    # this makes a crisp sound
    #x_out = 0.7

    sound_lib = DefaultDict(Sound)
    for i in range(0, num_sounds):
        frequency = C4 * np.pow(2., (i + shift) * 1./12)
        opt = StringOptions(base_frequency=frequency)
        string = String(opt)

        sound = convert_for_sounddevice(normalize(np.sum(string.sound(t, x_out), axis=1), scale=0.2), mode='clip')

        cur_octave = (octave * 12 + i + shift ) // 12
        sound_lib[keybindings[i]] = Sound(name=f'{notes[(i + shift) % 12]}{cur_octave}', sound=sound)
    return sound_lib


def on_press(key, samplerate, sound_lib):
    try:
        sound = sound_lib[key.char]
        if not sound.is_empty():
            print(f'{sound.name}', end=' ')
            sound.reset()
    except AttributeError:
        #print('special key {0} pressed'.format(key))
        if key is keyboard.Key.enter:
            print('Stopping Synthesizer')
            return False


def fill_output_stream(outdata, req_frames, time, status, sound_lib):
    if status:
        print(status, file=sys.stderr)

    buf = np.zeros(req_frames)
    for sound in sound_lib.values():
        sound.advance(output=buf, req_frames=req_frames)
    outdata[:] = buf.reshape(-1, 1)


if __name__ == '__main__':
    main()