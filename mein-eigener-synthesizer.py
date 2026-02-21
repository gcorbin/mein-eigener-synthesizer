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

from eigensynth.sounds import write_soundfile, play_sound, normalize
from eigensynth.instruments.string import String, StringOptions
from eigensynth.sounds.play import convert_for_sounddevice
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
    sound_lib = DefaultDict(Sound, {k:Sound(name=n, sound=s) for k,n,s in zip(keys, notes, sounds)})

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


def parse_program_args():
    parser = ArgumentParser(description="Play sounds on key presses")
    parser.add_argument('--device')
    args = parser.parse_args()


def on_press(key, samplerate, sound_lib):
    try:
        sound = sound_lib[key.char]
        if not sound.is_empty():
            print(f'{sound.name}')
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