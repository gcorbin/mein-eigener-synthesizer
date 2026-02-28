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
from eigensynth.instrument import Instrument
from eigensynth.space import String, Beam, CylindricalShell
from eigensynth.sounds import convert_for_sounddevice
from eigensynth.time import samples


@dataclass
class Sound:
    name: str = '0'
    sound: np.typing.NDArray = field(default_factory=lambda: np.zeros(0, dtype=np.int16))
    max_stack : int = 1
    playback_pos: np.typing.NDArray = field(default_factory=lambda : np.zeros(0, dtype=int), init=False)

    def __post_init__(self):
        self.playback_pos = np.zeros(self.max_stack, dtype=int)
        self.playback_pos[:] = self.sound.size

    def is_empty(self):
        return self.sound.size == 0

    def is_playing(self):
        return np.any(self.playback_pos < self.sound.size)

    def advance(self, output, req_frames: int=0):
        for i in range(self.playback_pos.size):
            frames = min(req_frames, self.sound.size - self.playback_pos[i])
            output[:frames] += self.sound[self.playback_pos[i]:self.playback_pos[i] + frames]
            self.playback_pos[i] += frames

    def play(self):
        self.playback_pos[np.argmax(self.playback_pos)] = 0



def main():
    args = parse_program_args()
    if args.list_devices:
        try:
            print(sd.query_devices(args.device))
        except ValueError as err:
            print(err)
        return 0

    samplerate = 44100  # Hz
    sound_lib = make_sound_lib(samplerate, args)

    def _fill_output_stream(outdata, req_frames, time, status):
        fill_output_stream(outdata, req_frames, time, status, sound_lib)

    with sd.OutputStream(samplerate=samplerate,
                         device=args.device,
                         dtype=np.int16,
                         channels=1, #  mono for all
                         callback=_fill_output_stream) as stream:
        print(f"Synthesizer streaming to device {stream.device}: '{sd.query_devices(stream.device)["name"]}'.\n"
              f"Press Enter to exit the program.")
        listener = keyboard.Listener(
            on_press=lambda key: on_press(key, sound_lib),
            suppress=True)
        listener.start()
        listener.join()
    return 0


keybindings = ('a', 'w', 's', 'e', 'd', 'f', 't', 'g', 'y', 'h', 'u', 'j', 'k', 'o', 'l', 'p')
notes = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')
note_indices = {n: i for i,n in enumerate(notes)}

oscillators = {'string': String(L=1., N=10),
               'beam': Beam(L=1., N=10),
               'pipe': CylindricalShell(L=(1., 1./(4.*np.pi)), N=(8,8), shell_constant=1e5),}

sound_pickup = {
    'clean': np.array([0.7]),
    'muffled': np.arange(0.6, 0.8, 0.02)
}

excitation = {
    'string': {'soft': 0.5, 'medium': 0.85, 'hard': 0.95},
    'beam': {'soft': 1., 'medium': 0.5, 'hard': 0.2},
    'pipe': {'soft': 0.5, 'medium': 0.85, 'hard': 0.95},
}



def parse_program_args():
    parser = ArgumentParser(description="Simple electronic keyboard / synthesizer")
    parser.add_argument('--device', type=int_or_str,
                        help='output device (numeric ID or substring)')
    parser.add_argument('--list-devices', '-l', action='store_true',
                        help='print devices matching the --devices arg and exit')
    parser.add_argument('--base', default='C', choices=notes,
                        help='the key of the music scale')
    parser.add_argument('--octave', type=int, default=4,
                        help='the octave of the music scale')
    parser.add_argument('--max-stack', type=int, default=3,
                        help='times a sound can be played over itself')
    parser.add_argument('--instrument', default='string', choices=oscillators.keys(),
                        help='the instrument to play')
    parser.add_argument('--pickup', default='clean', choices=sound_pickup.keys(),
                        help="whether the produced sounds are clean or muffled")
    parser.add_argument('--excitation', default='medium', choices=excitation['string'].keys(),
                        help="whether to produce a soft, medium, or hard sound")
    parser.add_argument('--duration', type=float, default=2.,
                        help='duration of played sounds')
    args = parser.parse_args()
    return args


def make_sound_lib(samplerate, args):
    num_sounds = len(keybindings)

    damping_halflife = args.duration / 10  # seconds
    amplitude = 0.4

    A4 = 440.  # Hz
    C4 = A4 * np.pow(2., -9./12.)
    shift = note_indices[args.base] + 12 * (args.octave - 4)

    t = samples(samplerate, args.duration)
    x_out = sound_pickup[args.pickup]
    x0 = excitation[args.instrument][args.excitation]

    print("")
    sound_lib = DefaultDict(Sound)
    for i in range(0, num_sounds):
        print("\r"+" "*80+f"\rComputing sound {i+1} / {num_sounds}", end='', flush=True)
        cur_octave = (4 * 12 + i + shift) // 12
        sound_name = f'{notes[(i + shift) % 12]}{cur_octave}'

        frequency = C4 * np.pow(2., (i + shift) * 1./12)
        oscillator = oscillators[args.instrument]
        instrument = Instrument(oscillator, base_frequency=frequency, halflife=damping_halflife)

        sound = convert_for_sounddevice(normalize(instrument.sound(t, x_out, x0), scale=amplitude), mode='clip')
        sound_lib[keybindings[i]] = Sound(name=sound_name, sound=sound, max_stack=args.max_stack)
    print("\r"+" "*80+f"\rComputing sounds done")
    print("Key bindings:")
    print("  ".join([f"'{k}' = {s.name}" for k,s in sound_lib.items()]))
    return sound_lib



def on_press(key, sound_lib):
    try:
        sound = sound_lib[key.char]
        if not sound.is_empty():
            print(f'{sound.name}', end=' ', flush=True)
            sound.play()
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



def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


if __name__ == '__main__':
    return_code = main()
    exit(return_code)