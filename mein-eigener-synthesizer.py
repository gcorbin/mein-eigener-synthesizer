# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
from argparse import ArgumentParser
from pynput import keyboard


def on_press(key):
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
    except AttributeError:
        print('special key {0} pressed'.format(
            key))
        if key is keyboard.Key.enter:
            print('received enter, stopping')
            return False


def main():
    parser = ArgumentParser(description="Play sounds on key presses")
    args = parser.parse_args()

    listener = keyboard.Listener(
        on_press=on_press,
        suppress=True)

    listener.start()
    listener.join()


if __name__ == '__main__':
    main()