# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
from argparse import ArgumentParser


def main():
    parser = ArgumentParser(description="Play sounds on key presses")
    args = parser.parse_args()


if __name__ == '__main__':
    main()