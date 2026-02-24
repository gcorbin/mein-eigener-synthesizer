# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
from pathlib import Path

import numpy as np
import soundfile as sf

__all__ = ['write_soundfile']


def write_soundfile(filename: str | Path, data, samplerate):
    filepath = Path(filename).with_suffix('.wav')
    sf.write(filepath.name, np.clip(data, -1., 1.), samplerate)
