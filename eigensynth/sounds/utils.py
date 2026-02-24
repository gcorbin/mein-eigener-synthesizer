# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
import numpy as np

__all__ = ['normalize']


def normalize(a, scale=1., axis=None):
    maxval = np.max(np.abs(a), axis)
    if maxval > 0.:
        return a * scale / np.max(np.abs(a), axis)
    else:
        return a
