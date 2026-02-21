# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
import numpy as np

__all__=['normalize']

def normalize(a, scale=1., axis=None):
    return a * scale / np.max(np.abs(a), axis)
