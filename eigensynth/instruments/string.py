# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
import numpy as np
from dataclasses import dataclass

from eigensynth.instruments.instrument import InstrumentOptions, Instrument
from eigensynth.space import laplace_1d_eigen


def _hat_fun(x, L, x0):
    assert L > 0.
    assert x0 >= 0. and x0 <= L
    return np.where(x < x0, (L - x0) / L * x, x0 * (L - x) / L)


@dataclass(init=True)
class StringOptions(InstrumentOptions):
    pick_pos: float = 0.85


class String(Instrument):
    def compute_eigen(self, x):
        return laplace_1d_eigen(x, N=self.options.Nk, L=self.options.L)

    def _compute_initial_coefficients(self):
        x = np.linspace(0., self.options.L, self.options.Nx)
        U0 = _hat_fun(x, x0=self.options.pick_pos * self.options.L, L=self.options.L)  # Initial condition, evaluated at x
        return np.matmul(self.e_k, U0)
