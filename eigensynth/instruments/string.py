# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
from dataclasses import dataclass

import numpy as np

from eigensynth.instruments.instrument import InstrumentOptions, Instrument
from eigensynth.space import laplace_1d_eigen

__all__ = ['StringOptions', 'String']


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

    def compute_coefficients(self, fun):
        u = fun(self.x)  # Initial condition, evaluated at x
        # This is integration of the piecewise-constant reconstructions of U0 and e_k
        integration_weights = np.ones_like(u) * (self.options.L / (self.options.Nx - 1))
        integration_weights[0] *= 0.25
        integration_weights[-1] *= 0.25
        return np.matmul(self.e_k.transpose(), u * integration_weights)
