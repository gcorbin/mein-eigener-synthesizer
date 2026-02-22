# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
import numpy as np
from dataclasses import dataclass

from eigensynth.instruments.instrument import InstrumentOptions, Instrument
from eigensynth.space import cantilevered_beam_eigen


@dataclass(init=True)
class BeamOptions(InstrumentOptions):
    pass


class Beam(Instrument):
    def compute_eigen(self, x):
        return cantilevered_beam_eigen(x, N=self.options.Nk, L=self.options.L, EI=1.)

    def _compute_initial_coefficients(self):
        """
        u0_k = 1 / beta_k^4 / || e_k ||^2 * e_k(L)
        """
        e_k_L, _ = self.compute_eigen(np.array([self.options.L]))
        return e_k_L.reshape((-1,))
