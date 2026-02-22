# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
import numpy as np
from dataclasses import dataclass

from eigensynth.instruments.instrument import InstrumentOptions, Instrument
from eigensynth.space import cantilevered_beam_eigen


@dataclass(init=True)
class BeamOptions(InstrumentOptions):
    pick_pos : float = 1.


class Beam(Instrument):
    def compute_eigen(self, x):
        e_k, lam_k = cantilevered_beam_eigen(x, N=self.options.Nk, L=self.options.L)
        return e_k, -1. * lam_k

    def _compute_initial_coefficients(self):
        """
        u0_k = 1 / beta_k^4 / || e_k ||^2 * e_k(L)

        lambda_k = beta_k^4
        ||e_k||^2 = 1
        """
        e_k_L, _ = self.compute_eigen(np.array([self.options.pick_pos * self.options.L]))
        return 1. / self.lam_k * e_k_L.reshape((-1,))
