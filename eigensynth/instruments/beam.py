# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
from dataclasses import dataclass

from eigensynth.instruments.instrument import InstrumentOptions, Instrument
from eigensynth.space import cantilevered_beam_eigen

__all__ = ['BeamOptions', 'Beam']


@dataclass(init=True)
class BeamOptions(InstrumentOptions):
    pick_pos: float = 1.


class Beam(Instrument):
    def compute_eigen(self, x):
        e_k, lam_k = cantilevered_beam_eigen(x, N=self.options.Nk, L=self.options.L)
        return e_k, lam_k
