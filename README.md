<!--
SPDX-FileCopyrightText: None

SPDX-License-Identifier: CC0-1.0
-->

# Mein eigener Synthesizer

Makes sounds from oscillators.

An oscillator is a differential equation of the form 

d_tt u + d * d_t u + D(u) = 0

If u = u(t), D(u) = c^2 * u, this is an ordinary differential equation

d_tt u + d * d_t u + c^2 u = 0


If u = u(t, x), this is a partial differential equation with some differential operator D(u) in x.
The operator D should allow diagonalization via eigenfunctions. 