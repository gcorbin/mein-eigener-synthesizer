# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
import numpy as np


def c2_from_frequency(freq):
    return (freq * 2. * np.pi)**2


def oscillator(t, c2, x0, dx0):
    """
    Evaluate solutions of the harmonic oscillator equation at the given times.

    The oscillator equation is

    x'' + c^2 * x = 0

    with initial condition

    x(0)  = x0
    x'(0) = dx0

    and wave speed c, c > 0.

    The solution is

    x(t) = alpha * cos(c * t) + beta * sin(c * t)
    alpha = x0
    beta = dx0 / c

    :param t: Array. The times at which the oscillator is evaluated.
    :param c2: Coefficient for the zeroth-order term, oscillator frequency is sqrt(c2) / ( 2 * pi)
    :param x0: Initial condition for x.
    :param dx0: Initial condition for dx/dt.
    :return: x(t), i.e. solution of the oscillator evaluated at t.
    """
    assert np.all(c2 > 0)
    c = np.sqrt(c2)
    return x0 * np.cos(c * t) + dx0 / c * np.sin(c * t)


def damped_oscillator_coefficients(freq, halflife):
    """
    Coefficients for a damped oscillator that oscillates with given oscillation
    frequency and halflife for dampening.
    :param freq: Oscillator frequency in Hz.
    :param halflife: Dampening halflife in seconds.
    :return:  Coefficients c2, d for the damped_oscillator function.
    """
    assert np.all(freq > 0)
    assert np.all(halflife > 0)

    d = np.where(np.isinf(halflife),
                0,
                 2. * np.log(2.) / halflife)

    # freq = sqrt( c2 - d^2 / 4 ) / (2 * pi)
    c2 = (freq * 2 * np.pi )**2 + d**2 / 4
    return c2, d


def damped_oscillator(t, c2, d, x0, dx0):
    """
    Evaluate solutions of the damped harmonic oscillator equation at the given times.

    The damped oscillator equation is

    x'' + c^2 * x + d * x' = 0

    with initial condition

    x(0)  = x0
    x'(0) = dx0

    wave speed c, c > 0 and dampening coefficient 0 <= d < 2c.

    The solution is

    x(t) = ( alpha * cos( gamma * t ) + beta * sin( gamma * t ) ) exp ( -d/2 * t ),

    where

    gamma = sqrt( c^2 - d^2 / 4),

    and alpha, beta follow from initial conditions as

    alpha = x0
    beta  = ( dx0 + d / 2 * x0 ) / gamma

    The special case d == 0 leads to the harmonic oscillator without dampening.

    :param t: Array. The times at which the oscillator is evaluated.
    :param c2: Coefficient for the zeroth-order term, oscillator frequency is sqrt( c2 - d^2 / 4 ) / (2 * pi)
    :param d: Coefficient for the first-order term, i.e. dampening coefficient. Must fulfill 0 <= d < 2c.
    :param x0: Initial condition for x.
    :param dx0: Initial condition for dx/dt.
    :return: x(t), i.e. solution of the damped oscillator evaluated at t.
    """
    assert np.all(c2 > 0.), "c^2 must be positive"
    assert np.all(d >= 0.), "Damping coefficient d must be positive"
    assert np.all(d**2 < 4. * c2), "No oscillations for d > 2c"
    gamma = np.sqrt(c2 - d**2 / 4.)

    alpha = x0
    beta = ( dx0 + d / 2 * x0 ) / gamma

    return ( alpha * np.cos(gamma * t) + beta * np.sin(gamma * t) ) * np.exp( - d / 2. * t)



