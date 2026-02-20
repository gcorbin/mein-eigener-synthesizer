# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
import numpy as np


def oscillator_stiffness_from_frequency(freq):
    return (freq * 2. * np.pi)**2


def oscillator_frequency(k):
    assert k > 0
    return np.sqrt(k) / ( 2 * np.pi)


def oscillator(t, k, x0=1., dx0=0.):
    """
    Evaluate solutions of the harmonic oscillator equation at the given times.

    The oscillator equation is

    x'' + k * x = 0

    with initial condition

    x(0)  = x0
    x'(0) = dx0

    The constant k > 0 is the (normalized) stiffness coefficient.

    The solution is

    x(t) = alpha * cos(c * t) + beta * sin(c * t)
    c     = sqrt(k)
    alpha = x0
    beta  = dx0 / c

    :param t: Array. The times at which the oscillator is evaluated.
    :param k: Stiffness coefficient k > 0. Controls the oscillator frequency f = sqrt(k) / ( 2 * pi)
    :param x0: Initial condition for x.
    :param dx0: Initial condition for dx/dt.
    :return: x(t), i.e. solution of the oscillator evaluated at t.
    """
    assert np.all(k > 0)
    c = np.sqrt(k)
    return x0 * np.cos(c * t) + dx0 / c * np.sin(c * t)


def damped_oscillator_coefficients(freq, halflife):
    """
    Calculate coefficients for a damped oscillator from given frequency and halflife
    for damping.
    :param freq: Oscillator frequency in Hz.
    :param halflife: Damping halflife in seconds.
    :return: Coefficients k, d for the damped_oscillator function.
    """
    assert np.all(freq > 0)
    assert np.all(halflife > 0)

    d = np.where(np.isinf(halflife),
                0,
                 2. * np.log(2.) / halflife)

    # freq = sqrt( k - d^2 / 4 ) / (2 * pi)
    k = (freq * 2 * np.pi )**2 + d**2 / 4
    return k, d


def damped_oscillator(t, k, d, x0=1., dx0=0.):
    """
    Evaluate solutions of the damped harmonic oscillator equation at the given times.

    The damped oscillator equation is

    x'' + k * x + d * x' = 0

    with initial condition

    x(0)  = x0,
    x'(0) = dx0,

    and constant (normalized) stiffness k > 0 and damping 0 <= d < 2c coefficients.

    The solution is

    x(t) = ( alpha * cos( gamma * t ) + beta * sin( gamma * t ) ) exp ( -d/2 * t ),

    where

    gamma = sqrt( k - d^2 / 4),

    and alpha, beta follow from initial conditions as

    alpha = x0
    beta  = ( dx0 + d / 2 * x0 ) / gamma

    The special case d == 0 leads to the harmonic oscillator without damping.

    :param t: Array. The times at which the oscillator is evaluated.
    :param k: Stiffness coefficient k > 0. Controls the oscillator frequency f = sqrt( k - d^2 / 4 ) / (2 * pi).
    :param d: Damping coefficient 0 <= d < 2 * sqrt(k).
    :param x0: Initial condition for x.
    :param dx0: Initial condition for dx/dt.
    :return: x(t), i.e. solution of the damped oscillator evaluated at t.
    """
    assert np.all(k > 0.), "k must be positive"
    assert np.all(d >= 0.), "d must be positive"
    assert np.all(d ** 2 < 4. * k), "d < 2 * sqrt(k)"
    gamma = np.sqrt(k - d ** 2 / 4.)

    alpha = x0
    beta = ( dx0 + d / 2 * x0 ) / gamma

    return ( alpha * np.cos(gamma * t) + beta * np.sin(gamma * t) ) * np.exp( - d / 2. * t)
