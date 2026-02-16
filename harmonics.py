# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
"""
Solve the 1-D wave equation

    d_tt u = c d_xx u on Omega = [0,L]

with Dirichlet boundary conditions

    u(t,0) = u(t,L) = 0

and initial condition corresponding to a point force

    d_xx u(0,x) = -1 * delta(x - x_0)
    d_x(0, x)   = 0


With the non-dimensional quantities x_r, t_r

    x = L * x_r
    t = L / sqrt(c) * t_r

this is transformed (and dropping the _r) to

    d_tt u = d_xx u on [0,1]

Orthonormal eigenfunctions of the 1-D Laplacian
(d_xx on [0,1], with the 0 Dirichlet BC) are:

    u_k(x) = sin(lambda_k x) = sin(pi * k * x)
    <u_k, u_l> = integral u_k(x) u_l(x) dx from 0 to 1
               = | 1/2 if k==l,
                 | 0   if k!=l

This gives the solution for u as an infinite series

    u(t,x) = sum_{k=1} A_k(t) u_k
           = sum_{k=1} A_k(t) sin(pi * k * x)
           = sum_{k=1} a_k cos(pi * k * t) * sin(pi * k * x)

The form of the time-dependent coefficients A_k(t) = a_k cos(pi * k * t) follows
from d_tt u = d_xx u with the IC d_x = 0.
Finally, it remains to determine the coefficients a_k from the first IC:

    d_xx u(0,x) = sum_{k=1} a_k * -1 * pi^2 * k^2 * sin(pi * k * x) == -1 * delta(x - x_0)

Multiply with the l-th eigenfunction and integrate to get

    <d_xx u, u_l> = 1/2 * a_l * pi^2 * l^2 = <delta(x-x0), u_l> = u_l(x_0) = sin(pi * l * x_0)
    a_l = 2 / ( pi^2 * l^2 ) sin(pi * l * x_0)
"""
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


def main():
    args = parse_args()
    print(f'pick position = {args.pick_position}')
    print(f'hole position = {args.hole_position[0]}, {args.hole_position[1]}')

    idx = np.linspace(1,args.num_terms, args.num_terms)
    coeffs = coefficients(args.num_terms, args.pick_position)
    x = np.linspace(0., 1., 201)
    basis = eval_basis(args.num_terms, x)
    u0 = np.matmul(basis, coeffs)

    fig, ax = plt.subplots(4,1)

    ax[0].xaxis.set_major_locator(ticker.MaxNLocator(10, integer=True))
    ax[0].bar(idx, np.abs(coeffs))

    ax[1].plot(x, initial_condition(args.pick_position, x), label=f'Exact initial condition')
    for n in range(1, min(args.num_terms,3)+1):
        ax[1].plot(x, np.matmul(basis[:, 0:n], coeffs[0:n]), label=f'Expansion with {n} terms')
    ax[1].plot(x, u0, label=f'Series expansion of u_0 with {args.num_terms} terms')
    ax[1].legend()

    mask = np.where(np.logical_and(args.hole_position[0] < x, x < args.hole_position[1]),
                          1.,
                          0.)
    time_coeffs = coeffs * np.matmul(mask, basis)
    t = np.linspace(0, 10, 201)
    time_basis = eval_time_basis(args.num_terms, t)

    ax[2].xaxis.set_major_locator(ticker.MaxNLocator(10, integer=True))
    ax[2].bar(idx, np.abs(time_coeffs))

    ax[3].plot(t, np.matmul(time_basis, time_coeffs))

    plt.show()


def parse_args():
    parser = ArgumentParser(description='Visualize the vibration modes of a guitar string')
    parser.add_argument('--num-terms', '-N', type=int, default=20)
    parser.add_argument('--pick-position', '-x0', type=float_in_interval(0.,1.), default='0.5')
    parser.add_argument('--hole_position', '-H', type=float_in_interval(0., 1.), nargs=2, default=(0.666, 0.8) )
    args = parser.parse_args()
    return args


def float_in_interval(lb: float, ub: float):
    def float_in_interval(f: str):
        number = float(f)
        if number < lb or number > ub:
            raise ValueError(f'{number} is outside the range [{lb},{ub}]')
        return number
    return float_in_interval


def coefficients(num_terms: int, x0: float):
    idx = np.linspace(1,num_terms,num_terms)
    return 2. * np.pi**-2 * np.power(idx, -2) * np.sin(np.pi * idx * x0)


def eval_basis(num_terms: int, x):
    b = np.zeros((x.size, num_terms))
    for i in range(num_terms):
        b[:, i] = np.sin(np.pi * (i+1) * x)
    return b


def eval_time_basis(num_terms: int, t):
    b = np.zeros((t.size, num_terms))
    for i in range(num_terms):
        b[:, i] = np.cos(np.pi * (i+1) * t)
    return b


def initial_condition(x0, x):
    return np.where(x<x0,(1-x0)*x, x0*(1.-x))

if __name__ == '__main__':
    main()