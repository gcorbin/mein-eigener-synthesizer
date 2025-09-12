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
def main():
    pass

if __name__ == '__main__':
    main()