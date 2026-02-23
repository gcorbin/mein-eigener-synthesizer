# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from eigensynth.space.cylindrical_shell import cylindrical_shell_eigen


def main():
    args = parse_command_line_args()

    visualize_initial_condition()
    return 0


def visualize_initial_condition():
    L = 2
    a = 0.5
    shell_constant = 5e5
    Nx = [100,100]
    z = L * np.linspace(0., 1., Nx[0] + 1)
    phi = 2. * np.pi * np.linspace(0., 1., Nx[1] + 1)
    Z, Phi = np.meshgrid(z, phi)

    (m,n) = (20,20)
    e_k, gamma_k = cylindrical_shell_eigen((Z,Phi), (m,n), (L,a), shell_constant)
    e0, _ = cylindrical_shell_eigen(np.meshgrid(np.array([0.8*L]), np.array([0.])), (m,n), (L,a), shell_constant)
    c0 = e0.reshape(-1) / gamma_k
    w0 = np.sum(c0.reshape(1,1,-1) * e_k, axis=2)

    #print(gamma_k.reshape(m,n))
    print(np.amax(w0))
    R = a * np.ones_like(Z) +  0.1 / np.amax(w0) * w0
    X = R * np.cos(Phi)
    Y = R * np.sin(Phi)

    # Plot the surface
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X, Y, Z, vmin=Z.min() * 2, facecolors=cm.Blues(R/np.amax(R)), rcount=100, ccount=100)
    ax.set_xlabel('x')
    ax.set_ylabel('y')


    plt.show()


def parse_command_line_args():
    parser = ArgumentParser(description='Console app stub')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    return_code = main()
