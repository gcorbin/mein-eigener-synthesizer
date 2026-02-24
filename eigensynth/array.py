# SPDX-FileCopyrightText: None
#
# SPDX-License-Identifier: CC0-1.0
from numpy.typing import NDArray


def move_axes_after(a: NDArray, b: NDArray) -> NDArray:
    return a.reshape((1,) * b.ndim + a.shape)


def move_axes_before(a: NDArray, b: NDArray) -> NDArray:
    return a.reshape(a.shape + (1,) * b.ndim)


def outer_product_nd(a: NDArray, b: NDArray) -> NDArray:
    """
    Outer product of two Nd arrays.

    If a.shape = (r0,...,rN) and b.shape = (s0,...,sM), the result has
    shape (r0,...,rN,s0,...,sM)
    :param a: Shape (r0,...,rN)
    :param b: Shape (s0,...,sM)
    :return: Shape (r0,...,rN,s0,...,sM)
    """
    return move_axes_before(a, b) * move_axes_after(b, a)
