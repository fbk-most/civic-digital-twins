"""Shared numpy kernels for operations with no single built-in numpy equivalent.

Every other node kind the numpy backend evaluates maps onto one existing
numpy function (``np.sum``, ``np.roll``, ``np.quantile``, ...), so
``executor.py`` and ``numpy_ast.py`` each just name that function — there is
nothing to share. ``shift`` (fill-padded, unlike the single-call ``np.roll``)
and ``laplacian`` (a multi-axis finite-difference stencil) are the first
operations whose actual numeric logic does not reduce to one numpy call, so
that logic needs to live *somewhere* both the interpreter (``executor.py``)
and the debug codegen (``numpy_ast.py``) can reach without duplicating it —
two independent copies would drift the first time either one got a bugfix or
optimization the other didn't. This module is that one place.

``executor.py`` already imports ``numpy_ast`` (for trace printing), so
``numpy_ast`` importing back from ``executor`` would cycle; both instead
import from here.
"""

# SPDX-License-Identifier: Apache-2.0

from typing import Literal

import numpy as np


def shift(x: np.ndarray, periods: int, *, axis: int, fill_value: float) -> np.ndarray:
    """Shift *x* along *axis* by *periods*, filling exposed positions with *fill_value*.

    Args:
        x: The input array to shift.
        periods: Number of positions to shift by (may be negative).
        axis: The axis along which to shift.
        fill_value: Value used for positions exposed at the boundary.

    Returns
    -------
        Array of the same shape as *x*, with values moved by *periods*
        positions along *axis* and the exposed boundary filled.
    """
    if periods == 0:
        return x.copy()
    result = np.roll(x, periods, axis=axis)
    idx: list[slice] = [slice(None)] * x.ndim
    idx[axis] = slice(0, periods) if periods > 0 else slice(periods, None)
    result[tuple(idx)] = fill_value
    return result


_PAD_MODE_FOR_BOUNDARY: dict[str, Literal["reflect", "constant", "wrap", "edge"]] = {
    "reflect": "reflect",
    "constant": "constant",
    "wrap": "wrap",
    "nearest": "edge",
}
"""Maps a SpaceType boundary policy to the corresponding numpy.pad mode.

"nearest" maps to numpy's "edge" — numpy has no mode named "nearest";
"edge" (repeat the boundary value) is what "nearest" describes."""


def _laplacian_1axis(x: np.ndarray, axis: int, spacing: float, boundary: str) -> np.ndarray:
    """Second derivative of *x* along *axis* via central differences, respecting *boundary*.

    Args:
        x: The input array to differentiate.
        axis: The axis along which to compute the second derivative.
        spacing: Grid spacing between samples along *axis*.
        boundary: One of "reflect", "constant", "wrap", "nearest" — how
            values just outside the array are extrapolated for the
            central-difference stencil at the boundary.

    Returns
    -------
        Array of the same shape as *x*: the discrete second derivative
        ``(x[i-1] - 2*x[i] + x[i+1]) / spacing**2`` along *axis*.
    """
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (1, 1)
    padded = np.pad(x, pad_width, mode=_PAD_MODE_FOR_BOUNDARY[boundary])
    n = x.shape[axis]
    left = np.take(padded, range(0, n), axis=axis)
    right = np.take(padded, range(2, n + 2), axis=axis)
    return (left - 2.0 * x + right) / spacing**2


def laplacian(
    x: np.ndarray, axes: tuple[int, ...], spacings: tuple[float, ...], boundaries: tuple[str, ...]
) -> np.ndarray:
    """Sum of second partial derivatives of *x* over *axes* (an isotropic Laplacian).

    Args:
        x: The input array to differentiate.
        axes: The axes to sum the second derivative over.
        spacings: Grid spacing for each axis in *axes*, same length and order.
        boundaries: Boundary policy for each axis in *axes*, same length and order.
    """
    result = np.zeros_like(x, dtype=float)
    for axis, spacing, boundary in zip(axes, spacings, boundaries, strict=True):
        result = result + _laplacian_1axis(x, axis, spacing, boundary)
    return result
