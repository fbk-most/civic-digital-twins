"""Shared numpy kernels for operations with no single built-in numpy equivalent.

Every other node kind the numpy backend evaluates maps onto one existing
numpy function (``np.sum``, ``np.roll``, ``np.quantile``, ...), so
``executor.py`` and ``numpy_ast.py`` each just name that function — there is
nothing to share. ``shift`` (fill-padded, unlike the single-call ``np.roll``),
``gradient``, and ``laplacian`` (finite-difference stencils honoring a
:class:`~..axes.BoundaryCondition`, sharing the ghost-padding logic in
``_ghost_pad``) are the first operations whose actual numeric logic does not
reduce to one numpy call, so that logic needs to live *somewhere* both the
interpreter (``executor.py``) and the debug codegen (``numpy_ast.py``) can
reach without duplicating it — two independent copies would drift the first
time either one got a bugfix or optimization the other didn't. This module is
that one place.

``executor.py`` already imports ``numpy_ast`` (for trace printing), so
``numpy_ast`` importing back from ``executor`` would cycle; both instead
import from here.
"""

# SPDX-License-Identifier: Apache-2.0

import numpy as np

from ...axes import BoundaryCondition, Constant, Linear, Nearest, Neumann, Wrap


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


def _neumann_pad(x: np.ndarray, axis: int, value: float, spacing: float) -> np.ndarray:
    """Pad *x* by one ghost cell on each side of *axis*, per a ``Neumann(value)`` boundary.

    The ghost value on each side is chosen so the central-difference
    derivative *at the border node* — ``(f[1] - ghost_left) / (2*spacing)``
    on the left, ``(ghost_right - f[-2]) / (2*spacing)`` on the right —
    equals *value* exactly, with the sign convention that *value* is the
    derivative in the direction of increasing index (so it has the *same*
    sign on both borders for a field with a genuinely constant slope):

        ghost_left  = neighbor_of_neighbor - 2 * spacing * value
        ghost_right = neighbor_of_neighbor + 2 * spacing * value

    ``value=0.0`` is the classical "reflecting" boundary on both sides
    (``ghost = neighbor_of_neighbor``, sign is irrelevant when it's zero).

    Raises
    ------
    IndexError
        If *x* has fewer than 2 points along *axis* (there is no
        "neighbor of neighbor" to anchor the ghost value to).
    """
    left_neighbor = np.take(x, [1], axis=axis)
    right_neighbor = np.take(x, [-2], axis=axis)
    offset = 2.0 * spacing * value
    return np.concatenate([left_neighbor - offset, x, right_neighbor + offset], axis=axis)


def _ghost_pad(x: np.ndarray, axis: int, spacing: float, boundary: BoundaryCondition) -> np.ndarray:
    """Pad *x* with one ghost cell on each side of *axis*, per *boundary*.

    Both :func:`gradient` and :func:`laplacian` read from this same
    ghost-padded array with a central-difference stencil, including at the
    two boundary positions — a boundary condition is really just a rule for
    filling in one value past each edge.

    Args:
        x: The input array to pad.
        axis: The axis to pad.
        spacing: Grid spacing along *axis* (only used by ``Neumann``).
        boundary: The boundary-condition policy.

    Returns
    -------
        Array shaped like *x* except *axis*, which grows by 2.
    """
    if isinstance(boundary, Neumann):
        return _neumann_pad(x, axis, boundary.value, spacing)
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (1, 1)
    if isinstance(boundary, Nearest):
        return np.pad(x, pad_width, mode="edge")
    if isinstance(boundary, Wrap):
        return np.pad(x, pad_width, mode="wrap")
    if isinstance(boundary, Linear):
        if x.shape[axis] < 2:
            # Unlike Neumann's manual indexing, np.pad's odd-reflect mode
            # doesn't error here on its own — it silently degrades to a
            # flat extrapolation instead, which would hide a genuine
            # "not enough points" mistake rather than surface it.
            raise IndexError(f"kernels: Linear boundary needs at least 2 points along axis {axis}, got {x.shape[axis]}")
        return np.pad(x, pad_width, mode="reflect", reflect_type="odd")
    if isinstance(boundary, Constant):
        return np.pad(x, pad_width, mode="constant", constant_values=boundary.value)
    raise TypeError(f"kernels: unsupported boundary condition: {boundary!r}")


def gradient(x: np.ndarray, axis: int, spacing: float, boundary: BoundaryCondition) -> np.ndarray:
    """First derivative of *x* along *axis* via central differences, respecting *boundary*.

    Args:
        x: The input array to differentiate.
        axis: The axis along which to differentiate.
        spacing: Grid spacing between samples along *axis*.
        boundary: How values just outside the array are extrapolated for
            the central-difference stencil at the boundary.

    Returns
    -------
        Array of the same shape as *x*: the discrete first derivative
        ``(x[i+1] - x[i-1]) / (2 * spacing)`` along *axis*, including at the
        two boundary positions (unlike ``np.gradient``, which falls back to
        a one-sided formula there instead of honoring *boundary*).
    """
    padded = _ghost_pad(x, axis, spacing, boundary)
    n = x.shape[axis]
    left = np.take(padded, range(0, n), axis=axis)
    right = np.take(padded, range(2, n + 2), axis=axis)
    return (right - left) / (2.0 * spacing)


def _laplacian_1axis(x: np.ndarray, axis: int, spacing: float, boundary: BoundaryCondition) -> np.ndarray:
    """Second derivative of *x* along *axis* via central differences, respecting *boundary*.

    Args:
        x: The input array to differentiate.
        axis: The axis along which to compute the second derivative.
        spacing: Grid spacing between samples along *axis*.
        boundary: How values just outside the array are extrapolated for
            the central-difference stencil at the boundary.

    Returns
    -------
        Array of the same shape as *x*: the discrete second derivative
        ``(x[i-1] - 2*x[i] + x[i+1]) / spacing**2`` along *axis*.
    """
    padded = _ghost_pad(x, axis, spacing, boundary)
    n = x.shape[axis]
    left = np.take(padded, range(0, n), axis=axis)
    right = np.take(padded, range(2, n + 2), axis=axis)
    return (left - 2.0 * x + right) / spacing**2


def laplacian(
    x: np.ndarray,
    axes: tuple[int, ...],
    spacings: tuple[float, ...],
    boundaries: tuple[BoundaryCondition, ...],
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
