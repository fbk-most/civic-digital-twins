"""Guarded-region array operations over the leading evaluation layout.

Regional plan execution evaluates guarded regions only at the leading-axis
coordinates (``(*PARAMETER, *ENSEMBLE)``) selected by their guards.  This
module owns the node-aware array operations that support it: masking
coordinates by selector value, gathering already-known values into a
branch-local state, scattering branch results back into the full layout, and
the fill policy for inactive coordinates.

The pure shape arithmetic lives in the leading-array primitives of
:class:`~simulation.axis_layout.AxisLayout`; this module adds the
region-execution semantics on top: ``variant_selector`` sentinels pass
through untouched, DOMAIN-carrying nodes get broadcast treatment based on
their declared ``output_axes``, and scalar values are padded with a trailing
singleton when the evaluation carries timeseries.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..axes import DOMAIN, Axis
from ..engine.frontend import graph
from .axis_layout import AxisLayout

__all__ = ["RegionArrayOps"]


def _has_domain_axis(node: graph.Node) -> bool:
    """Return True when the node's output carries a DOMAIN axis."""
    return any(ax.role == DOMAIN for ax in node.output_axes)


def _branch_fill_value(dtype: np.dtype) -> tuple[np.dtype, Any]:
    """Return an output dtype and inactive-branch fill value for *dtype*."""
    if dtype.kind in {"f", "c"}:
        return dtype, np.nan
    if dtype.kind == "b":
        return dtype, False
    if dtype.kind in {"i", "u"}:
        return dtype, 0
    return np.dtype(object), None


@dataclass(frozen=True)
class RegionArrayOps:
    """Node-aware mask/gather/scatter operations for guarded-region execution.

    Bound to one execution context: the leading (PARAMETER + ENSEMBLE)
    layout and the DOMAIN axes the evaluation carries as trailing
    dimensions.

    Parameters
    ----------
    layout:
        The leading evaluation layout (no DOMAIN axes yet — they are
        appended to the result layout only after execution).
    domain_axes:
        The evaluation's DOMAIN axes, in canonical order.  Scalar
        (non-DOMAIN) values are padded with one trailing singleton per axis
        so they broadcast against domain-carrying values.
    """

    layout: AxisLayout
    domain_axes: tuple[Axis, ...] = ()

    @property
    def n_domain(self) -> int:
        """Number of trailing DOMAIN dimensions this evaluation reserves."""
        return len(self.domain_axes)

    def _align_to_leading(self, node: graph.Node, value: Any) -> np.ndarray:
        """Return *value* normalised and broadcast over the leading layout.

        Values without explicit leading dimensions gain them first: a raw
        DOMAIN-only value (e.g. a timeseries constant with shape ``(T,)``)
        is recognised by its node's ``output_axes`` and prepended even when
        ``T`` accidentally equals a leading-axis size.
        """
        arr = np.asarray(value)
        if self.layout.n_leading == 0 or isinstance(node, graph.variant_selector):
            return arr
        if _has_domain_axis(node) and arr.ndim == len(node.output_axes):
            arr = self.layout.prepend_leading(arr)
        elif not self.layout.spans_leading(arr.shape):
            arr = self.layout.prepend_leading(arr)
        return self.layout.broadcast_leading(arr)

    def selector_mask(self, node: graph.Node, value: Any, branch_key: str) -> np.ndarray:
        """Return a boolean mask over the leading layout where *value* equals *branch_key*.

        Raises
        ------
        NotImplementedError
            If the selector value varies along a DOMAIN axis (non-singleton
            trailing dimensions).
        """
        sel = self._align_to_leading(node, value)
        n = self.layout.n_leading
        if n == 0:
            mask = np.asarray(sel == branch_key)
            if mask.shape != ():
                if any(dim > 1 for dim in mask.shape):
                    raise NotImplementedError(
                        "Regional execution does not support selectors with non-singleton DOMAIN axes."
                    )
                mask = mask.reshape(())
            return mask
        trailing = sel.shape[n:]
        if trailing:
            if any(dim > 1 for dim in trailing):
                raise NotImplementedError(
                    "Regional execution does not support selectors with non-singleton DOMAIN axes."
                )
            sel = sel.reshape(sel.shape[:n])
        return np.broadcast_to(sel == branch_key, self.layout.leading_shape)

    def gather(self, node: graph.Node, value: Any, flat_idx: np.ndarray) -> np.ndarray:
        """Gather selected leading-axis coordinates into a branch-local first axis."""
        if self.layout.n_leading == 0 or isinstance(node, graph.variant_selector):
            return np.asarray(value)
        return self.layout.take_leading(self._align_to_leading(node, value), flat_idx)

    def scatter(self, node: graph.Node, value: Any, flat_idx: np.ndarray) -> np.ndarray:
        """Scatter a branch-local value back into the full leading layout.

        Inactive coordinates are filled with a branch-neutral placeholder
        chosen by dtype (NaN for floats, False for booleans, 0 for integers,
        None for object arrays).

        Raises
        ------
        ValueError
            If a non-DOMAIN value's first dimension does not enumerate the
            selected coordinates.
        """
        arr = np.asarray(value)
        if self.layout.n_leading == 0 or isinstance(node, graph.variant_selector):
            return arr
        k = int(flat_idx.size)
        if arr.ndim == 0:
            arr = np.broadcast_to(arr, (k,)).copy()
        elif arr.shape[0] == 1 and k != 1:
            arr = np.broadcast_to(arr, (k,) + arr.shape[1:]).copy()
        elif arr.shape[0] != k:
            # DOMAIN-only values produced inside the branch (shape (T,)) are
            # invariant across selected leading coordinates.
            if _has_domain_axis(node):
                arr = np.broadcast_to(arr, (k,) + arr.shape).copy()
            else:
                raise ValueError(
                    f"Regional scatter for node {getattr(node, 'name', repr(node))!r}: "
                    f"branch result first dimension {arr.shape[0]} does not match selected size {k}."
                )
        if self.n_domain and not _has_domain_axis(node) and arr.ndim == 1:
            arr = arr.reshape(arr.shape + (1,) * self.n_domain)
        out_dtype, fill_value = _branch_fill_value(arr.dtype)
        full_flat = np.full((self.layout.leading_size,) + arr.shape[1:], fill_value, dtype=out_dtype)
        full_flat[flat_idx] = arr.astype(out_dtype, copy=False)
        return self.layout.unflatten_leading(full_flat)

    def empty_branch_value(self) -> np.ndarray:
        """Create a broadcast-compatible inactive value for an unselected branch."""
        trailing = (1,) * self.n_domain
        return np.full(self.layout.leading_shape + trailing, np.nan, dtype=float)
