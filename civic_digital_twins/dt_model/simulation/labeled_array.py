"""Optional self-describing wrapper around a marginalized result array.

:class:`LabeledArray` pairs a plain :class:`numpy.ndarray` with the
:class:`~simulation.axis_layout.AxisLayout` that names its dimensions.  It is
purely additive: :meth:`~simulation.evaluation.EvaluationResult.__getitem__`
and :meth:`~simulation.evaluation.EvaluationResult.expected_value` keep
returning bare ``np.ndarray`` — this type is reached only via
:meth:`~simulation.evaluation.EvaluationResult.labeled`, for callers who want
name-based selection or interop with a labeled-array ecosystem.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import numpy as np

from .axis_layout import AxisLayout

__all__ = ["LabeledArray"]


class LabeledArray:
    """A numpy array paired with the :class:`~simulation.axis_layout.AxisLayout` naming its dimensions.

    Parameters
    ----------
    values:
        The array. Its rank and per-dimension sizes must match *layout*.
    layout:
        The axis layout describing *values*' dimensions, in order.
    """

    __slots__ = ("layout", "values")

    def __init__(self, values: np.ndarray, layout: AxisLayout) -> None:
        if values.shape != layout.full_shape:
            raise ValueError(
                f"LabeledArray: values.shape {values.shape} does not match layout.full_shape {layout.full_shape}."
            )
        self.values = values
        self.layout = layout

    @property
    def dims(self) -> tuple[str, ...]:
        """Axis names, in dimension order."""
        return tuple(ax.name for ax in self.layout.axes)

    def sel(self, **selection: int | slice) -> LabeledArray:
        """Select along axes named by keyword, by integer position or slice.

        An integer selection drops that dimension from the result; a slice
        keeps it, at its (possibly reduced) size. Unlike xarray's ``.sel``,
        selection is by *position*, not by coordinate value — layouts carry
        axis names and sizes, not per-index coordinate labels.

        Examples
        --------
        >>> import numpy as np
        >>> from civic_digital_twins.dt_model.axes import DOMAIN, Axis
        >>> from civic_digital_twins.dt_model.simulation.axis_layout import AxisLayout
        >>> time, x, y = Axis("time", DOMAIN), Axis("x", DOMAIN), Axis("y", DOMAIN)
        >>> values = np.arange(2 * 3 * 4).reshape(2, 3, 4)
        >>> la = LabeledArray(values, AxisLayout([(time, 2), (x, 3), (y, 4)]))
        >>> la.dims
        ('time', 'x', 'y')

        An integer selection fixes and drops that axis:

        >>> la.sel(x=0).dims
        ('time', 'y')

        A slice keeps the axis, just shortened — the same distinction as
        plain numpy indexing with an int versus a slice:

        >>> la.sel(y=slice(0, 2)).dims
        ('time', 'x', 'y')
        >>> la.sel(y=slice(0, 2)).values.shape
        (2, 3, 2)

        Raises
        ------
        KeyError
            If a keyword names an axis not present in :attr:`dims`.
        """
        indexer: list[Any] = [slice(None)] * len(self.layout)
        kept_axes = list(self.layout.axes)
        for name, value in selection.items():
            axis = self.layout.find_axis(name)
            if axis is None:
                raise KeyError(f"LabeledArray.sel: no axis named {name!r}; have {self.dims}.")
            indexer[self.layout.position_of(axis)] = value
            if isinstance(value, int):
                kept_axes.remove(axis)
        new_values = self.values[tuple(indexer)]
        new_layout = AxisLayout(zip(kept_axes, new_values.shape, strict=True))
        return LabeledArray(new_values, new_layout)

    def to_xarray(self) -> Any:
        """Convert to an :class:`xarray.DataArray`.

        Requires the optional ``xarray`` package — not a dependency of this
        library, so the import happens here, lazily, rather than at module
        load time.

        Raises
        ------
        ImportError
            If ``xarray`` is not installed.
        """
        try:
            import xarray as xr  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError(
                "LabeledArray.to_xarray() requires the optional 'xarray' package, which is not installed."
            ) from exc
        return xr.DataArray(self.values, dims=self.dims)  # pragma: no cover

    def __repr__(self) -> str:
        """Return a concise, dims-and-shape representation."""
        return f"LabeledArray(dims={self.dims!r}, shape={self.values.shape!r})"
