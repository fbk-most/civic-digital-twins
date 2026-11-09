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

from ..axes import Axis, declared_labels
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

    Notes
    -----
    A dimension whose axis is typed ``SetType(labels=...)``, and whose size
    equals the number of labels, carries those labels (:attr:`labels`) and
    can be selected by label or by a list of labels. Selecting a slice or a
    list of labels keeps the matching labels.
    """

    __slots__ = ("layout", "values", "_labels")

    def __init__(self, values: np.ndarray, layout: AxisLayout) -> None:
        if values.shape != layout.full_shape:
            raise ValueError(
                f"LabeledArray: values.shape {values.shape} does not match layout.full_shape {layout.full_shape}."
            )
        self.values = values
        self.layout = layout
        self._labels: dict[str, tuple[str, ...]] = {}
        for ax, size in layout.entries:
            labels = declared_labels(ax)
            if labels is not None and len(labels) == size:
                self._labels[ax.name] = labels

    @property
    def labels(self) -> dict[str, tuple[str, ...]]:
        """Labels of the labelled dimensions, by axis name (a fresh dict; unlabelled axes are absent)."""
        return dict(self._labels)

    @property
    def dims(self) -> tuple[str, ...]:
        """Axis names, in dimension order."""
        return tuple(ax.name for ax in self.layout.axes)

    def sel(self, **selection: int | slice | str | list[str] | tuple[str, ...]) -> LabeledArray:
        """Select along axes named by keyword, by integer position, slice, label, or list of labels.

        An integer or a label selects one position and drops that dimension
        from the result; a slice keeps it, at its (possibly reduced) size,
        together with the matching labels. A list (or tuple) of labels keeps
        the dimension with those members, in the order given. Labels (``str``)
        are accepted on labelled dimensions only (see :attr:`labels`); slices
        select by position.

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

        A label selects by name on a labelled dimension:

        >>> from civic_digital_twins.dt_model.axes import DomainAxis, SetType
        >>> zone = DomainAxis("zone", type=SetType(labels=("centre", "north", "south")))
        >>> la = LabeledArray(np.array([10.0, 20.0, 30.0]), AxisLayout([(zone, 3)]))
        >>> float(la.sel(zone="north").values)
        20.0
        >>> la.sel(zone=slice(1, 3)).labels
        {'zone': ('north', 'south')}
        >>> la.sel(zone=["south", "centre"]).values
        array([30., 10.])

        Raises
        ------
        KeyError
            If a keyword names an axis not present in :attr:`dims`, or a label
            is not among the labels of its dimension, or the dimension has no
            labels.
        ValueError
            If a list of labels is empty or repeats a label.
        TypeError
            If a slice bound is a label, or a list contains a non-label.
        """
        indexer: list[Any] = [slice(None)] * len(self.layout)
        kept_axes = list(self.layout.axes)
        new_labels = dict(self._labels)
        picks: dict[Axis, list[int]] = {}
        for name, value in selection.items():
            axis = self.layout.find_axis(name)
            if axis is None:
                raise KeyError(f"LabeledArray.sel: no axis named {name!r}; have {self.dims}.")
            if isinstance(value, (list, tuple)):
                picks[axis] = [self._label_position(name, label) for label in self._check_label_list(name, value)]
                new_labels[name] = tuple(value)
                continue
            if isinstance(value, slice) and any(isinstance(b, str) for b in (value.start, value.stop)):
                raise TypeError(
                    f"LabeledArray.sel: axis {name!r}: slices select by position; "
                    f"to select several labels, pass a list of labels."
                )
            if isinstance(value, str):
                value = self._label_position(name, value)
            indexer[self.layout.position_of(axis)] = value
            if isinstance(value, int):
                kept_axes.remove(axis)
                new_labels.pop(name, None)
            elif name in new_labels:
                new_labels[name] = new_labels[name][value]
        new_values = self.values[tuple(indexer)]
        # Lists are applied one axis at a time: indexing several axes with
        # lists in one step would pair their entries instead of crossing them.
        for axis, positions in picks.items():
            new_values = np.take(new_values, positions, axis=kept_axes.index(axis))
        new_layout = AxisLayout(zip(kept_axes, new_values.shape, strict=True))
        result = LabeledArray(new_values, new_layout)
        result._labels = new_labels
        return result

    def _label_position(self, name: str, label: str) -> int:
        labels = self._labels.get(name)
        if labels is None:
            raise KeyError(f"LabeledArray.sel: axis {name!r} has no labels; select it by position.")
        if label not in labels:
            raise KeyError(f"LabeledArray.sel: {label!r} is not a label of axis {name!r}; labels are {labels}.")
        return labels.index(label)

    @staticmethod
    def _check_label_list(name: str, labels: list[str] | tuple[str, ...]) -> list[str] | tuple[str, ...]:
        if not labels:
            raise ValueError(f"LabeledArray.sel: axis {name!r}: the list of labels is empty.")
        for label in labels:
            if not isinstance(label, str):
                raise TypeError(
                    f"LabeledArray.sel: axis {name!r}: a list selects by label, but {label!r} is not a label."
                )
        if len(set(labels)) != len(labels):
            raise ValueError(f"LabeledArray.sel: axis {name!r}: the list of labels {list(labels)} repeats a label.")
        return labels

    def to_xarray(self) -> Any:
        """Convert to an :class:`xarray.DataArray`.

        Requires the optional ``xarray`` package — not a dependency of this
        library, so the import happens here, lazily, rather than at module
        load time. Labelled dimensions (:attr:`labels`) become coordinates.

        Raises
        ------
        ImportError
            If ``xarray`` is not installed.
        """
        coords = {name: list(labels) for name, labels in self._labels.items()}
        try:
            import xarray as xr  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError(
                "LabeledArray.to_xarray() requires the optional 'xarray' package, which is not installed."
            ) from exc
        return xr.DataArray(self.values, dims=self.dims, coords=coords)  # pragma: no cover

    def __repr__(self) -> str:
        """Return a concise, dims-and-shape representation."""
        return f"LabeledArray(dims={self.dims!r}, shape={self.values.shape!r})"
