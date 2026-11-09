"""Self-describing result array.

:class:`LabeledArray` pairs a plain :class:`numpy.ndarray` with the axes
that name its dimensions, in any order. The
result readers :meth:`~simulation.evaluation.EvaluationResult.expected` and
:meth:`~simulation.evaluation.EvaluationResult.samples` return one, with
its axes in the order given by their ``axes=``. It supports selection by
axis name (:meth:`LabeledArray.isel` by position, :meth:`LabeledArray.sel`
by label), changing the order of its axes (:meth:`LabeledArray.transpose`)
and conversion to xarray.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import operator
import warnings
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from ..axes import Axis, declared_labels
from .axis_layout import AxisLayout

__all__ = ["LabeledArray"]


class LabeledArray:
    """A numpy array paired with the axes naming its dimensions.

    Parameters
    ----------
    values:
        The array. Its rank and per-dimension sizes must match *layout*.
    layout:
        The ``(axis, size)`` pairs describing *values*' dimensions, in
        dimension order: an :class:`~simulation.axis_layout.AxisLayout`, or
        any iterable of pairs. Axes of different roles may come in any
        order; axis names must be distinct.

    Notes
    -----
    A dimension whose axis is typed ``SetType(labels=...)``, and whose size
    equals the number of labels, carries those labels (:attr:`labels`) and
    can be selected by label or by a list of labels (:meth:`sel`). Every
    dimension can be selected by position (:meth:`isel`). Selections that
    keep a dimension keep its matching labels.
    """

    __slots__ = ("values", "_axes", "_sizes", "_labels", "_label_index")

    def __init__(self, values: np.ndarray, layout: AxisLayout | Iterable[tuple[Axis, int]]) -> None:
        entries = layout.entries if isinstance(layout, AxisLayout) else tuple(layout)
        self._axes: tuple[Axis, ...] = tuple(ax for ax, _ in entries)
        self._sizes: tuple[int, ...] = tuple(int(size) for _, size in entries)
        names = self._names()
        if len(set(names)) != len(names):
            raise ValueError(f"LabeledArray: duplicate axis names in {list(names)}.")
        if values.shape != self._sizes:
            raise ValueError(f"LabeledArray: values.shape {values.shape} does not match the axis sizes {self._sizes}.")
        self.values = values
        self._labels: dict[str, tuple[str, ...]] = {}
        # Per axis name: the labels tuple it was built from, and label -> position.
        self._label_index: dict[str, tuple[tuple[str, ...], dict[str, int]]] = {}
        for ax, size in entries:
            labels = declared_labels(ax)
            if labels is not None and len(labels) == size:
                self._labels[ax.name] = labels

    @property
    def axes(self) -> tuple[Axis, ...]:
        """The axes of the dimensions, in dimension order."""
        return self._axes

    @property
    def sizes(self) -> tuple[int, ...]:
        """The sizes of the dimensions, in dimension order (``values.shape``)."""
        return self._sizes

    @property
    def layout(self) -> AxisLayout:
        """The axes and sizes as an :class:`~simulation.axis_layout.AxisLayout`.

        Available while the axes are grouped by role (PARAMETER, then
        ENSEMBLE, then DOMAIN), as an ``AxisLayout`` requires; after
        :meth:`transpose` crosses roles, use :attr:`axes` and :attr:`sizes`.

        Raises
        ------
        ValueError
            If the axes are not grouped by role.
        """
        try:
            return AxisLayout(zip(self._axes, self._sizes, strict=True))
        except ValueError as exc:
            raise ValueError(
                f"LabeledArray.layout: the axes {list(self._names())} are not grouped by role "
                f"(PARAMETER, ENSEMBLE, DOMAIN), so they have no AxisLayout; use axes and sizes. ({exc})"
            ) from None

    @property
    def labels(self) -> dict[str, tuple[str, ...]]:
        """Labels of the labelled dimensions, by axis name (a fresh dict; unlabelled axes are absent)."""
        return dict(self._labels)

    @property
    def dims(self) -> tuple[str, ...]:
        """Axis names, in dimension order.

        Deprecated, and staged for removal in a future milestone: use
        :attr:`axes` instead (``tuple(ax.name for ax in la.axes)`` for the
        names). Emits a :class:`DeprecationWarning`.
        """
        warnings.warn(
            "LabeledArray.dims is deprecated and will be removed in a future milestone. "
            "Use LabeledArray.axes instead (tuple(ax.name for ax in la.axes) for the names).",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._names()

    def _names(self) -> tuple[str, ...]:
        return tuple(ax.name for ax in self._axes)

    def isel(self, **positions: int | np.integer | slice) -> LabeledArray:
        """Select along axes named by keyword, by integer position or slice.

        An integer (a Python or NumPy integer) selects one position and drops
        that dimension from the result; a slice keeps it, at its (possibly
        reduced) size, together with the matching labels — the same
        distinction as plain numpy indexing with an int versus a slice.

        Examples
        --------
        >>> import numpy as np
        >>> from civic_digital_twins.dt_model.axes import DOMAIN, Axis
        >>> from civic_digital_twins.dt_model.simulation.axis_layout import AxisLayout
        >>> time, x, y = Axis("time", DOMAIN), Axis("x", DOMAIN), Axis("y", DOMAIN)
        >>> values = np.arange(2 * 3 * 4).reshape(2, 3, 4)
        >>> la = LabeledArray(values, AxisLayout([(time, 2), (x, 3), (y, 4)]))
        >>> [ax.name for ax in la.isel(x=0).axes]
        ['time', 'y']
        >>> la.isel(y=slice(0, 2)).values.shape
        (2, 3, 2)
        >>> int(la.isel(time=np.int64(1), x=2, y=3).values)
        23

        Raises
        ------
        KeyError
            If a keyword names an axis not present in :attr:`axes`.
        TypeError
            If a value is neither an integer nor a slice (labels select
            through :meth:`sel`).
        """
        for name, value in positions.items():
            if isinstance(value, str) or (
                isinstance(value, slice) and any(isinstance(b, str) for b in (value.start, value.stop))
            ):
                raise TypeError(f"LabeledArray.isel: axis {name!r}: isel selects by position; use sel() for labels.")
            if not isinstance(value, slice) and _as_int(value) is None:
                raise TypeError(
                    f"LabeledArray.isel: axis {name!r}: expected an integer or a slice, got {type(value).__name__}."
                )
        return self._select("isel", positions)

    def sel(self, **selection: str | list[str] | tuple[str, ...] | int | np.integer | slice) -> LabeledArray:
        """Select along axes named by keyword, by label or list of labels.

        A label selects one member and drops that dimension from the result.
        A list (or tuple) of labels keeps the dimension with those members, in
        the order given; lists on several axes cross. Labels are accepted on
        labelled dimensions only (see :attr:`labels`).

        Selecting by integer position or slice through ``sel`` still works,
        but is deprecated and staged for removal in a future milestone, after
        which ``sel`` selects by label only: use :meth:`isel` instead. It
        emits a :class:`DeprecationWarning`.

        Examples
        --------
        >>> import numpy as np
        >>> from civic_digital_twins.dt_model.axes import DomainAxis, SetType
        >>> from civic_digital_twins.dt_model.simulation.axis_layout import AxisLayout
        >>> zone = DomainAxis("zone", type=SetType(labels=("centre", "north", "south")))
        >>> la = LabeledArray(np.array([10.0, 20.0, 30.0]), AxisLayout([(zone, 3)]))
        >>> float(la.sel(zone="north").values)
        20.0
        >>> la.sel(zone=["south", "centre"]).values
        array([30., 10.])

        Raises
        ------
        KeyError
            If a keyword names an axis not present in :attr:`axes`, or a label
            is not among the labels of its dimension, or the dimension has no
            labels.
        ValueError
            If a list of labels is empty or repeats a label.
        TypeError
            If a slice bound is a label, a list contains a non-label, or a
            value is neither a label, a list of labels, an integer nor a slice.
        """
        for name, value in selection.items():
            if isinstance(value, (str, list, tuple)):
                continue
            if isinstance(value, slice) and any(isinstance(b, str) for b in (value.start, value.stop)):
                raise TypeError(
                    f"LabeledArray.sel: axis {name!r}: slices select by position; "
                    f"to select several labels, pass a list of labels."
                )
            if isinstance(value, slice) or _as_int(value) is not None:
                warnings.warn(
                    f"LabeledArray.sel: selecting axis {name!r} by position is deprecated and will be removed "
                    f"in a future milestone. Use isel({name}=...) instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                continue
            raise TypeError(
                f"LabeledArray.sel: axis {name!r}: expected a label or a list of labels, got {type(value).__name__}."
            )
        return self._select("sel", selection)

    def transpose(self, *axes: Axis | str) -> LabeledArray:
        """Return this array with its dimensions in the order of *axes*.

        *axes* are axis objects or axis names, and must be all the axes of
        this array, each once. The values are a transposed view; labels go
        with their dimensions. To read a result in a given order, pass
        ``axes=`` to :meth:`~simulation.evaluation.EvaluationResult.expected`
        or :meth:`~simulation.evaluation.EvaluationResult.samples` instead.

        Examples
        --------
        >>> import numpy as np
        >>> from civic_digital_twins.dt_model.axes import DOMAIN, Axis
        >>> from civic_digital_twins.dt_model.simulation.axis_layout import AxisLayout
        >>> time, tree = Axis("time", DOMAIN), Axis("tree", DOMAIN)
        >>> la = LabeledArray(np.zeros((4, 3)), AxisLayout([(time, 4), (tree, 3)]))
        >>> la.transpose("tree", time).values.shape
        (3, 4)

        Axes of different roles may be interleaved:

        >>> from civic_digital_twins.dt_model.axes import PARAMETER
        >>> factor = Axis("factor", PARAMETER)
        >>> la = LabeledArray(np.zeros((2, 4)), AxisLayout([(factor, 2), (time, 4)]))
        >>> [ax.name for ax in la.transpose(time, factor).axes]
        ['time', 'factor']

        Raises
        ------
        ValueError
            If an axis is missing, unknown or repeated.
        """
        try:
            return self._reorder(axes)
        except ValueError as exc:
            raise ValueError(f"LabeledArray.transpose: {exc}") from None

    def _reorder(self, axes: tuple[Axis | str, ...] | list[Axis | str]) -> LabeledArray:
        """Implement :meth:`transpose`; errors carry no caller prefix, so that other readers can add theirs."""
        order = self._resolve_order(axes)
        if order == list(range(len(self._axes))):
            return self
        entries = [(self._axes[i], self._sizes[i]) for i in order]
        result = LabeledArray(np.transpose(self.values, order), entries)
        result._labels = dict(self._labels)
        return result

    def _resolve_order(self, axes: tuple[Axis | str, ...] | list[Axis | str]) -> list[int]:
        """Positions of *axes* in this array; error unless they are all its axes, each once."""
        names = [ax if isinstance(ax, str) else ax.name for ax in axes]
        have = self._names()
        unknown = [n for n in names if n not in have]
        repeated = sorted({n for n in names if names.count(n) > 1})
        missing = [n for n in have if n not in names]
        if unknown or repeated or missing:
            problems = []
            if missing:
                problems.append(f"missing {missing}")
            if unknown:
                problems.append(f"unknown {unknown}")
            if repeated:
                problems.append(f"repeated {repeated}")
            raise ValueError(f"the axes must be all of {list(have)}, each once; " + ", ".join(problems) + ".")
        for ax in axes:
            if isinstance(ax, Axis) and ax not in self._axes:
                found = self._axes[have.index(ax.name)]
                raise ValueError(f"axis {ax!r} does not match this array's axis {found!r}.")
        return [have.index(n) for n in names]

    def _select(self, caller: str, selection: dict[str, Any]) -> LabeledArray:
        """Apply a validated selection: integers and labels drop a dimension, slices and lists keep it."""
        indexer: list[Any] = [slice(None)] * len(self._axes)
        kept_axes = list(self._axes)
        names = self._names()
        new_labels = dict(self._labels)
        picks: dict[Axis, list[int]] = {}
        for name, value in selection.items():
            if name not in names:
                raise KeyError(f"LabeledArray.{caller}: no axis named {name!r}; have {names}.")
            axis = self._axes[names.index(name)]
            if isinstance(value, (list, tuple)):
                picks[axis] = self._label_positions(name, self._check_label_list(name, value))
                new_labels[name] = tuple(value)
                continue
            if isinstance(value, slice):
                indexer[names.index(name)] = value
                if name in new_labels:
                    new_labels[name] = new_labels[name][value]
                continue
            position = self._label_position(name, value) if isinstance(value, str) else _as_int(value)
            indexer[names.index(name)] = position
            kept_axes.remove(axis)
            new_labels.pop(name, None)
        new_values = self.values[tuple(indexer)]
        # Lists are applied one axis at a time: indexing several axes with
        # lists in one step would pair their entries instead of crossing them.
        for axis, positions in picks.items():
            new_values = np.take(new_values, positions, axis=kept_axes.index(axis))
        result = LabeledArray(np.asarray(new_values), zip(kept_axes, new_values.shape, strict=True))
        result._labels = new_labels
        return result

    def _label_position(self, name: str, label: str) -> int:
        return self._label_positions(name, (label,))[0]

    def _label_positions(self, name: str, selected: Sequence[str]) -> list[int]:
        labels = self._labels.get(name)
        if labels is None:
            raise KeyError(f"LabeledArray.sel: axis {name!r} has no labels; select it by position with isel().")
        cached = self._label_index.get(name)
        if cached is None or cached[0] is not labels:
            cached = (labels, {label: i for i, label in enumerate(labels)})
            self._label_index[name] = cached
        position = cached[1]
        missing = [label for label in selected if label not in position]
        if missing:
            raise KeyError(f"LabeledArray.sel: {missing[0]!r} is not a label of axis {name!r}; labels are {labels}.")
        return [position[label] for label in selected]

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
        return xr.DataArray(self.values, dims=self._names(), coords=coords)  # pragma: no cover

    def __repr__(self) -> str:
        """Return a concise representation: axis names and shape."""
        return f"LabeledArray(axes={self._names()!r}, shape={self.values.shape!r})"


def _as_int(value: object) -> int | None:
    """Return *value* as a Python ``int`` if it is a Python or NumPy integer (not a bool), else ``None``."""
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        return operator.index(value)  # pyright: ignore[reportArgumentType]
    except TypeError:
        return None
