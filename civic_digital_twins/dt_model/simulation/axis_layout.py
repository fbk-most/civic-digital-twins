"""Axis layout management for evaluation results.

An :class:`AxisLayout` maps each semantic :class:`~dt_model.axes.Axis` to its
numpy dimension position and size in a result array.  It is the single owner
of the layout logic previously spread across ``evaluation.py`` (construction,
contraction, domain dropping), ``handle.py`` (merge signature validation,
axis growth), and ``runner.py`` (serialization).

Layout invariant
----------------
Result dimensions are ordered by role: all PARAMETER axes first, then all
ENSEMBLE axes, then all DOMAIN axes.  The constructor enforces this grouping,
turning what used to be scattered index arithmetic into a construction-time
invariant.

The class also provides *leading-array primitives* — pure shape arithmetic
over the leading ``(*PARAMETER, *ENSEMBLE)`` dimensions of value arrays
(prepend, broadcast, flatten, gather).  This module depends only on the axis
vocabulary and numpy; the node-aware guarded-region operations built on top
of these primitives live in :mod:`~simulation.region_execution`.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Container, Iterable, Mapping
from typing import Any

import numpy as np

from ..axes import DOMAIN, ENSEMBLE, PARAMETER, Axis, AxisRole

__all__ = ["AxisLayout"]

_ROLE_RANK: dict[AxisRole, int] = {PARAMETER: 0, ENSEMBLE: 1, DOMAIN: 2}
"""Canonical role ordering of result dimensions."""


class AxisLayout:
    """Immutable mapping from semantic axes to numpy dimension positions and sizes.

    Positions are implicit: the *entries* iterable lists ``(axis, size)``
    pairs in dimension order, so the axis at index ``i`` occupies numpy
    dimension ``i``.

    Parameters
    ----------
    entries:
        ``(axis, size)`` pairs in dimension order.

    Raises
    ------
    ValueError
        If two axes share a name (names are globally unique within a
        layout, across roles), if an axis has a role other than PARAMETER,
        ENSEMBLE, or DOMAIN, if the roles are not grouped in canonical
        order (PARAMETER, then ENSEMBLE, then DOMAIN), or if a size is not
        a positive integer.
    """

    __slots__ = ("_axes", "_sizes", "_positions")

    def __init__(self, entries: Iterable[tuple[Axis, int]]) -> None:
        axes: list[Axis] = []
        sizes: list[int] = []
        for ax, size in entries:
            axes.append(ax)
            sizes.append(int(size))
        names = [ax.name for ax in axes]
        if len(set(names)) != len(names):
            raise ValueError(f"AxisLayout: duplicate axis names in {names}.")
        ranks: list[int] = []
        for ax in axes:
            rank = _ROLE_RANK.get(ax.role)
            if rank is None:
                raise ValueError(
                    f"AxisLayout: unsupported role {ax.role!r} for axis {ax.name!r}; "
                    f"expected one of {sorted(_ROLE_RANK)}."
                )
            ranks.append(rank)
        if any(ranks[i] > ranks[i + 1] for i in range(len(ranks) - 1)):
            raise ValueError(
                "AxisLayout: axes must be grouped by role in canonical order "
                f"(PARAMETER, ENSEMBLE, DOMAIN); got {[ax.role for ax in axes]}."
            )
        if any(size <= 0 for size in sizes):
            raise ValueError(f"AxisLayout: sizes must be positive; got {sizes}.")
        self._axes = tuple(axes)
        self._sizes = tuple(sizes)
        self._positions = {ax: i for i, ax in enumerate(axes)}

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        parameters: Iterable[tuple[Axis, int]] = (),
        ensemble: Iterable[tuple[Axis, int]] = (),
        domain: Iterable[tuple[Axis, int]] = (),
    ) -> AxisLayout:
        """Build a layout from role groups, enforcing the canonical ordering.

        Parameters
        ----------
        parameters:
            PARAMETER ``(axis, size)`` pairs (named axes before anonymous
            ones, in caller order).
        ensemble:
            ENSEMBLE ``(axis, size)`` pairs.
        domain:
            DOMAIN ``(axis, size)`` pairs.
        """
        return cls((*parameters, *ensemble, *domain))

    @classmethod
    def from_positions(cls, positions: Mapping[Axis, int], sizes: Mapping[Axis, int]) -> AxisLayout:
        """Build a layout from explicit position and size mappings.

        Parameters
        ----------
        positions:
            Maps each axis to its dimension position.  Positions must be
            exactly ``0..n-1`` with no gaps.
        sizes:
            Maps each axis to its size.

        Raises
        ------
        ValueError
            If positions are not contiguous starting at zero.
        KeyError
            If an axis in *positions* is missing from *sizes*.
        """
        ordered = sorted(positions.items(), key=lambda item: item[1])
        if [pos for _, pos in ordered] != list(range(len(ordered))):
            raise ValueError(f"AxisLayout: positions must be contiguous 0..n-1; got {dict(positions)}.")
        return cls((ax, sizes[ax]) for ax, _ in ordered)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AxisLayout:
        """Reconstruct a layout from the dict produced by :meth:`to_dict`.

        The format matches the persisted :class:`~simulation.runner.ModelOutput`
        payload: ``axis_layout`` is a list of ``[name, role, position]`` rows
        and ``axis_sizes`` maps ``"name:role"`` keys to sizes.
        """
        positions = {Axis(row[0], row[1]): int(row[2]) for row in data["axis_layout"]}
        sizes: dict[Axis, int] = {}
        for key, size in data["axis_sizes"].items():
            ax_name, ax_role = key.split(":", 1)
            sizes[Axis(ax_name, ax_role)] = int(size)
        return cls.from_positions(positions, sizes)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def axes(self) -> tuple[Axis, ...]:
        """The axes in dimension order."""
        return self._axes

    @property
    def entries(self) -> tuple[tuple[Axis, int], ...]:
        """``(axis, size)`` pairs in dimension order."""
        return tuple(zip(self._axes, self._sizes))

    def __len__(self) -> int:
        """Return the number of dimensions."""
        return len(self._axes)

    def __contains__(self, axis: Axis) -> bool:
        """Return True when *axis* (by value equality) is in the layout."""
        return axis in self._positions

    def position_of(self, axis: Axis) -> int:
        """Return the dimension position of *axis* (KeyError if absent)."""
        return self._positions[axis]

    def size_of(self, axis: Axis) -> int:
        """Return the size of *axis* (KeyError if absent)."""
        return self._sizes[self._positions[axis]]

    def find_axis(self, name: str, role: AxisRole | None = None) -> Axis | None:
        """Return the axis named *name* (optionally matching *role*), or None."""
        for ax in self._axes:
            if ax.name == name and (role is None or ax.role == role):
                return ax
        return None

    def axes_by_role(self, role: AxisRole) -> tuple[tuple[Axis, int], ...]:
        """Return ``(axis, position)`` pairs with the given role, in position order."""
        return tuple((ax, pos) for pos, ax in enumerate(self._axes) if ax.role == role)

    @property
    def positions(self) -> dict[Axis, int]:
        """A fresh ``{axis: position}`` dict (dimension order preserved)."""
        return dict(self._positions)

    @property
    def sizes(self) -> dict[Axis, int]:
        """A fresh ``{axis: size}`` dict (dimension order preserved)."""
        return {ax: size for ax, size in zip(self._axes, self._sizes)}

    # ------------------------------------------------------------------
    # Derived shapes
    # ------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        """Number of PARAMETER dimensions."""
        return sum(1 for ax in self._axes if ax.role == PARAMETER)

    @property
    def n_ensemble(self) -> int:
        """Number of ENSEMBLE dimensions."""
        return sum(1 for ax in self._axes if ax.role == ENSEMBLE)

    @property
    def n_domain(self) -> int:
        """Number of DOMAIN dimensions."""
        return sum(1 for ax in self._axes if ax.role == DOMAIN)

    @property
    def n_leading(self) -> int:
        """Number of leading (PARAMETER + ENSEMBLE) dimensions."""
        return self.n_params + self.n_ensemble

    @property
    def full_shape(self) -> tuple[int, ...]:
        """Shape of a fully-broadcast result array in dimension order."""
        return self._sizes

    @property
    def leading_axes(self) -> tuple[Axis, ...]:
        """The PARAMETER and ENSEMBLE axes, in dimension order."""
        return self._axes[: self.n_leading]

    @property
    def leading_shape(self) -> tuple[int, ...]:
        """Sizes of the leading (PARAMETER + ENSEMBLE) dimensions."""
        return self._sizes[: self.n_leading]

    @property
    def leading_size(self) -> int:
        """Total number of leading-axis coordinates (1 when there are none)."""
        shape = self.leading_shape
        return int(np.prod(shape, dtype=np.int64)) if shape else 1

    def compatible_with(self, shape: tuple[int, ...]) -> bool:
        """Return True when *shape* is broadcast-compatible with this layout.

        A compatible shape has exactly one dimension per layout axis, each
        either 1 (the value does not vary along that axis) or the axis size.
        """
        if len(shape) != len(self._axes):
            return False
        return all(dim in {1, size} for dim, size in zip(shape, self._sizes))

    # ------------------------------------------------------------------
    # Leading-array primitives
    #
    # Pure shape arithmetic over arrays whose first n_leading dimensions
    # map onto the leading (PARAMETER + ENSEMBLE) axes.  Trailing (DOMAIN)
    # dimensions are always preserved untouched.
    # ------------------------------------------------------------------

    def spans_leading(self, shape: tuple[int, ...]) -> bool:
        """Return True when *shape* has explicit leading dimensions.

        The shape must have at least ``n_leading`` dimensions and each of
        the first ``n_leading`` must be 1 or the corresponding axis size.
        """
        leading_shape = self.leading_shape
        return len(shape) >= self.n_leading and all(shape[i] in {1, leading_shape[i]} for i in range(self.n_leading))

    def prepend_leading(self, arr: np.ndarray) -> np.ndarray:
        """Return *arr* with ``n_leading`` singleton dimensions prepended."""
        return arr.reshape((1,) * self.n_leading + arr.shape)

    def broadcast_leading(self, arr: np.ndarray) -> np.ndarray:
        """Broadcast the first ``n_leading`` dims of *arr* to the leading shape.

        Trailing dimensions are preserved untouched.

        Raises
        ------
        ValueError
            If *arr* does not span the leading layout (:meth:`spans_leading`).
        """
        if not self.spans_leading(arr.shape):
            raise ValueError(
                f"broadcast_leading: shape {arr.shape} does not span the "
                f"leading layout {self.leading_shape} (each leading dim must "
                f"be 1 or the axis size)."
            )
        return np.broadcast_to(arr, self.leading_shape + arr.shape[self.n_leading :])

    def flatten_leading(self, arr: np.ndarray) -> np.ndarray:
        """Reshape *arr* to ``(leading_size, *trailing)``.

        Raises
        ------
        ValueError
            If the leading dimensions of *arr* are not exactly the leading
            shape (broadcast first with :meth:`broadcast_leading` when
            needed).  A plain reshape could silently scramble coordinates
            whenever the element counts happen to match, so the precondition
            is checked explicitly.
        """
        if arr.shape[: self.n_leading] != self.leading_shape:
            raise ValueError(
                f"flatten_leading: shape {arr.shape} does not start with the "
                f"full leading shape {self.leading_shape}; broadcast_leading first."
            )
        return arr.reshape((self.leading_size,) + arr.shape[self.n_leading :])

    def unflatten_leading(self, arr: np.ndarray) -> np.ndarray:
        """Inverse of :meth:`flatten_leading`: restore the leading dimensions.

        Raises
        ------
        ValueError
            If the first dimension of *arr* is not ``leading_size``.
        """
        if arr.ndim < 1 or arr.shape[0] != self.leading_size:
            raise ValueError(
                f"unflatten_leading: shape {arr.shape} does not start with leading_size {self.leading_size}."
            )
        return arr.reshape(self.leading_shape + arr.shape[1:])

    def take_leading(self, arr: np.ndarray, flat_idx: np.ndarray) -> np.ndarray:
        """Gather flattened leading coordinates of *arr* into a new first axis.

        Equivalent to indexing rows of :meth:`flatten_leading` with
        *flat_idx*; trailing dimensions are preserved.
        """
        return np.take(self.flatten_leading(arr), flat_idx, axis=0)

    # ------------------------------------------------------------------
    # Signatures (merge compatibility)
    # ------------------------------------------------------------------

    def role_signature(self, role: AxisRole, *, exclude_name: str | None = None) -> frozenset[tuple[str, int, int]]:
        """Return a ``{(name, position, size)}`` signature of the axes with *role*.

        Parameters
        ----------
        role:
            The role to include.
        exclude_name:
            Optional axis name to leave out (e.g. the axis being grown by a
            merge, whose size is allowed to differ).
        """
        return frozenset(
            (ax.name, pos, self._sizes[pos])
            for pos, ax in enumerate(self._axes)
            if ax.role == role and ax.name != exclude_name
        )

    def parameter_signature(self) -> frozenset[tuple[str, AxisRole, int, int]]:
        """Return a ``{(name, role, position, size)}`` signature of all non-ENSEMBLE axes.

        Includes DOMAIN axes so that two layouts only compare equal when
        their PARAMETER grids *and* domain dimensions match.
        """
        return frozenset(
            (ax.name, ax.role, pos, self._sizes[pos]) for pos, ax in enumerate(self._axes) if ax.role != ENSEMBLE
        )

    # ------------------------------------------------------------------
    # Transformations
    # ------------------------------------------------------------------

    def with_grown_axis(self, name: str, new_size: int) -> AxisLayout:
        """Return a copy with the axis named *name* resized to *new_size*.

        This is the single primitive behind ENSEMBLE and PARAMETER merge
        growth (KeyError if no axis has that name).
        """
        ax = self.find_axis(name)
        if ax is None:
            raise KeyError(f"AxisLayout: no axis named {name!r}.")
        return AxisLayout((a, new_size if a == ax else size) for a, size in self.entries)

    def with_axis_appended(self, axis: Axis, size: int) -> AxisLayout:
        """Return a copy with *axis* appended as the last dimension.

        Used to register the DOMAIN time axis once its length is known
        (post-execution).  The canonical role ordering is re-validated.
        """
        return AxisLayout((*self.entries, (axis, size)))

    # ------------------------------------------------------------------
    # Array operations
    # ------------------------------------------------------------------

    def contract_ensemble(self, arr: np.ndarray, weights: Mapping[Axis, np.ndarray]) -> np.ndarray:
        """Contract all ENSEMBLE dimensions of *arr* and return the ``(*P, *D)`` array.

        For each ENSEMBLE axis (in descending position order so earlier
        squeezes do not shift later positions): if the dimension has size 1
        the singleton is squeezed away directly; otherwise a weighted average
        is taken using ``weights[axis]``.

        Raises
        ------
        ValueError
            If *arr* is not compatible with this layout (one dimension per
            axis, each 1 or the axis size) — averaging positionally over a
            mismatched array would contract the wrong dimension.
        """
        if not self.compatible_with(arr.shape):
            raise ValueError(f"contract_ensemble: shape {arr.shape} is incompatible with layout {self!r}.")
        for ax, pos in sorted(self.axes_by_role(ENSEMBLE), key=lambda t: t[1], reverse=True):
            arr = arr.squeeze(axis=pos) if arr.shape[pos] == 1 else np.average(arr, weights=weights[ax], axis=pos)
        return arr

    def drop_stray_domain(self, arr: np.ndarray, carried_axes: Container[Axis]) -> np.ndarray:
        """Drop the DOMAIN dimensions of *arr* that are not in *carried_axes*.

        *arr* must already be ENSEMBLE-contracted (shape ``(*P, *D)``).  The
        result carries exactly the PARAMETER dimensions plus the DOMAIN
        dimensions listed in *carried_axes*.

        Raises
        ------
        ValueError
            If *arr* is not ENSEMBLE-contracted (its rank must be
            ``n_params + n_domain`` — on a full-layout array the positional
            arithmetic would treat ENSEMBLE dimensions as DOMAIN ones), or
            if a non-carried DOMAIN dimension has size greater than 1: the
            value varies along an axis its node does not declare in
            ``output_axes``, so the declared and actual axes disagree.
            Dropping data or silently returning an undeclared dimension
            would both be wrong; the inconsistency is reported instead.
        """
        n_params = self.n_params
        if arr.ndim != n_params + self.n_domain:
            raise ValueError(
                f"drop_stray_domain: shape {arr.shape} is not ENSEMBLE-contracted "
                f"(expected {n_params + self.n_domain} dims: {n_params} PARAMETER + "
                f"{self.n_domain} DOMAIN)."
            )
        stray: list[int] = []
        for i, (ax, _) in enumerate(self.axes_by_role(DOMAIN)):
            if ax in carried_axes:
                continue
            dim = n_params + i
            if arr.shape[dim] != 1:
                raise ValueError(
                    f"drop_stray_domain: DOMAIN axis {ax.name!r} is not carried by the "
                    f"node's output_axes, but the value varies along it "
                    f"(size {arr.shape[dim]}); declared and actual axes disagree."
                )
            stray.append(dim)
        return np.squeeze(arr, axis=tuple(stray)) if stray else arr

    # ------------------------------------------------------------------
    # Serialization and dunder support
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize to the JSON-friendly dict format used by :meth:`from_dict`."""
        return {
            "axis_layout": [[ax.name, ax.role, pos] for pos, ax in enumerate(self._axes)],
            "axis_sizes": {f"{ax.name}:{ax.role}": size for ax, size in zip(self._axes, self._sizes)},
        }

    def __eq__(self, other: object) -> bool:
        """Return True when both layouts have equal entries in the same order."""
        if isinstance(other, AxisLayout):
            return self.entries == other.entries
        return NotImplemented

    def __hash__(self) -> int:
        """Hash based on the ordered entries."""
        return hash(self.entries)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        inner = ", ".join(f"{ax!r}: {size}" for ax, size in self.entries)
        return f"AxisLayout({inner})"
