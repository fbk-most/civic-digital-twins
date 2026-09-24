"""Axis identity: role constants, the Axis class, and axis set operations.

This module is the canonical home for the axis vocabulary shared by all
layers (engine, model, simulation).  Always import axis types and utilities
from here; user code can equivalently use the re-exports in the top-level
``civic_digital_twins.dt_model`` package.  (See the README's "Conceptual
Overview" for the module-role convention behind this layout.)
"""

# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Sequence
from typing import Protocol

__all__ = [
    "AxisRole",
    "DOMAIN",
    "PARAMETER",
    "ENSEMBLE",
    "Axis",
    "DomainAxis",
    "DomainType",
    "SetType",
    "SequenceType",
    "TimeType",
    "SpaceType",
    "BoundaryCondition",
    "Constant",
    "Dirichlet",
    "Neumann",
    "Reflect",
    "Nearest",
    "Wrap",
    "Linear",
    "TIME_AXIS",
    "domain_axis_position",
    "filter_by_role",
    "union_axes",
]

# Open string type alias — users can define additional roles as plain strings
# following the UPPER_CASE convention.
AxisRole = str

# Built-in role constants.
DOMAIN: AxisRole = "DOMAIN"
PARAMETER: AxisRole = "PARAMETER"
ENSEMBLE: AxisRole = "ENSEMBLE"


class Axis:
    """A named, role-tagged axis object with value-based equality.

    Parameters
    ----------
    name:
        Lower-case string; globally unique within an :class:`EvaluationResult`.
        Names starting with ``_`` are reserved for framework use (e.g.
        ``_ensemble`` for the default ENSEMBLE axis created by
        :class:`~civic_digital_twins.dt_model.simulation.ensemble.DistributionEnsemble`).
    role:
        One of the built-in constants :data:`DOMAIN`, :data:`PARAMETER`,
        :data:`ENSEMBLE`, or a user-defined UPPER_CASE string.

    Notes
    -----
    Equality and hashing are value-based on ``(name, role)``.  Two ``Axis``
    objects with the same *name* and *role* are equal regardless of identity.
    This allows ``Axis("time", DOMAIN)`` constructed at graph-build time to
    match the one in ``axis_layout`` constructed at evaluation time, which is
    required for ``output_axes`` comparisons in
    :meth:`~civic_digital_twins.dt_model.simulation.evaluation.EvaluationResult.expected_value`.
    """

    __slots__ = ("name", "role")

    def __init__(self, name: str, role: AxisRole) -> None:
        self.name = name
        self.role = role

    def __eq__(self, other: object) -> bool:
        """Return True if name and role match."""
        if isinstance(other, Axis):
            return self.name == other.name and self.role == other.role
        return NotImplemented

    def __hash__(self) -> int:
        """Hash based on name and role."""
        return hash((self.name, self.role))

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return f"Axis({self.name!r}, role={self.role!r})"


class DomainAxis(Axis):
    """A DOMAIN axis carrying an optional static domain type.

    ``DomainAxis`` adds exactly one piece of static modeling metadata over
    :class:`Axis`: a :class:`DomainType` describing what operator vocabulary
    the axis supports (unordered set, ordered sequence, time, space, ...).
    Like ``role``, ``type`` is immutable and functionally determined by the
    axis ``name`` — it is not runtime/execution state. In particular ``size``
    deliberately stays off the axis: an axis identifies a dimension, while its
    extent is a per-result concern that lives in the execution layout.

    Identity is load-bearing and inherited unchanged from :class:`Axis`:
    equality and hashing stay on ``(name, role)`` only. ``type`` is
    deliberately *excluded* from ``__eq__``/``__hash__`` so a ``DomainAxis``
    remains interchangeable with a plain ``Axis(name, DOMAIN)`` — required
    for ``AxisLayout`` lookups, ``union_axes`` dedup, and cross-version
    serialization round-trips (a snapshot saved with an untyped axis must
    still match the live typed singleton).

    Parameters
    ----------
    name:
        See :class:`Axis`.
    type:
        Optional :class:`DomainType` instance (e.g. ``TimeType()``,
        ``SpaceType(spacing=10.0)``). ``None`` means "untyped", which the
        type lattice treats as :class:`SequenceType` by default.
    """

    __slots__ = ("type",)

    def __init__(self, name: str, *, type: "DomainType | None" = None) -> None:
        super().__init__(name, DOMAIN)
        self.type = type

    def __repr__(self) -> str:
        """Return a round-trippable string representation."""
        return f"DomainAxis({self.name!r}, type={self.type!r})"


class DomainType(Protocol):
    """Capability descriptor a :class:`DomainAxis` carries; operators dispatch on it.

    Reductions and element-wise operations are universal for any domain
    axis. Higher-level operations are gated by the concrete type, forming a
    lattice: ``SetType`` (reductions + selection only) is extended by
    ``SequenceType`` (adds ``shift``/``roll``/``diff``/``cumulative`` for
    ordered 1-D domains, gated by ``isinstance(axis.type, SequenceType)``),
    which is in turn extended by ``TimeType`` (calendar semantics — mostly
    deferred, currently a plain marker) and ``SpaceType`` (+ a metric and
    boundary condition, gating ``gradient``/``laplacian``).

    An untyped (``type=None``) :class:`DomainAxis` behaves as
    :class:`SequenceType`.
    """


class SetType:
    """Unordered domain: reductions and selection only."""

    __slots__ = ()

    def __repr__(self) -> str:
        """Return a round-trippable string representation."""
        return "SetType()"


class SequenceType:
    """Ordered, 1-D domain: gates ``shift``/``roll``/``diff``/``cumulative``.

    A plain marker, carrying no fields.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Return a round-trippable string representation."""
        return "SequenceType()"


class TimeType(SequenceType):
    """:class:`SequenceType` specialized for calendar time.

    Currently a plain marker, like :class:`SequenceType`: no calendar-specific
    metadata (e.g. a sampling interval) is represented yet.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Return a round-trippable string representation."""
        return "TimeType()"


class Constant:
    """Boundary condition: the field value just outside the domain is fixed to a constant.

    ``field[border+1] = value`` — the ghost point beyond the domain edge is
    pinned to *value*, independent of the field's actual data (the classical
    Dirichlet boundary condition — :data:`Dirichlet` is a plain alias for
    this class, for callers who prefer the PDE-standard term).

    Parameters
    ----------
    value:
        The fixed field value at the border.
    """

    __slots__ = ("value",)

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __repr__(self) -> str:
        """Return a round-trippable string representation."""
        return f"Constant(value={self.value!r})"


Dirichlet = Constant
"""Alias for :class:`Constant` — the classical PDE name for the same boundary condition."""


class Neumann:
    """Boundary condition: the spatial derivative just outside the domain is fixed to a constant.

    On the right border: ``field[N+1] = field[N-1] + 2 * spacing * value``.
    On the left border: ``field[0] = field[2] - 2 * spacing * value``. Each
    ghost point is chosen so the central-difference derivative *at that
    border node* equals *value* exactly. *value*'s sign is the derivative
    in the direction of increasing index — the same on both borders, so a
    field with a genuinely constant slope gets that slope back at both ends
    (the minus sign on the left is not a special case: it falls out of
    solving the same central-difference equation at the left node).
    ``Neumann()`` (``value=0.0``, the zero-flux/"reflecting" case) is
    :class:`SpaceType`'s default boundary — :data:`Reflect` is a plain
    alias for that specific instance, for callers who prefer the
    descriptive name over ``Neumann(0.0)``.

    Parameters
    ----------
    value:
        The fixed spatial derivative at the border, in the direction of
        increasing index.
    """

    __slots__ = ("value",)

    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __repr__(self) -> str:
        """Return a round-trippable string representation."""
        return f"Neumann(value={self.value!r})"


Reflect = Neumann(0.0)
"""Alias for the zero-flux ``Neumann(0.0)`` instance — :class:`SpaceType`'s default boundary."""


class Nearest:
    """Boundary condition: values just outside the domain repeat the outermost cell.

    ``field[border+1] = field[border]``.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Return a round-trippable string representation."""
        return "Nearest()"


class Wrap:
    """Boundary condition: opposite borders of the domain connect to each other (periodic).

    ``field[border+1] = field[opposite_border]``.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Return a round-trippable string representation."""
        return "Wrap()"


class Linear:
    """Boundary condition: values just outside the domain extend the local linear trend.

    ``field[border+1] = 2 * field[border] - field[border-1]``.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Return a round-trippable string representation."""
        return "Linear()"


BoundaryCondition = Constant | Neumann | Nearest | Wrap | Linear
"""Closed vocabulary of boundary-condition policies for :class:`SpaceType`.

The finite-difference operators that consume this (``gradient``,
``laplacian``) own the runtime semantics; this module only carries the
declared policy and its parameter, if any.
"""


class SpaceType(SequenceType):
    """:class:`SequenceType` with a metric (grid spacing) and boundary condition.

    Parameters
    ----------
    spacing:
        Grid spacing used by finite-difference operators (gradient,
        laplacian).
    boundary:
        Boundary-condition policy for neighbourhood operators — a
        :data:`BoundaryCondition` instance. Defaults to :data:`Reflect`
        (``Neumann(0.0)``, zero-flux / "reflecting").
    """

    __slots__ = ("spacing", "boundary")

    def __init__(
        self,
        spacing: float = 1.0,
        boundary: BoundaryCondition = Reflect,
    ) -> None:
        self.spacing = spacing
        self.boundary = boundary

    def __repr__(self) -> str:
        """Return a round-trippable string representation."""
        return f"SpaceType(spacing={self.spacing!r}, boundary={self.boundary!r})"


TIME_AXIS: DomainAxis = DomainAxis("time", type=TimeType())
"""Singleton for the time DOMAIN axis carried by timeseries nodes.

This is the canonical instance: every module that needs the time axis must
import it from here rather than constructing ``Axis("time", DOMAIN)`` locally.
(Value-based equality makes local copies *work*, but a single singleton keeps
the definition in one place.) It still compares and hashes equal to a plain
``Axis("time", DOMAIN)`` — see :class:`DomainAxis`.
"""


def domain_axis_position(domain_axes: Sequence[Axis], axis: Axis) -> int:
    """Return the numpy dimension index of *axis* within the trailing DOMAIN block.

    CDT lays arrays out as ``(*PARAMETER, *ENSEMBLE, *DOMAIN)``, so the DOMAIN
    axes always occupy the **last** ``len(domain_axes)`` dimensions, in the
    order given.  Because they are trailing, the index is returned as a
    **negative** offset from the end, which stays valid no matter how many
    leading dimensions a particular array happens to carry — arrays are
    right-aligned by numpy broadcasting, and mid-evaluation they have not yet
    been padded to a uniform rank.

    This generalizes the previous hard-coded ``-1``: with a single DOMAIN axis
    the sole axis is at ``-1``, exactly as before.

    Parameters
    ----------
    domain_axes:
        The evaluation's DOMAIN axes in canonical (layout) order.
    axis:
        The axis to locate.

    Returns
    -------
    The negative numpy dimension index of *axis*.

    Raises
    ------
    ValueError:
        If *axis* is not one of *domain_axes*.  Callers translate this into
        whichever "unsupported" error their layer reports.
    """
    try:
        position = list(domain_axes).index(axis)
    except ValueError:
        raise ValueError(f"axis {axis!r} is not one of the DOMAIN axes {[ax.name for ax in domain_axes]}") from None
    return position - len(domain_axes)


def union_axes(*seqs: tuple[Axis, ...]) -> tuple[Axis, ...]:
    """Merge axis tuples, preserving first-seen order and deduplicating."""
    seen: set[Axis] = set()
    result: list[Axis] = []
    for seq in seqs:
        for ax in seq:
            if ax not in seen:
                seen.add(ax)
                result.append(ax)
    return tuple(result)


def filter_by_role(axes: Iterable[Axis], role: AxisRole) -> tuple[Axis, ...]:
    """Return the axes whose role equals *role*, preserving input order."""
    matching = tuple(ax for ax in axes if ax.role == role)
    return matching
