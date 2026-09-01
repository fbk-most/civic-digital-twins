"""Axis identity: role constants, the Axis class, and axis set operations.

This module is the canonical home for the axis vocabulary shared by all
layers (engine, model, simulation).  Always import axis types and utilities
from here; user code can equivalently use the re-exports in the top-level
``civic_digital_twins.dt_model`` package.  (See the README's "Conceptual
Overview" for the module-role convention behind this layout.)
"""

# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable, Sequence
from typing import Literal, Protocol

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
    "MeshType",
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
    ``SequenceType`` (adds shift/lag/cumulative/diff for ordered 1-D
    domains), which is in turn extended by ``TimeType`` (+ calendar /
    periodicity) and ``SpaceType`` (+ a metric for gradient/laplacian).

    This is a metadata-only protocol in this step: the gated operator
    vocabulary itself is introduced in a later step. An untyped
    (``type=None``) :class:`DomainAxis` behaves as :class:`SequenceType`.
    """


class SetType:
    """Unordered domain: reductions and selection only."""

    __slots__ = ()

    def __repr__(self) -> str:
        """Return a round-trippable string representation."""
        return "SetType()"


class SequenceType:
    """Ordered, 1-D domain: adds shift/lag/cumulative/diff.

    Parameters
    ----------
    periodic:
        Whether the sequence wraps around (circular shift/roll) rather
        than being fill-padded at the boundary.
    """

    __slots__ = ("periodic",)

    def __init__(self, periodic: bool = False) -> None:
        self.periodic = periodic

    def __repr__(self) -> str:
        """Return a round-trippable string representation."""
        return f"SequenceType(periodic={self.periodic!r})"


class TimeType(SequenceType):
    """:class:`SequenceType` specialized for calendar / periodic time.

    ``periodic=True`` is the circular-shift convention used by recurrences
    such as a periodic ``np.roll``-based solver.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        """Return a round-trippable string representation."""
        return f"TimeType(periodic={self.periodic!r})"


class SpaceType(SequenceType):
    """:class:`SequenceType` with a metric (grid spacing) and boundary condition.

    Parameters
    ----------
    spacing:
        Grid spacing used by finite-difference operators (gradient,
        laplacian).
    boundary:
        Boundary-condition policy for neighbourhood operators: one of
        ``"reflect"``, ``"constant"``, ``"wrap"``, ``"nearest"``. The value
        is validated at authoring time by the ``Literal`` annotation; the
        finite-difference operators that consume it own the runtime semantics.
    """

    __slots__ = ("spacing", "boundary")

    def __init__(
        self,
        spacing: float = 1.0,
        boundary: Literal["reflect", "constant", "wrap", "nearest"] = "reflect",
    ) -> None:
        super().__init__(periodic=False)
        self.spacing = spacing
        self.boundary = boundary

    def __repr__(self) -> str:
        """Return a round-trippable string representation."""
        return f"SpaceType(spacing={self.spacing!r}, boundary={self.boundary!r})"


class MeshType:
    """Irregular domain of N cells with explicit adjacency (graph Laplacian).

    Parameters
    ----------
    adjacency:
        Opaque adjacency structure (e.g. a sparse matrix) consumed by
        graph-Laplacian operators. Not interpreted in this step.
    """

    __slots__ = ("adjacency",)

    def __init__(self, adjacency: object = None) -> None:
        self.adjacency = adjacency

    def __repr__(self) -> str:
        """Return a round-trippable string representation."""
        return f"MeshType(adjacency={self.adjacency!r})"


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
    return tuple(ax for ax in axes if ax.role == role)
