"""Axis identity: role constants, the Axis class, and axis set operations.

This module is the canonical home for the axis vocabulary shared by all
layers (engine, model, simulation).  Always import axis types and utilities
from here; user code can equivalently use the re-exports in the top-level
``civic_digital_twins.dt_model`` package.  (See the README's "Conceptual
Overview" for the module-role convention behind this layout.)
"""

# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable

__all__ = [
    "AxisRole",
    "DOMAIN",
    "PARAMETER",
    "ENSEMBLE",
    "Axis",
    "TIME_AXIS",
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


TIME_AXIS: Axis = Axis("time", DOMAIN)
"""Singleton for the time DOMAIN axis carried by timeseries nodes.

This is the canonical instance: every module that needs the time axis must
import it from here rather than constructing ``Axis("time", DOMAIN)`` locally.
(Value-based equality makes local copies *work*, but a single singleton keeps
the definition in one place.)
"""


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
