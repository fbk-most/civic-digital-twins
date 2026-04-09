"""Axis identity: role constants and the Axis class."""

# SPDX-License-Identifier: Apache-2.0

__all__ = ["AxisRole", "DOMAIN", "PARAMETER", "ENSEMBLE", "Axis"]

# Open string type alias — users can define additional roles as plain strings
# following the UPPER_CASE convention.
AxisRole = str

# Built-in role constants.
DOMAIN: AxisRole = "DOMAIN"
PARAMETER: AxisRole = "PARAMETER"
ENSEMBLE: AxisRole = "ENSEMBLE"


class Axis:
    """A named, role-tagged axis object with identity-based equality.

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
    Equality and hashing are identity-based (the default Python behaviour
    when neither ``__eq__`` nor ``__hash__`` is overridden).  Two ``Axis``
    objects with the same *name* and *role* are **not** equal unless they
    are the same instance.  This mirrors the convention used by
    :class:`~civic_digital_twins.dt_model.model.index.GenericIndex`.
    """

    __slots__ = ("name", "role")

    def __init__(self, name: str, role: AxisRole) -> None:
        self.name = name
        self.role = role

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return f"Axis({self.name!r}, role={self.role!r})"
