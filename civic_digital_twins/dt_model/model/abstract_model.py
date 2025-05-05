"""Abstract model definition."""

from __future__ import annotations

from ..symbols.constraint import Constraint
from ..symbols.context_variable import ContextVariable
from ..symbols.index import Index
from ..symbols.presence_variable import PresenceVariable


class AbstractModel:
    """Abstract model definition."""

    def __init__(
        self,
        name,
        cvs: list[ContextVariable],
        pvs: list[PresenceVariable],
        indexes: list[Index],
        capacities: list[Index],
        constraints: list[Constraint],
    ) -> None:
        self.name = name
        self.cvs = cvs
        self.pvs = pvs
        self.indexes = indexes
        self.capacities = capacities
        self.constraints = constraints
