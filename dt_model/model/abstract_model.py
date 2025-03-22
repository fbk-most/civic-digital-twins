from __future__ import annotations

from dt_model.symbols.context_variable import ContextVariable
from dt_model.symbols.presence_variable import PresenceVariable
from dt_model.symbols.index import Index
from dt_model.symbols.constraint import Constraint


class AbstractModel:
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


