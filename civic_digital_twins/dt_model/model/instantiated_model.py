"""Allow instantiation of AbstractModel."""

from ..model.abstract_model import AbstractModel
from ..model.legacy_model import LegacyModel


class InstantiatedModel:
    """Instantiation of AbstractModel."""

    def __init__(self, abs: AbstractModel, name: str | None = None, values: dict | None = None) -> None:
        self.abs = abs
        self.name = name if name is not None else abs.name
        self.values = values
        self.legacy = LegacyModel(name, abs.cvs, abs.pvs, abs.indexes, abs.capacities, abs.constraints)

    def get_values(self, all: bool = False) -> dict:
        values = self.values.copy() if self.values is not None else {}
        if all:
            for i in self.abs.indexes + self.abs.capacities:
               if i.name not in values:
                   values[i.name] = i.value
        return values