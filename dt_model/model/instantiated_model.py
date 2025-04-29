"""Allow instantiation of AbstractModel."""

from dt_model.model.abstract_model import AbstractModel
from dt_model.model.legacy_model import LegacyModel


class InstantiatedModel:
    """Instantiation of AbstractModel."""

    def __init__(self, abs: AbstractModel, name: str = None, values: dict = None) -> None:
        self.abs = abs
        self.name = name if name is not None else abs.name
        self.values = values
        self.legacy = LegacyModel(name, abs.cvs, abs.pvs, abs.indexes, abs.capacities, abs.constraints)
