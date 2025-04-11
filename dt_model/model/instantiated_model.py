"""Allow instantiation of AbstractModel."""

from dt_model.model.abstract_model import AbstractModel
from dt_model.model.legacy_model import LegacyModel


class InstantiatedModel:
    """Instantiation of AbstractModel."""

    def __init__(self, abs: AbstractModel) -> None:
        self.abs = abs
        self.legacy = LegacyModel(abs.name, abs.cvs, abs.pvs, abs.indexes, abs.capacities, abs.constraints)
