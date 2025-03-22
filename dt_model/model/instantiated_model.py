from dt_model.model.legacy_model import LegacyModel
from dt_model.model.abstract_model import AbstractModel
from dt_model.symbols.context_variable import ContextVariable
from dt_model.symbols.presence_variable import PresenceVariable
from dt_model.symbols.index import Index


class InstantiatedModel():
    def __init__(self, abs: AbstractModel) -> None:
        self.abs = abs
        self.legacy = LegacyModel(abs.name, abs.cvs, abs.pvs, abs.indexes, abs.capacities, abs.constraints)
