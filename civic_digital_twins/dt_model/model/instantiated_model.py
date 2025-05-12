"""Allow instantiation of AbstractModel."""

from ..model.abstract_model import AbstractModel


class InstantiatedModel:
    """Instantiation of AbstractModel."""

    def __init__(self, abs: AbstractModel, name: str | None = None, values: dict | None = None) -> None:
        self.abs = abs
        self.name = name if name is not None else abs.name
        self.values = values
