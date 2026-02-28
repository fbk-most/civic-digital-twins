"""Core model definition."""

from __future__ import annotations

from .index import Distribution, GenericIndex, Index


class Model:
    """A named collection of GenericIndex objects.

    A model is *abstract* if it contains indexes that need external values
    before evaluation — placeholder nodes (``value is None``) or
    distribution-backed indexes (``isinstance(value, Distribution)``).
    It is *instantiated* when every index has a concrete, evaluable value.

    Parameters
    ----------
    name:
        Human-readable name for the model.
    indexes:
        All indexes that belong to this model, in any order.
    """

    def __init__(self, name: str, indexes: list[GenericIndex]) -> None:
        self.name = name
        self.indexes = indexes

    def abstract_indexes(self) -> list[GenericIndex]:
        """Return indexes that require external values before evaluation.

        An index is abstract when its ``value`` is ``None`` (explicit
        placeholder) or a ``Distribution`` (needs sampling).  Constant and
        formula-based indexes are concrete and are not returned here.
        """
        result = []
        for index in self.indexes:
            if isinstance(index, Index):
                if index.value is None or isinstance(index.value, Distribution):
                    result.append(index)
        return result

    def is_instantiated(self) -> bool:
        """Return True when all indexes have concrete, evaluable values."""
        return len(self.abstract_indexes()) == 0
