"""Constraint symbol definition.

A constraint represents the relationship between the capacity of a given
resource and the usage of that resource. We model two types of constraints:

1. deterministic constraints where the capacity is a graph node

2. probabilistic constraints where the capacity is a random variable
"""

from __future__ import annotations

from ..engine.frontend import graph
from .index import Index, SymIndex


class Constraint:
    """
    Constraint class.

    This class is used to define constraints for the model.
    """

    def __init__(
        self,
        usage: graph.Node,
        capacity: Index,
        name: str = "",
    ) -> None:
        # For debuggability, assign the f'{name}_usage` name so it's
        # clear that this is the constraint's usage.
        self.usage = SymIndex(name=f"{name}_usage", value=usage)
        self.capacity = capacity
        self.name = name
