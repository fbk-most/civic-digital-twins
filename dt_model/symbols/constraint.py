"""
This module defines the Constraint symbol.

A constraint represents the relationship between the capacity of a given
resource and the usage of that resource. We model two types of constraints:

1. deterministic constraints where the capacity is a graph node

2. probabilistic constraints where the capacity is a random variable
"""

from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from sympy import Symbol


class Constraint:
    """
    Constraint class.

    This class is used to define constraints for the model.
    """

    def __init__(self, usage: Symbol, capacity: Symbol, group: str | None = None, name: str = "") -> None:
        self.usage = usage
        self.capacity = capacity
        self.name = name

        # TODO(bassosimone): this field is only used by the view. We could consider
        # deprecating it and moving the view mapping logic inside the view itself, which
        # would work as intended as long as we have a working __hash__. By doing this,
        # we would probably reduce the churn and coupling between the computational
        # model and the related view.
        self.group = group
