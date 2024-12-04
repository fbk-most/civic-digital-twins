from __future__ import annotations

import typing

if typing.TYPE_CHECKING:
    from sympy import Symbol


class Constraint:
    """
    Constraint class.

    This class is used to define constraints for the model.
    """

    def __init__(self, usage: Symbol, capacity: Symbol, group: str | None = None) -> None:
        self.usage = usage
        self.capacity = capacity
        self.group = group
