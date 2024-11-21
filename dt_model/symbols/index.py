from __future__ import annotations

from typing import Any

from sympy import lambdify

from dt_model.symbols._base import SymbolExtender
from dt_model.symbols.context_variable import ContextVariable


class Index(SymbolExtender):
    """
    Class to represent an index variable.
    """

    def __init__(self, name, value: Any, cvs: list[ContextVariable] | None = None) -> None:
        super().__init__(name)
        self.cvs = cvs
        if cvs is not None:
            self.value = lambdify(cvs, value, "numpy")
        else:
            self.value = value
