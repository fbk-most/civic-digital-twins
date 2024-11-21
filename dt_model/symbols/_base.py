from __future__ import annotations

from sympy import Symbol


class SymbolExtender(Symbol):
    """
    Utility class to extend sympy Simbol.
    """

    def __new__(cls, name, *args, **kwargs):
        obj = Symbol.__new__(cls, name)
        return obj

    def __init__(self, name, *args, **kwargs):
        pass
