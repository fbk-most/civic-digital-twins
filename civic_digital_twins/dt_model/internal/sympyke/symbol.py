"""Minimal support for symbols.

This module implements minimal support for sympy-like symbols
so that we can write dt_model models.
"""

import threading
from dataclasses import dataclass

from ...engine.frontend import graph


@dataclass(frozen=True)
class SymbolValue:
    """Contains the symbol graph node and the symbol name."""

    node: graph.placeholder
    name: str

    # TODO(bassosimone,pistore): the following is a hack to pretend that
    # we have strings in NumPy arrays rather than objects. While this
    # improves the debuggability story, it is also true that putting objects
    # inside NumPy arrays is not portable to backends like PyTorch.
    #
    # See https://github.com/fbk-most/civic-digital-twins/issues/90

    def __str__(self) -> str:
        """Pretend that the symbol is actually its name as a string.

        That is str(Symbol("a")) == "'a'".

        This simplifies the debuggability story.
        """
        return repr(self)

    def __repr__(self) -> str:
        """Pretend that the symbol is actually its name as a string.

        That is str(Symbol("a")) == "'a'".

        This simplifies the debuggability story.
        """
        return f"'{self.name}'"


class _SymbolTable:
    def __init__(self):
        self._table: dict[str, SymbolValue] = {}
        self._lock = threading.Lock()

    def get(self, name: str):
        with self._lock:
            if name not in self._table:
                self._table[name] = SymbolValue(graph.placeholder(name), name)
            return self._table[name]

    def values(self) -> list[SymbolValue]:
        with self._lock:
            values = list(self._table.values())
        return values


symbol_table = _SymbolTable()
"""Table containing all the defined symbols."""


def Symbol(name: str) -> SymbolValue:
    """Create a new SymbolValue by name.

    Subsequent invocations with the same name return the same SymbolValue.
    """
    return symbol_table.get(name)
