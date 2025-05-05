"""Sympy-like operators for creating symbolic expressions."""

from ...engine.frontend import graph
from .symbol import SymbolValue


def _ensure_node(value: graph.Node | SymbolValue | graph.Scalar) -> graph.Node:
    if isinstance(value, graph.Node):
        return value
    if isinstance(value, SymbolValue):
        return value.node
    return graph.constant(value)


def Eq(
    lhs: graph.Node | SymbolValue | graph.Scalar,
    rhs: graph.Node | SymbolValue | graph.Scalar,
) -> graph.Node:
    """Create an equality expression between nodes and/or symbols."""
    return graph.equal(_ensure_node(lhs), _ensure_node(rhs))
