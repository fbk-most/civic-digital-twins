"""
Piecewise Emulation
===================

This module emulates sympy.Piecewise using the tensor language frontend, by mapping
a Piecewise invocation to a graph.multi_clause_where tensor.
"""

from ...engine.frontend import graph


Cond = graph.Node | graph.Scalar
"""Condition for a piecewise clause."""

Expr = graph.Node | graph.Scalar
"""Expression for a piecewise clause."""

Clause = tuple[Expr, Cond]
"""Clause provided to piecewise."""


def Piecewise(*clauses: Clause) -> graph.Node:
    """Converts the provided clauses arranged according to the sympy.Piecewise
    convention into a graph.multi_clause_where computation tensor.

    Args:
        *clauses: The clauses to be converted.

    Returns:
        The computation tensor representing the piecewise function.

    Raises:
        ValueError: If no clauses are provided.
    """
    return _to_tensor(_filter_clauses(clauses))


def _filter_clauses(clauses: tuple[Clause, ...]) -> list[Clause]:
    """This function removes the clauses after the first true clause, to
    correctly emulate the sumpy.Piecewise behaviour.

    Args:
        clauses: The clauses to be filtered.

    Returns:
        The filtered clauses.
    """
    filtered: list[Clause] = []
    for expr, cond in clauses:
        filtered.append((expr, cond))
        if cond is True:
            break
    return filtered


def _to_tensor(clauses: list[Clause]) -> graph.Node:
    # 1. Bail if there are no remaining clauses
    if len(clauses) < 1:
        raise ValueError("piecewise: at least one clause is required")

    # 2. Check whether there is a default case and otherwise use NaN
    default_value: Expr = float("NaN")
    last_clause = clauses[-1]
    if last_clause[1] is True:
        default_value = last_clause[0]
        clauses = clauses[:-1]
    if isinstance(default_value, graph.Scalar):
        default_value = graph.constant(default_value)

    # 3. If no clauses remain after removing the default, just return the default value
    if len(clauses) <= 0:
        return default_value

    # 4. Prepare the reversed clauses adapting the types
    reversed: list[tuple[graph.Node, graph.Node]] = []
    for expr, cond in clauses:
        if isinstance(expr, graph.Scalar):
            expr = graph.constant(expr)
        if isinstance(cond, graph.Scalar):
            cond = graph.constant(cond)
        reversed.append((cond, expr))

    # 5. We're now all set call multi_clause_where
    return graph.multi_clause_where(reversed, default_value)
