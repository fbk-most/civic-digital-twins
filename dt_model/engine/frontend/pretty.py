"""
Pretty Printing for Computation Graphs
======================================

This module provides facilities for converting computation graphs into
readable string representations. It handles:

1. Operator precedence
2. Parentheses insertion
3. Special formatting for function-like operations
4. Named expressions

The main entry point is the format() function:

    >>> from dt_model.engine.frontend import graph, pretty
    >>> x = graph.placeholder("x")
    >>> y = graph.add(graph.multiply(x, 2), 1)
    >>> print(pretty.format(y))
    x * 2 + 1

Precedence Rules
----------------

The formatter follows standard mathematical precedence:

1. Function application (exp, log)
2. Unary operations (~)
3. Power (**)
4. Multiply/Divide (*, /)
5. Add/Subtract (+, -)
6. Comparisons (<, <=, >, >=, ==, !=)
7. Logical AND (&)
8. Logical XOR (^)
8. Logical OR (|)

Parentheses are automatically added when needed to preserve
the correct evaluation order:

    >>> x + y * z      # "x + y * z"
    >>> (x + y) * z    # "(x + y) * z"
    >>> ~x & y | z     # "(~x & y) | z"

Design Decisions
----------------

1. Precedence-based Formatting:
   - Uses numeric precedence levels to determine parenthesization
   - Follows standard mathematical conventions
   - Allows easy addition of new operators

2. Recursive Implementation:
   - Handles nested expressions naturally
   - Passes precedence information down the tree
   - Enables context-aware formatting decisions

3. Special Cases:
   - Function-like operations use function call syntax
   - Named nodes show assignment syntax
   - Placeholders use angle bracket notation for visibility

Implementation Notes
--------------------

The formatter uses a visitor-like pattern without explicitly implementing the
visitor pattern, which keeps the code simple and maintains extensibility.
"""

# SPDX-License-Identifier: Apache-2.0

from . import graph


def format(node: graph.Node) -> str:
    """Format a computation graph node as a string.

    Args:
        node: The node to format

    Returns:
        A string representation with appropriate parentheses
        and operator precedence.

    Examples:
        >>> x = graph.placeholder("x")
        >>> y = graph.add(graph.multiply(x, 2), 1)
        >>> print(pretty.format(y))
        x * 2 + 1
    """
    expr = _format(node, True, 0)  # Start with lowest precedence
    if node.name:
        expr = f"{node.name} = {expr}"
    return expr


def _format(node: graph.Node, toplevel: bool, parent_precedence: int) -> str:
    """Internal recursive formatter.

    Args:
        node: The node to format
        parent_precedence: The precedence of the parent operation

    Returns:
        Formatted string with appropriate parentheses based on
        operator precedence.
    """

    # If we're not at top-level and we have a named node,
    # stop formatting and return the node name, which means
    # we're printing formulae aligned with what the user
    # has written inside the input program/model
    if not toplevel and node.name:
        return node.name

    # Immediately flip toplevel to False after the first
    # iteration to avoid printing the whole graph
    toplevel = False

    # Precedence rules (higher binds tighter)
    PRECEDENCE = {
        # Function-like operators without corresponding infix operators
        graph.exp: 100,  # exp(x)
        graph.log: 100,  # log(x)
        # Unary operations
        graph.logical_not: 50,  # ~x
        # Binary operations
        graph.power: 40,  # x ** y
        graph.multiply: 30,  # x * y
        graph.divide: 30,  # x / y
        graph.add: 20,  # x + y
        graph.subtract: 20,  # x - y
        # Comparisons
        graph.less: 10,  # x < y
        graph.less_equal: 10,  # x <= y
        graph.greater: 10,  # x > y
        graph.greater_equal: 10,  # x >= y
        graph.equal: 10,  # x == y
        graph.not_equal: 10,  # x != y
        # Logical operations
        graph.logical_and: 5,  # x & y
        graph.logical_xor: 4,  # x ^ y
        graph.logical_or: 3,  # x | y
    }

    def needs_parens(expr_node: graph.Node) -> bool:
        """Determine if expression needs parentheses."""
        return PRECEDENCE.get(type(expr_node), 0) < parent_precedence

    def wrap(expr_node: graph.Node, expr: str) -> str:
        """Wrap expression in parentheses if needed."""
        return f"({expr})" if needs_parens(expr_node) else expr

    # Base cases
    if isinstance(node, graph.constant):
        return str(node.value)
    if isinstance(node, graph.placeholder):
        return node.name

    # Binary operations
    if isinstance(node, graph.BinaryOp):
        op_precedence = PRECEDENCE.get(type(node), 0)
        left = _format(node.left, toplevel, op_precedence)
        right = _format(node.right, toplevel, op_precedence)

        # Arithmetic operators
        if isinstance(node, graph.add):
            return wrap(node, f"{left} + {right}")
        if isinstance(node, graph.subtract):
            return wrap(node, f"{left} - {right}")
        if isinstance(node, graph.multiply):
            return wrap(node, f"{left} * {right}")
        if isinstance(node, graph.divide):
            return wrap(node, f"{left} / {right}")
        if isinstance(node, graph.power):
            return wrap(node, f"{left} ** {right}")
        if isinstance(node, graph.maximum):
            return f"maximum({left}, {right})"
        if isinstance(node, graph.logical_and):
            return wrap(node, f"{left} & {right}")
        if isinstance(node, graph.logical_or):
            return wrap(node, f"{left} | {right}")
        if isinstance(node, graph.logical_xor):
            return wrap(node, f"{left} ^ {right}")

        # Comparison operators
        if isinstance(node, graph.less):
            return wrap(node, f"{left} < {right}")
        if isinstance(node, graph.less_equal):
            return wrap(node, f"{left} <= {right}")
        if isinstance(node, graph.greater):
            return wrap(node, f"{left} > {right}")
        if isinstance(node, graph.greater_equal):
            return wrap(node, f"{left} >= {right}")
        if isinstance(node, graph.equal):
            return wrap(node, f"{left} == {right}")
        if isinstance(node, graph.not_equal):
            return wrap(node, f"{left} != {right}")

    # Unary operations
    if isinstance(node, graph.UnaryOp):
        op_precedence = PRECEDENCE.get(type(node), 0)
        inner = _format(node.node, toplevel, op_precedence)

        if isinstance(node, graph.logical_not):
            return wrap(node, f"~{inner}")
        if isinstance(node, graph.exp):
            return f"exp({inner})"
        if isinstance(node, graph.log):
            return f"log({inner})"

    # Conditional operations
    if isinstance(node, graph.where):
        condition = _format(node.condition, toplevel, 0)
        then_expr = _format(node.then, toplevel, 0)
        else_expr = _format(node.otherwise, toplevel, 0)
        return f"where({condition}, {then_expr}, {else_expr})"

    if isinstance(node, graph.multi_clause_where):
        clauses_str = ", ".join(
            f"({_format(cond, toplevel, 0)}, {_format(val, toplevel, 0)})" for cond, val in node.clauses
        )
        default_str = _format(node.default_value, toplevel, 0)
        return f"multi_clause_where([{clauses_str}], {default_str})"

    # Shape operations
    if isinstance(node, graph.AxisOp):
        inner = _format(node.node, toplevel, 0)
        axis_str = str(node.axis) if isinstance(node.axis, int) else str(tuple(node.axis))

        if isinstance(node, graph.expand_dims):
            return f"expand_dims({inner}, {axis_str})"
        if isinstance(node, graph.squeeze):
            return f"squeeze({inner}, {axis_str})"
        if isinstance(node, graph.reduce_sum):
            return f"reduce_sum({inner}, {axis_str})"
        if isinstance(node, graph.reduce_mean):
            return f"reduce_mean({inner}, {axis_str})"

    return f"<unknown:{type(node).__name__}>"
