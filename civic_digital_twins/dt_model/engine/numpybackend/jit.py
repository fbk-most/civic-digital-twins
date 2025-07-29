"""NumPy JIT compiler infrastructure.

This module provides support for:

1. transforming graph.Node, forest.Tree, and ir.DAG into a Python AST
that uses NumPy function calls (e.g., `graph.add` -> `np.add`).

2. compiling the Python AST to Python bytecode.

3. JIT compiling the Python bytecode using Numba.

This code is still experimental and also incomplete. For now, we only
implement the bare minimum to pretty print nodes for debugging.
"""

# SPDX-License-Identifier: Apache-2.0

import ast

import numpy as np

from ..frontend import graph


class UnsupportedNodeType(Exception):
    """Raised when the executor encounters an unsupported node type."""


class UnsupportedOperation(Exception):
    """Raised when the executor encounters an unsupported operation."""


_operation_names: dict[type[graph.Node], str] = {
    # placeholder
    graph.placeholder: "asarray",
    # constant
    graph.constant: "asarray",
    # binary
    graph.add: "add",
    graph.subtract: "subtract",
    graph.multiply: "multiply",
    graph.divide: "divide",
    graph.equal: "equal",
    graph.not_equal: "not_equal",
    graph.less: "less",
    graph.less_equal: "less_equal",
    graph.greater: "greater",
    graph.greater_equal: "greater_equal",
    graph.logical_and: "logical_and",
    graph.logical_or: "logical_or",
    graph.logical_xor: "logical_xor",
    graph.power: "power",
    graph.maximum: "maximum",
    # unary
    graph.logical_not: "logical_not",
    graph.exp: "exp",
    graph.log: "log",
    # where
    graph.multi_clause_where: "select",
    graph.where: "where",
    # axis operations
    graph.expand_dims: "expand_dims",
    graph.reduce_sum: "sum",
    graph.reduce_mean: "mean",
}


def _node_name(node: graph.Node) -> str:
    return f"n{node.id}"


def _np_attr_name(name: str) -> ast.expr:
    return ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr=name, ctx=ast.Load())


def _axis_as_tuple(axis: graph.Axis) -> tuple[int, ...]:
    return (axis,) if isinstance(axis, int) else axis


def _np_ndarray_to_ast_expr(value: graph.Scalar | list) -> ast.expr:
    if isinstance(value, list):
        return ast.List(elts=[_np_ndarray_to_ast_expr(v) for v in value], ctx=ast.Load())
    else:
        return ast.Constant(value=value)


def graph_node_to_ast_stmt(node: graph.Node, value: np.ndarray | None) -> ast.stmt:
    """Transform a graph.Node to a Python ast.expr."""
    # 2. get the operation name
    try:
        opname = _operation_names[type(node)]
    except KeyError:
        raise UnsupportedOperation(f"jit: unsupported operation: {type(node)}")

    # 3. prepare for args and kwargs
    posargs: list[ast.expr] = []
    kwargs: list[ast.keyword] = []

    # 4. evaluate placeholders
    if isinstance(node, graph.placeholder):
        assert value is not None
        posargs.append(_np_ndarray_to_ast_expr(value.tolist()))

    # 5. evaluate constants
    elif isinstance(node, graph.constant):
        posargs.append(ast.Constant(value=node.value))

    # 6. evaluate unary operations
    elif isinstance(node, graph.UnaryOp):
        posargs.append(ast.Name(id=_node_name(node.node), ctx=ast.Load()))

    # 7. evaluate binary operations
    elif isinstance(node, graph.BinaryOp):
        posargs.append(ast.Name(id=_node_name(node.left), ctx=ast.Load()))
        posargs.append(ast.Name(id=_node_name(node.right), ctx=ast.Load()))

    # 8. evaluate where operations
    elif isinstance(node, graph.where):
        posargs.append(ast.Name(id=_node_name(node.condition), ctx=ast.Load()))
        posargs.append(ast.Name(id=_node_name(node.then), ctx=ast.Load()))
        posargs.append(ast.Name(id=_node_name(node.otherwise), ctx=ast.Load()))

    # 9. evaluate multi_clause_where
    elif isinstance(node, graph.multi_clause_where):
        condlist: list[ast.expr] = []
        choicelist: list[ast.expr] = []
        for cond, choice in node.clauses:
            condlist.append(ast.Name(id=_node_name(cond), ctx=ast.Load()))
            choicelist.append(ast.Name(id=_node_name(choice), ctx=ast.Load()))
        default: ast.expr = ast.Name(id=_node_name(node.default_value), ctx=ast.Load())
        posargs.extend([ast.List(condlist), ast.List(choicelist), default])

    # 10. evaluate axis operations
    elif isinstance(node, graph.AxisOp):
        posargs.append(ast.Name(id=_node_name(node.node), ctx=ast.Load()))
        kwargs.append(ast.keyword("axis", ast.Tuple(elts=[ast.Constant(value=x) for x in _axis_as_tuple(node.axis)])))

    # 11. catch all for not implemented operations
    else:
        raise UnsupportedNodeType(f"jit: unsupported node type: {type(node)}")

    # 12. create function call expr
    expr = ast.Call(func=_np_attr_name(opname), args=posargs, keywords=kwargs)

    # 13. assign the result of the function call
    assign = ast.Assign(
        targets=[ast.Name(id=_node_name(node), ctx=ast.Store())],
        value=expr,
    )

    # 14. Fixup the resulting piece of AST recursively
    ast.fix_missing_locations(assign)
    return assign
