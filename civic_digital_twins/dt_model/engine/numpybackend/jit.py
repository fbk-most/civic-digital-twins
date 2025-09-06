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


class UnsupportedNodeArguments(Exception):
    """Raised when the JIT compiler does not know how to AST-compile a node arguments."""


class UnsupportedNodeType(Exception):
    """Raised when the JIT compiler does not know the NumPy function a node corresponds to."""


class _InternalTestingNode(graph.Node):
    """Node type used for writing tests."""


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
    graph.squeeze: "squeeze",
    graph.expand_dims: "expand_dims",
    graph.reduce_sum: "sum",
    graph.reduce_mean: "mean",
    # internal
    _InternalTestingNode: "_internal_testing",
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


def graph_node_to_ast_stmt(node: graph.Node, value: np.ndarray | None = None) -> ast.stmt:
    """Transform a graph.Node to a Python ast.ast.

    The value is only required for placeholders, whose value is known
    ahead of evaluation and isn't embedded into the graph. We verify
    whether this is the case with a runtime assertion checking that the
    value is only specified for placeholders.

    This function is useful to transform multiple nodes to valid
    Python AST, which in turn can be JIT compiled with Numba.

    This function calls ast.fix_missing_locations before returning.
    """
    # 1. distinguish between user-defined functions and other nodes
    if isinstance(node, graph.function):
        assert value is None
        expr = _graph_function_to_ast_expr(node)
    else:
        expr = _simple_graph_node_to_ast_expr(node, value)

    # 2. assign the result of the function call
    assign = ast.Assign(
        targets=[ast.Name(id=_node_name(node), ctx=ast.Store())],
        value=expr,
    )

    # 3. Fixup the resulting piece of AST recursively
    ast.fix_missing_locations(assign)
    return assign


def _graph_function_to_ast_expr(node: graph.function) -> ast.expr:
    # 1. get the operation name
    opname = node.name

    # 2. prepare for args and kwargs
    posargs: list[ast.expr] = []
    kwargs: list[ast.keyword] = []

    # 3. fill the positional arguments
    for argument in node.args:
        posargs.append(ast.Name(id=_node_name(argument), ctx=ast.Load()))

    # 4. fill the keyword arguments
    for key, value in node.kwargs.items():
        kwargs.append(ast.keyword(key, ast.Name(id=_node_name(value), ctx=ast.Load())))

    # 5. create function call expr
    return ast.Call(func=ast.Name(id=opname, ctx=ast.Load()), args=posargs, keywords=kwargs)


def _simple_graph_node_to_ast_expr(node: graph.Node, value: np.ndarray | None = None) -> ast.expr:
    # 0. ensure value is only given for placeholders
    assert (isinstance(node, graph.placeholder) and value is not None) or value is None

    # 1. get the operation name
    try:
        opname = _operation_names[type(node)]
    except KeyError:
        raise UnsupportedNodeType(f"jit: unsupported operation: {type(node)}")

    # 2. prepare for args and kwargs
    posargs: list[ast.expr] = []
    kwargs: list[ast.keyword] = []

    # 3. evaluate placeholders
    if isinstance(node, graph.placeholder):
        assert value is not None  # make the typechecker really happy
        posargs.append(_np_ndarray_to_ast_expr(value.tolist()))

    # 4. evaluate constants
    elif isinstance(node, graph.constant):
        posargs.append(ast.Constant(value=node.value))

    # 5. evaluate unary operations
    elif isinstance(node, graph.UnaryOp):
        posargs.append(ast.Name(id=_node_name(node.node), ctx=ast.Load()))

    # 6. evaluate binary operations
    elif isinstance(node, graph.BinaryOp):
        posargs.append(ast.Name(id=_node_name(node.left), ctx=ast.Load()))
        posargs.append(ast.Name(id=_node_name(node.right), ctx=ast.Load()))

    # 7. evaluate where operations
    elif isinstance(node, graph.where):
        posargs.append(ast.Name(id=_node_name(node.condition), ctx=ast.Load()))
        posargs.append(ast.Name(id=_node_name(node.then), ctx=ast.Load()))
        posargs.append(ast.Name(id=_node_name(node.otherwise), ctx=ast.Load()))

    # 8. evaluate multi_clause_where
    elif isinstance(node, graph.multi_clause_where):
        condlist: list[ast.expr] = []
        choicelist: list[ast.expr] = []
        for cond, choice in node.clauses:
            condlist.append(ast.Name(id=_node_name(cond), ctx=ast.Load()))
            choicelist.append(ast.Name(id=_node_name(choice), ctx=ast.Load()))
        default: ast.expr = ast.Name(id=_node_name(node.default_value), ctx=ast.Load())
        posargs.extend([ast.List(condlist), ast.List(choicelist), default])

    # 9. evaluate axis operations
    elif isinstance(node, graph.AxisOp):
        posargs.append(ast.Name(id=_node_name(node.node), ctx=ast.Load()))
        kwargs.append(ast.keyword("axis", ast.Tuple(elts=[ast.Constant(value=x) for x in _axis_as_tuple(node.axis)])))

    # 10. catch all for not implemented operations
    else:
        raise UnsupportedNodeArguments(f"jit: unsupported node type: {type(node)}")

    # 11. create function call expr
    return ast.Call(func=_np_attr_name(opname), args=posargs, keywords=kwargs)


def graph_node_to_numpy_code(node: graph.Node, value: np.ndarray | None = None) -> str:
    """Transform a node to numpy source code.

    This functionality is mainly useful for debugging and for compiling
    the graph to Python source code.
    """
    return ast.unparse(graph_node_to_ast_stmt(node, value))
