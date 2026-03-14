"""NumPy AST code generator.

Transforms ``graph.Node`` objects into a Python AST that uses NumPy
function calls (e.g. ``graph.add`` → ``np.add``), and unparses the
AST back to Python source code.

The primary use case is debugging: the generated source shows the
exact NumPy calls that would evaluate a computation graph node,
making it easy to inspect and verify graph construction.
"""

# SPDX-License-Identifier: Apache-2.0

import ast
import types
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..frontend import forest, graph


class UnsupportedNodeArguments(Exception):
    """Raised when the AST generator does not know how to compile a node's arguments."""


class UnsupportedNodeType(Exception):
    """Raised when the AST generator does not know the NumPy function a node corresponds to."""


class _InternalTestingNode(graph.Node):
    """Node type used for writing tests."""


_operation_names: dict[type[graph.Node], str] = {
    # timeseries
    graph.timeseries_constant: "asarray",
    graph.timeseries_placeholder: "asarray",
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
    graph.negate: "negative",
    graph.logical_not: "logical_not",
    graph.exp: "exp",
    graph.log: "log",
    # where
    graph.multi_clause_where: "select",
    graph.where: "where",
    # axis operations
    graph.squeeze: "squeeze",
    graph.expand_dims: "expand_dims",
    graph.project_using_sum: "sum",
    graph.project_using_mean: "mean",
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
    """Transform a graph.Node to a Python AST assignment statement.

    The value is only required for placeholder nodes (``graph.placeholder``
    and ``graph.timeseries_placeholder``), whose value is known ahead of
    evaluation and is not embedded in the graph.  We verify this invariant
    at runtime.

    This function calls ast.fix_missing_locations before returning.
    """
    # 1. distinguish between user-defined functions and other nodes
    if isinstance(node, graph.function_call):
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


def _graph_function_to_ast_expr(node: graph.function_call) -> ast.expr:
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
    _placeholders = (graph.placeholder, graph.timeseries_placeholder)

    # 0. ensure value is only given for placeholder nodes
    assert (isinstance(node, _placeholders) and value is not None) or value is None

    # 1. get the operation name
    try:
        opname = _operation_names[type(node)]
    except KeyError:
        raise UnsupportedNodeType(f"numpy_ast: unsupported operation: {type(node)}")

    # 2. prepare for args and kwargs
    posargs: list[ast.expr] = []
    kwargs: list[ast.keyword] = []

    # 3. evaluate timeseries constants (values embedded in the node)
    if isinstance(node, graph.timeseries_constant):
        posargs.append(_np_ndarray_to_ast_expr(np.asarray(node.values).tolist()))

    # 4. evaluate placeholder nodes (value provided externally)
    elif isinstance(node, _placeholders):
        assert value is not None  # make the typechecker really happy
        posargs.append(_np_ndarray_to_ast_expr(value.tolist()))

    # 5. evaluate scalar constants
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
        if isinstance(node, (graph.project_using_sum, graph.project_using_mean)):
            kwargs.append(ast.keyword("keepdims", ast.Constant(value=True)))

    # 11. catch all for not implemented operations
    else:
        raise UnsupportedNodeArguments(f"numpy_ast: unsupported node type: {type(node)}")

    # 12. create function call expr
    return ast.Call(func=_np_attr_name(opname), args=posargs, keywords=kwargs)


def graph_node_to_numpy_code(node: graph.Node, value: np.ndarray | None = None) -> str:
    """Transform a node to NumPy source code.

    This is mainly useful for debugging: the returned string shows the
    exact NumPy call that would evaluate the given graph node.
    """
    return ast.unparse(graph_node_to_ast_stmt(node, value))


# === Forest / JIT layer ===


def _ast_generate_nodes(*nodes: graph.Node) -> list[ast.stmt]:
    return [graph_node_to_ast_stmt(node) for node in nodes]


def _ast_generate_tree_body(*nodes: graph.Node) -> list[ast.stmt]:
    assert len(nodes) > 0
    stmts = _ast_generate_nodes(*nodes)
    root = nodes[-1]  # by definition of tree this is the root node
    stmts.append(ast.Return(value=ast.Name(id=_node_name(root), ctx=ast.Load())))
    return stmts


def _tree_name(tree: forest.Tree) -> str:
    return f"t{tree.root().id}"


def forest_tree_to_ast_function_def(tree: forest.Tree) -> ast.FunctionDef:
    """Transform a ``forest.Tree`` into an AST function definition.

    Parameters
    ----------
    tree:
        The tree to transform.

    Returns
    -------
        An AST function definition whose arguments are the tree's inputs and
        whose body evaluates the tree's nodes, returning the root value.
    """
    assert len(tree.body) > 0 and tree.root() is tree.body[-1]  # tree invariant
    func = ast.FunctionDef(
        _tree_name(tree),
        args=ast.arguments(
            posonlyargs=list(),
            args=[ast.arg(f"n{x.id}", _np_attr_name("ndarray")) for x in tree.inputs],
            vararg=None,
            kwonlyargs=list(),
            kw_defaults=list(),
            kwarg=None,
            defaults=list(),
        ),
        body=_ast_generate_tree_body(*tree.body),
        decorator_list=list(),
        returns=_np_attr_name("ndarray"),
    )
    ast.fix_missing_locations(func)
    return func


@runtime_checkable
class State(Protocol):
    """Read-only executor state protocol used by the JIT layer."""

    def get_node_value(self, node: graph.Node) -> np.ndarray:
        """Return the value associated with the node or raise an exception."""
        ...  # pragma: no cover


def _mod_compile(stmts: list[ast.stmt], filename: str) -> types.CodeType:
    mod = ast.Module(body=stmts, type_ignores=list())
    return compile(source=mod, filename=filename, mode="exec")


def _python_compile_function(tree: forest.Tree, filename: str = "<generated>") -> types.FunctionType:
    code = _mod_compile([forest_tree_to_ast_function_def(tree)], filename)
    context: dict[str, Any] = {"np": np}
    exec(code, context)
    return context[_tree_name(tree)]  # type: ignore[return-value]


def compile_function(tree: forest.Tree, filename: str = "<generated>") -> types.FunctionType:
    """JIT-compile a ``forest.Tree`` using Numba.

    Parameters
    ----------
    tree:
        The tree to compile.
    filename:
        The filename to embed in the generated code object (for tracebacks).

    Returns
    -------
        A Numba-JIT-compiled callable that accepts the tree's input arrays and
        returns the root value.
    """
    import numba  # lazy import — numba is an optional dependency

    return numba.njit(_python_compile_function(tree, filename))  # type: ignore[return-value]


def call_function(state: State, tree: forest.Tree, func: types.FunctionType) -> np.ndarray:
    """Call a JIT-compiled tree function using values from *state*.

    Parameters
    ----------
    state:
        An executor state that provides ``get_node_value`` for each input.
    tree:
        The tree whose inputs should be fetched from *state*.
    func:
        The compiled function returned by :func:`compile_function`.

    Returns
    -------
        The computed root value.
    """
    return func(*[state.get_node_value(node) for node in tree.inputs])  # type: ignore[return-value]
