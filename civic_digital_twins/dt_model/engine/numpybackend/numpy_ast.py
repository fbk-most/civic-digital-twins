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
from collections.abc import Sequence

import numpy as np

from ...axes import domain_axis_position
from ..frontend import graph
from .kernels import laplacian as _laplacian  # noqa: F401 (re-exported: generated code calls it by this name)
from .kernels import shift as _shift  # noqa: F401 (re-exported: generated code calls it by this name)


class UnsupportedNodeArguments(Exception):
    """Raised when the AST generator does not know how to compile a node's arguments."""


class UnsupportedNodeType(Exception):
    """Raised when the AST generator does not know the NumPy function a node corresponds to."""


class _InternalTestingNode(graph.Node):
    """Node type used for writing tests."""


_operation_names: dict[type[graph.Node], str] = {
    # domain-carrying arrays
    graph.array_constant: "asarray",
    graph.array_placeholder: "asarray",
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
    # projection operations
    graph.project_using_sum: "sum",
    graph.project_using_mean: "mean",
    graph.project_using_min: "min",
    graph.project_using_max: "max",
    graph.project_using_std: "std",
    graph.project_using_var: "var",
    graph.project_using_median: "median",
    graph.project_using_prod: "prod",
    graph.project_using_any: "any",
    graph.project_using_all: "all",
    graph.project_using_count_nonzero: "count_nonzero",
    graph.project_using_quantile: "quantile",
    # axis operations
    graph.roll: "roll",
    graph.cumulative: "cumsum",
    graph.shift: "shift",
    graph.gradient: "gradient",
    graph.laplacian: "laplacian",
    # internal
    _InternalTestingNode: "_internal_testing",
}

_BARE_NAME_OPERATIONS: tuple[type[graph.Node], ...] = (graph.shift, graph.laplacian)
"""Node types whose generated call targets a bare, underscore-prefixed name
(``_<name>``) rather than ``np.<name>``.

``shift`` (fill-padded, unlike the circular ``roll``) and ``laplacian`` (a
multi-axis finite-difference stencil) have no single-call NumPy equivalent,
so they render as calls to :func:`kernels.shift`/:func:`kernels.laplacian`
(imported above as ``_shift``/``_laplacian``) instead — mirroring how
:func:`_graph_function_to_ast_expr` already renders user-defined functions
as bare-name calls. The leading underscore is added at the call site (see
"create function call expr" below), not stored in ``_operation_names``. See
``kernels.py`` for why the underlying logic lives in one shared module
rather than being reimplemented here."""


def _node_name(node: graph.Node) -> str:
    return f"n{node.id}"


def _np_attr_name(name: str) -> ast.expr:
    return ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr=name, ctx=ast.Load())


def _axis_as_tuple(axis: graph.Axis, domain_axes: Sequence[graph.Axis]) -> tuple[int, ...]:
    try:
        return (domain_axis_position(domain_axes, axis),)
    except ValueError:
        raise UnsupportedNodeArguments(
            f"numpy_ast: numpybackend only supports projection along this evaluation's DOMAIN axes "
            f"{[ax.name for ax in domain_axes]}; got {axis!r}"
        ) from None


def _np_ndarray_to_ast_expr(value: graph.Scalar | list) -> ast.expr:
    if isinstance(value, list):
        return ast.List(elts=[_np_ndarray_to_ast_expr(v) for v in value], ctx=ast.Load())
    else:
        return ast.Constant(value=value)


def graph_node_to_ast_stmt(
    node: graph.Node,
    value: np.ndarray | None = None,
    *,
    domain_axes: Sequence[graph.Axis] = (),
) -> ast.stmt:
    """Transform a graph.Node to a Python AST assignment statement.

    The value is only required for placeholder nodes (``graph.placeholder``
    and ``graph.array_placeholder``), whose value is known ahead of
    evaluation and is not embedded in the graph.  We verify this invariant
    at runtime.

    This function calls ast.fix_missing_locations before returning.
    """
    # 1. distinguish between user-defined functions and other nodes
    if isinstance(node, graph.function_call):
        assert value is None
        expr = _graph_function_to_ast_expr(node)
    else:
        expr = _simple_graph_node_to_ast_expr(node, value, domain_axes)

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


def _simple_graph_node_to_ast_expr(
    node: graph.Node,
    value: np.ndarray | None = None,
    domain_axes: Sequence[graph.Axis] = (),
) -> ast.expr:
    _placeholders = (graph.placeholder, graph.array_placeholder)

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
    if isinstance(node, graph.array_constant):
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

    # 10. evaluate projection operations
    elif isinstance(node, graph.ProjectionOp):
        posargs.append(ast.Name(id=_node_name(node.node), ctx=ast.Load()))
        if isinstance(node, graph.project_using_quantile):
            # For quantile, the q parameter comes first
            posargs.insert(0, ast.Constant(value=node.q))
        positions = _axis_as_tuple(node.axis, domain_axes)
        kwargs.append(ast.keyword("axis", ast.Tuple(elts=[ast.Constant(value=x) for x in positions])))
        if isinstance(
            node,
            (
                graph.project_using_sum,
                graph.project_using_mean,
                graph.project_using_min,
                graph.project_using_max,
                graph.project_using_std,
                graph.project_using_var,
                graph.project_using_median,
                graph.project_using_prod,
                graph.project_using_any,
                graph.project_using_all,
                graph.project_using_count_nonzero,
                graph.project_using_quantile,
            ),
        ):
            kwargs.append(ast.keyword("keepdims", ast.Constant(value=True)))

    # 11. evaluate axis operations (shape-preserving)
    elif isinstance(node, graph.AxisOp):
        posargs.append(ast.Name(id=_node_name(node.node), ctx=ast.Load()))
        (position,) = _axis_as_tuple(node.axis, domain_axes)
        if isinstance(node, (graph.shift, graph.roll)):
            posargs.append(ast.Constant(value=node.periods))
        if isinstance(node, graph.gradient):
            posargs.append(ast.Constant(value=node.spacing))
        kwargs.append(ast.keyword("axis", ast.Constant(value=position)))
        if isinstance(node, graph.shift):
            kwargs.append(ast.keyword("fill_value", ast.Constant(value=node.fill_value)))

    # 12. evaluate laplacian (shape-preserving, multi-axis)
    elif isinstance(node, graph.laplacian):
        posargs.append(ast.Name(id=_node_name(node.node), ctx=ast.Load()))
        positions = tuple(_axis_as_tuple(ax, domain_axes)[0] for ax in node.axes)
        kwargs.append(ast.keyword("axes", ast.Tuple(elts=[ast.Constant(value=p) for p in positions])))
        kwargs.append(ast.keyword("spacings", ast.Tuple(elts=[ast.Constant(value=s) for s in node.spacings])))
        kwargs.append(ast.keyword("boundaries", ast.Tuple(elts=[ast.Constant(value=b) for b in node.boundaries])))

    # 13. catch all for not implemented operations
    else:
        raise UnsupportedNodeArguments(f"numpy_ast: unsupported node type: {type(node)}")

    # 14. create function call expr
    is_bare_name = type(node) in _BARE_NAME_OPERATIONS
    func = ast.Name(id=f"_{opname}", ctx=ast.Load()) if is_bare_name else _np_attr_name(opname)
    return ast.Call(func=func, args=posargs, keywords=kwargs)


def graph_node_to_numpy_code(
    node: graph.Node,
    value: np.ndarray | None = None,
    *,
    domain_axes: Sequence[graph.Axis] = (),
) -> str:
    """Transform a node to NumPy source code.

    This is mainly useful for debugging: the returned string shows the
    exact NumPy call that would evaluate the given graph node.
    """
    return ast.unparse(graph_node_to_ast_stmt(node, value, domain_axes=domain_axes))
