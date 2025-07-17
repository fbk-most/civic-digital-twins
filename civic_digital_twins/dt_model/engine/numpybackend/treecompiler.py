"""Python-bytecode and Numba-JIT `forest.Tree` compiler.

The compiler takes in input a `forest.Tree` and executes the compilation
pipeline, which consists of the following stages:

    1. generate_ast: stops after converting the tree to Python AST
    2. python_compile: stops after compiling the tree to Python bytecode
    3. jit_compile: stops after setting up JIT compilation using Numba

The output of stage 2 or stage 3 can be executed using call_function.
"""

# SPDX-License-Identifier: Apache-2.0

from typing import Any, ItemsView, Protocol, cast, runtime_checkable
import ast
import types

import numba
import numpy as np

from ..frontend import forest, graph

# === generate_ast ===

class UnsupportedNodeType(Exception):
    """Raised when the executor encounters an unsupported node type."""


class UnsupportedOperation(Exception):
    """Raised when the executor encounters an unsupported operation."""


_operation_names: dict[type[graph.Node], "str"] = {
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


def _ast_generate_single_node(node: graph.Node) -> ast.expr:
    # 1. invariant: placeholders are arguments and should not appear here
    assert not isinstance(node, graph.placeholder)

    # 2. get the operation name
    try:
        opname = _operation_names[type(node)]
    except KeyError:
        raise UnsupportedOperation(f"astgen: unsupported operation: {type(node)}")

    # 3. prepare for args and kwargs
    posargs: list[ast.expr] = []
    kwargs: list[ast.keyword] = []

    # 4. evaluate constants
    if isinstance(node, graph.constant):
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
        raise UnsupportedNodeType(f"astgen: unsupported node type: {type(node)}")

    # 11. return the function call AST
    return ast.Call(func=_np_attr_name(opname), args=posargs, keywords=kwargs)


def _ast_generate_nodes(*nodes: graph.Node) -> list[ast.stmt]:
    stmts: list[ast.stmt] = []
    for node in nodes:
        # 1. skip placeholder nodes because they are injected into the context
        if isinstance(node, graph.placeholder):
            continue

        # 2. generate the expression for evaluating the node
        expr = _ast_generate_single_node(node)

        # 3. transform the expression for evaluating into a statement
        assign = ast.Assign(
            targets=[ast.Name(id=_node_name(node), ctx=ast.Store())],
            value=expr,
        )
        ast.fix_missing_locations(assign)
        stmts.append(assign)

    return stmts


def _ast_generate_tree_body(*nodes: graph.Node) -> list[ast.stmt]:
    # We need to add a return statement at the end
    assert len(nodes) > 0
    stmts = _ast_generate_nodes(*nodes)
    root = nodes[-1]  # by definition of tree this is the root node
    stmts.append(ast.Return(value=ast.Name(id=_node_name(root), ctx=ast.Load())))
    return stmts


def _tree_name(tree: forest.Tree) -> str:
    return f"t{tree.root().id}"


def generate_ast(tree: forest.Tree) -> ast.FunctionDef:
    """Transform a `forest.Tree` into an AST function definition.

    This function does not generate code for placeholders. The caller of the
    generated code will need to inject them at evaluation time.

    Args:
        tree: The tree to generate AST for.

    Raises
    ------
        UnsupportedNodeType: If the executor does not support the given node type.
        UnsupportedOperation: If the executor does not support a specific operation.
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


# === generate_function_type ===


@runtime_checkable
class State(Protocol):
    """Read-only executor state protocol."""

    def get_node_value(self, node: graph.Node) -> np.ndarray:
        """Return the value associate with the node or throws an exception."""
        ...

    def items(self) -> ItemsView[graph.Node, np.ndarray]:
        """Return all the values present in the state."""
        ...


def _mod_compile(stmts: list[ast.stmt], filename: str) -> types.CodeType:
    mod = ast.Module(body=stmts, type_ignores=list())
    return compile(source=mod, filename=filename, mode="exec")


def python_compile(state: State, tree: forest.Tree, filename: str = "<generated>") -> types.FunctionType:
    """Compile a `forest.Tree` into an the corresponding Python bytecode.

    This function uses `generate_ast` to generate the AST.

    We use the given state to provide the placeholder values as global variables.

    The returned function expects `np.ndarray` arguments in the same order
    in which inputs are listed in the tree instance.
    """
    code = _mod_compile([generate_ast(tree)], filename)
    context: dict[str, Any] = {"np": np}
    for node, value in state.items():
        context[f"n{node.id}"] = value
    exec(code, context)
    return cast(types.FunctionType, context[_tree_name(tree)])


def jit_compile(state: State, tree: forest.Tree, filename: str = "<generated>") -> types.FunctionType:
    """JIT compile a `forest.Tree` using Numba.

    This function uses `python_compile` to generate the Python function first
    and then wraps it with `numba.njit` to enable JIT compilation."""
    return numba.njit(python_compile(state, tree, filename))


def call_function(state: State, tree: forest.Tree, func: types.FunctionType) -> np.ndarray:
    """Call the given func associated with the given tree using the given state."""
    return func(*[state.get_node_value(node) for node in tree.inputs])
