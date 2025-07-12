"""
Transform a `forest.Tree` into an `ast.ASTFunctionDef` wrapped by a `ASTFunctionDef`.

The generated `ast.ASTFunctionDef` does not depend on numpy proper. We only assume that
numpy-compatible names will be available in scope under the `np` namespace.

The generated AST code can then be compiled to Python bytecode or transpiled to
more efficient representations, e.g., using Numba.
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
from dataclasses import dataclass

from ..frontend import forest, graph

FunctionArgs = dict[graph.Node, str]
"""Map a node to the corresponding argument name."""


@dataclass(frozen=True)
class FunctionDef:
    """Function evaluating a `compile.Tree` root node value."""

    # Maps nodes to argument names
    args: FunctionArgs

    # AST of the function definition
    func: ast.FunctionDef

    # Tree from which we generated
    tree: forest.Tree

    def name(self) -> str:
        """Return the function name."""
        return self.func.name

    def unparse(self) -> str:
        """Return the Python source code implementing this function."""
        return ast.unparse(self.func)


def function_def(tree: forest.Tree, name="") -> FunctionDef:
    """Transform a `forest.Tree` into a `ASTFunctionDef` wrapping Python AST."""
    # 1. ensure that the tree has been correctly compiled
    assert len(tree.nodes) > 0 and tree.root() is tree.nodes[-1]

    # 2. auto-assign the function name if not assigned
    if not name:
        name = f"t{tree.root().id}"

    # 3. create the arguments mapping
    args = _new_function_args(tree)

    # 4. build the AST function definition
    return FunctionDef(
        args=args,
        func=ast.FunctionDef(
            name,
            args=ast.arguments(
                posonlyargs=list(),
                args=[ast.arg(x, _np_attr_name("ndarray")) for x in args.values()],
                vararg=None,
                kwonlyargs=list(),
                kw_defaults=list(),
                kwarg=None,
                defaults=list(),
            ),
            body=_new_function_body(args, tree.nodes),
            decorator_list=list(),
            returns=_np_attr_name("ndarray"),
        ),
        tree=tree,
    )


def _node_name(node: graph.Node) -> str:
    return f"n{node.id}"


def _new_function_args(tree: forest.Tree) -> FunctionArgs:
    return {x: _node_name(x) for x in sorted(tree.inputs, key=lambda x: x.id)}


def _new_function_body(args: FunctionArgs, nodes: list[graph.Node]) -> list[ast.stmt]:
    # 1. more sanity checks on the availability of nodes
    assert len(nodes) > 0

    # 2. process each node in the topological sorting
    statements: list[ast.stmt] = []
    last: ast.expr | None = None
    for idx, node in enumerate(nodes):
        # 2.1. if the node is given as argument, nothing to do
        if node in args:
            continue

        # 2.2. otherwise we need to evaluate the node
        expr = _eval_node(node)

        # 2.3. assign the node to a local value
        assign = ast.Assign(
            targets=[ast.Name(id=_node_name(node), ctx=ast.Store())],
            value=expr,
        )
        last = expr

        # 2.4. remember the statement
        if idx < len(nodes) - 1:
            statements.append(assign)

    # 3. ensure we have an expression to return
    assert last is not None

    # 4. create the return statement
    statements.append(ast.Return(value=last))

    # 5. return the body
    return statements


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
    graph.where: "where",
    # axis operations
    graph.expand_dims: "expand_dims",
    graph.reduce_sum: "sum",
    graph.reduce_mean: "mean",
}
"""Maps graph operations to their numpy names."""


def _eval_node(node: graph.Node) -> ast.expr:
    # 1. invariant: placeholders are arguments and should not appear here
    assert not isinstance(node, graph.placeholder)

    # 2. get the operation name
    opname = _operation_names[type(node)]  # KeyError if missing

    # 3. prepare for args and kwargs
    posargs: list[ast.expr] = []
    kwargs: list[ast.keyword] = []

    # 4. evaluate constants
    if isinstance(node, graph.constant):
        posargs.append(ast.Constant(value=node.value))

    # 5. evaluate unary operations
    elif isinstance(node, graph.UnaryOp):
        posargs.append(
            ast.Name(
                id=_node_name(node.node),
                ctx=ast.Load(),
            ),
        )

    # 6. evaluate binary operations
    elif isinstance(node, graph.BinaryOp):
        posargs.append(
            ast.Name(
                id=_node_name(node.left),
                ctx=ast.Load(),
            ),
        )
        posargs.append(
            ast.Name(
                id=_node_name(node.right),
                ctx=ast.Load(),
            ),
        )

    # 7. evaluate where operations
    elif isinstance(node, graph.where):
        posargs.append(
            ast.Name(
                id=_node_name(node.condition),
                ctx=ast.Load(),
            ),
        )
        posargs.append(
            ast.Name(
                id=_node_name(node.then),
                ctx=ast.Load(),
            ),
        )
        posargs.append(
            ast.Name(
                id=_node_name(node.otherwise),
                ctx=ast.Load(),
            ),
        )

    # 8. evaluate multi_clause_where
    elif isinstance(node, graph.multi_clause_where):
        # TODO(bassosimone): implement this
        raise NotImplementedError

    # 9. evaluate axis operations
    elif isinstance(node, graph.AxisOp):
        posargs.append(
            ast.Name(
                id=_node_name(node.node),
                ctx=ast.Load(),
            ),
        )
        kwargs.append(
            ast.keyword(
                "axis",
                ast.Tuple(elts=[ast.Constant(value=x) for x in _axis_as_tuple(node.axis)]),
            ),
        )

    # 10. catch all for not implemented operations
    else:
        raise NotImplementedError

    # 11. return the function call AST
    return ast.Call(
        func=_np_attr_name(opname),
        args=posargs,
        keywords=kwargs,
    )


def _np_attr_name(name: str) -> ast.expr:
    return ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr=name, ctx=ast.Load())


def _axis_as_tuple(axis: graph.Axis) -> tuple[int, ...]:
    if isinstance(axis, int):
        axis = (axis,)
    return axis
