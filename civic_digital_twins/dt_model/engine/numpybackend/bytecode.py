"""Tree-shaped-graph bytecode based evaluator.

This module implements evaluation for a tree-shaped graph. You can obtain
a tree shaped graph using the `frontend.forest` module. This module partitions
the graph into independents trees. We transform a tree into Python AST using
the `numpybackend.astgen` module. Then we bytecode-compile the AST into a
function using `numpy` and we cache the code. Evaluation of a tree entails
invoking the corresponding bytecode function with the proper arguments.

Compared to the `backend.executor` evaluator, this evaluator evaluates each
tree as an atomic unit. Therefore, there is less transparency regarding
what happens inside each unit of bytecode. However, since we're using bytecode,
obviously this evaluator is potentially more efficient. Whether it is really
more efficient depends on the workload and may not be true for the common case
of small digital-twin models to evaluate (as of 2025-07-14).
"""

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import copy
import dis
import io
import threading
import types
from dataclasses import dataclass, field
from typing import Callable, Protocol, cast, runtime_checkable

import numpy as np

from ..frontend import forest, graph
from . import astgen, debug, executor

NodeValueNotFound = executor.NodeValueNotFound
"""Type alias re-exporting `executor.NodeValueNotFound`."""


@runtime_checkable
class StateMap(Protocol):
    """Protocol allowing `FunctionDef` to access the state."""

    def get_node_value(self, node: graph.Node) -> np.ndarray:
        """Return the value of the node or throws `NodeValueNotFound`."""
        ...


@dataclass(frozen=True)
class FunctionDef:
    """Bytecode-compiled function definition."""

    # The callable obtained evaluating the code
    callable: Callable[..., np.ndarray]

    # The bytecode for the compiled function
    code: types.CodeType

    # The original AST-level function definition
    fdef: astgen.FunctionDef

    def call(self, statemap: StateMap) -> np.ndarray:
        """Call the function with the given arguments and returns the evaluation result.

        The statemap MUST include a value for each node required by the function.
        """
        sorted_inputs = sorted(self.fdef.args.items(), key=lambda x: x[0].id)
        arg_values = [statemap.get_node_value(node) for node, _ in sorted_inputs]
        return self.callable(*arg_values)

    def disassemble(self) -> str:
        """Return a string containing the disassembled bytecode."""
        out = io.StringIO()
        dis.dis(self.code, file=out)
        return out.getvalue()


def compile_function_def(
    fdef: astgen.FunctionDef,
    *,
    filename: str = "<generated>",
    no_deep_copy: bool = False,
) -> FunctionDef:
    """Compile an `astgen.FunctionDef` to a `FunctionDef` ready to be evaluated."""
    # 1. optionally skip cloning the function AST before mutating it with fix_missing_locations
    astcode = fdef.func if no_deep_copy else copy.deepcopy(fdef.func)

    # 2. create a fake module for compiling the function to Python bytecode
    mod = ast.Module(body=[astcode], type_ignores=list())

    # 3. mutate the module by fixing the missing code locations
    ast.fix_missing_locations(mod)

    # 4. obtain the bytecode for the compiled function
    code = compile(source=mod, filename=filename, mode="exec")

    # 5. evaluate the bytecode in a namespace with numpy defined
    namespace = {"np": np}
    exec(code, namespace)

    # 6. obtain the function to execute based on its name
    # Note: it seems it's not possible in Python 3.11 to express a function
    # taking an arbitrary number of np.ndarray as parameters
    callable: Callable[..., np.ndarray] = cast(Callable[..., np.ndarray], namespace[fdef.name()])

    # 7. return the result to the caller
    return FunctionDef(code=code, fdef=fdef, callable=callable)


def compile_tree(
    tree: forest.Tree,
    *,
    filename: str = "<generated>",
    funcname: str = "",
) -> FunctionDef:
    """Compile a `forest.Tree` to a `FunctionDef` ready to be evaluated."""
    return compile_function_def(
        astgen.function_def(
            tree=tree,
            name=funcname,
        ),
        filename=filename,
        no_deep_copy=True,  # we don't need deepcopy since the AST is not shared
    )


@runtime_checkable
class CodeCache(Protocol):
    """Facility to cache compiled bytecode."""

    def compile(self, tree: forest.Tree) -> FunctionDef:
        """Compile a `forest.Tree` to a `FunctionDef` containing bytecode."""
        ...


class ThreadSafeCodeCache:
    """Code cache with thread safety guarantees."""

    def __init__(self):
        self.__cache: dict[forest.Tree, FunctionDef] = {}
        self.__lock = threading.Lock()

    def all(self) -> list[FunctionDef]:
        """Return a copy of the functions inside the cache."""
        result: list[FunctionDef] = []
        with self.__lock:
            result = list(self.__cache.values())
        return result

    def compile(self, tree: forest.Tree) -> FunctionDef:
        """Compile the given `Tree` to a `FunctionDef`.

        This function uses the internal codecache and memoizes the
        compiled code to avoid compiling multiple times.
        """
        with self.__lock:
            return self.__compile_locked(tree)

    def __compile_locked(self, tree: forest.Tree) -> FunctionDef:
        # 1. attempt to use the cache first
        fdef = self.__cache.get(tree)
        if fdef:
            return fdef

        # 2. otherwise actually compile to bytecode
        fdef = compile_tree(tree)
        self.__cache[tree] = fdef
        return fdef


@dataclass(frozen=True)
class State:
    """
    The graph executor state.

    Make sure to provide values for placeholder nodes ahead of the evaluation
    by initializing the `values` dictionary accordingly.

    Note that, if graph.NODE_FLAG_TRACE is set, the State will print the
    nodes provided to the constructor in its __post_init__ method.

    Attributes
    ----------
        values: A dictionary caching the result of the computation.
        flags: Bitmask containing debug flags (e.g., graph.NODE_FLAG_BREAK).
        codecache: Cache for storing byte-compiled functions.
    """

    values: dict[graph.Node, np.ndarray]
    flags: int = 0
    codecache: CodeCache = field(default_factory=ThreadSafeCodeCache)

    def get_node_value(self, node: graph.Node) -> np.ndarray:
        """Access the value associated with a node.

        Args:
            node: The node whose value to retrieve.

        Returns
        -------
            The value associated with the node.

        Raises
        ------
            NodeValueNotFound: If the node has not been evaluated.
        """
        try:
            return self.values[node]
        except KeyError:
            raise executor.NodeValueNotFound(f"executor: node '{node.name}' has not been evaluated")

    def compile(self, tree: forest.Tree) -> FunctionDef:
        """Compile the given `Tree` to a `FunctionDef`.

        This function uses the internal codecache and memoizes the
        compiled code to avoid compiling multiple times.
        """
        return self.codecache.compile(tree)


def evaluate_tree(state: State, tree: forest.Tree) -> np.ndarray:
    """Evaluate a tree given the current state.

    This function assumes you used `frontend.forest` to transform the
    graph into a set of `forest.Tree` trees to evaluate in order. If this
    is not the case, the evaluation will fail.

    Args:
        state: The current executor state.
        tree: The tree to evaluate.

    Raises
    ------
        NodeValueNotFound: If a dependent node has not been evaluated
            and therefore its value cannot be found in the state.
    """
    # TODO(bassosimone): the list of exceptions that could be raised here
    # is incomplete and, generally, we should improve the DX here

    # 1. check whether the tree root node has been already evaluated
    root = tree.root()
    if root in state.values:
        return state.values[root]

    # 2. check whether we need to trace this node
    flags = root.flags | state.flags
    tracing = flags & graph.NODE_FLAG_TRACE
    if tracing:
        debug.print_graph_node(root)

    # 3. possibly bytecode-compile the function associated with the tree
    # trying to use the cache if we have already compiled the node
    func = state.compile(tree)

    # 4. evaluate the node function proper
    result = func.call(state)

    # 5. check whether we need to print the computation result
    if tracing:
        debug.print_evaluated_node(result, cached=False)

    # 6. check whether we need to stop after evaluating this node
    if flags & graph.NODE_FLAG_BREAK != 0:
        input("executor: press any key to continue...")
        print("")

    # 7. store the node result in the state
    state.values[root] = result

    # 8. return the result
    return result
